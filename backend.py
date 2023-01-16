import re
from collections import defaultdict
import numpy as np
from loader import BucketIndexLoader
from helperClasses import QueryProcessing, Calculator, ResultProcessor
from BM25 import BM25
from time import time
import hashlib

epsilon = .0000001


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
start = time()
bucket_loader = BucketIndexLoader("project_bucket_316533942")
print(f"loading indices -- {time() - start}")
text_inverted_index, title_inverted_index, anchor_inverted_index = bucket_loader.load_all_indices()
print(f"loading pageviews -- {time() - start}")
page_views = bucket_loader.loda_page_views()
print(f"normalizing pageviews -- {time() - start}")
page_views_norm = QueryProcessing.normalize_pageviews(page_views)
print(f"loading pagerank -- {time() - start}")
page_rank, pr_norm = bucket_loader.load_page_rank_to_df()
page_rank = page_rank.to_dict(defaultdict(int))
pr_norm = pr_norm.to_dict(defaultdict(int))
print(f"loading other dictionaries -- {time() - start}")
doc_titles = bucket_loader.load_doc_titles()
doc_norms = bucket_loader.load_doc_norms()
DL = bucket_loader.load_doc_len()


class SearchHandler:
    @staticmethod
    def search_page_view(doc_ids, normalized=False):
        """
        Handles the `get_pageview` request from the frontend.
        :param doc_ids: List of doc_id values, that we want to find their matching page views value.
        :param normalized: Boolean, determines if to use the normalized page views dictionary.
        :return: List of matching page views.
        """
        if normalized:
            res = [page_views_norm.get(doc_id, 0) for doc_id in doc_ids]
        else:
            res = [page_views.get(doc_id, 0) for doc_id in doc_ids]
        return res

    @staticmethod
    def search_page_rank(doc_ids, normalized=False):
        """
        Handles the `get_pagerank` request from the frontend.
        :param doc_ids: List of doc_id values, that we want to find their matching page rank value.
        :param normalized: Boolean, determines if to use the normalized page views dictionary.
        :return: List of matching page views.
        """
        if normalized:
            res = [page_views_norm[doc_id] for doc_id in doc_ids]
        else:
            res = [page_rank[doc_id] for doc_id in doc_ids]
        return res

    @staticmethod
    def search_title(query):
        """
        Handles the `search_title` request from the frontend.
        :param query: String - the query to be searched.
        :return: List of (doc_id, doc_title) of the title result.
        """
        res_dict = defaultdict(int)
        query_tokens = [token.group() for token in RE_WORD.finditer(query.lower())]  # Tokenize the query
        dist_query_list = []
        for word in query_tokens:
            if word not in dist_query_list:
                dist_query_list.append(word)  # get a distinct token query
        for token in query_tokens:
            if token not in title_inverted_index.posting_locs:
                continue
            pl = BucketIndexLoader.load_posting_lists_for_token(token, title_inverted_index,  # load posting list
                                                                "title_inverted_index_with_stemming")
            for doc_id, _ in pl:
                if doc_id in doc_titles:
                    res_dict[doc_id] += 1
        res = [doc_id for doc_id, val in res_dict.items() if val > 0]
        res = sorted(res, reverse=True, key=lambda x: res_dict[x])
        res = [(doc_id, doc_titles[doc_id]) for doc_id in res]
        return res

    @staticmethod
    def search_anchor(query):
        """
        Handles the `search_anchor` request from the frontend.
        :param query: String - the query to be searched.
        :return: List of (doc_id, doc_title) of the anchor result.
        """
        res_dict = defaultdict(int)
        query_tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
        for token in query_tokens:
            if token not in title_inverted_index.posting_locs:
                continue
            pl = BucketIndexLoader.load_posting_lists_for_token(token, anchor_inverted_index, "anchor_inverted_index")
            for doc_id, _ in pl:
                if doc_id in doc_titles:
                    res_dict[doc_id] += 1
        res = [doc_id for doc_id, val in res_dict.items() if val > 0]
        res = sorted(res, reverse=True, key=lambda x: res_dict[x])
        res = [(doc_id, doc_titles[doc_id]) for doc_id in res]
        return res

    @staticmethod
    def search_body(query):
        """
        Handles the `search_body` request from the frontend.
        :param query: String - the query to be searched.
        :return: List of (doc_id, doc_title) of the body result.
        """
        query_tokens = QueryProcessing.tokenize_ass3(query)
        Q = Calculator.get_tfidf_for_query(query_tokens, text_inverted_index)
        D = SearchHandler.get_candidate_docs_with_scores(query_tokens, text_inverted_index)
        sims = Calculator.dict_cosine_similarity(D, Q, doc_norms)
        top_100 = ResultProcessor.get_top_n(sims)
        res = [(tup[0], doc_titles[tup[0]]) for tup in top_100]
        return res

    @staticmethod
    def get_candidate_docs_with_scores(query, index):
        """
        Returns a dictionary of doc_id: tfidf_score, of relevant documents for the query.
        :param query: List of tokens.
        :param index: InvertedIndex object.
        :param DL: Dictionary - doc_id: doc_len.
        :return: dictionary of doc_id: tfidf_score
        """
        res = {}
        for token in np.unique(query):
            if token in index.df.keys():
                pl = BucketIndexLoader.load_posting_lists_for_token(token, index, "text_inverted_index")
                for doc_id, tf in pl:
                    res[doc_id] = {}
                    normalized_tfidf = (tf / DL.get(doc_id, 1 / epsilon)) * (
                        np.log10((len(DL) / index.df.get(token, 1 / epsilon))))
                    res[doc_id][token] = normalized_tfidf
        return res

    @staticmethod
    def search(query):
        """
        Handles the `search` request from the frontend. Using optimized combination of BM25 models and other parameters.
        :param query: List of tokens.
        :return: List of (doc_id, doc_title) of the best results.
        """
        tokenized_query = QueryProcessing.tokenize_with_stem(query)
        body_bm25 = BM25(text_inverted_index, DL, k1=5, b=0.2,
                         folder_name="text_inverted_index")
        body_res = body_bm25.search(tokenized_query)
        body_res = body_bm25.normalize_score(body_res)
        title_bm25 = BM25(title_inverted_index, DL, k1=2, b=0.05,
                          folder_name="title_inverted_index_with_stemming")
        title_res = title_bm25.search(tokenized_query)
        title_res = title_bm25.normalize_score(title_res)
        page_rank_res = SearchHandler.search_page_rank(
            set([doc_id for doc_id, _ in body_res] + [doc_id for doc_id, _ in title_res]), normalized=True)
        page_rank_res = [(body_res[i][0], page_rank_res[i]) for i in range(len(body_res))]
        page_views_res = SearchHandler.search_page_view(
            set([doc_id for doc_id, _ in body_res] + [doc_id for doc_id, _ in title_res]), normalized=True)
        page_views_res = [(body_res[i][0], page_views_res[i]) for i in range(len(body_res))]
        ws = [0.6, 0.3, 0.15, 0.15]
        N = min([10 * len(tokenized_query), 30])
        res = SearchHandler.multi_merge_results([body_res, title_res, page_rank_res, page_views_res], ws)[:N]
        res = [(tup[0], doc_titles[tup[0]]) for tup in res]
        return res

    @staticmethod
    def multi_merge_results(scores, weights, N=100):
        """
        Merging the results of multiple sources to one list.
        :param scores: List of lists, each contains (doc_id, score) tuples.
        :param weights: List of matching weights.
        :param N: Maximum number of results.
        :return: List of the merged results, with maximum length of N.
        """
        merged_scores = defaultdict(int)
        for tuples_list, weight in zip(scores, weights):
            for doc_id, score in tuples_list:
                merged_scores[doc_id] += score * weight

        merged_list = [(k, v) for k, v in sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)]

        return merged_list[:N]

    @staticmethod
    def search_config(query, config):
        """
        Allows search using different configurations.
        :param query: List of tokens.
        :param config: Dictionary, specifying the configuration.
        :return: List of merged results of the given configuration.
        """
        tokenized_query = QueryProcessing.tokenize_with_stem(query)
        body_bm25 = BM25(text_inverted_index, DL, k1=config['body_k'], b=config['body_b'],
                         folder_name="text_inverted_index")
        body_res = body_bm25.search(tokenized_query)
        body_res = body_bm25.normalize_score(body_res)
        title_bm25 = BM25(title_inverted_index, DL, k1=config['title_k'], b=config['title_b'],
                          folder_name="title_inverted_index_with_stemming")
        title_res = title_bm25.search(tokenized_query)
        title_res = title_bm25.normalize_score(title_res)
        page_rank_res = SearchHandler.search_page_rank(set([doc_id for doc_id, _ in body_res] +
                                                           [doc_id for doc_id, _ in title_res]), normalized=True)
        page_rank_res = [(body_res[i][0], page_rank_res[i]) for i in range(len(body_res))]
        page_views_res = SearchHandler.search_page_view(
            set([doc_id for doc_id, _ in body_res] + [doc_id for doc_id, _ in title_res]), normalized=True)
        page_views_res = [(body_res[i][0], page_views_res[i]) for i in range(len(body_res))]
        ws = [config['body_w'], config['title_w'], config['page_rank_w'], config['page_views_w']]
        N = min([10 * len(tokenized_query), 30])
        res = SearchHandler.multi_merge_results([body_res, title_res, page_rank_res, page_views_res], ws)[:N]
        res = [(tup[0], doc_titles[tup[0]]) for tup in res]
        return res
