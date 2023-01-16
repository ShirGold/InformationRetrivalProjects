from collections import Counter, defaultdict
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from inverted_index_gcp import InvertedIndex
# import gensim.downloader as api
from sklearn.preprocessing import MinMaxScaler
epsilon = .0000001


class QueryProcessing:

    @staticmethod
    def tokenize_ass3(query):
        """
        Tokenizes a query based on the tokenizer from assignment 3.
        :param query: String, query to be searched.
        :return: List of individual tokens.
        """
        english_stopwords = frozenset(stopwords.words('english'))
        toked_q = query.lower().split()
        new_query = []
        for token in toked_q:
            if token in english_stopwords:
                continue
            new_query.append(token)
        return new_query

    @staticmethod
    def tokenize_with_stem(query):
        """
        Tokenizes a query using SnowBall stemmer.
        :param query: String, query to be searched.
        :return: List of individual tokens.
        """
        english_stopwords = frozenset(stopwords.words('english'))
        snowball_stemmer = SnowballStemmer(language='english')
        toked_q = query.lower().split()
        new_query = []
        for token in toked_q:
            if token in english_stopwords:
                continue
            if token.endswith('?') or token.endswith('!'):
                token = token[:-1]
            new_query.append(snowball_stemmer.stem(token))
        return new_query

    @staticmethod
    def normalize_pageviews(page_views: dict):
        """
        Creates a normalized page views dictionary using MinMaxScaler.
        :param page_views: dictionary - doc_id: page_view
        :return: dictionary - doc_id: normalized_page_view
        """
        views = pd.Series(page_views)
        norm_views = pd.Series(MinMaxScaler().fit_transform(np.array(views).reshape(-1, 1))[:, 0])
        norm_views.index = views.index
        return norm_views.to_dict(defaultdict(int))


class Calculator:
    @staticmethod
    def get_tfidf_for_query(query, index: InvertedIndex):
        """
        Calculates the TfIdf value for a query.
        :param query: list of tokens.
        :param index: InvertedIndex object.
        :return: dictionary - term: tfidf
        """
        Q = {}
        counter = Counter(query)
        for token in np.unique(query):
            if token in index.df.keys():  # avoid terms that do not appear in the index.
                tf = counter.get(token, 0) / len(query)  # term frequency divided by the length of the query
                df = index.df.get(token, 0)
                idf = np.log10((len(index.df)) / (df + epsilon))  # smoothing
                Q[token] = tf * idf
        return Q

    @staticmethod
    def calc_idf_for_token(token, index: InvertedIndex):
        """
        Calculates the idf value for a given token.
        :param token: String.
        :param index: InvertedIndex.
        :return: idf score for the given token.
        """
        if token not in index.df.keys():
            return 0
        eps = 10 ** -6
        idf = np.log10(len(index.df) / index.df[token] + eps)
        return idf

    @staticmethod
    def dict_cosine_similarity(D: dict, Q: dict, doc_norms: dict):
        """
        Calculates the cosine similarity between the query and the documents, represented as dictionaries.
        :param D: Documents dictionary - doc_id: {token: tfidf}
        :param Q: Query dictionary - token: tfidf
        :param doc_norms: Dictionary containing precalculated documents normals.
        :return: dictionary of cosine similarity between the document and the query.
        """
        sims = {}
        for doc_id in D.keys():
            dot = 0
            for token in Q.keys():
                if token in D[doc_id]:
                    dot += Q[token] * D[doc_id][token]
            sims[doc_id] = dot / (doc_norms[doc_id] * np.linalg.norm(list(Q.values())))
        return sims


class ResultProcessor:
    @staticmethod
    def get_top_n(sim_dict, N=100):
        """
        Sorting and slicing a list of (doc_id, score) tuples.
        :param sim_dict: Dictionary of doc_id: score.
        :param N: Number of results wanted.
        :return: Sorted list of (doc_id, score) with the size of N.
        """
        return sorted([(doc_id, np.round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                      reverse=True)[:N]
