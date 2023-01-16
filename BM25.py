from helperClasses import *
from loader import BucketIndexLoader
from sklearn.preprocessing import MaxAbsScaler


class BM25:
    def __init__(self, index: InvertedIndex, DL, k1=1.5, b=0.75, folder_name="text_inverted_index"):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(DL)
        self.DL = DL
        self.AVGDL = sum(DL.values()) / self.N
        self.idf = None
        self.freqs = {}
        self.folder_name = folder_name

    def calc_idf(self, list_of_tokens):
        """
        Calculates the idf score for given tokens.
        :param list_of_tokens: List of tokens.
        :return: Dictionary - token: idf_score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df.get(term, 0.5-self.N)
                idf[term] = np.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def get_candidate_docs(self, query, index):
        """
        Returns all the possible relevant documents for the query.
        :param query: List of tokens.
        :param index: InvertedIndex object.
        :return: Set of relevant doc_ids
        """
        res = set()
        for token in np.unique(query):
            if token in index.df.keys():
                pl = BucketIndexLoader.load_posting_lists_for_token(token, index, self.folder_name)
                self.freqs[token] = {}
                for doc_id, tf in pl:
                    self.freqs[token][doc_id] = tf
                    res.add(doc_id)
        return res

    def search(self, tokenized_query, N=100):
        """
        Searches for the best matches for the query.
        :param tokenized_query: List of tokens.
        :param N: Maximum length of the result.
        :return: Sorted list of the result, with maximum length of N.
        """
        self.idf = self.calc_idf(tokenized_query)
        candid_list = self.get_candidate_docs(tokenized_query, self.index)
        res = sorted([(doc_id, self._score(tokenized_query, doc_id)) for doc_id in candid_list], key=lambda x: x[1],
                     reverse=True)[:N]
        return res

    def _score(self, query, doc_id):
        """
        Scoring a document based on the BM25 scoring formula.
        :param query: List of tokens.
        :param doc_id: The document to be scored.
        :return: float - The score of the document.
        """
        score = 0.0
        doc_len = self.DL.get(doc_id, 0)

        for term in query:
            if term in self.freqs.keys():
                if doc_id in self.freqs[term]:
                    freq = self.freqs[term][doc_id]
                    numerator = self.idf.get(term, 0) * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score

    @staticmethod
    def normalize_score(scores):
        """
        Normalizes the scores of a List of (doc_id, score)
        :param scores: List of (doc_id, score)
        :return: List of (doc_id, norm_score)
        """
        vals = np.array([score for _, score in scores])
        norm_scores = MaxAbsScaler().fit_transform(vals.reshape(-1, 1))[:, 0]
        return [(scores[i][0], norm_scores[i]) for i in range(len(scores))]