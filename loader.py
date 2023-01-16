import hashlib
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from google.cloud import storage
from inverted_index_gcp import InvertedIndex, TUPLE_SIZE
from multifilereader import MultiFileReader
from contextlib import closing


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


class BucketIndexLoader:

    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_name)

    def load_all_indices(self):
        """
        Loading all indices from the .pkl files.
        :return: 3 InvertedIndex object.
        """
        text_inverted_index = self.load_index_from_folder("text_inverted_index")
        # self.load_all_bin_files("text_inverted_index")
        title_inverted_index = self.load_index_from_folder("title_inverted_index_with_stemming")
        # self.load_all_bin_files("title_inverted_index_with_stemming")
        anchor_inverted_index = self.load_index_from_folder("anchor_inverted_index")
        # self.load_all_bin_files("anchor_inverted_index")
        return text_inverted_index, title_inverted_index, anchor_inverted_index

    def load_index_from_folder(self, folder_name):
        """
        Loading an index from the .pkl file in a given folder.
        :param folder_name: folder name.
        :return: InvertedIndex object.
        """
        path = f"postings_gcp/{folder_name}/index.pkl"
        if not os.path.exists(f'{folder_name}.pkl'):
            blob = self.bucket.get_blob(path)
            blob.download_to_filename(f'{folder_name}.pkl')
        with open(f'{folder_name}.pkl', 'rb') as f:
            return pickle.load(f)

    def load_all_bin_files(self, folder_name):
        """
        Loading all the bin files in a folder to the local memory.
        :param folder_name: folder_name
        """
        for blob in self.client.list_blobs(self.bucket_name, prefix=f'postings_gcp/{folder_name}'):
            if blob.name.endswith('.bin'):
                if not os.path.exists(f'{folder_name}_{os.path.basename(blob.name)}'):
                    blob.download_to_filename(f'{folder_name}_{os.path.basename(blob.name)}')

    def load_page_rank_to_df(self):
        """
        Loading the pagerank from the .pkl file to a pd.Series object, and normalizing it.
        :return:
        """
        blobs = self.client.list_blobs(self.bucket_name, prefix='pr')
        for blob in blobs:
            if blob.name.endswith('csv.gz'):
                if not os.path.exists('pr.csv.gz'):
                    blob.download_to_filename('pr.csv.gz')
                df = pd.read_csv("./pr.csv.gz", header=None)
                df.columns = ['doc_id', 'page_rank']
                pr = df['page_rank']
                pr_norm = pd.Series(MinMaxScaler().fit_transform(np.array(pr).reshape(-1, 1))[:, 0])
                pr.index = df['doc_id']
                pr_norm.index = df['doc_id']
                return pr, pr_norm

    def loda_page_views(self):
        """
        Loading the page views dictionary from the .pkl file.
        :return: page views dictionary.
        """
        blob = self.bucket.get_blob('pv/page_views.pkl')
        blob.download_to_filename(f'page_views.pkl')
        with open(f'page_views.pkl', 'rb') as f:
            pv = pickle.load(f)
        return pv

    @staticmethod
    def load_posting_lists_for_token(token, index: InvertedIndex, folder_name):
        """
        Loading the posting list of a specific token.
        :param token: String
        :param index: InvertedIndex object
        :param folder_name: folder name
        :return: posting list of the token
        """
        if token not in index.posting_locs:
            return []
        locs = index.posting_locs[token]
        for f_name, pos in locs:
            name = f"{folder_name}_{f_name}"
            if not os.path.exists(name):
                loader = BucketIndexLoader("project_bucket_316533942")
                loader.bucket.get_blob(f"postings_gcp/{folder_name}/{f_name}").download_to_filename(name)
        with closing(MultiFileReader(folder_name)) as reader:
            b = reader.read(locs, index.df[token] * TUPLE_SIZE)
            posting_list = []
            for i in range(index.df[token]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    def load_doc_titles(self):
        """
        Loading the document titles dictionary from the .pkl file.
        :return: dictionary of document titles.
        """
        if not os.path.exists("doc_titles.pkl"):
            blob = self.bucket.get_blob('titles/doc_titles.pkl')
            blob.download_to_filename(f'doc_titles.pkl')
        with open(f'doc_titles.pkl', 'rb') as f:
            doc_titles = pickle.load(f)
        return doc_titles

    def load_doc_norms(self):
        """
        Loading the document normals dictionary from the .pkl file.
        :return: dictionary of document normals.
        """
        if not os.path.exists("doc_norms.pkl"):
            blob = self.bucket.get_blob('doc_norms/doc_norms.pkl')
            blob.download_to_filename(f'doc_norms.pkl')
        with open(f'doc_norms.pkl', 'rb') as f:
            doc_norms = pickle.load(f)
        return doc_norms

    def load_doc_len(self):
        """
        Loading the document length dictionary from the .pkl file.
        :return: dictionary of document lengths.
        """
        if not os.path.exists("doc_len.pkl"):
            blob = self.bucket.get_blob('doc_len/doc_len.pkl')
            blob.download_to_filename(f'doc_len.pkl')
        with open(f'doc_len.pkl', 'rb') as f:
            doc_len = pickle.load(f)
        return doc_len
