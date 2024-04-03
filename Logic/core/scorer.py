import collections
import math

import numpy as np

from indexer import index_reader, indexes_enum


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        # TODO
        idf = self.idf.get(term, None)
        if idf is None:
            df = len(self.index.get(term).keys())
            idf = np.log(self.N / df)
            self.idf[term] = idf

        print(f"idf of term {term} is {idf}")
        return idf

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        # TODO
        query_tfs = collections.defaultdict(int)
        for term in query:
            query_tfs[term] += 1
        return query_tfs

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        document_method, query_method = method[:3], method[4:]
        scores = {}
        for document_id in self.get_list_of_documents(query):  # get all documents having at least one word of bag
            query_tfs = self.get_query_tfs(query)
            scores[document_id] = self.get_vector_space_model_score(query, query_tfs, document_id, document_method,
                                                                    query_method)
        return scores

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """

        # TODO
        document_vector = []
        query_vector = []

        for term in query:
            # tf handling for document
            tf_in_document = self.index.get(term, {}).get(document_id, 0)  # TODO: search in index using Trie
            if document_method[0] == 'l':
                tf_in_document = np.log(tf_in_document + 1)
            # df handling for document
            df_in_document = 1
            if document_method[1] == 't':
                df_in_document = self.get_idf(term)
            document_vector.append(tf_in_document * df_in_document)

            # tf handling for query
            tf_in_query = query_tfs[term]
            if query_method[0] == 'l':
                tf_in_query = np.log(tf_in_query + 1)
            # df handling for query
            df_in_query = 1
            if query[1] == 't':
                df_in_query = self.get_idf(term)
            query_vector.append(tf_in_query * df_in_query)

        if document_method[2] == 'c':
            document_vector = np.array(self.cosine_normalize(document_vector))
        if query_method[2] == 'c':
            query_vector = np.array(self.cosine_normalize(query_vector))

        score = np.dot(document_vector, query_vector)
        return score

    def cosine_normalize(self, vector: list):
        w = 0.0
        for v in vector:
            w += v ** 2
        w = math.sqrt(w)
        return [v / w for v in vector]

    def compute_socres_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        pass

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """

        # TODO
        pass


def main():
    query = "spider man in wonderland"
    method = "lnc.ltn"
    path = "D:/Sharif/Daneshgah stuff/term 6/mir/project/IMDb-IR/Logic/core/indexer/saved_indexes/"
    reader = index_reader.Index_reader(path, indexes_enum.Indexes.SUMMARIES)
    query = "peter parker"
    query_as_list = query.split()
    scorer = Scorer(reader.index, 3253)
    print(scorer.get_list_of_documents(query_as_list))
    res = scorer.compute_scores_with_vector_space_model(query_as_list, method)
    res_as_list = []
    for k, v in res.items():
        res_as_list.append((k, v))
    res_as_list.sort(key=lambda x: x[1], reverse=True)
    print(res_as_list)


if __name__ == "__main__":
    main()
