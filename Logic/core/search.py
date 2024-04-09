import collections
import json
import numpy as np

from Logic.core import path_access
from Logic.core.preprocess import Preprocessor
from Logic.core.scorer import Scorer
from Logic.core.indexer.indexes_enum import Indexes, Index_types
from Logic.core.indexer.index_reader import Index_reader
import Logic.core.path_access

class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        # path = './saved_indexes/'
        path = path_access.path_to_logic() + 'core/indexer/saved_indexes/'
        self.document_indexes = {
            Indexes.STARS: Index_reader(path, Indexes.STARS),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES)
        }
        self.tiered_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.TIERED),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.TIERED),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.TIERED)
        }
        self.document_lengths_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.DOCUMENT_LENGTH),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH)
        }
        self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA)
        self.number_of_documents = self.metadata_index.index["document_count"]

    def search(self, query, method, weights, safe_ranking=True, max_results=10):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results. 
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """

        preprocessor = Preprocessor([query])
        # query = preprocessor.preprocess()[0].split()
        query = preprocessor.preprocess_one_text(query).split()  # tokenized preprocessed query list

        scores = {}
        if safe_ranking:
            scores = self.find_scores_with_safe_ranking(query, method, weights, scores)
        else:
            scores = self.find_scores_with_unsafe_ranking(query, method, weights, max_results, scores)

        final_scores = {}

        self.aggregate_scores(weights, scores, final_scores)

        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results is not None:
            result = result[:max_results]

        return result

    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregates the scores of the fields.

        Parameters
        ----------
        weights : dict
            The weights of the fields.
        scores : dict
            The scores of the fields.
        final_scores : dict
            The final scores of the documents.
        """
        # TODO
        for document_id in scores.keys():
            document_scores = scores[document_id]
            final_document_score = 0
            for field in document_scores.keys():
                if field not in weights:
                    raise ValueError("Invalid field ! please provide valid fields to search")
                final_document_score += document_scores[field] * weights[field]
            final_scores[document_id] = final_document_score

    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results, scores):
        """
        Finds the scores of the documents using the unsafe ranking method using the tiered index.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        max_results : int
            The maximum number of results to return.
        scores : dict
            The scores of the documents.
        """
        for tier in ["first_tier", "second_tier", "third_tier"]:
            tier_scores = {}
            for field in weights:
                # TODO
                tiered_index = self.tiered_index[field].index
                normal_index = self.document_indexes[field].index  # will be used to compute dfs, tf and idf
                index_scorer = Scorer(tiered_index[tier], self.number_of_documents, index_needed_for_dfs=normal_index)
                print(f"field {field} and tier {tier} index loaded {index_scorer}")
                if method == "OkapiBM25":
                    scoring_result = index_scorer.compute_socres_with_okapi_bm25 \
                        (query, self.metadata_index.index["average_document_length"][field.value],
                         self.document_lengths_index[field].index)
                else:
                    scoring_result = index_scorer.compute_scores_with_vector_space_model(query, method)
                for document_id in scoring_result.keys():
                    if document_id not in tier_scores.keys():
                        tier_scores[document_id] = {}
                    tier_scores[document_id][field] = scoring_result[document_id]
            # if max results found, stop
            scores = self.merge_scores(tier_scores, scores)
            if len(scores.keys()) >= max_results:
                print("reached enough results")
                return scores
            else:
                scores = self.merge_scores(tier_scores, scores)
        return scores

    def merge_scores(self, scores1, scores2):
        """
        Merges two dictionaries of scores.

        Parameters
        ----------
        scores1 : dict
            The first dictionary of scores.
        scores2 : dict
            The second dictionary of scores.

        Returns
        -------
        dict
            The merged dictionary of scores.
        """
        # TODO
        result = {}
        for key, value in scores1.items():
            if key not in result.keys():
                result[key] = collections.defaultdict(int)
            for field, score in value.items():
                result[key][field] = max(result[key][field], score)
        for key, value in scores2.items():
            if key not in result.keys():
                result[key] = collections.defaultdict(int)
            for field, score in value.items():
                result[key][field] = max(result[key][field], score)

        return result

    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Finds the scores of the documents using the safe ranking method.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        scores : dict
            The scores of the documents.
        """
        print(f"weights is {weights}")
        for field in weights:
            # TODO
            index = self.document_indexes[field].index
            index_scorer = Scorer(index, self.number_of_documents, index_needed_for_dfs=index)
            if method == "OkapiBM25":
                scoring_result = index_scorer.compute_socres_with_okapi_bm25 \
                    (query, self.metadata_index.index["average_document_length"][field.value],
                     self.document_lengths_index[field].index)
            else:
                scoring_result = index_scorer.compute_scores_with_vector_space_model(query, method)
            # print(f"scoring result for query <{query}> in field <{field}> is \n {scoring_result}")
            for document_id in scoring_result.keys():
                if document_id not in scores.keys():
                    scores[document_id] = {}
                scores[document_id][field] = scoring_result[document_id]
            # print(f"scores dict is : {scores}")
        return scores


if __name__ == '__main__':
    search_engine = SearchEngine()
    # query = "spider man in wonderland"
    # query = "spiderman"
    query = "matrix"
    query = "the dune atreides "
    query = "harry potter"

    # method = "lnc.ltc"
    method = "OkapiBM25"

    weights = {
        Indexes.STARS: 1,
        Indexes.GENRES: 1,
        Indexes.SUMMARIES: 1
    }

    result = search_engine.search(query, method, weights, safe_ranking=True, max_results=10)
    # result = search_engine.search(query, method, weights, safe_ranking=False, max_results=20)

    print(f"final search result is \n {result}")

