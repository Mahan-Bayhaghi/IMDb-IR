import re
from Logic.core.preprocess import Preprocessor


class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        # TODO: remove stop words from the query.
        preprocessor = Preprocessor(None)
        filtered_query = preprocessor.remove_stopwords(query)

        return filtered_query

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        preprocessor = Preprocessor(None)
        doc = preprocessor.remove_stopwords(doc)
        doc = preprocessor.remove_punctuations(doc)
        doc = doc.lower()
        doc_terms = doc.split()
        query = preprocessor.remove_stopwords(query)
        query = preprocessor.remove_punctuations(query)
        query = query.lower()
        query_terms = query.split()

        print(f"doc terms : {doc_terms}")
        print(f"query terms : {query_terms}")
        doc_occurrence = []
        for doc_term in doc_terms:
            print(f"doc term is : {doc_term}")
            if doc_term in query_terms:
                doc_occurrence.append(1)
            else:
                doc_occurrence.append(0)

        print(doc_occurrence)
        max_window_size, window_indices = find_smallest_window(doc_occurrence, self.number_of_words_on_each_side)
        for idx, doc_term in enumerate(doc_terms):
            doc_terms[idx] = f'***{doc_term}***' if idx in window_indices else doc_term

        final_snippet = ' '.join(doc_terms)
        not_exist_words = None
        return final_snippet, not_exist_words
        # ------------------------------------------------------------
        # not_exist_words = []
        #
        # # TODO: Extract snippet and the tokens which are not present in the doc.
        # filtered_query = self.remove_stop_words_from_query(query)
        # preprocessor = Preprocessor(None)
        # filtered_query = preprocessor.normalize(filtered_query)
        # filtered_doc = self.remove_stop_words_from_query(doc).lower()
        # filtered_doc = preprocessor.normalize(filtered_doc)
        # query_tokens = filtered_query.split()
        # occurrences = {}
        # for token in query_tokens:
        #     occurrences[token] = [m.start() for m in re.finditer(r'\b%s\b' % re.escape(token), filtered_doc)]
        #
        # # generate snippet using occurrences
        # final_snippet = ""
        # for token in query_tokens:
        #     if token in occurrences:
        #         for index in occurrences[token]:
        #             start = max(0, index - self.number_of_words_on_each_side)
        #             end = min(len(doc), index + len(token) + self.number_of_words_on_each_side)
        #             snippet = filtered_doc[start:end].strip()
        #             snippet = snippet.replace(token, f' ***{token}*** ')
        #             final_snippet += snippet + " ... "
        #     else:
        #         not_exist_words.append(token)
        #
        # return final_snippet.strip(), not_exist_words
    # ------------------------------------------------------------


def find_smallest_window(arr, k):
    left, right = 0, 0
    max_ones = 0
    max_ones_start = 0
    max_window_size = 0
    count = 0
    ones_indices = []
    for i in range(len(arr)):
        if arr[i] == 1:
            ones_indices.append(i)
    for i in range(len(arr)):
        if arr[i] == 1:
            count += 1
        if i - left + 1 - count > k:
            if arr[left] == 1:
                count -= 1
            left += 1
        if i - left + 1 > max_window_size:
            max_window_size = i - left + 1
            max_ones = count
            max_ones_start = left
    return max_window_size, ones_indices[max_ones_start:max_ones_start + max_ones]


if __name__ == "__main__":
    doc = "The lives of two mob hitmen, a boxer, a gangster and his wife, " \
          "and a pair of diner bandits intertwine in four tales of violence and redemption."
    # doc = "Pulp novelist Holly Martins travels to shadowy, postwar Vienna, only to find himself " \
    #       "investigating the mysterious death of an old friend, Harry Lime."
    # query = "gangster boxer black man coming up"
    query = "boxer bandit pulp The"
    snippet = Snippet(number_of_words_on_each_side=10)
    final_snippet, not_exist_words = snippet.find_snippet(doc, query)
    print(f"Final snippet : {final_snippet}")
    print(f"Words not found : {not_exist_words}")
