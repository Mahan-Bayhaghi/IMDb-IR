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
        final_snippet = ""
        not_exist_words = []

        # TODO: Extract snippet and the tokens which are not present in the doc.
        filtered_query = self.remove_stop_words_from_query(query)
        query_tokens = filtered_query.split()
        occurrences = {}
        for token in query_tokens:
            occurrences[token] = [m.start() for m in re.finditer(r'\b%s\b' % re.escape(token), doc.lower())]
        # generate snippet using occurrences
        final_snippet = ""
        for token in query_tokens:
            if token in occurrences:
                for index in occurrences[token]:
                    start = max(0, index - self.number_of_words_on_each_side)
                    end = min(len(doc), index + len(token) + self.number_of_words_on_each_side)
                    snippet = doc[start:end].strip()
                    snippet = snippet.replace(token, f'***{token}***')
                    final_snippet += snippet + " ... "
            else:
                not_exist_words.append(token)

        return final_snippet.strip(), not_exist_words


if __name__ == "__main__":
    doc = "The lives of two mob hitmen, a boxer, a gangster and his wife, " \
          "and a pair of diner bandits intertwine in four tales of violence and redemption."

    query = "gangster and boxer and a black man coming up"
    snippet = Snippet()
    final_snippet, not_exist_words = snippet.find_snippet(doc, query)
    print(f"Final snippet : {final_snippet}")
    print(f"Words not found : {not_exist_words}")
