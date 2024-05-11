import collections
import json
from Logic.core.utility.preprocess import Preprocessor


class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of json objects
            The input documents.
        """
        valuable_data = self.extract_valuable_data(all_documents)
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(valuable_data)

    def extract_valuable_data(self, all_documents):
        """

        Parameters
        ----------
        all_documents: list of str
            The input documents.

        Returns
        -------
        valuable_fields: list of str
            A list of long strings consisting of title, first page summary, stars, directors, summaries and synopsis for each movie

        """

        fields = ["title", "stars", "first_page_summary", "summaries", "synopsis"]
        valuable_fields = []
        preprocessor = Preprocessor(None)
        for document in all_documents:
            long_str = []
            for field in fields:
                if document[field] is None:
                    continue
                if field == "first_page_summary" or field == "title":
                    long_str.append(preprocessor.light_preprocess_one_text(document[field]))
                else:
                    for item in document[field]:
                        long_str.append(preprocessor.light_preprocess_one_text(item) + " ")
            # print(f"valuable data extracted : {''.join(long_str)}")
            valuable_fields.append(''.join(long_str))
        return valuable_fields

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """

        # TODO: Create shingle here
        shingles = set()
        char_count = len(word)
        for i in range(char_count - k + 1):
            shingles.add(word[i:i + k])
        return shingles

    def jaccard_score(self, first_set: set, second_set: set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        # TODO: Calculate jaccard score here.
        score = 0.00
        intersection = first_set.intersection(second_set)
        union = first_set.union(second_set)
        if len(union) > 0:
            score = len(intersection) / len(union)
        return score

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = collections.defaultdict(int)

        # TODO: Create shingled words dictionary and word counter dictionary here.
        for documents in all_documents:
            for word in documents.split():
                word_counter[word] += 1
                word_shingles = self.shingle_word(word)
                all_shingled_words[word] = word_shingles
        return all_shingled_words, word_counter

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """

        # TODO: Find 5 nearest candidates here.
        all_candidates = []
        word_shingles = self.shingle_word(word, k=2)
        for candidate in self.all_shingled_words.keys():
            candidate_shingles = self.all_shingled_words[candidate]
            score = self.jaccard_score(candidate_shingles, word_shingles)
            all_candidates.append((candidate, score))

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        top5_candidates = [candidate for candidate in all_candidates[:5]]
        return top5_candidates

    def spell_check(self, query: str):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = []

        # TODO: Do spell correction here.
        query_tokens = query.split()
        for token in query_tokens:
            top5_candidates = self.find_nearest_words(token)
            top5_candidate_words = [candidate[0] for candidate in top5_candidates]
            max_tf = max([self.word_counter[word] for word in top5_candidate_words])
            normalized_tf_scores = {candidate: self.word_counter[candidate] / max_tf for candidate in top5_candidate_words}

            # jaccard score * normalized-tf if word does not really exist !
            candidates_with_scores = []
            for candidate in top5_candidates:
                if candidate[1] == 1:   # word really exists
                    # TODO: decide what to do in the next to lines !!!
                    # candidates_with_scores.append((candidate[0], 2*candidate[1]*normalized_tf_scores[candidate[0]]))
                    candidates_with_scores.append((candidate[0], 1))
                else:
                    candidates_with_scores.append((candidate[0], candidate[1]*normalized_tf_scores[candidate[0]]))
            candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

            final_result.append(candidates_with_scores[0][0])

        return ' '.join(final_result)


def import_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def main():
    crawled_data = import_data('../IMDB_crawled.json')[:]
    spell_checker = SpellCorrection(crawled_data)
    print(spell_checker.find_nearest_words("abslutly"))
    print(spell_checker.spell_check("abslutly"))
    print(spell_checker.spell_check("andre garfild"))
    print(spell_checker.find_nearest_words("multiverse"))
    print(spell_checker.spell_check("multiverse"))


if __name__ == "__main__":
    main()
