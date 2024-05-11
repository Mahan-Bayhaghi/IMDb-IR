import hashlib
import mmh3
import numpy as np
import itertools
import random
import json
import time

from Logic.core.utility.preprocess import Preprocessor


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        words = document.split()
        words_count = len(words)
        for i in range(words_count - k + 1):
            shingles.add(' '.join(words[i:i + k]))
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        # TODO
        all_shingles = [self.shingle_document(doc) for doc in self.documents]
        unique_shingles = set()
        for shingle in all_shingles:
            for item in shingle:
                unique_shingles.add(item)
        print(f"number of all shingles : {sum([len(lst) for lst in all_shingles])}")
        print(f"number of unique shingles : {len(unique_shingles)}")

        characteristic_matrix = np.zeros((len(self.documents), len(unique_shingles)), dtype=bool)
        for i, doc in enumerate(all_shingles):
            for j, shingle in enumerate(unique_shingles):
                # if we search in doc, "is a" can be matched to "is another" so search in set of shingles
                # in other words, see doc as set of it's shingles
                characteristic_matrix[i, j] = shingle in doc
        return characteristic_matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        # TODO
        characteristic_matrix = self.build_characteristic_matrix()
        num_docs, num_shingles = characteristic_matrix.shape

        # random permutations
        # using method of chapter 3.3.5 (Mining of Massive Datasets- Leskovec,Rajaraman,Ullman)
        hash_permutations = np.array([np.random.permutation(num_shingles) for _ in range(self.num_hashes)])

        signatures_matrix = np.full((self.num_hashes, num_docs), np.inf)

        # for hash_index in range(self.num_hashes):
        #     for doc_index in range(num_docs):
        #         # signatures_matrix[hash_index][num_docs] = idx
        #         for idx in range(self.num_hashes):
        #             if characteristic_matrix[doc_index][hash_permutations[hash_index][idx]]:
        #                 signatures_matrix[hash_index][doc_index] = idx

        for i in range(num_docs):
            for j in range(num_shingles):
                if characteristic_matrix[i, j]:
                    hash_values = hash_permutations[:, j]
                    signatures_matrix[:, i] = np.minimum(signatures_matrix[:, i], hash_values)
        return signatures_matrix

    def lsh_buckets(self, signature, bands=50, rows_per_band=None):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        num_hashes, num_docs = signature.shape
        if rows_per_band is None:
            rows_per_band = int(num_hashes / bands)
        print(f"threshold of similarity : {(1 / bands) ** (1 / rows_per_band)}")
        bucket_dict = {}
        for b in range(bands):
            starting_row = b * rows_per_band
            ending_row = starting_row + rows_per_band
            band_hash = {}
            for doc_index in range(num_docs):
                # TODO: use a hash function for band_signature -> done
                band_signature = tuple(signature[starting_row:ending_row, doc_index])

                if band_signature in band_hash:
                    bucket_id = band_hash[band_signature]
                    if doc_index not in bucket_dict[bucket_id]:
                        bucket_dict[bucket_id].append(doc_index)
                        # print(
                        #     f"added doc {doc_index} with band signature {band_signature} to bucket with id {bucket_id} that already had {bucket_dict[bucket_id]}")
                else:
                    bucket_id = len(bucket_dict)
                    band_hash[band_signature] = bucket_id
                    bucket_dict[bucket_id] = [doc_index]

        return bucket_dict

    def hash_band_signature(self, band_signature, num_buckets=200, hash_function='md5'):
        hash_value = 0
        if hash_function == 'md5':
            hash_value = hashlib.md5(str(band_signature).encode()).hexdigest()
        elif hash_function == 'sha1':
            hash_value = hashlib.sha1(str(band_signature).encode()).hexdigest()
        elif hash_function == 'murmur':
            hash_value = mmh3.hash(str(band_signature))
        return int(hash_value, 16) % num_buckets

    def perform_lsh(self, num_bands=50, num_rows_per_band=None):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO: change values of r and b to reach a valid score
        signature_matrix = self.min_hash_signature()
        all_buckets = self.lsh_buckets(signature_matrix, bands=num_bands, rows_per_band=num_rows_per_band)
        # aggregate all buckets
        buckets_dict = {}
        temp = set()
        for v in all_buckets.keys():
            if len(all_buckets[v]) > 1:
                temp.add(tuple(all_buckets[v]))
        for idx, v in enumerate(temp):
            buckets_dict[idx] = v

        return buckets_dict

    def jaccard_score(self, first_set: set, second_set: set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        score = 0.00
        intersection = first_set.intersection(second_set)
        union = first_set.union(second_set)
        if len(union) > 0:
            score = len(intersection) / len(union)
        return score

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score >= 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)


def main():
    def import_all_summaries(filepath, restrict=False, restrict_num=200):
        with open(filepath, 'r') as file:
            data = json.load(file)
        all_movies = [movie for movie in data]
        preprocessor = Preprocessor(None)
        temp_summaries = []
        for movie in all_movies:
            if len(movie['summaries']) >= 1:
                temp_summaries.append(movie["summaries"])
        if restrict:
            temp_summaries = temp_summaries[:restrict_num]
        summaries = [preprocessor.preprocess_one_text(' '.join(summary)) for summary in temp_summaries]
        return summaries

    all_summaries = import_all_summaries("./LSHFakeData_preprocessed.json")
    all_summaries += import_all_summaries("../IMDB_crawled_preprocessed.json", restrict=True, restrict_num=1000)

    minhash_lsh = MinHashLSH(all_summaries, num_hashes=500)
    t = time.time()
    # if num_rows_per_band is not defined, it will match with needed number itself
    buckets = minhash_lsh.perform_lsh(num_bands=50, num_rows_per_band=None)
    t = time.time() - t
    minhash_lsh.jaccard_similarity_test(buckets, all_summaries)
    print(f"elapsed time for LSH: {t} seconds")


if __name__ == "__main__":
    main()


# *******************************************************
# the result of above code on my test run :             *
# number of all shingles : 262234                       *
# number of unique shingles : 146160                    *
# threshold of similarity : 0.6762433378062414          *
# your final score in near duplicate detection: 1.0     *
# elapsed time for LSH: 67.08891892433167 seconds       *
# *******************************************************
