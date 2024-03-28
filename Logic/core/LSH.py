import numpy as np
import itertools
import random


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
        # unique_shingles = sorted(list(set().union(*all_shingles)))
        unique_shingles = set()
        for shingle in all_shingles:
            for item in shingle:
                unique_shingles.add(item)

        characteristic_matrix = np.zeros((len(self.documents), len(unique_shingles)), dtype=bool)
        for i, doc in enumerate(self.documents):
            for j, shingle in enumerate(unique_shingles):
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
        _, num_shingles = characteristic_matrix.shape
        # random permutations
        hash_permutations = np.array([np.random.permutation(num_shingles) for _ in range(self.num_hashes)])
        signatures_matrix = np.full((self.num_hashes, len(self.documents)), np.inf)
        for i in range(len(self.documents)):
            for j in range(num_shingles):
                if characteristic_matrix[i, j]:
                    hash_values = hash_permutations[:, j]
                    signatures_matrix[:, i] = np.minimum(signatures_matrix[:, i], hash_values)
        return signatures_matrix

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
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
        bucket_dict = {}
        for b in range(bands):
            starting_row = b * rows_per_band
            ending_row = starting_row + rows_per_band
            band_hash = {}
            for doc_index in range(num_docs):
                band_signature = tuple(signature[starting_row:ending_row, doc_index])
                if band_signature in band_hash:
                    bucket_id = band_hash[band_signature]
                    bucket_dict[bucket_id].append(doc_index)
                else:
                    bucket_id = len(bucket_dict)
                    band_hash[band_signature] = bucket_id
                    bucket_dict[bucket_id] = [doc_index]
        return bucket_dict

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO: change values of r and b to reach a valid score
        signature_matrix = self.min_hash_signature()
        buckets_dict = self.lsh_buckets(signature_matrix, 5, 5)
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

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)
