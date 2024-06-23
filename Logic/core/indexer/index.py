import time
import os
import json
import copy

from nltk import word_tokenize

from Logic.core.indexer.indexes_enum import Indexes
# import Logic.core.preprocess as preprocess
import Logic.core.utility.preprocess as preprocess

class TrieNode:
    def __init__(self):
        self.children = {}
        self.movie_ids = []
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert_term(self, term, movie_tf):
        term = term.lower()
        node = self.root
        length = len(term)
        for level in range(length):
            index = term[level]
            # if current character is not present
            if index not in node.children.keys():
                node.children[index] = TrieNode()
            node = node.children[index]
        node.movie_ids.append(movie_tf)
        node.isEndOfWord = True

    def create_trie(self, dictionary: dict):
        for term in dictionary.keys():
            # print(dictionary[term])
            self.insert_term(term, dictionary[term])
        return self.root

    def search_trie(self, term):
        node = self.root
        length = len(term)
        for level in range(length):
            index = term[level]
            if index not in node.children.keys():
                return None
            node = node.children[index]
        return node.movie_ids


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index_trie_root = {
            Indexes.DOCUMENTS.value: Trie(),
            Indexes.STARS.value: Trie(),
            Indexes.GENRES.value: Trie(),
            Indexes.SUMMARIES.value: Trie()
        }

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        #         TODO
        for document in self.preprocessed_documents:
            current_index[document['id']] = document

        self.index_trie_root[Indexes.DOCUMENTS.value] = Trie()
        self.index_trie_root[Indexes.DOCUMENTS.value].create_trie(current_index)

        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO
        current_index = {}
        for document in self.preprocessed_documents:
            try:
                # according to quera, first name and last name should be separated
                stars = []
                for full_name in document['stars']:
                    stars += full_name.split()
                for star in stars:
                    if star in current_index:  # update tf
                        current_index[star][document['id']] = current_index[star].get(document['id'], 0) + 1
                    else:  # create with tf = 1
                        current_index[star] = {document['id']: 1}
            except Exception as e:
                print(e)

        self.index_trie_root[Indexes.STARS.value] = Trie()
        self.index_trie_root[Indexes.STARS.value].create_trie(current_index)

        return current_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO
        current_index = {}
        for document in self.preprocessed_documents:
            try:
                genres = document['genres']
                for genre in genres:
                    if genre in current_index:  # update tf
                        current_index[genre][document['id']] = current_index[genre].get(document['id'], 0) + 1
                    else:  # create with tf = 1
                        current_index[genre] = {document['id']: 1}
            except Exception as e:
                print(e)

        self.index_trie_root[Indexes.GENRES.value] = Trie()
        self.index_trie_root[Indexes.GENRES.value].create_trie(current_index)

        return current_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        for document in self.preprocessed_documents:
            try:
                summaries = document['summaries']
                # if document['first_page_summary'] is not None:
                #     summaries.append(document['first_page_summary'])
                # if document['title'] is not None:
                #     summaries.append(document['title'])
                for summary in summaries:
                    # split into tokens
                    tokens = summary.split()
                    for token in tokens:
                        if token in current_index:  # update tf
                            current_index[token][document['id']] = current_index[token].get(document['id'], 0) + 1
                        else:  # create with tf = 1
                            current_index[token] = {document['id']: 1}
            except Exception as e:
                print(e)

        self.index_trie_root[Indexes.SUMMARIES.value] = Trie()
        self.index_trie_root[Indexes.SUMMARIES.value].create_trie(current_index)

        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """
        if index_type not in self.index:
            raise ValueError('Invalid index type')

        else:
            posting_list = self.index_trie_root[index_type].search_trie(word)
            if posting_list is not None:
                posting_list = posting_list[0]
                return sorted(posting_list.keys())
            else:
                return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        #         TODO
        try:
            self.index[Indexes.DOCUMENTS.value][document['id']] = document

            for index_type in [Indexes.STARS.value, Indexes.GENRES.value, Indexes.SUMMARIES.value]:
                if index_type in document:
                    for term in document[index_type]:
                        if term not in self.index[index_type]:
                            self.index[index_type][term] = {}
                        self.index[index_type][term][document['id']] = 1  # Assuming tf is always 1 for now
        except Exception as e:
            print(e)

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        #         TODO
        try:
            if document_id in self.index[Indexes.DOCUMENTS.value]:
                del self.index[Indexes.DOCUMENTS.value][document_id]

            for index_type in [Indexes.STARS.value, Indexes.GENRES.value, Indexes.SUMMARIES.value]:
                for term, posting_list in self.index[index_type].items():
                    if document_id in posting_list:
                        del posting_list[document_id]
        except Exception as e:
            print(e)

    def delete_dummy_keys(self, index_before_add, index, key):
        if len(index_before_add[index][key]) == 0:
            del index_before_add[index][key]

    def check_if_key_exists(self, index_before_add, index, key):
        if not index_before_add[index].__contains__(key):
            index_before_add[index].setdefault(key, {})


    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(
                set(index_before_add[Indexes.STARS.value]['tim']))

        self.check_if_key_exists(index_before_add, Indexes.STARS.value, 'tim')

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(
                set(index_before_add[Indexes.STARS.value]['henry']))
        self.check_if_key_exists(index_before_add, Indexes.STARS.value, 'henry')

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(
                set(index_before_add[Indexes.GENRES.value]['drama']))

        self.check_if_key_exists(index_before_add, Indexes.GENRES.value, 'drama')

        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(
                set(index_before_add[Indexes.GENRES.value]['crime']))
        self.check_if_key_exists(index_before_add, Indexes.GENRES.value, 'crime')

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        self.check_if_key_exists(index_before_add, Indexes.SUMMARIES.value, 'good')

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        # Change the index_before_remove to its initial form if needed

        self.delete_dummy_keys(index_before_add, Indexes.STARS.value, 'tim')
        self.delete_dummy_keys(index_before_add, Indexes.STARS.value, 'henry')
        self.delete_dummy_keys(index_before_add, Indexes.GENRES.value, 'drama')
        self.delete_dummy_keys(index_before_add, Indexes.GENRES.value, 'crime')
        self.delete_dummy_keys(index_before_add, Indexes.SUMMARIES.value, 'good')

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        filepath = os.path.join(path, f"{index_name}_index.json")
        with open(filepath, 'w') as file:
            json.dump(self.index[index_name], file, indent=4)

    def store_all_index(self, path: str):
        self.store_index(path, Indexes.DOCUMENTS.value)
        self.store_index(path, Indexes.STARS.value)
        self.store_index(path, Indexes.GENRES.value)
        self.store_index(path, Indexes.SUMMARIES.value)

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """

        #         TODO
        try:
            loaded_index = {}
            for index_type in Indexes:
                filepath = os.path.join(path, f"{index_type.value}_index.json")
                with open(filepath, 'r') as file:
                    loaded_index[index_type.value] = json.load(file)

            if not loaded_index:
                raise FileNotFoundError("No index files found in the specified directory.")

            self.index = loaded_index
        except Exception as e:
            print(e)

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field.split():
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        # TODO: based on your implementation, you may need to change the following line
        # tokenize for multiple word queries
        word_tokens = word_tokenize(check_word)
        posting_lists = []

        start = time.time()
        for word in word_tokens:
            posting_lists += self.get_posting_list(word, index_type)
        posting_list = list(set(posting_lists))

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False


# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods


def import_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def main():
    preprocessed_documents = import_data("../../IMDB_crawled_preprocessed.json")[:]
    index = Index(preprocessed_documents)
    index.store_all_index(path="./saved_indexes/")

    # check methods
    # index.check_add_remove_is_correct()
    index.load_index("./saved_indexes/")
    print(
        f"index loaded correctly : "
        f"{index.check_if_index_loaded_correctly(Indexes.GENRES.value, index.index[Indexes.GENRES.value])}")
    index.check_if_indexing_is_good(index_type=Indexes.STARS.value, check_word='al pacino')


if __name__ == "__main__":
    main()
