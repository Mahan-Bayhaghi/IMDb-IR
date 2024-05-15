import json
from Logic.core.indexer.indexes_enum import Indexes, Index_types
from Logic.core.indexer.index_reader import Index_reader


class DocumentLengthsIndex:
    def __init__(self, path='./saved_indexes/'):
        """
        Initializes the DocumentLengthsIndex class.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.

        """

        self.documents_index = Index_reader(path, index_name=Indexes.DOCUMENTS).index
        print(f"number of all docs : {len(self.documents_index.keys())}")
        self.document_length_index = {
            Indexes.STARS: self.get_documents_length(Indexes.STARS.value),
            Indexes.GENRES: self.get_documents_length(Indexes.GENRES.value),
            Indexes.SUMMARIES: self.get_documents_length(Indexes.SUMMARIES.value)
        }
        self.store_document_lengths_index(path, Indexes.STARS)
        self.store_document_lengths_index(path, Indexes.GENRES)
        self.store_document_lengths_index(path, Indexes.SUMMARIES)

    def get_documents_length(self, where):
        """
        Gets the documents' length for the specified field.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.

        Returns
        -------
        dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field (where).
        """
        # TODO:
        # print(f"===========================> now getting field {where}")
        document_lengths = {}
        for document_id, document in self.documents_index.items():
            # print(f"where is : <{document[where]}>")
            if where == "summaries":
                if len(document[where]) > 0:
                    # lst = document[where][0]
                    # print(f"lst is {lst} and it's len is {len(lst.strip().split())}")
                    # if len(lst.strip().split()) == 0:
                    #     print(f"DOC {document_id} has summary empty")
                    document_lengths[document_id] = sum(len(lst.strip().split()) for lst in document[where])
                else:
                    document_lengths[document_id] = 0
            else:
                print(f"now for field {where}")
                if len(document[where]) > 0:
                    document_lengths[document_id] = len(document[where])
                else:
                    document_lengths[document_id] = 0
        return document_lengths

    def store_document_lengths_index(self, path, index_name):
        """
        Stores the document lengths index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        index_name : Indexes
            The name of the index to store.
        """
        path = path + index_name.value + '_' + Index_types.DOCUMENT_LENGTH.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.document_length_index[index_name], file, indent=4)
        print(f'Document lengths index {path} stored successfully.')


if __name__ == '__main__':
    document_lengths_index = DocumentLengthsIndex()
