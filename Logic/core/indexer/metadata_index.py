from index_reader import Index_reader
from indexes_enum import Indexes, Index_types
import json


class Metadata_index:
    def __init__(self, path='./saved_indexes/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        # TODO
        self.path = path
        self.documents = self.read_documents()
        self.metadata_index = None

    def read_documents(self):
        """
        Reads the documents.
        
        """
        # TODO
        document_index = Index_reader(self.path, index_name=Indexes.DOCUMENTS).index
        return document_index

    def create_metadata_index(self):
        """
        Creates the metadata index.
        """
        metadata_index = {'average_document_length': {
            'stars': self.get_average_document_field_length('stars'),
            'genres': self.get_average_document_field_length('genres'),
            'summaries': self.get_average_document_field_length('summaries')
        }, 'document_count': len(self.documents)}
        self.metadata_index = metadata_index
        return metadata_index

    def get_average_document_field_length(self, where):
        """
        Returns the average of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """
        # TODO
        total, counter = 0, 0
        for document_id, document in self.documents.items():
            if where in document:
                if where == "summaries":
                    total += sum(len(lst) for lst in document[where])
                else:
                    total += len(document[where])
                counter += 1

        return 0 if counter == 0 else (float(total)/float(counter))

    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path = path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)


if __name__ == "__main__":
    meta_index = Metadata_index()
    meta_index.create_metadata_index()
    meta_index.store_metadata_index('./saved_indexes/')
