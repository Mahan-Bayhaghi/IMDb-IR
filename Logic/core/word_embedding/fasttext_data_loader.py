import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import Logic.core.path_access as path_access
from Logic.core.indexer.index_reader import Index_reader
from Logic.core.indexer.indexes_enum import Indexes


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                    punctuation_removal=True, str_output=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    pass
    if pd.isna(text):
        text = ""
    if lower_case:
        text = text.lower()
    if punctuation_removal:
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r',', ' ', text)
    if stopword_removal:
        stopwords_ = set(stopwords.words("english"))
        tokens = word_tokenize(text)
        text = [word for word in tokens if word not in stopwords_]
    if minimum_length > 1:
        text = [word for word in text if len(word) >= minimum_length]

    if str_output:
        return " ".join(text)
    return text


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path


    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        index = Index_reader(path=self.file_path, index_name=Indexes.DOCUMENTS).index
        # print(f"Index is {index}")
        synopses = []
        summaries = []
        reviews = []
        titles = []
        genres = []
        # counter = 0
        for movie_id, movie in index.items():
            # counter += 1
            # if counter == 50:
            #     break

            # print("snp is ", ))
            titles.append(preprocess_text((movie.get("title", ""))))
            genres.append(preprocess_text(" ".join(movie.get("genres", []))))
            synopses.append(preprocess_text(" ".join(movie["synposis"])))
            summaries.append(preprocess_text(" ".join(movie["summaries"])))

            all_reviews = movie.get("reviews")
            rev = []
            for review in all_reviews:
                rev.append(review[0])
                rev.append(review[1])
            reviews.append(preprocess_text(" ".join(rev)))

        return pd.DataFrame({
            'title': titles,
            'genre': genres,
            'synopsis': synopses,
            'summary': summaries,
            'reviews': reviews
        })

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        # df = self.read_data_to_df()
        # read from file
        df = pd.read_csv('./training_data.csv')
        texts = df['synopsis'] + ' ' + df['summary'] + ' ' + df['reviews']  # Concatenate text data
        labels = df['genre']
        return texts, labels

    def create_train_data_for_cluster(self):
        df = pd.read_csv(self.file_path)
        texts = df['synopsis'] + ' ' + df['summary'] + ' ' + df['reviews']  # Concatenate text data
        labels = df['genre']
        return texts, labels
        pass


if __name__ == "__main__":
    path = path_access.path_to_logic() + "core/indexer/saved_indexes/"
    print(f"path is {path}")
    ftdl = FastTextDataLoader(path)
    df = ftdl.read_data_to_df()
    # df.to_csv("training_data.csv")
