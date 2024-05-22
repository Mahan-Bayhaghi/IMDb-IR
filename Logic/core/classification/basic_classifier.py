import numpy as np
from tqdm import tqdm
from Logic.core.word_embedding.fasttext_model import FastText


class BasicClassifier:
    def __init__(self):
        self.fasttext_model = FastText()
        self.model = None  # Placeholder for the actual classifier model
        # raise NotImplementedError()

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def prediction_report(self, x, y):
        raise NotImplementedError()

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        embeddings = []
        for sentence in tqdm(sentences):
            embedding = self.fasttext_model.get_query_embedding(sentence)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)

        # predict the sentiment for each sentence
        predictions = self.predict(embeddings)

        # calculate the percentage of positive reviews
        positive_count = np.sum(predictions == 1)  # 1 represents positive sentiment
        total_count = len(predictions)
        percent_positive = (positive_count / total_count) * 100

        return percent_positive
        # pass

