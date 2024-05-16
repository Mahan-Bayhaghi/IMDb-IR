import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Logic.core.word_embedding.fasttext_model import FastText


class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = FastText()
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self, load_fasttext_model=False, fasttext_model_path=None, train_epochs=5):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """

        # if we have already trained a fasttext model, use it
        if load_fasttext_model and fasttext_model_path is not None:
            self.fasttext_model.load_model(fasttext_model_path)

        # if not, train it and save it
        else:
            data = pd.read_csv("IMDB Dataset small.csv")
            # preprocess text and save normalized tokens and sentiment labels
            for review, sentiment in tqdm(zip(data['review'], data['sentiment']), total=len(data)):
                tokens = review.split()
                tokens = [token.lower() for token in tokens]
                self.review_tokens.append(tokens)
                self.sentiments.append(sentiment)

            # self.review_tokens = self.review_tokens[:1000]
            # self.sentiments = self.sentiments[:1000]

            # convert text labels to numerical
            label_encoder = LabelEncoder()
            self.sentiments = label_encoder.fit_transform(self.sentiments)

            # now train it and then save it
            # write tokens and sentiments to a txt file for fasttext access
            with open('./tokens.txt', 'w', encoding="utf-8") as f:
                for tokens in self.review_tokens:
                    for token in tokens:
                        f.write(f"{token} ")
                    f.write("\n")
            with open('./sentiments.txt', 'w', encoding="utf-8") as f:
                for sentiment in self.sentiments:
                    f.write(f"{sentiment}\n")

            self.fasttext_model.prepare("./tokens.txt", mode="train", epochs=train_epochs)
            self.fasttext_model.prepare(None, mode="save", save=True, path=fasttext_model_path)
            print("fasttext model trained and saved")

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        if not self.review_tokens:
            with open("./tokens.txt", encoding="utf-8") as file:
                for line in file:
                    self.review_tokens += line.rstrip().split()
            with open("./sentiments.txt", encoding="utf-8") as file:
                for line in file:
                    self.sentiments += line.rstrip()

        for tokens in tqdm(self.review_tokens):
            embeddings = []
            for token in tokens:
                embedding = self.fasttext_model.model[token]  # get embedding of token from model
                embeddings.append(embedding)
            self.embeddings.append(embeddings)
        pass

    def save_embeddings(self):
        print("should save embeddings")

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(self.embeddings, self.sentiments, test_size=test_data_ratio,
                                                            random_state=40)

        return np.array(x_train, dtype="object"), np.array(x_test, dtype="object"), np.array(y_train, dtype="object"), np.array(y_test, dtype="object")


if __name__ == "__main__":
    review_loader = ReviewLoader(None)
    review_loader.load_data(load_fasttext_model=False, fasttext_model_path="./IMDB_dataset_FastText_small.bin")
    # review_loader.load_data(load_fasttext_model=True, fasttext_model_path="./IMDB_dataset_FastText_small.bin")
    review_loader.get_embeddings()
    print("embeddings loaded")
    review_loader.save_embeddings()
    review_loader.split_data()
    print("data split")
