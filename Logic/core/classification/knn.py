import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader
from collections import Counter


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        predictions = []
        for i in tqdm(range(x.shape[0])):
            distances = []
            for j in range(self.x_train.shape[0]):
                distance = np.linalg.norm(self.x_train[j] - x[i])
                # print(f"Distance is {distance}")
                distances.append(distance)
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_neighbors]
            # print(f"nearest labels is : {nearest_labels}")
            most_common_label = Counter(nearest_labels).most_common(1)[0][0]
            # print(f"label {most_common_label} predicted for i = {i}")
            predictions.append(most_common_label)
        return np.array(predictions)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        return classification_report(y.astype("int"), y_pred.astype("int"))


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    file_path = "IMDB Dataset small.csv"
    review_loader = ReviewLoader(file_path)
    review_loader.load_data(dataset_path=file_path, load_fasttext_model=False, fasttext_model_path="./IMDB_dataset_FastText_small.bin")
    # review_loader.load_data(load_fasttext_model=True, fasttext_model_path="./IMDB_dataset_FastText_small_small.bin")
    review_loader.get_embeddings()

    x_train, x_test, y_train, y_test = review_loader.split_data(test_data_ratio=0.1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    knn_classifier = KnnClassifier(n_neighbors=5)
    knn_classifier.fit(x_train, y_train)
    print("KNN fitted")

    print(knn_classifier.prediction_report(x_test, y_test))

# for the following test, I've used small dataset which is first 25k rows of original dataset to reduce the time
# by increasing size of dataset, classification works on larger data as well
##################### test results ##########################
# 100%|██████████| 25000/25000 [00:00<00:00, 26019.86it/s]
# Read 5M words
# Number of words:  45612
# Number of labels: 0
# Progress: 100.0% words/sec/thread:   22562 lr:  0.000000 avg.loss:  2.235495 ETA:   0h 0m 0s
# fasttext model trained and saved
# 100%|██████████| 25000/25000 [00:18<00:00, 1379.94it/s]
# x_train is 22500 objects each with shape 50
# KNN fitted
# 100%|██████████| 2500/2500 [04:03<00:00, 10.27it/s]
#               precision    recall  f1-score   support
#
#            0       0.74      0.82      0.78      1220
#            1       0.81      0.73      0.77      1280
#
#     accuracy                           0.77      2500
#    macro avg       0.77      0.77      0.77      2500
# weighted avg       0.78      0.77      0.77      2500
