import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC()
        self.label_encoder = LabelEncoder()

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        y = y.astype("int")
        self.model.fit(x, y)

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
        return self.model.predict(x)

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
        # I have use dtype = 'object' and thus, classificatn_metrics can't quite understand it should be treated as
        # an int, so the type casting is much needed !
        return classification_report(y.astype("int"), y_pred.astype("int"))


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    file_path = "IMDB dataset.csv"
    review_loader = ReviewLoader(file_path)
    review_loader.load_data(dataset_path=file_path, load_fasttext_model=False, fasttext_model_path="./IMDB_dataset_FastText.bin")

    # review_loader.load_data(load_fasttext_model=True, fasttext_model_path="./IMDB_dataset_FastText.bin")

    review_loader.get_embeddings()

    x_train, x_test, y_train, y_test = review_loader.split_data(test_data_ratio=0.2)

    svm_classifier = SVMClassifier()
    svm_classifier.fit(x_train, y_train)
    print("SVM fitted")
    print(svm_classifier.prediction_report(x_test, y_test))

##################### test results ##########################
# 100%|██████████| 50000/50000 [00:01<00:00, 27385.94it/s]
# Read 11M words
# Number of words:  71131
# Number of labels: 0
# Progress: 100.0% words/sec/thread:   26662 lr:  0.000000 avg.loss:  2.183791 ETA:   0h 0m 0s
#   0%|          | 0/50000 [00:00<?, ?it/s]fasttext model trained and saved
# 100%|██████████| 50000/50000 [00:29<00:00, 1686.14it/s]
# x_train is 40000 objects each with shape 50
# SVM fitted
#               precision    recall  f1-score   support
#
#            0       0.87      0.87      0.87      5100
#            1       0.87      0.87      0.87      4900
#
#     accuracy                           0.87     10000
#    macro avg       0.87      0.87      0.87     10000
# weighted avg       0.87      0.87      0.87     10000
