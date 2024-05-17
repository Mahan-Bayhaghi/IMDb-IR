import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        self.classes, y_indices = np.unique(y, return_inverse=True)
        self.num_classes = len(self.classes)
        self.number_of_samples, self.number_of_features = x.shape
        print(f"num classes is {self.num_classes}")
        print(f"num samples is {self.number_of_samples}")

        self.prior = np.zeros(self.num_classes)
        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))

        for class_index, class_value in enumerate(self.classes):
            x_class = x[y_indices == class_index]
            self.prior[class_index] = x_class.shape[0] / self.number_of_samples
            # we are using standard laplace smoothing
            self.feature_probabilities[class_index, :] = (x_class.sum(axis=0) + self.alpha) / (
                    x_class.sum() + self.alpha * self.number_of_features)

        self.log_probs = np.log(self.feature_probabilities)
        return self

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
        log_prior = np.log(self.prior)
        log_likelihood = x @ self.log_probs.T
        log_posterior = log_prior + log_likelihood
        return self.classes[np.argmax(log_posterior, axis=1)]

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
        return classification_report(y, y_pred)
        pass

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        embeddings = self.cv.transform(sentences)
        predictions = self.predict(embeddings.toarray())
        positive_count = np.sum(predictions == 'positive')
        total_count = len(predictions)
        percent_positive = (positive_count / total_count) * 100
        return percent_positive
        pass


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    file_path = 'IMDB dataset.csv'
    review_loader = ReviewLoader(file_path)
    review_loader.load_data \
        (dataset_path="IMDB Dataset.csv", load_fasttext_model=False,
         fasttext_model_path=None, dont_train=True)

    count_vectorizer = CountVectorizer(dtype="uint16")
    x = count_vectorizer.fit_transform(review_loader.review_texts)
    print("count vectorizer done")
    print(f"cv : {x.toarray()}")
    print(f"cv.shape : {x.toarray().shape}")

    y = np.array(review_loader.sentiments)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=40)

    nb_classifier = NaiveBayes(count_vectorizer)
    nb_classifier.fit(x_train, y_train)

    print(nb_classifier.prediction_report(x_test, y_test))

##################### test results ##########################
#   100%|██████████| 50000/50000 [00:03<00:00, 14679.57it/s]
#   count_vectorizer done
#   cv : [[0 0 0 ... 0 0 0]
#         [0 0 0 ... 0 0 0]
#         [0 0 0 ... 0 0 0]
#         ...
#         [0 0 0 ... 0 0 0]
#         [0 0 0 ... 0 0 0]
#         [0 0 0 ... 0 0 0]]
#   cv.shape : (50000, 101895)
#   num classes is 2
#   num samples is 45000
#               precision    recall  f1-score   support
#
#            0       0.84      0.89      0.86      2563
#            1       0.88      0.82      0.85      2437
#
#     accuracy                           0.86      5000
#    macro avg       0.86      0.86      0.86      5000
# weighted avg       0.86      0.86      0.86      5000
