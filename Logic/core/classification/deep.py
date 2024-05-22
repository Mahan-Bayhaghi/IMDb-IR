import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Logic.core.classification.data_loader import ReviewLoader
from Logic.core.classification.basic_classifier import BasicClassifier


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):

        # print(f"types of embeddings is {type(embeddings.astype('float'))}")
        self.embeddings = torch.FloatTensor(embeddings.astype('float32'))
        self.labels = torch.LongTensor(labels.astype('int'))

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embeddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = 'mps' if torch.backends.mps.is_available else 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else self.device
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def fit(self, x, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        train_dataset = ReviewDataSet(x, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            # one step of training done
            train_loss = 0
            for embeddings, labels in train_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            if self.test_loader is not None:
                self._eval_epoch(self.test_loader, self.model)
        self.model.load_state_dict(self.best_model)
        return self

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        test_dataset = ReviewDataSet(x, np.zeros((x.shape[0],), dtype=np.int64))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for embeddings, _ in tqdm(test_loader):
                embeddings = embeddings.to(self.device)
                outputs = self.model(embeddings)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        return np.array(predictions)
        # pass

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        model.eval()
        eval_loss = 0
        predicted_labels = []
        true_labels = []

        with torch.no_grad():
            for embeddings, labels in dataloader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                outputs = model(embeddings)
                loss = self.criterion(outputs, labels)
                eval_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        f1 = f1_score(true_labels, predicted_labels, average='macro')
        # print(f"Validation Loss: {eval_loss / len(dataloader)}, F1 Score: {f1}")
        if f1 > getattr(self, "best_f1", 0):
            self.best_f1 = f1
            self.best_model = model.state_dict()

        return eval_loss / len(dataloader), predicted_labels, true_labels, f1

    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        test_dataset = ReviewDataSet(X_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return self

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        y_pred = self.predict(x)
        # I have use dtype = 'object' and thus, classificatn_metrics can't quite understand it should be treated as
        # an int, so the type casting is much needed !
        return classification_report(y.astype("int"), y_pred.astype("int"))


# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    pass
    file_path = "IMDB Dataset small.csv"
    review_loader = ReviewLoader(file_path)
    review_loader.load_data(dataset_path=file_path, load_fasttext_model=False, fasttext_model_path="./IMDB_dataset_FastText_small.bin")
    # review_loader.load_data(load_fasttext_model=True, fasttext_model_path="./IMDB_dataset_FastText.bin")
    review_loader.get_embeddings()

    x_train, x_test, y_train, y_test = review_loader.split_data(test_data_ratio=0.2)

    deep_model_classifier = DeepModelClassifier(in_features=x_train.shape[1], num_classes=2, batch_size=32, num_epochs=50)
    deep_model_classifier.set_test_dataloader(x_test, y_test)
    deep_model_classifier.fit(x_train, y_train)
    print("Model fitted")

    print(deep_model_classifier.prediction_report(x_test, y_test))

# for the following test, I've used small dataset which is first 25k rows of original dataset to reduce the time
# by increasing size of dataset, classification works on larger data as well
##################### test results ##########################
# 100%|██████████| 25000/25000 [00:01<00:00, 14910.63it/s]
# Read 5M words
# Number of words:  45612
# Number of labels: 0
# Progress: 100.0% words/sec/thread:   16956 lr:  0.000000 avg.loss:  2.234760 ETA:   0h 0m 0s
# fasttext model trained and saved
# 100%|██████████| 25000/25000 [00:26<00:00, 940.72it/s]
# x_train is 20000 objects each with shape 50
# Using device: cuda
# 100%|██████████| 50/50 [04:29<00:00,  5.38s/it]
#   0%|          | 0/157 [00:00<?, ?it/s]Model fitted
# 100%|██████████| 157/157 [00:00<00:00, 474.01it/s]
#               precision    recall  f1-score   support
#
#            0       0.86      0.88      0.87      2524
#            1       0.88      0.86      0.87      2476
#
#     accuracy                           0.87      5000
#    macro avg       0.87      0.87      0.87      5000
# weighted avg       0.87      0.87      0.87      5000
