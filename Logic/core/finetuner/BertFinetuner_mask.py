import json
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import seaborn as sns


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        # TODO: Implement initialization logic
        self.top_n_genres = top_n_genres
        self.file_path = file_path
        self.dataset = []
        self.top_genres = []

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("tokenizer initialized")
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=top_n_genres, problem_type="multi_label_classification")
        print("model initialized")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device is {self.device}")
        self.model.to(self.device)
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        # TODO: Implement dataset loading logic
        with open(self.file_path, 'r') as f:
            dataset = json.load(f)
        for movie in dataset:
            if len(movie['genres']) > 0 and movie['first_page_summary'] is not None:
                movie_pair = {'genres': movie['genres'], 'first_page_summary': movie['first_page_summary']}
                self.dataset.append(movie_pair)
        self.dataset = self.dataset[:1000]
        print("dataset loaded")

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        # first, let's get the count of each genre in the dataset
        genre_distribution = {}
        for movie in self.dataset:
            genres = movie.get('genres', [])
            for genre in genres:
                genre_distribution[genre] = genre_distribution.get(genre, 0) + 1

        # now we will sort them in descending order
        sorted_genres = sorted(genre_distribution.items(), key=lambda x: x[1], reverse=True)
        # we needed only top_n_genres
        self.top_genres = [genre for genre, count in sorted_genres[:self.top_n_genres]]
        # filter dataset entries to include only the top genres
        filtered_dataset = [movie for movie in self.dataset if any(genre in movie.get('genres', []) for genre in self.top_genres)]
        self.dataset = filtered_dataset
        print(f"Filtered dataset to include top {self.top_n_genres} genres: {self.top_genres}")

    def split_dataset(self, test_size=0.3, val_size=0.4):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        # split dataset sizes
        total_len = len(self.dataset)
        test_len = int(total_len * test_size)
        train_len = int(total_len - test_len)
        train_dataset, test_dataset = random_split(self.dataset, [train_len, test_len])
        val_len = int(len(test_dataset) * val_size)
        test_len = int(len(test_dataset) - val_len)
        val_dataset, test_dataset = random_split(test_dataset, [val_len, test_len])

        self.train_set = train_dataset
        self.val_set = val_dataset
        self.test_set = test_dataset
        print(f"Dataset split: Train={len(self.train_set)}, Validation={len(self.val_set)}, Test={len(self.test_set)}")

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        return IMDbDataset(encodings=encodings, labels=labels)

    def extract_text_and_labels(self, dataset):
        dataset_texts = [movie['first_page_summary'] for movie in dataset]
        dataset_labels = [list(set(movie['genres']).intersection(set(self.top_genres))) for movie in dataset]
        return dataset_texts, dataset_labels

    def fine_tune_bert(self, epochs=20, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        mlb = MultiLabelBinarizer(classes=self.top_genres)

        training_texts, training_labels = self.extract_text_and_labels(self.train_set)
        training_labels = mlb.fit_transform(training_labels)
        training_encodings = self.tokenizer(training_texts, truncation=True, padding=True, max_length=512)
        # print("training encodings : ", training_encodings[:4])
        training_set = self.create_dataset(training_encodings, training_labels)

        val_texts, val_labels = self.extract_text_and_labels(self.val_set)
        val_labels = mlb.transform(val_labels)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=512)
        # print("validation encodings:", val_encodings[:4])
        val_set = self.create_dataset(val_encodings, val_labels)

        # defining training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            # metric_for_best_model="f1",
            # greater_is_better=True
        )
        # our main trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=training_set,
            eval_dataset=val_set,
            compute_metrics=self.compute_metrics
        )
        trainer.train()
        print("Finished Fine-tuning")

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        predictions, labels = pred.predictions, pred.label_ids
        probabilities = torch.sigmoid(torch.tensor(predictions))    # softmax
        predictions = (probabilities > 0.5).cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=predictions, average='samples')
        return {
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy_score(labels, predictions),
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        mlb = MultiLabelBinarizer(classes=self.top_genres)

        test_text, test_labels = self.extract_text_and_labels(self.test_set)
        test_encodings = self.tokenizer(test_text, truncation=True, padding=True, max_length=512)
        test_labels = mlb.fit_transform(test_labels)
        test_dataset = self.create_dataset(test_encodings, test_labels)

        trainer = Trainer(model=self.model, compute_metrics=self.compute_metrics)
        results = trainer.evaluate(test_dataset)
        print(f"Test set evaluation results: {results}")
        predictions = trainer.predict(test_dataset).predictions
        probabilities = torch.sigmoid(torch.tensor(predictions))
        predicted_labels = (probabilities > 0.5).cpu().numpy()

        # Compute the confusion matrix
        cm = confusion_matrix(test_labels.argmax(axis=1), predicted_labels.argmax(axis=1))

        # Visualize the results and confusion matrix
        self.plot_confusion_matrix(cm)

    def plot_confusion_matrix(self, cm):
        """
        Plot the confusion matrix using Seaborn.

        Args:
            cm (ndarray): The confusion matrix.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.top_genres, yticklabels=self.top_genres)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        # TODO: Implement model saving logic
        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)


class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)
