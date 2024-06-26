import json

import torch
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup


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
        self.model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=top_n_genres, problem_type='multi_label_classification')
        print("model initialized")

        self.train_set = None
        self.val_set = None
        self.test_set = None


    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        # TODO: Implement dataset loading logic
        with open(self.file_path, 'r') as f:
            self.dataset = json.load(f)
        print("dataset loaded")

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        # TODO: Implement genre filtering and visualization logic
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

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        # TODO: Implement dataset splitting logic
        # split dataset sizes
        total_len = len(self.dataset)
        test_len = int(total_len * test_size)
        val_len = int(total_len * val_size)
        train_len = total_len - test_len - val_len
        train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_len, val_len, test_len])

        self.train_set = train_dataset
        self.val_set = val_dataset
        self.test_set = test_dataset
        print(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        # TODO: Implement dataset creation logic
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        # TODO: Implement BERT fine-tuning logic
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size, shuffle=False)

        # setting up optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=weight_decay)
        training_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)

        # training loop of fine-tuning
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                optimizer.zero_grad()
                # forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                scheduler.step()

            avg_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f}")

            # validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    labels = batch['labels']
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                avg_val_loss = val_loss / len(val_loader)
                print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {avg_val_loss:.4f}")
        print("Finished Fine-tuning")

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        # TODO: Implement metric computation logic

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        # TODO: Implement model evaluation logic

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        # TODO: Implement model saving logic


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
        # TODO: Implement initialization logic
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
        # TODO: Implement item retrieval logic
        input_ids = torch.tensor(self.encodings['input_ids'][idx])
        attention_mask = torch.tensor(self.encodings['attention_mask'][idx])
        label = torch.tensor(self.labels[idx])

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label}

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        # TODO: Implement length computation logic
        return len(self.labels)