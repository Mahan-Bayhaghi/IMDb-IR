import fasttext
from scipy.spatial import distance

from Logic.core import path_access
from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader


class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None

    def train(self, texts_path, epochs=5):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        epochs : int
            How many epoches to train the fasttext unsupervised training
        texts_path : str
            Address to the training file of the FastText model.
        """
        self.model = fasttext.train_unsupervised(input=texts_path, model=self.method, epoch=epochs)
        # pass

    def get_query_embedding(self, query):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        return self.model.get_sentence_vector(query)
        # pass

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        analogies = self.model.get_analogies(word1, word2, word3)
        print(analogies)
        return analogies[0][1]

    def detailed_analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        # Obtain word embeddings for the words in the analogy
        embeddings_word1 = self.model[word1]
        embeddings_word2 = self.model[word2]
        embeddings_word3 = self.model[word3]

        # Perform vector arithmetic
        analogy_vector = embeddings_word2 - embeddings_word1 + embeddings_word3
        word_embeddings = {word: self.model[word] for word in self.model.words}

        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        candidate_words = set(self.model.words) - {word1, word2, word3}  # don't use input words for candidate words !

        # Find the word whose vector is closest to the result vector
        closest_word = None
        min_distance = float('inf')
        for word in candidate_words:
            distance_to_analogy_vector = distance.cosine(word_embeddings[word], analogy_vector)
            if distance_to_analogy_vector < min_distance:
                min_distance = distance_to_analogy_vector
                closest_word = word

        return closest_word

    def save_model(self, path='FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)
        # pass

    def load_model(self, path="./FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)
        # pass

    def prepare(self, dataset_path, mode, epochs=5, save=False, path='./FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        epochs : int
            How many epoches to train
        dataset_path : str
            Address to the training data of the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset_path, epochs=epochs)
        if mode == 'load':
            self.load_model(path)
        if mode == 'save' or save:
            self.save_model(path)


if __name__ == "__main__":

    # importing non-preprocessed training data

    path = path_access.path_to_logic() + "core/indexer/saved_indexes/"
    ft_data_loader = FastTextDataLoader(path)
    texts, _ = ft_data_loader.create_train_data()
    print(f"training dataset loaded successfully")

    # preprocessing training data and saving it

    # preprocessed_texts = [preprocess_text(text) for text in texts]
    # with open('preprocessed_training_data.txt', 'w', encoding="utf-8") as f:
    #     for line in preprocessed_texts:
    #         f.write("%s\n" % line)
    # print(f"training dataset preprocessed and saved successfully")

    ft_model = FastText(method='skipgram')

    # loading model if it has already been trained (bin file)
    # ft_model.load_model()
    # ft_model.prepare(None, mode="load")

    # training model
    ft_model.train("./preprocessed_training_data.txt")
    ft_model.prepare("./preprocessed_training_data.txt", mode="train", epochs=50)
    print(f"model trained successfully")

    # saving model
    ft_model.prepare(None, mode="save", save=True)
    print(f"model saved successfully")

    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "king"
    word3 = "queen"
    print(
        f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
    print(
        f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.detailed_analogy(word1, word2, word3)}")
