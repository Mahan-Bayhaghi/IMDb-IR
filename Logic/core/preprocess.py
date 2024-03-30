import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize


def load_stopwords(filepath):
    stopwords = []
    with open(filepath, 'r') as file:
        for line in file:
            stopwords.append(line.strip())
    return stopwords


class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        # TODO
        self.documents = documents
        self.stopwords = load_stopwords("./stopwords.txt")
        # download nltk modules
        nltk.download('omw-1.4')
        nltk.download('wordnet')

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        # TODO
        return

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        text = text.lower()
        words = text.split()
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stemmed_words = [stemmer.stem(word) for word in words]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]
        return ' '.join(lemmatized_words)

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*',
                    r'<span[^>]*>.*?</span>', r'<a[^>]*>.*?</a>']
        combined_pattern = '|'.join(patterns)
        cleaned_text = re.sub(combined_pattern, '', text)
        return cleaned_text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        punctuation_pattern = r'[^\w\s]'
        cleaned_text = re.sub(punctuation_pattern, '', text)
        return cleaned_text

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        return

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        words = text.split()
        cleaned_words = [word for word in words if word.lower() not in self.stopwords]
        return ' '.join(cleaned_words)
