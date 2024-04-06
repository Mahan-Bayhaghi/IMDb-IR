import json
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def load_stopwords(filepath):
    stopwords = []
    absolute_path = "D:/Sharif/Daneshgah stuff/term 6/mir/project/IMDb-IR/Logic/core/"
    with open(absolute_path + filepath, 'r') as file:
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
        self.stopwords = load_stopwords("/stopwords.txt")
        # download nltk modules
        # nltk.download('omw-1.4')
        # nltk.download('wordnet')
        # nltk.download('punkt')

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        preprocessed_documents = []
        for document in self.documents:
            fields_to_preprocess = [("summaries", True), ("synopsis", True), ("first_page_summary", False), ("genres", True), ("stars", True)]
            preprocessed_document = self.preprocess_one_document(document, fields_to_preprocess=fields_to_preprocess)
            preprocessed_documents.append(preprocessed_document)
        return preprocessed_documents

    def light_preprocess_one_text(self, text):
        text = self.remove_links(text)
        text = self.remove_punctuations(text)
        text = self.remove_stopwords(text)
        text = ''.join(text).lower()
        return text

    def preprocess_one_text(self, text):
        text = self.remove_links(text)
        text = self.remove_punctuations(text)
        text = self.remove_stopwords(text)
        text = self.normalize(''.join(text))
        # the way I implemented the class, tokenization is really not needed !
        # text = self.tokenize(text)
        return text

    def preprocess_one_document(self, document, fields_to_preprocess=None):
        if fields_to_preprocess is None:  # document is a text
            return self.preprocess_one_text(text=document)
        else:  # document is dict
            for field, is_list in fields_to_preprocess:
                try:
                    if is_list:
                        items = document[field]
                        new_items = []
                        for item in items:
                            new_items.append(self.preprocess_one_text(item))
                        document[field] = new_items
                    else:
                        document[field] = self.preprocess_one_text(document[field])
                except Exception as e:
                    print(e)
            return document

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
        return word_tokenize(text)

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        str
            The string of words with stopwords removed.
        """
        words = text.split()
        cleaned_words = [word for word in words if word.lower() not in self.stopwords]
        return ' '.join(cleaned_words)


def preprocess_dataset(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    all_movies = [movie for movie in data]
    preprocessor = Preprocessor(all_movies)
    preprocessed_movies = preprocessor.preprocess()
    with open(filepath.replace(".json", "_preprocessed.json"), 'w') as file:
        json.dump(preprocessed_movies, file, indent=4)


def main():
    preprocess_dataset("../IMDB_crawled.json")
    preprocess_dataset("./LSHFakeData.json")


if __name__ == "__main__":
    main()
