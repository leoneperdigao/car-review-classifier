import logging
from typing import List

import pandas as pd
import re

from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


class TextPreprocessingPipelineBase:
    """
    A base pipeline for ingesting and cleaning text data, splitting it into train and test datasets.
    """
    def __init__(
            self,
            text_column="text",
            label_column="label",
            positive_label="Pos",
            negative_label="Neg",
            language="english",
            test_size=0.2,
            num_samples_to_log=10,
    ):
        self.data_source = None
        self.text_column = text_column
        self.label_column = label_column
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.language = language
        self.test_size = test_size
        self.num_samples_to_log = num_samples_to_log

        self.stop_words = set(stopwords.words(self.language))

        # configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)

        # The default download location is the user's home folder.
        # Download NLTK resources is set to be done quietly to not expose the user's home folder name.
        nltk.download('punkt', quiet=True)  # download Punkt Sentence Tokenizer
        nltk.download('wordnet', quiet=True)  # download lexical database
        nltk.download('stopwords', quiet=True)  # download list of stopwords


    def pre_process(self):
        raise NotImplemented("This method must be implemented in the child class")

    def vectorize_text(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        raise NotImplemented("This method must be implemented in the child class")

    def add_data_source(self, file_path: str):
        """
        Adds an input dataset to the pipelines.

        Args:
            file_path (str): The file path of the input dataset.
        """
        if not self.data_source:
            self.data_source = pd.read_csv(file_path)
        else:
            self.data_source.append(pd.read_csv(file_path), ignore_index=True)

    def split_train_test(self, data: pd.DataFrame):
        """
        Splits the input dataset into training and testing data.

        Args:
            data (pandas.DataFrame): The input dataset.

        Returns:
            tuple: A tuple containing the training data, testing data, training labels, and
                testing labels.
        """
        return train_test_split(data, test_size=self.test_size, shuffle=True, stratify=data[self.label_column])

    @staticmethod
    def remove_punctuation(text: str) -> str:
        return re.sub(r'\W', ' ', text).strip()

    @staticmethod
    def stem_tokens(tokens: List[str]) -> List[str]:
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

    @staticmethod
    def lemmatize_tokens(tokens: List[str]) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        stop_words = set(stopwords.words(self.language))
        return [token for token in tokens if token not in stop_words]

    def clean_text(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the text data in the input dataset by removing punctuation, stemming, lemmatizing,
        and removing stop words.

        Args:
            data (pandas.DataFrame): The input dataset.

        Returns:
            pandas.DataFrame: The cleaned input dataset.
        """
        # make text case-insensitive
        data[self.text_column] = data[self.text_column].apply(lambda x: x.lower())
        # remove punctuation
        data[self.text_column] = data[self.text_column].apply(TextPreprocessingPipelineBase.remove_punctuation)

        data[self.text_column] = data[self.text_column].apply(lambda x: word_tokenize(x))
        data[self.text_column] = data[self.text_column].apply(TextPreprocessingPipelineBase.stem_tokens)
        data[self.text_column] = data[self.text_column].apply(TextPreprocessingPipelineBase.lemmatize_tokens)
        data[self.text_column] = data[self.text_column].apply(self.remove_stopwords)

        data[self.text_column] = data[self.text_column].apply(lambda x: ' '.join(x))

        return data
