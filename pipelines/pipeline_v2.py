from typing import List

import pandas as pd
import re

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


class PipelineV2:
    """
    A pipelines for preprocessing text data, splitting it into train and test datasets,
    vectorizing the text data, and performing feature selection.
    """
    def __init__(
            self,
            text_column="text",
            label_column="label",
            positive_label="Pos",
            negative_label="Neg",
            language="english",
            test_size=0.2,
            max_features=2000,
            ngram_range=(1, 2)
    ):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')

        self.data_source = None
        self.text_column = text_column
        self.label_column = label_column
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.language = language
        self.test_size = test_size
        self.max_features = max_features
        self.ngram_range = ngram_range

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

    def pre_process(self):
        """
        Preprocesses the text data in the input dataset by cleaning, splitting into train and test
        datasets, vectorizing the text data, performing feature selection, and reducing dimensionality.

        Returns:
            tuple: A tuple containing the processed training data, training labels, processed
                testing data, and testing labels.
        """
        cleaned_data = self.__clean_text(self.data_source)
        train_data, test_data = self.__split_train_test(cleaned_data)
        X_train, y_train, X_test, y_test = self.__vectorize_text(train_data, test_data)

        # Reduce dimensionality
        X_train_reduced, X_test_reduced = PipelineV2.__reduce_dimensionality(X_train, X_test)

        return X_train_reduced, y_train, X_test_reduced, y_test

    def __split_train_test(self, data: pd.DataFrame):
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
    def __remove_punctuation(text: str) -> str:
        return re.sub(r'\W', ' ', text).strip()

    @staticmethod
    def __stem_tokens(tokens: List[str]) -> List[str]:
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

    @staticmethod
    def __lemmatize_tokens(tokens: List[str]) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    def __remove_stopwords(self, tokens: List[str]) -> List[str]:
        stop_words = set(stopwords.words(self.language))
        return [token for token in tokens if token not in stop_words]

    def __clean_text(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the text data in the input dataset by removing punctuation, stemming, lemmatizing,
        and removing stop words.

        Args:
            data (pandas.DataFrame): The input dataset.

        Returns:
            pandas.DataFrame: The cleaned input dataset.
        """
        data[self.text_column] = data[self.text_column].apply(PipelineV2.__remove_punctuation)

        data[self.text_column] = data[self.text_column].apply(lambda x: word_tokenize(x))
        data[self.text_column] = data[self.text_column].apply(PipelineV2.__stem_tokens)
        data[self.text_column] = data[self.text_column].apply(PipelineV2.__lemmatize_tokens)
        data[self.text_column] = data[self.text_column].apply(self.__remove_stopwords)

        data[self.text_column] = data[self.text_column].apply(lambda x: ' '.join(x))

        return data

    @staticmethod
    def __reduce_dimensionality(X_train, X_test, n_components=100):
        """
        Reduces the dimensionality of the input data using TruncatedSVD.

        Args:
            X_train (array-like): The training data.
            X_test (array-like): The testing data.
            n_components (int, optional): The number of components to keep.
                If not provided, the method will keep 100 components by default.

        Returns:
            tuple: A tuple containing the dimensionality reduced training data, and testing data.
        """
        pca = TruncatedSVD(n_components=n_components)
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)

        return X_train_reduced, X_test_reduced

    def __vectorize_text(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
        """
        Vectorizes the text data in the input datasets using TfidfVectorizer and performs feature
        selection using SelectKBest.

        Args:
            train_data (pandas.DataFrame): The training data.
            test_data (pandas.DataFrame): The testing data.

        Returns:
            tuple: A tuple containing the vectorized training data, training labels, vectorized
                testing data, and testing labels.
        """
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
        X_train_raw = vectorizer.fit_transform(train_data[self.text_column])
        y_train = train_data[self.label_column]

        # Address data leakage
        vocabulary = pd.DataFrame.sparse.from_spmatrix(X_train_raw, columns=vectorizer.get_feature_names_out())
        # Remove words that only appear in one class of the training data
        pos_reviews = ' '.join(train_data[train_data[self.label_column] == self.positive_label][self.text_column])
        neg_reviews = ' '.join(train_data[train_data[self.label_column] == self.negative_label][self.text_column])

        pos_words_freq = pos_reviews.lower().split().count
        neg_words_freq = neg_reviews.lower().split().count

        filtered_vocabulary = [word for word in vocabulary if pos_words_freq(word) > 0 and neg_words_freq(word) > 0]

        # Use the filtered_vocabulary from the training data for the test data
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, vocabulary=filtered_vocabulary)
        X_train = vectorizer.fit_transform(train_data[self.text_column])
        X_test = vectorizer.transform(test_data[self.text_column])
        y_test = test_data[self.label_column]

        # Perform feature selection using SelectKBest
        k = min(X_train.shape[1], self.max_features)
        selector = SelectKBest(mutual_info_classif, k=k)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        return X_train, y_train, X_test, y_test
