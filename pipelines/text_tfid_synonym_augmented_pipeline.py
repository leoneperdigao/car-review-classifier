import random

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from .text_processing_pipeline_base import TextPreprocessingPipelineBase


class TextTfidfSynonymAugmentedPipeline(TextPreprocessingPipelineBase):
    """
    An improved pipeline for preprocessing text data, splitting it into train and test datasets,
    vectorizing the text data, and performing synonym replacement.
    """
    def __init__(
            self,
            text_column="text",
            label_column="label",
            positive_label="Pos",
            negative_label="Neg",
            language="english",
            test_size=0.2,
            ngram_range=(1, 2),
            load_saved_augmented_data=False,
    ):
        super().__init__(
            text_column=text_column,
            label_column=label_column,
            positive_label=positive_label,
            negative_label=negative_label,
            language=language,
            test_size=test_size
        )
        self.synonym_cache = {}
        self.stop_words = set(stopwords.words(self.language))
        self.ngram_range = ngram_range
        self.load_saved_augmented_data = load_saved_augmented_data

    def pre_process(self):
        """
        Preprocesses the text data in the input dataset by cleaning, splitting into train and test
        datasets, vectorizing the text data and synonym replacement.

        Returns:
            tuple: A tuple containing the processed training data, training labels, processed
                testing data, and testing labels.
        """
        cleaned_data = self.clean_text(self.data_source)

        if not self.load_saved_augmented_data:
            # Add an augmentation step
            augmented_data = cleaned_data.copy()
            augmented_data[self.text_column] = cleaned_data[self.text_column].apply(lambda x: self.__synonym_replacement(x, n=1))
            augmented_data.to_csv("data/augmented_data.csv", index=False)
        else:
            # Load augmented_data from the CSV file
            augmented_data = pd.read_csv("data/augmented_data.csv")

        # Combine the original and augmented data
        combined_data = pd.concat([cleaned_data, augmented_data], ignore_index=True)

        train_data, test_data = self.split_train_test(combined_data)
        X_train, y_train, X_test, y_test = self.vectorize_text(train_data, test_data)

        return X_train, y_train, X_test, y_test

    def vectorize_text(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
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
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, min_df=0.05, max_df=0.95)
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
        vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, vocabulary=filtered_vocabulary, min_df=0.05, max_df=0.95)
        X_train = vectorizer.fit_transform(train_data[self.text_column])
        X_test = vectorizer.transform(test_data[self.text_column])
        y_test = test_data[self.label_column]

        return X_train, y_train, X_test, y_test

    def __synonym_replacement(self, text: str, n: int) -> str:
        """
        Replaces n words in the input text with their synonyms.

        Args:
            text (str): The input text.
            n (int): The number of words to replace.

        Returns:
            str: The augmented text.
        """
        words = word_tokenize(text)
        num_words = len(words)

        if n > num_words:
            n = num_words

        non_stopword_idxs = [i for i, word in enumerate(words) if word.lower() not in self.stop_words]
        if n > len(non_stopword_idxs):
            n = len(non_stopword_idxs)
        idxs = random.sample(non_stopword_idxs, n)

        new_words = words.copy()

        for i in idxs:
            if words[i] not in self.synonym_cache:
                synonyms = []
                for syn in wordnet.synsets(words[i]):
                    for lemma in syn.lemmas():
                        synonyms.append(lemma.name())
                self.synonym_cache[words[i]] = synonyms
            else:
                synonyms = self.synonym_cache[words[i]]

            if len(synonyms) > 0:
                new_word = random.choice(synonyms)
                new_words[i] = new_word.replace('_', ' ')

        return ' '.join(new_words)
