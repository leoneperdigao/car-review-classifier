import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from .text_processing_pipeline_base import TextPreprocessingPipelineBase


class TextCountVectorizerPipeline(TextPreprocessingPipelineBase):
    """
    A pipeline for preprocessing text data, splitting it into train and test datasets,
    vectorizing the text data.
    """
    def __init__(
            self,
            text_column="text",
            label_column="label",
            positive_label="Pos",
            negative_label="Neg",
            language="english",
            test_size=0.2
    ):
        super().__init__(
            text_column=text_column,
            label_column=label_column,
            positive_label=positive_label,
            negative_label=negative_label,
            language=language,
            test_size=test_size
        )

    def pre_process(self):
        """
        Preprocesses the text data in the input dataset by cleaning, splitting into train and test
        datasets, vectorizing the text data.

        Returns:
            tuple: A tuple containing the vectorized training data, training labels, vectorized
                testing data, and testing labels.
        """
        cleaned_data = self.clean_text(self.data_source)
        train_data, test_data = self.split_train_test(cleaned_data)
        X_train, y_train, X_test, y_test = self.vectorize_text(train_data, test_data)
        return X_train, y_train, X_test, y_test

    def vectorize_text(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
        """
        Vectorizes the text data in the input datasets using CountVectorizer and performs feature
        selection using SelectKBest.

        The CountVectorizer from scikit-learn handle unseen words in the test set by default.
        Therefore, this pipeline handles unseen words in the test set by default. When transforming the test data,
        any words that are not part of the training data vocabulary are ignored. This means that the model
        does not attempt to process or classify based on unseen words.

        Args:
            train_data (pandas.DataFrame): The training data.
            test_data (pandas.DataFrame): The testing data.

        Returns:
            tuple: A tuple containing the vectorized training data, training labels, vectorized
                testing data, and testing labels.
        """
        vectorizer = CountVectorizer(binary=True)
        X_train_raw = vectorizer.fit_transform(train_data[self.text_column])
        y_train = train_data[self.label_column]

        # Address data leakage
        vocabulary = pd.DataFrame.sparse.from_spmatrix(X_train_raw, columns=vectorizer.get_feature_names_out())

        # Remove words that only appear in one class of the training data
        pos_reviews = ' '.join(train_data[train_data[self.label_column] == self.positive_label][self.text_column])
        neg_reviews = ' '.join(train_data[train_data[self.label_column] == self.negative_label][self.text_column])

        pos_word_freq = pos_reviews.lower().split().count
        neg_word_freq = neg_reviews.lower().split().count

        filtered_vocabulary = [word for word in vocabulary if pos_word_freq(word) > 0 and neg_word_freq(word) > 0]

        # Use the filtered_vocabulary from the training data for the test data
        vectorizer = CountVectorizer(binary=True, vocabulary=filtered_vocabulary)
        X_train = vectorizer.fit_transform(train_data[self.text_column])
        X_test = vectorizer.transform(test_data[self.text_column])
        y_test = test_data[self.label_column]

        # Log the first 5 samples of the vectorized data
        self.logger.info(f"Vectorized data (first {self.num_samples_to_log} samples):\n%s", X_train_raw[:5].toarray())

        return X_train, y_train, X_test, y_test
