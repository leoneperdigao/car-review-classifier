import pandas as pd

from sklearn.naive_bayes import MultinomialNB


class NaiveBayesSentimentClassifier:
    """This class is a simple implementation of Multinomial Naive Bayes model
    and acts just a wrapper.
    """

    def __init__(self, lambda_reg=0.1389):
        self.clf = MultinomialNB(alpha=lambda_reg)

    def get_classifier(self):
        return self.clf

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame):
        return self.clf.predict(X_test)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        y_pred = self.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        return accuracy
