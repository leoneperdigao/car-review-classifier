import pandas as pd
from sklearn import svm


class SVMSentimentClassifier:
    def __init__(self, C=0.1, coef0=1, degree=4, gamma=1.0, kernel='poly'):
        self.clf = svm.SVC(C=C, coef0=coef0, degree=degree, gamma=gamma, kernel=kernel)

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame):
        return self.clf.predict(X_test)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        y_pred = self.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        return accuracy
