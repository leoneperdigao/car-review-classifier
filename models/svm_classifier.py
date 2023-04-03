import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder


class SVMSentimentClassifier:
    def __init__(self, C=0.1, coef0=1, degree=4, gamma=1.0, kernel='poly'):
        self.clf = svm.SVC(C=C, coef0=coef0, degree=degree, gamma=gamma, kernel=kernel)

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame):
        y_pred = self.clf.predict(X_test)
        label_encoder = LabelEncoder()
        label_encoder.fit(self.clf.classes_)
        return label_encoder.inverse_transform(y_pred)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        y_pred = self.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        return accuracy
