import pandas as pd
import re

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

clf = MultinomialNB(alpha=0.05)


def read_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def preprocess(review):
    # Remove redundant formatting
    review = re.sub(r'\W', ' ', review)
    review = re.sub(r'\s+', ' ', review)

    # Stemming, lemmatization, and stop words removal
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(review)

    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens if token not in stop_words]

    return ' '.join(lemmatized_tokens)


def wrangling(data: pd.DataFrame):
    data['Review'] = data['Review'].apply(preprocess)


def split_train_test(data: pd.DataFrame, test_size=0.2):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=21, stratify=data['Sentiment'])
    return train_data, test_data


def vectorize_text(train_data: pd.DataFrame, test_data: pd.DataFrame):
    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    X_train_raw = vectorizer.fit_transform(train_data['Review'])
    y_train = train_data['Sentiment']

    # Address data leakage
    vocabulary = vectorizer.get_feature_names_out(input_features=X_train_raw)

    # Remove words that only appear in one class of the training data
    pos_reviews = ' '.join(train_data[train_data['Sentiment'] == 'Pos']['Review'])
    neg_reviews = ' '.join(train_data[train_data['Sentiment'] == 'Neg']['Review'])

    pos_word_freq = pos_reviews.lower().split().count
    neg_word_freq = neg_reviews.lower().split().count

    filtered_vocabulary = [word for word in vocabulary if pos_word_freq(word) > 0 and neg_word_freq(word) > 0]

    # Use the filtered_vocabulary from the training data for the test data
    vectorizer = CountVectorizer(binary=True, vocabulary=filtered_vocabulary, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_data['Review'])
    X_test = vectorizer.transform(test_data['Review'])
    y_test = test_data['Sentiment']

    # Feature selection
    percentage_of_features = 0.6
    k_value = int(len(filtered_vocabulary) * percentage_of_features)
    k_best = SelectKBest(chi2, k=k_value)  # Select the 5000 best features
    X_train = k_best.fit_transform(X_train, y_train)
    X_test = k_best.transform(X_test)

    return X_train, y_train, X_test, y_test


def train(X_train, y_train):
    clf.fit(X_train, y_train)


def predict(X_test):
    return clf.predict(X_test)
