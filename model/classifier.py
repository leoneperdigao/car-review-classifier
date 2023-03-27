import pandas as pd
import re
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


# Data ingestion
data = pd.read_csv('../data/car-reviews.csv')

# Data wrangling
data['Review'] = data['Review'].apply(preprocess)

# Analysis and modeling
train_data, test_data = train_test_split(data, test_size=0.2, random_state=99, stratify=data['Sentiment'])

vectorizer = CountVectorizer(binary=True)
X_train_raw = vectorizer.fit_transform(train_data['Review'])
y_train = train_data['Sentiment']

# Address data leakage
vocabulary = vectorizer.get_feature_names_out()

# Remove words that only appear in one class of the training data
pos_reviews = ' '.join(train_data[train_data['Sentiment'] == 'Pos']['Review'])
neg_reviews = ' '.join(train_data[train_data['Sentiment'] == 'Neg']['Review'])

pos_word_freq = pos_reviews.lower().split().count
neg_word_freq = neg_reviews.lower().split().count

filtered_vocabulary = [word for word in vocabulary if pos_word_freq(word) > 0 and neg_word_freq(word) > 0]

vectorizer = CountVectorizer(binary=True, vocabulary=filtered_vocabulary)
X_train = vectorizer.fit_transform(train_data['Review'])
X_test = vectorizer.transform(test_data['Review'])
y_test = test_data['Sentiment']

clf = MultinomialNB(alpha=0.5)
clf.fit(X_train, y_train)
