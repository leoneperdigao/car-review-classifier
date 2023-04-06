import os
import time

from models.svm_sentiment_classifier import SVMSentimentClassifier
from models.naive_bayes_sentiment_classifier import NaiveBayesSentimentClassifier
from optimisation.naive_bayes_multinomialnb_tuner import fine_tune as naive_bayes_fine_tune
from optimisation.svm_tuner import fine_tune as svc_fine_tune
from pipelines import TextCountVectorizerPipeline, TextTfidfSynonymAugmentedPipeline
from report import report

if __name__ == '__main__':
    pipeline_v1 = TextCountVectorizerPipeline(
        text_column="Review",
        label_column="Sentiment",
        positive_label="Pos",
        negative_label="Neg",
        language="english",
        test_size=0.2
    )
    pipeline_v1.add_data_source('./data/car-reviews.csv')
    X_train, y_train, X_test, y_test = pipeline_v1.pre_process()

    naive_model = NaiveBayesSentimentClassifier()
    naive_model.train(X_train, y_train)
    y_pred = naive_model.predict(X_test)
    report.report(naive_model.get_classifier(), X_test, y_pred, y_test)

    pipeline_v2 = TextTfidfSynonymAugmentedPipeline(
        text_column="Review",
        label_column="Sentiment",
        positive_label="Pos",
        negative_label="Neg",
        language="english",
        test_size=0.2,
        ngram_range=(1, 2),
    )
    pipeline_v2.add_data_source('./data/car-reviews.csv')
    start_time = time.time()
    X_train, y_train, X_test, y_test = pipeline_v2.pre_process()
    pre_processing_time = time.time() - start_time
    print("Pre-processing time: %.5f seconds" % pre_processing_time)

    svc_model = SVMSentimentClassifier()
    start_time = time.time()
    svc_model.train(X_train, y_train)
    train_time = time.time() - start_time
    print("Training time: %.5f seconds" % train_time)

    y_pred = svc_model.predict(X_test)
    report.report(svc_model.get_classifier(), X_test, y_pred, y_test)

    if os.environ.get("FINE_TUNE"):
        naive_bayes_fine_tune(X_train, y_train)
        svc_fine_tune(X_train, y_train)







