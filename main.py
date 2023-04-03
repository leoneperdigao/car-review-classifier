from optimisation.svm_tuner import fine_tune
from models.svm_classifier import SVMSentimentClassifier
from pipelines import PipelineV2
from report import report

if __name__ == '__main__':
    pipeline = PipelineV2(
        text_column="Review",
        label_column="Sentiment",
        positive_label="Pos",
        negative_label="Neg",
        language="english",
        test_size=0.2
    )
    pipeline.add_data_source('./data/car-reviews.csv')
    X_train, y_train, X_test, y_test = pipeline.pre_process()

    model = SVMSentimentClassifier()

    model.train(X_train, y_train)

    # Fine-tune the hyperparameters and get the best classifier
    # fine_tune(X_train, y_train)

    y_pred = model.predict(X_test)
    report.report(y_pred, y_test)

