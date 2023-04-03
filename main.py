from hyperparameters_tuning.tuner import fine_tune_hyperparameters
from model import SentimentClassifier
from pipeline import Pipeline
from report import report

if __name__ == '__main__':
    pipeline = Pipeline(
        text_column="Review",
        label_column="Sentiment",
        positive_label="Pos",
        negative_label="Neg",
        language="english",
        test_size=0.2
    )
    pipeline.add_data_source('./data/car-reviews.csv')
    X_train, y_train, X_test, y_test = pipeline.pre_process()

    model = SentimentClassifier(
        lambda_reg=1.00
    )

    model.train(X_train, y_train)


    # Fine-tune the hyperparameters and get the best classifier
    # fine_tune_hyperparameters(X_train, y_train)

    y_pred = model.predict(X_test)
    print(model.evaluate(X_test, y_test))

    report.report(y_pred, y_test)

