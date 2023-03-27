from hyperparameters_tuning.tuner import fine_tune_hyperparameters
from model import classifier
from report import report

if __name__ == '__main__':
    dataframe = classifier.read_data('./data/car-reviews.csv')
    classifier.wrangling(dataframe)
    train_data, test_data = classifier.split_train_test(dataframe)
    X_train, y_train, X_test, y_test = classifier.vectorize_text(train_data, test_data)
    classifier.train(X_train, y_train)

    # Fine-tune the hyperparameters and get the best classifier
    fine_tune_hyperparameters(X_train, y_train)

    # y_pred = classifier.predict(X_test)
    #
    # report.report(y_pred, y_test)

