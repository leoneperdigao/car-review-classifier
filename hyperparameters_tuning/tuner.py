import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


def fine_tune_hyperparameters(X_train, y_train):
    # Define the classifier
    clf = MultinomialNB()

    # Define the hyperparameter grid
    param_grid = {
        'alpha': np.logspace(-3, 0, 20)
    }

    # Set up the grid search with cross-validation
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and their corresponding accuracy
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best accuracy:", grid_search.best_score_)

    # Return the best classifier
    return grid_search.best_estimator_
