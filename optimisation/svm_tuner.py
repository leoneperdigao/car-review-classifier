import numpy as np

from sklearn.svm import SVC

from optimisation.hyperparameter_tuner import HyperparameterTuner


def fine_tune(X_train, y_train):
    # Define the hyperparameter grid
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': np.logspace(-3, 3, 20),
        'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 20)),
        'degree': [2, 3, 4],
        'coef0': [0, 1, 2, 3],
    }

    clf = SVC()

    tuner = HyperparameterTuner(model_instance=clf, param_grid=param_grid)

    return tuner.random_search(X_train, y_train, n_iterations=5376)
