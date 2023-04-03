import numpy as np

from sklearn.svm import SVC

from optimisation.hyperparameter_tuner import HyperparameterTuner


def fine_tune(X_train, y_train):
    # Define the hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7)),
        'coef0': [-1, 0, 1]
    }

    clf = SVC()
    tuner = HyperparameterTuner(model_instance=clf, param_grid=param_grid)

    return tuner.grid_search(X_train, y_train)
