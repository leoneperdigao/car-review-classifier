import numpy as np
from sklearn.naive_bayes import MultinomialNB

from optimisation.hyperparameter_tuner import HyperparameterTuner


def fine_tune(X_train, y_train):
    # Define the hyperparameter grid
    param_grid = {
        'alpha': np.logspace(-3, 0, 20)
    }

    clf = MultinomialNB()
    tuner = HyperparameterTuner(model_instance=clf, param_grid=param_grid)

    return tuner.grid_search(X_train, y_train)
