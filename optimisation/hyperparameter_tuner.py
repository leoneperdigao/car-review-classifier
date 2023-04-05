from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class HyperparameterTuner:
    def __init__(self, model_instance: BaseEstimator, param_grid):
        self.param_grid = param_grid
        self.model_instance = model_instance

    def random_search(self, X_train, y_train, n_iterations=100):
        # Set up the random search with cross-validation
        random_search = RandomizedSearchCV(
            self.model_instance,
            self.param_grid,
            n_iter=n_iterations,
            random_state=True,
            scoring='accuracy',
            n_jobs=-1,
            verbose=4,
        )

        # Fit the grid search to the training data
        random_search.fit(X_train, y_train)

        # Print the best hyperparameters and their corresponding accuracy
        print("Best hyperparameters:", random_search.best_params_)
        print("Best accuracy:", random_search.best_score_)

        # Return the best classifier
        return random_search.best_estimator_

    def grid_search(self, X_train, y_train):
        # Set up the grid search with cross-validation
        grid_search = GridSearchCV(
            self.model_instance, self.param_grid, scoring='accuracy', n_jobs=-1, verbose=3,
        )

        # Fit the grid search to the training data
        grid_search.fit(X_train, y_train)

        # Print the best hyperparameters and their corresponding accuracy
        print("Best hyperparameters:", grid_search.best_params_)
        print("Best accuracy:", grid_search.best_score_)

        # Return the best classifier
        return grid_search.best_estimator_


