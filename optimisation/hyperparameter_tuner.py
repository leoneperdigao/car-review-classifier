from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


class HyperparameterTuner:
    def __init__(self, model_instance: BaseEstimator, param_grid):
        self.param_grid = param_grid
        self.model_instance = model_instance

    def grid_search(self, X_train, y_train):
        # Set up the grid search with cross-validation
        grid_search = GridSearchCV(
            self.model_instance, self.param_grid, scoring='accuracy'
        )

        # Fit the grid search to the training data
        grid_search.fit(X_train, y_train)

        # Print the best hyperparameters and their corresponding accuracy
        print("Best hyperparameters:", grid_search.best_params_)
        print("Best accuracy:", grid_search.best_score_)

        # Return the best classifier
        return grid_search.best_estimator_


