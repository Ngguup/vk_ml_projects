import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBCustomClassifier:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        params = {k: v for k, v in locals().items() if k != "self"}
        self.__dict__.update(params)

    def fit(self, x, y):
        self.estimators = np.array([
            DecisionTreeRegressor(
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            for _ in range(self.n_estimators)
        ], dtype=object)

        prediction = np.zeros_like(y, dtype=float)

        for tree in self.estimators:
            probability = self.sigmoid(prediction)
            r = y - probability

            tree.fit(x, r)
            prediction += self.learning_rate * tree.predict(x)

    def sigmoid(self, prediction):
        return 1 / (1 + np.exp(-prediction))

    def predict_proba_(self, x):
        return self.sigmoid(self.learning_rate *
                            np.sum([tree.predict(x) for tree in self.estimators], axis=0))

    def predict_proba(self, x):
        proba = self.predict_proba_(x)
        return np.vstack([1 - proba, proba]).T

    def predict(self, x):
        return (self.predict_proba_(x) >= 0.5).astype(int)

    @property
    def estimators_(self):
        return self.estimators