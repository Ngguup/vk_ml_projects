import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBCustomRegressor:
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

        self.y_mean = y.mean()
        self.prediction = np.full_like(y, self.y_mean)
        for tree in self.estimators:
            r = y - self.prediction

            tree.fit(x, r)
            self.prediction += self.learning_rate * tree.predict(x)

    def predict(self, x):
        return self.learning_rate * \
            np.sum([tree.predict(x) for tree in self.estimators], axis=0) + self.y_mean

    @property
    def estimators_(self):
        return self.estimators