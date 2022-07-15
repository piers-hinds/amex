from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression


class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def eval(self, X):
        pass


class LogReg(Model):
    def __init__(self):
        self.logreg = LogisticRegression()

    def fit(self, X, y):
        self.logreg.fit(X, y)

    def eval(self, X):
        return self.logreg.predict_proba(X)[:, 1]


class Baseline(Model):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def eval(self, X):
        raw_preds = np.array(1 - X[:, 1])
        return raw_preds / raw_preds.max()