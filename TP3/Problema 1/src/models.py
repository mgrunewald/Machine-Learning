import numpy as np

class LogisticRegression:
    def __init__(self, max_iter=10000, learning_rate=0.1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _add_intercept(self, X):
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X, y):
        X = np.array(X)
        X = self._add_intercept(X)
        y = np.array(y)
        self.coef_ = np.zeros(X.shape[1])
        for _ in range(self.max_iter):
            z = np.dot(X, self.coef_)
            y_hat = self._sigmoid(z)
            gradient = np.dot(X.T, (y_hat - y)) / y.size
            self.coef_ -= self.learning_rate * gradient
        
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]
    
    def predict_proba(self, X):
        X = self._add_intercept(X)
        prob_positive = self._sigmoid(np.dot(X, np.r_[self.intercept_, self.coef_]))
        prob_negative = 1 - prob_positive
        return np.vstack((prob_negative, prob_positive)).T

class LogisticRegressionWithCostReweighting:
    def __init__(self, max_iter=10000, learning_rate=0.1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def _compute_class_weights(self, y):
        n_total = len(y)
        pi_1 = np.sum(y == 1) / n_total 
        pi_2 = np.sum(y == 0) / n_total 
        C = pi_2 / pi_1  
        return C, pi_1, pi_2

    def fit(self, X, y):
        X = np.array(X)
        X = self._add_intercept(X)
        y = np.array(y)
        self.coef_ = np.zeros(X.shape[1])

        C, pi_1, pi_2 = self._compute_class_weights(y)

        for _ in range(self.max_iter):
            z = np.dot(X, self.coef_)
            y_hat = self._sigmoid(z)
            weights = np.where(y == 1, C, 1) 
            gradient = np.dot(X.T, weights * (y_hat - y)) / y.size
            self.coef_ -= self.learning_rate * gradient

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict_proba(self, X):
        X = self._add_intercept(X)
        prob_positive = self._sigmoid(np.dot(X, np.r_[self.intercept_, self.coef_]))
        prob_negative = 1 - prob_positive
        return np.vstack((prob_negative, prob_positive)).T