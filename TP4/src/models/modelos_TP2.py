import numpy as np
import pandas as pd
from scipy.optimize import minimize

class RidgeRegression:
    def __init__(self, lambda_=1.0):
        self.lambda_ = lambda_
        self.theta = None

    def _cost_function(self, theta, X, y):
        m = X.shape[0]
        predictions = X @ theta
        error = predictions - y
        squared_error = np.sum(error**2) / (2 * m)
        regularization_term = (self.lambda_ / (2 * m)) * np.sum(theta[1:]**2) 
        cost = squared_error + regularization_term
        return cost
    def _gradient_function(self, theta, X, y):
        m = X.shape[0]
        error = X @ theta - y
        gradient = (X.T @ error) / m
        gradient[1:] += (self.lambda_ / m) * theta[1:] 
        return gradient

    def fit(self, X, y):
        not_nan_indices = ~np.isnan(X).any(axis=1)
        X_clean = X[not_nan_indices]
        y_clean = y[not_nan_indices]
        X_clean_with_bias = np.c_[np.ones(X_clean.shape[0]), X_clean]
        X = X_clean_with_bias
        m, n = X.shape
        initial_theta = np.zeros(n)
        result = minimize(self._cost_function, initial_theta, args=(X, y_clean),
                          jac=self._gradient_function, method='BFGS')
        self.theta = result.x

    def predict(self, X):
        if self.theta is None:
            raise ValueError("El modelo no ha sido entrenado aún.")
        X = np.c_[np.ones(X.shape[0]), X] 
        y_pred = X @ self.theta
        return y_pred

class LocallyWeightedRegression:
    def __init__(self, tau):
        self.tau = tau

    def fit(self, X, y):
        self.X_train = X.values if isinstance(X, pd.DataFrame) else X
        self.y_train = y.values if isinstance(y, pd.Series) else y
    
    def predict(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y_pred = np.zeros(len(X))
        for i, x in enumerate(X):
            y_pred[i] = self._predict_single(x)
        return y_pred
    
    def _predict_single(self, x):
        weights = self._calculate_weights(x)
        theta = self._calculate_theta(weights)
        return np.dot(x, theta)
    
    def _calculate_weights(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        weights = np.exp(-distances**2 / (2 * self.tau**2))
        return weights
    
    def _calculate_theta(self, weights):
        W = np.diag(weights)
        XTWX = np.dot(self.X_train.T, np.dot(W, self.X_train))
        XTWy = np.dot(self.X_train.T, np.dot(W, self.y_train))
        try:
            theta = np.linalg.solve(XTWX, XTWy)
        except np.linalg.LinAlgError:
            theta = np.dot(np.linalg.pinv(XTWX), XTWy)
        return theta

class NonLinearRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.coefficients = None
        self.mean = None
        self.std = None

    def _polynomial_features(self, X):
        X_poly = X.copy()
        for d in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def fit(self, X, y):
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Los datos contienen valores NaN")
        X_poly = self._polynomial_features(X)
        self.mean = X_poly.mean(axis=0)
        self.std = X_poly.std(axis=0)
        X_poly = (X_poly - self.mean) / self.std
        X_poly_with_intercept = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))
        try:
            X_poly_T = X_poly_with_intercept.T
            self.coefficients = np.linalg.inv(X_poly_T @ X_poly_with_intercept) @ X_poly_T @ y
        except np.linalg.LinAlgError:
            raise ValueError("No se pudo invertir la matriz.")
    
    def predict(self, X):
        if np.any(np.isnan(X)):
            raise ValueError("Los datos contienen valores NaN")
        X_poly = self._polynomial_features(X)
        X_poly = (X_poly - self.mean) / self.std
        X_poly_with_intercept = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))
        return X_poly_with_intercept @ self.coefficients
