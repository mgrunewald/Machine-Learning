import numpy as np

def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae( y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_res / ss_total)

def mse(y_true, y_pred): 
        return np.mean((y_true - y_pred) ** 2)