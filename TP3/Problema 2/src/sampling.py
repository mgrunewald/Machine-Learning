import numpy as np

def undersampling(X, y):
    unique, counts = np.unique(y, return_counts=True)
    clase_minoritaria = unique[np.argmin(counts)]
    num_muestras_clase_minoritaria = counts.min()
    X_balanceado = []
    y_balanceado = []
    for clase in unique:
        clase_indices = np.where(y == clase)[0]
        clase_X = X[clase_indices]
        clase_y = y[clase_indices]
        if clase == clase_minoritaria:
            X_balanceado.append(clase_X)
            y_balanceado.append(clase_y)
        else:
            indices = np.random.choice(clase_X.shape[0], num_muestras_clase_minoritaria, replace=False)
            X_balanceado.append(clase_X[indices])
            y_balanceado.append(clase_y[indices])
    X_balanceado = np.vstack(X_balanceado)
    y_balanceado = np.hstack(y_balanceado)
    return X_balanceado, y_balanceado
