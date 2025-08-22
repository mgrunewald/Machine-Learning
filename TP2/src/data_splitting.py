import numpy as np

def train_val_split(data, train_size=0.8):
    total_filas = len(data)
    filas_entrenamiento = int(train_size * total_filas)
    training = data.iloc[:filas_entrenamiento]
    validation = data.iloc[filas_entrenamiento:]
    return training, validation

def create_k_train_val_splits(X, y, k=5):
    assert len(X) == len(y), "X e y deben tener la misma longitud"
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_sizes = np.ones(k, dtype=int) * len(X) // k
    fold_sizes[:len(X) % k] += 1
    current = 0
    splits = []

    for fold_size in fold_sizes:
        val_indices = indices[current:current + fold_size]
        train_indices = np.concatenate([indices[:current], indices[current + fold_size:]])
        X_train = X.iloc[train_indices]
        X_val = X.iloc[val_indices]
        y_train = y.iloc[train_indices]
        y_val = y.iloc[val_indices]
        splits.append((X_train, X_val, y_train, y_val))
        current += fold_size

    return splits
