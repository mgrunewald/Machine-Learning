import numpy as np

def train_val_split(data, train_size=0.8):
    total_filas = len(data)
    filas_entrenamiento = int(train_size * total_filas)
    training = data.iloc[:filas_entrenamiento]
    validation = data.iloc[filas_entrenamiento:]
    return training, validation

def split_k_folds(data, k):
    np.random.shuffle(data)
    fold_size = len(data) // k
    folds = [data[i:i + fold_size] for i in range(0, len(data), fold_size)]
    if len(folds) > k:
        folds[-2].extend(folds[-1])
        folds = folds[:-1]
    return folds