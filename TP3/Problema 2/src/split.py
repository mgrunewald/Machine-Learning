def train_val_split(data, train_size=0.8):
    total_filas = len(data)
    filas_entrenamiento = int(train_size * total_filas)
    training = data.iloc[:filas_entrenamiento]
    validation = data.iloc[filas_entrenamiento:]
    return training, validation