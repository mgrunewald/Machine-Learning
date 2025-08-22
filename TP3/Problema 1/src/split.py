import numpy as np

def train_val_split(data, train_size=0.8):
    clase_counts = data['target'].value_counts()
    train_counts = (clase_counts * train_size).astype(int)
    train_indices = []
    val_indices = []
    for clase in clase_counts.index:
        indices_clase = data[data['target'] == clase].index.tolist()
        np.random.shuffle(indices_clase) 
        train_indices.extend(indices_clase[:train_counts[clase]]) 
        val_indices.extend(indices_clase[train_counts[clase]:])
    training = data.loc[train_indices]
    validation = data.loc[val_indices]
    return training, validation