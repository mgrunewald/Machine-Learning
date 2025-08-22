import pandas as pd
import numpy as np
import random

def undersampling(df):
    clase_mayoritaria = df[df['target'] == 0]
    clase_minoritaria = df[df['target'] == 1]
    num_muestras_clase_mayoritaria = len(clase_mayoritaria)
    num_muestras_clase_minoritaria = len(clase_minoritaria)
    if num_muestras_clase_mayoritaria <= num_muestras_clase_minoritaria:
        print("Classes are already balanced or majority class is smaller than minority.")
        return df
    num_muestras_a_eliminar = num_muestras_clase_mayoritaria - num_muestras_clase_minoritaria
    indices_mayoritaria_a_eliminar = np.random.choice(clase_mayoritaria.index, size=num_muestras_a_eliminar, replace=False)
    df_balanceado = df.drop(indices_mayoritaria_a_eliminar)
    return df_balanceado
    
def oversampling(df):
    clase_mayoritaria = df[df['target'] == 0]
    clase_minoritaria = df[df['target'] == 1]
    num_muestras_clase_mayoritaria = len(clase_mayoritaria)
    num_muestras_clase_minoritaria = len(clase_minoritaria)
    proporcion = num_muestras_clase_mayoritaria / num_muestras_clase_minoritaria
    num_replicas = num_muestras_clase_mayoritaria - num_muestras_clase_minoritaria
    replicas = clase_minoritaria.sample(n=num_replicas, replace=True)
    df_balanceado = pd.concat([df, replicas], ignore_index=True)
    return df_balanceado

def distancia_euclidiana(sample, row):
    return np.sqrt(np.sum((sample - row) ** 2))

def knn(df, sample_index, k=3):
    sample = df.iloc[sample_index].to_numpy()
    distancias = []
    for i, row in df.iterrows():
        if i == sample_index:
            continue
        distancia = distancia_euclidiana(sample, row.to_numpy())
        distancias.append((i, distancia))
    distancias.sort(key=lambda x: x[1])
    k_indices = [distancias[i][0] for i in range(min(k, len(distancias)))]
    return k_indices

def smote(X, y): #ESTE SMOTE QUE HICE NO ESTÁ FUNCIONANDO CORRECTAMENTE, es por eso que se compara con el smote de una librería en el notebook
    labels = y.to_numpy()
    one_amount = np.count_nonzero(labels == 1)
    zero_amount = np.count_nonzero(labels == 0)
    ones = np.where(labels == 1)[0]
    zeros = np.where(labels == 0)[0]
    if one_amount > zero_amount:
        minoria = zeros
        mayoria = ones
    else: 
        minoria = ones
        mayoria = zeros
    ratio = len(mayoria) // len(minoria) if len(minoria) > 0 else 0
    samples_amount = len(mayoria) - len(minoria)
    new_samples = []
    for i in minoria:
        if len(new_samples) >= samples_amount:
            break
        for j in range(ratio):
            nn_index = knn(X, i, k=3)
            if not nn_index:
                continue
            new_index = random.choice(nn_index)
            if new_index < len(X):
                new_sample = X.iloc[new_index].tolist()
                new_sample.append(y.iloc[new_index])
                new_samples.append(new_sample)  
    old_samples = pd.concat([X, y], axis=1)   
    if new_samples:
        new_samples_df = pd.DataFrame(new_samples, columns=list(X.columns) + [y.name])
        modified_samples = pd.concat([old_samples, new_samples_df], axis=0, ignore_index=True)
    else:
        modified_samples = old_samples

    return modified_samples
