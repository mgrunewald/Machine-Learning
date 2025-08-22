import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def reasignar_muestras_pequenos_barrios(data, barrio_col='ITE_ADD_NEIGHBORHOOD_NAME', lat_col='LONGITUDE', lon_col='LATITUDE', min_muestras=150):
    barrio_counts = data[barrio_col].value_counts()
    barrios_pequenos = barrio_counts[barrio_counts <= min_muestras].index
    barrios_grandes = data[~data[barrio_col].isin(barrios_pequenos)][barrio_col].unique()
    centroides = data[~data[barrio_col].isin(barrios_pequenos)].groupby(barrio_col)[[lat_col, lon_col]].mean()
    nueva_asignacion = data[barrio_col].copy()
    for barrio in barrios_pequenos:
        muestras = data[data[barrio_col] == barrio]
        for idx, muestra in muestras.iterrows():
            lat_lon_muestra = np.array([radians(muestra[lat_col]), radians(muestra[lon_col])])
            distancias = centroides.apply(
                lambda x: haversine_distances(
                    [lat_lon_muestra, [radians(x[lat_col]), radians(x[lon_col])]]
                )[0][1],
                axis=1
            )
            nuevo_barrio = distancias.idxmin()
            nueva_asignacion.loc[idx] = nuevo_barrio
    data[barrio_col] = nueva_asignacion
    return data

def barrios_asigning(data):
    data_sin_mini_barrios = reasignar_muestras_pequenos_barrios(data)
    new_data = data_sin_mini_barrios
    muestras_nan = new_data[new_data['ITE_ADD_NEIGHBORHOOD_NAME'].isnull()]
    muestras_nan_lista = muestras_nan[['LONGITUDE', 'LATITUDE']].values.tolist()
    data_dev = new_data[new_data['ITE_ADD_NEIGHBORHOOD_NAME'].notnull()].copy()


    X = data_dev[['LONGITUDE', 'LATITUDE']].values  
    y = data_dev['ITE_ADD_NEIGHBORHOOD_NAME'].values 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    if len(muestras_nan_lista) > 0:
        predicciones_nan = knn.predict(muestras_nan_lista)
        new_data.loc[muestras_nan.index, 'ITE_ADD_NEIGHBORHOOD_NAME'] = predicciones_nan
    else:
        print("No hay muestras con NaN para predecir.")
    
    X2 = new_data[['LONGITUDE', 'LATITUDE']].values  
    y2 = new_data['ITE_ADD_NEIGHBORHOOD_NAME'].values 

    output_file_return = './data/final_data_barrios.csv'
    new_data.to_csv(output_file_return, index=False)

    return new_data