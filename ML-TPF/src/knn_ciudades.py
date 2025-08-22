import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

archivo_csv = './data/sin_hasta40.csv'
data = pd.read_csv(archivo_csv)
data = data.dropna(subset=['LONGITUDE', 'LATITUDE']).reset_index(drop=True)
muestras_nan = data[data['ITE_ADD_CITY_NAME'].isnull()]
muestras_nan_lista = muestras_nan[['LONGITUDE', 'LATITUDE']].values.tolist()
data_dev = data[data['ITE_ADD_CITY_NAME'].notnull()].copy()

X = data_dev[['LONGITUDE', 'LATITUDE']].values  
y = data_dev['ITE_ADD_CITY_NAME'].values 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))

plt.scatter(X[:, 0], X[:, 1], c=pd.factorize(y)[0], cmap='magma', alpha=0.7)
plt.title("Distribución de Ciudades por Coordenadas")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.colorbar(label="Barrios")
plt.show()

if len(muestras_nan_lista) > 0:
    predicciones_nan = knn.predict(muestras_nan_lista)
    data.loc[muestras_nan.index, 'ITE_ADD_CITY_NAME'] = predicciones_nan
    data.to_csv('./data/datos_modificados_ciudades.csv', index=False)
    print("Archivo actualizado con las predicciones.")
else:
    print("No hay muestras con NaN para predecir.")

