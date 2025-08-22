import numpy as np
from scipy.stats import mode
import pandas as pd
import re

def mapeo(data):
    # eliminamos features irrelevantes
    data.drop(['SitioOrigen', 'TIPOPROPIEDAD', 'MesListing', 'AreaJuegosInfantiles', 'SalonFiestas', 'PistaJogging',
            'Lobby', 'LocalesComerciales', 'AreaCine', 'BusinessCenter', 'Jacuzzi',
            'Chimenea', 'EstacionamientoVisitas', 'SistContraIncendios', 'Cisterna', 'year'],
            axis=1, inplace=True)

    boolean_features = [
            'Amoblado',
            'AccesoInternet',
            'Gimnasio',
            'Laundry',
            'Calefaccion',
            'SalonDeUsosMul',
            'AireAC',
            'Recepcion',
            'Estacionamiento',
            'Ascensor',
            'Seguridad',
            'Pileta',
            'Cocheras',
            'AreaParrillas',
            'CanchaTennis',
        ]
    
    for feature in boolean_features:
        data[feature] = data[feature].astype(str)

    # mapear si y no a 1 y 0
    for categoria in boolean_features:
        condicion_negativa = (data[categoria].isin(['0', '0.0', 'No', 'no', 'NO', 'nO', 'N', 'n']))
        condicion_positiva = (data[categoria].isin(['1', '1.0', 'si', 'sí', 'Sí', 'SÌ', 'sI', 'sÍ', 'S', 's', 'Y', 'y', 'yes', 'YES', 'Yes']))
        data.loc[condicion_negativa, categoria] = '0.0'
        data.loc[condicion_positiva, categoria] = '1.0'
        data[categoria] = data[categoria].astype(float)
    
    # mapear X años a X solo
    data['Antiguedad'] = data['Antiguedad'].astype(str).str.replace(' años', '', regex=False)
    data['Antiguedad'] = pd.to_numeric(data['Antiguedad'], errors='coerce')
    clean_antiguedad_column(data)

    # mapear a 0 datos faltantes 
    for ft in boolean_features:
        data[ft] = data[ft].fillna(0).astype(float)

    # mapear cochears a 0, 1 o 2
    data['Cocheras'] = data['Cocheras'].astype(float)
    condicion_cocheras_a_0 = (data['Cocheras'] < 0 )
    condicion_cocheras_a_2 = (data['Cocheras'] > 2 )
    data.loc[condicion_cocheras_a_0, 'Cocheras'] = 0.0
    data.loc[condicion_cocheras_a_2, 'Cocheras'] = 2.0

    numeric_features = [('Antiguedad', 10), ('STotalM2', 5), ('SConstrM2', 5), ('Dormitorios', 2), ('Banos', 2), ('Ambientes', 0)]

    # para valores faltantes en features numéricas reemplazamos con una muestra normal con la media=moda y una sd de 10
    for n_feat in numeric_features:
        moda = data[n_feat[0]].dropna().mode()[0]
        n_valores_nulos = data[n_feat[0]].isnull().sum()
        muestras = np.random.normal(loc=(moda), scale=n_feat[1], size=n_valores_nulos)
        muestras = np.round(muestras).astype(int).astype(float) 
        data.loc[data[n_feat[0]].isnull(), n_feat[0]] = muestras
        # caso valores negativos
        data.loc[data[n_feat[0]] < 0, n_feat[0]] = 0

    # si pusieron la fecha, calculamos la antigüedad
    data.loc[(data['Antiguedad'] > 1840) & (data['Antiguedad'] < 2025), 'Antiguedad'] = (2024 - data['Antiguedad'])
    data.loc[(data['Antiguedad'] > 2024), 'Antiguedad'] = 0
    data.loc[(data['Antiguedad'] > 250) & (data['Antiguedad'] <= 1840), 'Antiguedad'] = 130

    # caso 0 ambientes, baños, m^2
    data.loc[data['Ambientes'] <= 0, 'Ambientes'] = 1
    data.loc[data['Banos'] <= 0, 'Banos'] = 1
    data.loc[data['STotalM2'] <= 0, 'STotalM2'] = 10
    data.loc[data['SConstrM2'] <= 0, 'SConstrM2'] = 10

    # valores numéricos muy grandes
    data.loc[data['Ambientes'] > 10, 'Ambientes'] = 10
    data.loc[data['Banos'] > 8, 'Banos'] = 8
    data.loc[data['STotalM2'] > 1000, 'STotalM2'] = 1000
    data.loc[data['SConstrM2'] > 1000, 'SConstrM2'] = 1000
    data.loc[data['Dormitorios'] > 8, 'Dormitorios'] = 8

    # encodear features categóricas
    # CABA, zona norte, oeste y sur --> 0, 1, 2, 3
    data['ITE_ADD_STATE_NAME'] = data['ITE_ADD_STATE_NAME'].astype(str)
    data.loc[data['ITE_ADD_STATE_NAME'] == 'Capital Federal', 'ITE_ADD_STATE_NAME'] = 0.0
    data.loc[data['ITE_ADD_STATE_NAME'] == 'Bs.As. G.B.A. Norte', 'ITE_ADD_STATE_NAME'] = 1.0
    data.loc[data['ITE_ADD_STATE_NAME'] == 'Bs.As. G.B.A. Oeste', 'ITE_ADD_STATE_NAME'] = 2.0
    data.loc[data['ITE_ADD_STATE_NAME'] == 'Bs.As. G.B.A. Sur', 'ITE_ADD_STATE_NAME'] = 3.0
    data['ITE_ADD_STATE_NAME'] = data['ITE_ADD_STATE_NAME'].astype(float)


    # Usado y Null va a 1.0 y nuevo va a 0.0
    data['ITE_TIPO_PROD'] = data['ITE_TIPO_PROD'].astype(str)
    data.loc[data['ITE_TIPO_PROD'] == 'U', 'ITE_TIPO_PROD'] = 1.0
    data.loc[data['ITE_TIPO_PROD'] == 'N', 'ITE_TIPO_PROD'] = 0.0
    data.loc[data['ITE_TIPO_PROD'] == 'S', 'ITE_TIPO_PROD'] = 1.0
    data['ITE_TIPO_PROD'] = data['ITE_TIPO_PROD'].astype(float)


    #City lo vamos a pasar a un booleano de si es CABA o no 
    data['ITE_ADD_CITY_NAME'] = data['ITE_TIPO_PROD'].astype(str)
    data.loc[data['ITE_ADD_CITY_NAME'] == 'Capital Federal', 'ITE_ADD_CITY_NAME'] = 1.0
    data.loc[data['ITE_ADD_CITY_NAME'] != 'Capital Federal', 'ITE_ADD_CITY_NAME'] = 0.0
    data['ITE_ADD_CITY_NAME'] = data['ITE_TIPO_PROD'].astype(float)

    return data


def merge_barrios(dataset, barrio_col='ITE_ADD_NEIGHBORHOOD_NAME', barrios_a_normalizar=None):
    if barrios_a_normalizar is None:
        barrios_a_normalizar = ["Palermo", "Belgrano", "San Isidro", "Nordelta", "Pilar", "Tigre", "Don Torcuato", "Lanús", "Berazategui", "Bernal", "Florida", "Quilmes"]

    def normalizar_barrios(barrio):
        if isinstance(barrio, str):
            for barrio_estandar in barrios_a_normalizar:
                if barrio_estandar.lower() in barrio.lower():
                    return barrio_estandar
        return barrio
    dataset[barrio_col] = dataset[barrio_col].apply(normalizar_barrios)
    return dataset

def clean_antiguedad_column(data, column_name='Antiguedad'):

    if column_name not in data.columns:
        raise ValueError(f"La columna '{column_name}' no existe en el DataFrame.")

    def extract_years(value):
        """
        Extrae el número de años de un valor en la columna de Antiguedad.
        """
        if pd.isna(value):
            return np.nan  # Mantener valores NaN
        if isinstance(value, str):
            match = re.search(r'\d+', value)
            return int(match.group()) if match else np.nan
        elif isinstance(value, (int, float)):
            return int(value) if not pd.isna(value) else np.nan
        return np.nan

    data[column_name] = data[column_name].apply(extract_years)

    return data