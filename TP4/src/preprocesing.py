import numpy as np
import pandas as pd

def numerize(data):
    tipo_conversion = { # Después de un estudio de mercado, es el nivel de completitud del vehículo
    'Hilux SW4': 30.0,
    'RAV4': 20.0,
    'Corolla Cross': 10.0
}
    combustible_conversion = {
        'Diésel': 0.0,
        'Nafta': 2.0,
        'Híbrido': 3.0
    }
    transmision_conversion = { # Mayor tecnología de transmisión
        'Automática secuencial': 1.0,
        'Automática': 0.5,
        'Manual': 0.0
    }
    vendedor_conversion = {
        'concesionaria': 0.5,
        'tienda': 1.0,
        'particular': 0.0
    }

    data['Tipo'] = data['Tipo'].map(tipo_conversion)
    data['Tipo de combustible'] = data['Tipo de combustible'].map(combustible_conversion)
    data['Transmisión'] = data['Transmisión'].map(transmision_conversion)
    data['Tipo de vendedor'] = data['Tipo de vendedor'].map(vendedor_conversion)

    return data

def normalize(data):
    data = data.apply(pd.to_numeric, errors='coerce')
    range_ = data.max() - data.min()
    range_[range_ == 0] = 1 
    normalized = (data - data.min()) / range_
    return normalized
    
def handle_missing_values(data):
    #data['Color'] = data['Color'].astype(str)
    data['Tipo de combustible'] = data['Tipo de combustible'].astype(str)
    data['Transmisión'] = data['Transmisión'].astype(str)
    data['Motor'] = data['Motor'].astype(str)
    data['Tipo'] = data['Tipo'].astype(str)
    data['Kilómetros'] = data['Kilómetros'].str.replace(' km', '', regex=False).astype(float)

    # Tipo faltante
    condicion_tipo_faltante = (data['Tipo'] == 'nan')
    data.loc[condicion_tipo_faltante, 'Tipo'] = 'Hilux SW4'

    # Combustible faltante
    condicion_combustuble_hilux = (data['Tipo de combustible'] =='nan') & (data['Tipo'] == 'Hilux SW4')
    condicion_combustuble_GNC = (data['Tipo de combustible'] =='Nafta/GNC')
    condicion_combustuble_corolla = (data['Tipo de combustible'] =='nan') & (data['Tipo'] == 'Corolla Cross')    
    condicion_combustuble_rav4 = (data['Tipo de combustible'] =='nan') & (data['Tipo'] == 'RAV4')
    data.loc[condicion_combustuble_GNC, 'Tipo de combustible'] = 'Nafta'
    data.loc[condicion_combustuble_hilux, 'Tipo de combustible'] = 'Diésel'    
    data.loc[condicion_combustuble_corolla, 'Tipo de combustible'] = 'Nafta'
    data.loc[condicion_combustuble_rav4, 'Tipo de combustible'] = 'Nafta'

    # Transmisión faltante
    condicion_transmisión_faltante = (data['Transmisión'] =='nan') 
    data.loc[condicion_transmisión_faltante, 'Transmisión'] = 'Automática'    

    # Motor incorrecto
    for index, valor in data['Motor'].items():
        try:
            motor_float = float(valor)
            if motor_float >= 5 or motor_float <= 0:
                data.at[index, 'Motor'] = np.nan 
        except (ValueError, TypeError):
            data.at[index, 'Motor'] = np.nan 
    # Motor faltante
    data['Motor'] = data['Motor'].astype(str)
    condicion_motor_hilux = (data['Motor'] =='nan') & (data['Tipo'] == 'Hilux SW4')
    condicion_motor_corolla = (data['Motor'] =='nan') & (data['Tipo'] == 'Corolla Cross')
    condicion_motor_rav4 = (data['Motor'] == 'nan') & (data['Tipo'] == 'RAV4')
    data.loc[condicion_motor_hilux, 'Motor'] = '2.8'
    data.loc[condicion_motor_corolla, 'Motor'] = '2.0'
    data.loc[condicion_motor_rav4, 'Motor'] = '2.5'

    # Datos extraños o no esperados
    condicion_falta_combustible_Hilux = (data['Tipo de combustible'] == 'nan') & (data['Tipo'] == 'Hilux SW4')
    condicion_falta_combustible = (data['Tipo de combustible'] == 'nan') & (data['Tipo'].isin(['Corolla Cross', 'RAV4']))
    condicion_electric_hilux = (data['Tipo de combustible'].isin(['Eléctrico', 'Electric', 'Electrico', 'E', 'Elec'])) & (data['Tipo'] == 'Hilux SW4')
    condicion_electric_CC_R4 = (data['Tipo de combustible'].isin(['Eléctrico', 'Electric', 'Electrico', 'E', 'Elec'])) & (data['Tipo'].isin(['Corolla Cross', 'RAV4']))
    condicion_hybrid_corolla_rav4 = (data['Tipo de combustible'].isin(['Híbrido/Nafta', 'Híbrido/Diésel', 'Hibrido', 'Hybrid', 'Hyb'])) & \
                                    (data['Tipo'].isin(['Corolla Cross', 'RAV4']))
    condicion_hybrid_hilux = (data['Tipo de combustible'].isin(['Híbrido/Nafta', 'Híbrido/Diésel', 'Hibrido', 'Hybrid', 'Hyb'])) & \
                             (data['Tipo'] == 'Hilux SW4')
    
    data.loc[condicion_falta_combustible, 'Tipo de combustible'] = 'Nafta'
    data.loc[condicion_falta_combustible_Hilux, 'Tipo de combustible'] = 'Diesel'    
    data.loc[condicion_electric_hilux, 'Tipo de combustible'] = 'Nafta'
    data.loc[condicion_hybrid_corolla_rav4, 'Tipo de combustible'] = 'Híbrido'
    data.loc[condicion_hybrid_hilux, 'Tipo de combustible'] = 'Nafta'
    data.loc[condicion_electric_CC_R4, 'Tipo de combustible'] = 'Híbrido'

    data['Motor'] = data['Motor'].astype(float)
    data['Kilómetros'] = data['Kilómetros'].astype(float)
    return data