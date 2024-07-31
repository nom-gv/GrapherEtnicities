import dash
import requests
import base64
import io
import threading
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dash import dcc
from dash import html
from dash import dash_table, callback_context
from dash import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Función para descargar y guardar archivos desde Google Drive
def download_and_save(nombre, file_id):
    try:
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        response = requests.get(url)
        if response.status_code == 200:
            with open(nombre, 'wb') as f:
                f.write(response.content)
            df = pd.read_excel(nombre)
            return df
        else:
            print(f'Error al descargar el archivo {nombre}: Código de estado {response.status_code}')
            return None
    except Exception as e:
        print(f'Error al descargar y guardar el archivo {nombre}: {str(e)}')
        return None

# Lista de archivos para descargar desde Google Drive
archivos = [
    ('CasosCancer-Guaranis.xlsx', '1oRB3DMP1NtnnwfQcaYHo9a3bUcbQfB5U'),
    ('CasosDiabetes-Guaranis.xlsx', '1xHYonZp8RbPYCE9kihc3IthwOtgVNi1P'),
    ('CasosHipertensionArterial-Guaranis.xlsx', '1_jue36lk4iJim6btVh_tSUkR0i_QGeIk'),
    ('CasosObesidad-Guaranis.xlsx', '19aVPGne2nPm7_I0L9i_csyEBRw9geGea'),
    ('CasosNeumonia-Guaranis.xlsx', '1tK7dDEo1b7gWn-KHl1qE_WL62ztrygHw'),
    ('CasosChagas-Guaranis.xlsx', '1kAXyvg1cvLtl7w8a6D1AijMwFLJiialT'),
    ('CasosVIH-Guaranis.xlsx', '1xmnFEOBzaIZa3Ah4daAVEMo4HeLCVyZK'),
    ('CasosEstadoNutricional-Guaranis.xlsx', '1G8k9bqzJop0dSgFjigeVrzVQiuHuUFUp'),
    ('CasosEmbarazoAdolescente-Guaranis.xlsx', '1WGjRPOdiKjbblojvO96WpkfSITvbpvsH'),
    ('CasosConsultaExterna-Guaranis.xlsx', '1iA8HOY1nCGd62dqL1RU3MMgitXKT1a4q'),
    ('DatosPoblaciones-Guaranis.xlsx', '1Tkr9PBQJHAb5m8zq8k-EFl1bvUMvq-5B'),
    ('DatosEspeciales-Guaranis.xlsx', '1NoaMbxqsDrw3gtya91fnE2TPZo54Dxf6'),
    ('CasosCancer-Afrobolivianos.xlsx', '1ysmxBKWrHeC3xXmK1RzuL5-eveaNXqj1'),
    ('CasosDiabetes-Afrobolivianos.xlsx', '1L1XoqEI1ysMxq3TTNLgW1Ji5AUGPN4C4'),
    ('CasosHipertensionArterial-Afrobolivianos.xlsx', '1Rha7FxxGEDaJSLG-mjemzRZuS0rwWTLK'),
    ('CasosObesidad-Afrobolivianos.xlsx', '1V3W07eB4HwZOB-Tnn-Q1uU2MFB0hXpbV'),
    ('CasosNeumonia-Afrobolivianos.xlsx', '1dCVGa3sHmhlglO7j5thD0M8WXdrjpNV7'),
    ('CasosChagas-Afrobolivianos.xlsx', '1SgV1pzBKc2_5dCtQ4xDLgm2QNU-9HA5_'),
    ('CasosVIH-Afrobolivianos.xlsx', '11IWn0JXocoZ2Rh0zwbgV32ijI4zytQIN'),
    ('CasosEstadoNutricional-Afrobolivianos.xlsx', '1PGbomzjaufJt6mOSPtC3B7MIyKBymgxn'),
    ('DatosPoblaciones-Afrobolivianos.xlsx', '1_j05pCS_IeudCbt7oTm38tzbhYi2cbRc'),
    ('DatosEspeciales-Afrobolivianos.xlsx', '1TOHGe0-akhPcUgpFQN4uNFUvVbUeOiKo'),
    ('CaracteristicasSocieconomicas-Guaranis.xlsx', '1WTJkEGpCEkVoDmkGc_OoRfHpBvgDsUP1')
]

# Función para descargar todos los archivos en un hilo separado
def descargar_archivos():
    for nombre, file_id in archivos:
        download_and_save(nombre, file_id)

# Descargar archivos en un hilo separado
descarga_thread = threading.Thread(target=descargar_archivos)
descarga_thread.start()

# Inicializar la aplicación Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
# Definir el servidor
server = app.server

def get_casos(tipo, comunidad):
    file_map = {
        'cancer': {
            'guarani': 'CasosCancer-Guaranis.xlsx',
            'afroboliviano': 'CasosCancer-Afrobolivianos.xlsx'
        },
        'diabetes': {
            'guarani': 'CasosDiabetes-Guaranis.xlsx',
            'afroboliviano': 'CasosDiabetes-Afrobolivianos.xlsx'
        },
        'hipertension': {
            'guarani': 'CasosHipertensionArterial-Guaranis.xlsx',
            'afroboliviano': 'CasosHipertensionArterial-Afrobolivianos.xlsx'
        },
        'obesidad': {
            'guarani': 'CasosObesidad-Guaranis.xlsx',
            'afroboliviano': 'CasosObesidad-Afrobolivianos.xlsx'
        },
        'neumonia': {
            'guarani': 'CasosNeumonia-Guaranis.xlsx',
            'afroboliviano': 'CasosNeumonia-Afrobolivianos.xlsx'
        },
        'chagas': {
            'guarani': 'CasosChagas-Guaranis.xlsx',
            'afroboliviano': 'CasosChagas-Afrobolivianos.xlsx'
        },
        'vih': {
            'guarani': 'CasosVIH-Guaranis.xlsx',
            'afroboliviano': 'CasosVIH-Afrobolivianos.xlsx'
        },
        'nutricion': {
            'guarani': 'CasosEstadoNutricional-Guaranis.xlsx',
            'afroboliviano': 'CasosEstadoNutricional-Afrobolivianos.xlsx'
        },
        'embarazo': {
            'guarani': 'CasosEmbarazoAdolescente-Guaranis.xlsx',
            'afroboliviano': 'CasosEmbarazoAdolescente-Afrobolivianos.xlsx'
        },
        'consultas': {
            'guarani': 'CasosConsultaExterna-Guaranis.xlsx',
            'afrobolivianos': 'CasosConsultaExterna-Afrobolivianos.xlsx'
        },
        'poblacion': {
            'guarani': 'DatosPoblaciones-Guaranis.xlsx',
            'afroboliviano': 'DatosPoblaciones-Afrobolivianos.xlsx'
        },
        'poblacion-especial': {
            'guarani': 'DatosEspeciales-Guaranis.xlsx',
            'afroboliviano': 'DatosEspeciales-Afrobolivianos.xlsx'
        },
        'salud': {
            'guarani': 'CaracteristicasSocieconomicas-Guaranis.xlsx'
        }
    }
    
    sheets_map = {
        'cancer': {
            'guarani': ["CANCER-C", "CANCER-G", "CANCER-L", "CANCER-PC", "CANCER-SC", "CANCER-R-C", "CANCER-R-G", "CANCER-R-L", "CANCER-R-PC", "CANCER-R-SC"],
            'afroboliviano': ["CANCER-A", "CANCER-I", "CANCER-CO", "CANCER-CP", "CANCER-SY", "CANCER-NY", "CANCER-LP"]
        },
        'diabetes': {
            'guarani': ["DIABETES-C", "DIABETES-G", "DIABETES-L", "DIABETES-PC", "DIABETES-SC", "DIABETES-R-C", "DIABETES-R-G", "DIABETES-R-L", "DIABETES-R-PC", "DIABETES-R-SC"],
            'afroboliviano': ["DIABETES-A", "DIABETES-I", "DIABETES-CO", "DIABETES-CP", "DIABETES-SY", "DIABETES-NY", "DIABETES-LP"]
        },
        'hipertension': {
            'guarani': ["HIPERTENSION-C", "HIPERTENSION-G", "HIPERTENSION-L", "HIPERTENSION-PC", "HIPERTENSION-SC", "HIPERTENSION-R-C", "HIPERTENSION-R-G", "HIPERTENSION-R-L", "HIPERTENSION-R-PC", "HIPERTENSION-R-SC"],
            'afroboliviano': ["HIPERTENSION-A", "HIPERTENSION-I", "HIPERTENSION-CO", "HIPERTENSION-CP", "HIPERTENSION-SY", "HIPERTENSION-NY", "HIPERTENSION-LP"]
        },
        'obesidad': {
            'guarani': ["OBESIDAD-C", "OBESIDAD-G", "OBESIDAD-L", "OBESIDAD-PC", "OBESIDAD-SC", "OBESIDAD-R-C", "OBESIDAD-R-G", "OBESIDAD-R-L", "OBESIDAD-R-PC", "OBESIDAD-R-SC"],
            'afroboliviano': ["OBESIDAD-A", "OBESIDAD-I", "OBESIDAD-CO", "OBESIDAD-CP", "OBESIDAD-SY", "OBESIDAD-NY", "OBESIDAD-LP"]
        },
        'neumonia': {
            'guarani': ["NEUMONIA-C", "NEUMONIA-G", "NEUMONIA-L", "NEUMONIA-PC", "NEUMONIA-SC"],
            'afroboliviano': ["NEUMONIA-A", "NEUMONIA-I", "NEUMONIA-CO", "NEUMONIA-CP", "NEUMONIA-SY", "NEUMONIA-NY", "NEUMONIA-LP"]
        },
        'chagas': {
            'guarani': ["CHAGAS-C", "CHAGAS-G", "CHAGAS-L", "CHAGAS-PC", "CHAGAS-SC"],
            'afroboliviano': ["CHAGAS-A", "CHAGAS-I", "CHAGAS-CO", "CHAGAS-CP", "CHAGAS-SY", "CHAGAS-NY", "CHAGAS-LP"]
        },
        'vih': {
            'guarani': ["VIH-C", "VIH-G", "VIH-L", "VIH-PC", "VIH-SC"],
            'afroboliviano': ["VIH-A", "VIH-I", "VIH-CO", "VIH-CP", "VIH-SY", "VIH-NY", "VIH-LP"]
        },
        'nutricion': {
            'guarani': ["OBESIDAD-C", "OBESIDAD-G", "OBESIDAD-L", "OBESIDAD-PC", "OBESIDAD-SC", "SOBREPESO-C", "SOBREPESO-G", "SOBREPESO-L", "SOBREPESO-PC", "SOBREPESO-SC", "BAJOPESO-C", "BAJOPESO-G", "BAJOPESO-L", "BAJOPESO-PC", "BAJOPESO-SC"],
            'afroboliviano': ["OBESIDAD-A", "OBESIDAD-I", "OBESIDAD-CO", "OBESIDAD-CP", "OBESIDAD-SY", "OBESIDAD-NY", "OBESIDAD-LP", "SOBREPESO-A", "SOBREPESO-I", "SOBREPESO-CO", "SOBREPESO-CP", "SOBREPESO-SY", "SOBREPESO-NY", "SOBREPESO-LP", "BAJOPESO-A", "BAJOPESO-I", "BAJOPESO-CO", "BAJOPESO-CP", "BAJOPESO-SY", "BAJOPESO-NY", "BAJOPESO-LP"]
        },
        'embarazo': {
            'guarani': ["EMBARAZO-C", "EMBARAZO-G", "EMBARAZO-L", "EMBARAZO-PC", "EMBARAZO-SC"],
            'afroboliviano': ["EMBARAZO-A", "EMBARAZO-I", "EMBARAZO-CO", "EMBARAZO-CP", "EMBARAZO-SY", "EMBARAZO-NY", "EMBARAZO-LP"]
        },
        'consultas': {
            'guarani': ["CONSULTAS-C", "CONSULTAS-G", "CONSULTAS-L", "CONSULTAS-PC", "CONSULTAS-SC"],
            'afroboliviano': ["CONSULTAS-A", "CONSULTAS-I", "CONSULTAS-CO", "CONSULTAS-CP", "CONSULTAS-SY", "CONSULTAS-NY", "CONSULTAS-LP"]
        },
        'poblacion': {
            'guarani': ["POBLACION-C", "POBLACION-G", "POBLACION-L", "POBLACION-PC", "POBLACION-SC"],
            'afroboliviano': ["POBLACION-AFROS"]
        },
        'poblacion-especial': {
            'guarani': ["ESPECIALES-C", "ESPECIALES-G", "ESPECIALES-L", "ESPECIALES-PC", "ESPECIALES-SC"],
            'afroboliviano': ["ESPECIAL-AFROS"]
        },
        'salud': {
            'guarani': ["POBLACION-SC", "ETNICIDAD-SC", "ALFABETISMO-SC", "SERVICIOS-BASICOS-SC", "NUMERO-DORMITORIOS-SC", "CATEGORIA-OCUPACIONAL-SC", "ABAN-INTRA-ESC-C", "ABAN-INTRA-ESC-G", "ABAN-INTRA-ESC-L"]
        }
    }
    
    file_path = file_map[tipo][comunidad]
    sheet_names = sheets_map[tipo][comunidad]
    
    dataframes = [pd.read_excel(file_path, sheet_name=sheet) for sheet in sheet_names]
    
    return dataframes

def generate_total(df_o):
    df = df_o.copy()
    df = df.groupby('Año').sum()
    df_total = df.drop(columns=['Sexo']).reset_index()
    return df_total

def calculate_total(df_o, factor, p, column):
    poblacion_estimada = {
        2019: p[0],
        2020: p[1],
        2021: p[2],
        2022: p[3],
        2023: p[4]
    }
    df = df_o.copy()
    df['Incidencia'] = (((df[column]/(df['Año'].map(poblacion_estimada)-df[column])) * factor).round(0)).astype(int)
    suma_total = df[column].sum()
    suma_total = suma_total if suma_total != 0 else 1
    if column == '< 15' or column == '15-19' or column == "< 19":
        df['Porcentaje'] = (((df[column]/df['Total'])) * 100).round(2)
    else:
        df['Porcentaje'] = (((df[column]/suma_total)) * 100).round(2)
    
    return df

def calculate_gender(df_o, factor, m, h):
    df = df_o.copy()
    # Población estimada
    total_mujeres = {2019: m[0], 2020: m[1], 2021: m[2], 2022: m[3], 2023: m[4]}
    total_hombres = {2019: h[0], 2020: h[1], 2021: h[2], 2022: h[3], 2023: h[4]}

    # Calcular incidencias
    df['Incidencia'] = df.apply(
        lambda row: (row['Total'] / (total_hombres[row['Año']] - row['Total']) * factor) if row['Sexo'] == 'Hombre' else (row['Total'] / (total_mujeres[row['Año']] - row['Total']) * factor),
        axis=1
    ).round().astype(int)
    
    # Calcular los totales para hombres y mujeres
    total_hombres = df[df['Sexo'] == 'Hombre']['Total'].sum()
    total_mujeres = df[df['Sexo'] == 'Mujer']['Total'].sum()

    # Asegurarse de que los totales no sean cero
    total_hombres = total_hombres if total_hombres != 0 else 1
    total_mujeres = total_mujeres if total_mujeres != 0 else 1

    # Calcular el porcentaje y redondear a 2 decimales
    df['Porcentaje'] = df.apply(
        lambda row: (row['Total'] / total_hombres * 100) if row['Sexo'] == 'Hombre' else (row['Total'] / total_mujeres * 100),
        axis=1
    ).round(2)

    return df

def calculate_age(df_original, p, type):
    df = df_original.copy()
    if type == "neumonia":
      df['0-1'] = df['< 6'] + df['0-1']
      df['10-19'] = df['10-14'] + df['15-19']
      df.drop(columns=['< 6', '10-14', '15-19'], inplace=True)
      age_columns = ['0-1', '1-4', '5-9', '10-19', '20-39', '40-49', '50-59', '60+']
      
    elif type == "diabetes" or type == "hipertension":
      df['0-19'] = df['< 6'] + df['0-1'] + df['1-4'] + df['5-9'] + df['10-14'] + df['15-19']
      p['0-19'] = p['0-9'] + p['10-19']
      df.drop(columns=['< 6', '0-1', '1-4', '5-9', '10-14', '15-19'], inplace=True)
      age_columns = ['0-19', '20-39', '40-49', '50-59', '60+']
    
    elif type == "embarazo":
      age_columns = ['< 15', '15-19', '20-34', '35-49', '50+']
    
    else:
      df['0-9'] = df['< 6'] + df['0-1'] + df['1-4'] + df['5-9']
      df['10-19'] = df['10-14'] + df['15-19']
      df.drop(columns=['< 6', '0-1', '1-4', '5-9', '10-14', '15-19'], inplace=True)
      age_columns = ['0-9', '10-19', '20-39', '40-49', '50-59', '60+']
    
    if(type == "embarazo"):
        df = df.loc[:, ['Año', 'Tipo'] + age_columns + ['Total']] 
    else:
        df = df.loc[:, ['Año', 'Sexo'] + age_columns + ['Total']] 
    # Recorrer las columnas de edad y calcular incidencias para cada una
    if type != "embarazo":
        for age_col in age_columns:
            # Calcular incidencias por cada mil habitantes 
            df[f'I_{age_col}'] = round((df[age_col] / (p[age_col] - df[age_col])) * 10000, 0)
            df[f'I_{age_col}'] = df[f'I_{age_col}'].astype(int)

    for age_col in age_columns:
        percent_col = f"% {age_col}"
        df[percent_col] = df.apply(lambda row: round((row[age_col] / row['Total']) * 100, 2) if row['Total'] != 0 else 0, axis=1)
    
    return df

def calculate_age_total(df_o, type):
    df = df_o.copy()
    if type == "neumonia":
        age_columns = ['0-1', '1-4', '5-9', '10-19', '20-39', '40-49', '50-59', '60+']

    elif type == "diabetes" or type == "hipertension":
        age_columns = ['0-19', '20-39', '40-49', '50-59', '60+']

    elif type == "embarazo":
      age_columns = ['< 15', '15-19', '20-34', '35-49', '50+']

    else:
        age_columns = ['0-9', '10-19', '20-39', '40-49', '50-59', '60+']

    if type == "embarazo":
        df_summed = df.groupby('Tipo').sum().reset_index()
    else:
        df_summed = df.groupby('Sexo').sum().reset_index()
        df_summed = df_summed.drop(columns=['Año'])

    for age_col in age_columns:
            percent_col = f"% {age_col}"
            df_summed[percent_col] = df_summed.apply(lambda row: round((row[age_col] / row['Total']) * 100, 2) if row['Total'] != 0 else 0, axis=1)

    return df_summed

def calculate_population_group(df):
    # Definir los grupos etarios
    grupos_etarios = {
        '0-9 años': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        '10-19 años': ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19'],
        '20-29 años': ['20', '21', '22', '23', '24', '25', '26', '27', '28', '29'],
        '30-39 años': ['30-34', '35-39'],
        '40-49 años': ['40-44', '45-49'],
        '50-59 años': ['50-54', '55-59'],
        '60-69 años': ['60-64', '65-69'],
        '70-79 años': ['70-74', '75-79'],
        '80 o más': ['80 o más']
    }

    # Crear una copia del DataFrame para evitar modificaciones en el original
    df_grupos_etarios = df.copy()

    # Agrupar los datos por grupos etarios
    for grupo, edades in grupos_etarios.items():
        df_grupos_etarios[grupo] = df[edades].sum(axis=1)
    
    # Seleccionar las columnas relevantes para la gráfica
    df_grupos_etarios = df_grupos_etarios[['DEP_PROV_MUN', 'Sexo'] + list(grupos_etarios.keys())]
    
    return df_grupos_etarios

def table_total_percent(df):
    # Agrupar por año y sexo para obtener los totales
    df_totales = df.groupby(['Año', 'Sexo'])['Total'].sum().unstack(fill_value=0)

    # Calcular el total general por año
    df_totales['Total'] = df_totales.sum(axis=1)

    # Calcular los porcentajes
    df_totales['% Hombres'] = (df_totales['Hombre'] / df_totales['Total'] * 100).round(2).astype(str) + '%'
    df_totales['% Mujeres'] = (df_totales['Mujer'] / df_totales['Total'] * 100).round(2).astype(str) + '%'

    # Crear el nuevo DataFrame con los totales y porcentajes
    df_resultado = pd.DataFrame({
        'Año': df_totales.index,
        'Total Hombres': df_totales['Hombre'].astype(str) + ' (' + df_totales['% Hombres'] + ')',
        'Total Mujeres': df_totales['Mujer'].astype(str) + ' (' + df_totales['% Mujeres'] + ')',
        'Total': df_totales['Total']
    }).reset_index(drop=True)
    
    return df_resultado

def generate_graph_total(df_barras, df_tendencias, labels_barras, labels_tendencias,
                   title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    fig = go.Figure()

    texttemplate = "%{text:.2f}%" if y == 'Porcentajes' else None
    # Lista de colores
    color_barras = ['#011CA2', '#0B2FE4', '#2146FF', '#6E85F5', '#99AAFF']  # Puedes agregar más colores
    color_tendencias = ['#FFCDA6', '#FFAB5C', '#FF932E', '#F88011', '#E95D0C']  # Puedes agregar más colores

    for i, df in enumerate(df_tendencias):
        tend_color = color_tendencias[i % len(color_tendencias)]
        # Gráfica de tendencias
        fig.add_trace(go.Scatter(
            x=df[x].astype(str),
            y=df[y],
            text=df[y],
            texttemplate=texttemplate,
            textfont=dict(color=tend_color, family="sans-serif", size=size_graph, weight='bold'),
            #textfont=dict(family="sans-serif", size=size_graph, weight='bold'),
            textposition="top center",
            mode='lines+markers+text',
            name=labels_tendencias[i],
            line=dict(color=tend_color, width=2) 
        ))

    for i, df in enumerate(df_barras):
        # Obtener el color correspondiente
        bar_color = color_barras[i % len(color_barras)]
        # Gráfica de barras
        fig.add_trace(go.Bar(
            x=df[x].astype(str),
            y=df[y],
            text=df[y],
            texttemplate=texttemplate,
            textfont=dict(color=bar_color, family="sans-serif", size=size_graph, weight='bold'),
            #textfont=dict(family="sans-serif", size=size_graph, weight='bold'),
            textposition='outside',
            name=labels_barras[i],
            marker=dict(color=bar_color, line=dict(color='black', width=1))
        ))
    if y == "Incidencia":
        y = "Incidencia x 10000 personas"
    # Ajustes de layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title)
        },
        xaxis_title=x,
        yaxis_title=y,
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(font=dict(size=size_legend)),
        annotations=[
            dict(
                text=footer,
                xref="paper", yref="paper",
                x=0.5, y=-0.2,  # Ajustar posición según sea necesario
                showarrow=False,
                font=dict(size=size_footer)
            )
        ],
        width=800,  # Ancho de la gráfica
        height=500  # Alto de la gráfica
    )

    # Habilitar líneas de cuadrícula
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_x))
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_y))

    return fig

def generate_graph_join_gender(df_barras, df_tendencias, labels_barras, labels_tendencias, 
                         title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Hombres", "Mujeres"))

    texttemplate = "%{text:.2f}%" if y == 'Porcentajes' else None
    # Lista de colores
    color_barras_male = ['#011CA2', '#0B2FE4', '#2146FF', '#6E85F5', '#99AAFF']  # Puedes agregar más colores
    color_tendencias_male = ['#99AAFF', '#6E85F5', '#2146FF', '#0B2FE4', '#011CA2']
    color_barras_female = ['#E95D0C', '#F88011', '#FF932E', '#FFAB5C', '#FFCDA6']  # Puedes agregar más colores
    color_tendencias_female = ['#FFCDA6', '#FFAB5C', '#FF932E', '#F88011', '#E95D0C']
    
    # Separar en DataFrames para hombres y mujeres
    df_barras_male = [df.loc[df['Sexo'] == 'Hombre'].reset_index(drop=True) for df in df_barras]
    df_barras_female = [df.loc[df['Sexo'] == 'Mujer'].reset_index(drop=True) for df in df_barras]
    df_tendencias_male = [df.loc[df['Sexo'] == 'Hombre'].reset_index(drop=True) for df in df_tendencias]
    df_tendencias_female = [df.loc[df['Sexo'] == 'Mujer'].reset_index(drop=True) for df in df_tendencias]

    for i, df in enumerate(df_barras_male):
        # Obtener el color correspondiente
        bar_color = color_barras_male[i % len(color_barras_male)]
        # Gráfica de barras
        fig.add_trace(go.Bar(
            x=df[x].astype(str),
            y=df[y],
            text=df[y],
            texttemplate=texttemplate,
            textfont=dict(color=bar_color, family="sans-serif", size=size_graph, weight='bold'),
            textposition='outside',
            name='Hombre-'+labels_barras[i],
            marker=dict(color=bar_color, line=dict(color='black', width=1))
        ), row=1, col=1)

    for i, df in enumerate(df_tendencias_male):
        tend_color = color_tendencias_male[i % len(color_tendencias_male)]
        # Gráfica de tendencias
        fig.add_trace(go.Scatter(
            x=df[x].astype(str),
            y=df[y],
            text=df[y],
            texttemplate=texttemplate,
            textfont=dict(color=tend_color, family="sans-serif", size=size_graph, weight='bold'),
            textposition="top center",
            mode='lines+markers+text',
            name='Hombre-'+labels_tendencias[i],
            line=dict(color=tend_color, width=2) 
        ), row=1, col=1)

    for i, df in enumerate(df_barras_female):
        # Obtener el color correspondiente
        bar_color = color_barras_female[i % len(color_barras_female)]
        # Gráfica de barras
        fig.add_trace(go.Bar(
            x=df[x].astype(str),
            y=df[y],
            text=df[y],
            texttemplate=texttemplate,
            textfont=dict(color=bar_color, family="sans-serif", size=size_graph, weight='bold'),
            textposition='outside',
            name='Mujer-'+labels_barras[i],
            marker=dict(color=bar_color, line=dict(color='black', width=1))
        ), row=1, col=2)

    for i, df in enumerate(df_tendencias_female):
        tend_color = color_tendencias_female[i % len(color_tendencias_female)]
        # Gráfica de tendencias
        fig.add_trace(go.Scatter(
            x=df[x].astype(str),
            y=df[y],
            text=df[y],
            texttemplate=texttemplate,
            textfont=dict(color=tend_color, family="sans-serif", size=size_graph, weight='bold'),
            textposition="top center",
            mode='lines+markers+text',
            name='Mujer-'+labels_tendencias[i],
            line=dict(color=tend_color, width=2) 
        ), row=1, col=2)

    if y == "Incidencia":
        y = "Incidencia x 10000 personas"
    # Ajustes de layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title)
        },
        xaxis_title=x,
        yaxis_title=y,
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(font=dict(size=size_legend)),
        annotations=[
            dict(
                text=footer,
                xref="paper", yref="paper",
                x=0.5, y=-0.2,  # Ajustar posición según sea necesario
                showarrow=False,
                font=dict(size=size_footer)
            )
        ]
    )

    # Habilitar líneas de cuadrícula
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_x))
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_y), matches='y')

    return fig

def generate_graph_separate_gender(dfs, graph_type, labels, 
                                    title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    num_dfs = len(dfs)
    num_cols = 2
    num_rows = (num_dfs + 1) // 2  # Calcula el número de filas necesarias

    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=labels, vertical_spacing=0.15, horizontal_spacing=0.1)

    texttemplate = "%{text:.2f}%" if y == 'Porcentajes' else None

    # Lista de colores
    color_1_male = ['#011CA2', '#0B2FE4', '#2146FF', '#6E85F5', '#99AAFF']  # Puedes agregar más colores
    color_2_male = ['#99AAFF', '#6E85F5', '#2146FF', '#0B2FE4', '#011CA2']
    color_1_female = ['#E95D0C', '#F88011', '#FF932E', '#FFAB5C', '#FFCDA6']
    color_2_female = ['#FFCDA6', '#FFAB5C', '#FF932E', '#F88011', '#E95D0C']

    # Separar en DataFrames para hombres y mujeres
    df_male = [df.loc[df['Sexo'] == 'Hombre'].reset_index(drop=True) for df in dfs]
    df_female = [df.loc[df['Sexo'] == 'Mujer'].reset_index(drop=True) for df in dfs]

    if graph_type == 'barras':
        for i, df in enumerate(df_male):
            bar_color = color_1_male[i % len(color_1_male)]
            row = i // num_cols + 1
            col = i % num_cols + 1 

            fig.add_trace(go.Bar(
                x=df[x].astype(str),
                y=df[y],
                text=df[y],
                texttemplate=texttemplate,
                textfont=dict(color=bar_color, family="sans-serif", size=size_graph),
                textposition='outside',
                name=f'Hombre - {labels[i]}',
                marker=dict(color=bar_color, line=dict(color='black', width=1)),
                showlegend=False
            ), row=row, col=col)

        for i, df in enumerate(df_female):
            bar_color = color_1_female[i % len(color_1_female)]
            row = i // num_cols + 1
            col = i % num_cols + 1

            fig.add_trace(go.Bar(
                x=df[x].astype(str),
                y=df[y],
                text=df[y],
                texttemplate=texttemplate,
                textfont=dict(color=bar_color, family="sans-serif", size=size_graph),
                textposition='outside',
                name=f'Mujer - {labels[i]}',
                marker=dict(color=bar_color, line=dict(color='black', width=1)),
                showlegend=False
            ), row=row, col=col)

    else:
        for i, df in enumerate(df_male):
            tend_color = color_2_male[i % len(color_2_male)]
            row = i // num_cols + 1
            col = i % num_cols + 1

            fig.add_trace(go.Scatter(
                x=df[x].astype(str),
                y=df[y],
                text=df[y],
                texttemplate=texttemplate,
                textfont=dict(color=tend_color, family="sans-serif", size=size_graph),
                textposition="top center",
                mode='lines+markers+text',
                name=f'Hombre - {labels[i]}',
                line=dict(color=tend_color, width=2),
                showlegend=False
            ), row=row, col=col)

        for i, df in enumerate(df_female):
            tend_color = color_2_female[i % len(color_2_female)]
            row = i // num_cols + 1
            col = i % num_cols + 1

            fig.add_trace(go.Scatter(
                x=df[x].astype(str),
                y=df[y],
                text=df[y],
                texttemplate=texttemplate,
                textfont=dict(color=tend_color, family="sans-serif", size=size_graph),
                textposition="top center",
                mode='lines+markers+text',
                name=f'Mujer - {labels[i]}',
                line=dict(color=tend_color, width=2),
                showlegend=False
            ), row=row, col=col)

    if y == "Incidencia":
        y = "Incidencia x 10000 personas"
    # Ajustes de layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title)
        },
        xaxis_title=x,
        yaxis_title=y,
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        #annotations=[
        #    dict(
        #        text=footer,
        #        xref="paper", yref="paper",
        #        x=0.5, y=-0.1,  # Ajustar posición según sea necesario
        #        showarrow=False,
        #        font=dict(size=size_footer)
        #    )
        #],
        height=400 * num_rows,  # Ajustar la altura de las subplots
        width=1000,  # Ajustar la anchura de las subplots
    )

    # Habilitar líneas de cuadrícula y ajustar tamaños de fuentes de los ejes
    for i in range(1, num_rows + 1):
        for j in range(1, num_cols + 1):
            fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_text=x, title_font=dict(size=size_x), tickfont=dict(size=size_x), row=i, col=j)
            fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_text=y, title_font=dict(size=size_y), tickfont=dict(size=size_y), row=i, col=j, matches='y')

    return fig

def generate_graph_separate_age(df, graph_type, labels, 
                                   title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    # Filtrar datos por sexo
    df_male = df[df['Sexo'] == 'Hombre']
    df_female = df[df['Sexo'] == 'Mujer']

    # Crear figura con subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Hombres', 'Mujeres'])

    # Configuración de colores por año (puedes modificar o añadir más colores según necesites)
    colors_male = ['#011CA2', '#0B2FE4', '#2146FF', '#6E85F5', '#99AAFF']
    colors_female = ['#E95D0C', '#F88011', '#FF932E', '#FFAB5C', '#FFCDA6']
    
    age_columns = []
    if y == 'Total':
        if graph_type == "neumonia":
            age_columns = ['0-1', '1-4', '5-9', '10-19', '20-39', '40-49', '50-59', '60+']
        elif graph_type == "diabetes" or graph_type == "hipertension":
            age_columns = ['0-19', '20-39', '40-49', '50-59', '60+']
        else:
            age_columns = ['0-9', '10-19', '20-39', '40-49', '50-59', '60+']
    elif y == 'Incidencia':
        if graph_type == "neumonia":
            age_columns = ['I_0-1', 'I_1-4', 'I_5-9', 'I_10-19', 'I_20-39', 'I_40-49', 'I_50-59', 'I_60+']
        elif graph_type == "diabetes" or graph_type == "hipertension":
            age_columns = ['I_0-19', 'I_20-39', 'I_40-49', 'I_50-59', 'I_60+']
        else:
            age_columns = ['I_0-9', 'I_10-19', 'I_20-39', 'I_40-49', 'I_50-59', 'I_60+']
    elif y == 'Porcentaje':
        if graph_type == "neumonia":
            age_columns = ['% 0-1', '% 1-4', '% 5-9', '% 10-19', '% 20-39', '% 40-49', '% 50-59', '% 60+']
        elif graph_type == "diabetes" or graph_type == "hipertension":
            age_columns = ['% 0-19', '% 20-39', '% 40-49', '% 50-59', '% 60+']
        else:
            age_columns = ['% 0-9', '% 10-19', '% 20-39', '% 40-49', '% 50-59', '% 60+']


    # Añadir barras para hombres
    for i, year in enumerate(df[x].unique()):
        fig.add_trace(go.Bar(
            x=age_columns,  # Columnas desde < 6 en adelante son edades
            y=df_male[df_male[x] == year][age_columns].iloc[0],  # Fila correspondiente al año y hombres
            name=str(year),
            marker=dict(color=colors_male[i]),
            legendgroup='hombres',
            texttemplate='%{y}',
            textposition='outside', 
            textfont=dict(color=colors_male[i], size=size_graph, line=dict(color='black', width=1)),
        ), row=1, col=1)

    # Añadir barras para mujeres
    for i, year in enumerate(df[x].unique()):
        fig.add_trace(go.Bar(
            x=age_columns,  # Columnas desde < 6 en adelante son edades
            y=df_female[df_female[x] == year][age_columns].iloc[0],  # Fila correspondiente al año y mujeres
            name=str(year),
            marker=dict(color=colors_female[i]),
            legendgroup='mujeres',
            texttemplate='%{y}',
            textposition='outside', 
            textfont=dict(color=colors_female[i], size=size_graph, line=dict(color='black', width=1)),
        ), row=1, col=2)

    if y == "Incidencia":
        y = "Incidencia x 10000 personas"

    # Ajustes de layout
    fig.update_layout(
        title={
            'text': title + ' (' + labels + ')',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title)
        },
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            title=y,
            titlefont_size=size_y,
            showgrid=True,  # Mostrar cuadrícula en el eje y
        ),
        #font=dict(size=size_graph),
        legend=dict(
            x=1.02, y=1,  # Ajustar posición de la leyenda a la derecha y arriba
            orientation='v',  # Orientación vertical
            tracegroupgap=10,
            font=dict(size=size_legend),
        )
    )

    # Ajustar título del eje x para ambas subplots
    fig.update_xaxes(title_font=dict(size=size_x), title_text="Edad", row=1, col=1)
    fig.update_xaxes(title_font=dict(size=size_x), title_text="Edad", row=1, col=2)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_y), row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_y), row=1, col=2)

    return fig

def generate_graph_age_pregnans(df, graph_type, labels, 
                           title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    # Crear figura
    fig = go.Figure()

    df_female = df.copy()
    colors_female = ['#E95D0C', '#F88011', '#FF932E', '#FFAB5C', '#FFCDA6']
    
    if y != 'Incidencia':
        age_columns = []
        if y == 'Total':
            age_columns = ['< 15', '15-19', '20-34', '35-49', '50+']
        elif y == 'Porcentaje':
            age_columns = ['% < 15', '% 15-19', '% 20-34', '% 35-49', '% 50+']

        # Añadir barras para cada año
        for i, year in enumerate(df[x].unique()):
            fig.add_trace(go.Bar(
                x=age_columns,  # Columnas desde < 6 en adelante son edades
                y=df_female[df_female[x] == year][age_columns].iloc[0],  # Fila correspondiente al año y mujeres
                name=f'Año {year}',
                marker=dict(color=colors_female[i % len(colors_female)], line=dict(color='black', width=1)),
                texttemplate='%{y}',
                textposition='outside', 
                textfont=dict(color=colors_female[i % len(colors_female)], size=size_graph),
            ))

        # Ajustes de layout
        fig.update_layout(
            title={
                'text': title + ' (' + labels + ')',
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=size_title)
            },
            barmode='stack',  # Apilar barras
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(
                title=y,
                titlefont_size=size_y,
                showgrid=True,  # Mostrar cuadrícula en el eje y
            ),
            annotations=[
                dict(
                    text=footer,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.2,  # Ajustar posición según sea necesario
                    showarrow=False,
                    font=dict(size=size_footer)
                )
            ],
            width=800,  # Ancho de la gráfica
            height=500  # Alto de la gráfica
        )

        # Ajustar título del eje x
        fig.update_xaxes(title_font=dict(size=size_x), title_text="Edad")
        fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_y))

    return fig

def generate_graph_join_age(df, graph_type, labels, 
                                   title, size_title, x, size_x,y, size_y, footer, size_footer, size_legend, size_graph, matches='y'):
    # Filtrar datos por sexo
    df_male = df[df['Sexo'] == 'Hombre']
    df_female = df[df['Sexo'] == 'Mujer']

    # Crear figura con subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Hombres', 'Mujeres'])

    # Configuración de colores
    color_male = '#011FB7'
    color_female = '#FF810A'
    
    age_columns = []
    if y == 'Total':
        if graph_type == "neumonia":
            age_columns = ['0-1', '1-4', '5-9', '10-19', '20-39', '40-49', '50-59', '60+']
        elif graph_type == "diabetes" or graph_type == "hipertension":
            age_columns = ['0-19', '20-39', '40-49', '50-59', '60+']
        else:
            age_columns = ['0-9', '10-19', '20-39', '40-49', '50-59', '60+']
    elif y == 'Incidencia':
        if graph_type == "neumonia":
            age_columns = ['I_0-1', 'I_1-4', 'I_5-9', 'I_10-19', 'I_20-39', 'I_40-49', 'I_50-59', 'I_60+']
        elif graph_type == "diabetes" or graph_type == "hipertension":
            age_columns = ['I_0-19', 'I_20-39', 'I_40-49', 'I_50-59', 'I_60+']
        else:
            age_columns = ['I_0-9', 'I_10-19', 'I_20-39', 'I_40-49', 'I_50-59', 'I_60+']
    elif y == 'Porcentaje':
        if graph_type == "neumonia":
            age_columns = ['% 0-1', '% 1-4', '% 5-9', '% 10-19', '% 20-39', '% 40-49', '% 50-59', '% 60+']
        elif graph_type == "diabetes" or graph_type == "hipertension":
            age_columns = ['% 0-19', '% 20-39', '% 40-49', '% 50-59', '% 60+']
        else:
            age_columns = ['% 0-9', '% 10-19', '% 20-39', '% 40-49', '% 50-59', '% 60+']

    # Añadir barras para hombres
    fig.add_trace(go.Bar(
        x=age_columns,
        y=df_male[age_columns].iloc[0],
        name='Hombres',
        marker=dict(color=color_male, line=dict(color='black', width=1)),
        texttemplate='%{y}',
        textposition='outside',
        textfont=dict(color=color_male, size=size_graph),
    ), row=1, col=1)

    # Añadir barras para mujeres
    fig.add_trace(go.Bar(
        x=age_columns,
        y=df_female[age_columns].iloc[0],
        name='Mujeres',
        marker=dict(color=color_female, line=dict(color='black', width=1)),
        texttemplate='%{y}',
        textposition='outside',
        textfont=dict(color=color_female, size=size_graph),
    ), row=1, col=2)

    if y == "Incidencia":
        y = "Incidencia x 10000 personas"
    # Ajustes de layout
    fig.update_layout(
        title={
            'text': title + ' (' + labels + ')',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title)
        },
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            title=y,
            titlefont_size=size_y,
            showgrid=True,  # Mostrar cuadrícula en el eje y
        ),
        legend=dict(
            x=1.02, y=1,  # Ajustar posición de la leyenda a la derecha y arriba
            orientation='v',  # Orientación vertical
            tracegroupgap=10,
            font=dict(size=size_legend),
        )
    )

    # Ajustar título del eje x para ambas subplots
    fig.update_xaxes(title_font=dict(size=size_x), title_text="Edad", row=1, col=1)
    fig.update_xaxes(title_font=dict(size=size_x), title_text="Edad", row=1, col=2)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_y), row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_y), row=1, col=2)

    return fig

def generate_comparison_graph_by_year(df, labels, 
                                      title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    # Filtrar datos por sexo
    df_male = df[df['Sexo'] == 'Hombre']
    df_female = df[df['Sexo'] == 'Mujer']

    # Configuración de colores
    color_male = '#011CA2'
    color_female = '#F88011'
    
    fig = go.Figure()

    # Añadir barras para hombres
    fig.add_trace(go.Bar(
        x=df_male[x],
        y=df_male[y],
        name='Hombres',
        marker=dict(color=color_male, line=dict(color='black', width=1)),
        texttemplate='%{y}',
        textposition='outside',
        textfont=dict(color=color_male, size=size_graph),
    ))

    # Añadir barras para mujeres
    fig.add_trace(go.Bar(
        x=df_female[x],
        y=df_female[y],
        name='Mujeres',
        marker=dict(color=color_female, line=dict(color='black', width=1)),
        texttemplate='%{y}',
        textposition='outside',
        textfont=dict(color=color_female, size=size_graph),
    ))
    # Ajustes de layout
    fig.update_layout(
        title={
            'text': title + ' (' + labels + ')',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title)
        },
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title='Año',
            titlefont_size=size_x,
        ),
        yaxis=dict(
            title='Total',
            titlefont_size=size_y,
            showgrid=True,  # Mostrar cuadrícula en el eje y
        ),
        legend=dict(
            x=1.02, y=1,  # Ajustar posición de la leyenda a la derecha y arriba
            orientation='v',  # Orientación vertical
            tracegroupgap=10,
            font=dict(size=size_legend),
        ),
        annotations=[
            dict(
                text=footer,
                xref="paper", yref="paper",
                x=0.5, y=-0.2,  # Ajustar posición según sea necesario
                showarrow=False,
                font=dict(size=size_footer)
            )
        ],
        width=800,  # Ancho de la gráfica
        height=500
    )

    return fig

def generate_graph_by_years(df, label, 
                                      title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    # Configuración de colores
    color = '#011CA2'
    
    fig = go.Figure()

    # Añadir líneas de tendencia
    fig.add_trace(go.Scatter(
        x=df[x],
        y=df[y],
        mode='lines+markers+text',
        name=label,
        marker=dict(color=color),
        text=df[y],
        textposition='top center',
        textfont=dict(color=color, size=size_graph),
    ))

    # Ajustes de layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title)
        },
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title='Año',
            titlefont_size=size_x,
        ),
        yaxis=dict(
            title='Total',
            titlefont_size=size_y,
            showgrid=True,  # Mostrar cuadrícula en el eje y
        ),
        legend=dict(
            x=1.02, y=1,  # Ajustar posición de la leyenda a la derecha y arriba
            orientation='v',  # Orientación vertical
            tracegroupgap=10,
            font=dict(size=size_legend),
        ),
        annotations=[
            dict(
                text=footer,
                xref="paper", yref="paper",
                x=0.5, y=-0.2,  # Ajustar posición según sea necesario
                showarrow=False,
                font=dict(size=size_footer)
            )
        ],
        width=800,  # Ancho de la gráfica
        height=500
    )

    return fig

def generate_stacked_bar_chart(df, labels, title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    df_obesidad = df[0]
    df_sobrepeso = df[1]
    df_desnutricion = df[2]

    # Calcular el total de embarazadas por año
    df_obesidad['Total'] = df_obesidad['Embarazadas'] + df_sobrepeso['Embarazadas'] + df_desnutricion['Embarazadas']
    df_sobrepeso['Total'] = df_obesidad['Total']
    df_desnutricion['Total'] = df_obesidad['Total']

    # Calcular el porcentaje de cada categoría respecto al total del año
    df_obesidad['Porcentaje'] = (df_obesidad['Embarazadas'] / df_obesidad['Total']) * 100
    df_sobrepeso['Porcentaje'] = (df_sobrepeso['Embarazadas'] / df_sobrepeso['Total']) * 100
    df_desnutricion['Porcentaje'] = (df_desnutricion['Embarazadas'] / df_desnutricion['Total']) * 100

    # Combinar los DataFrames para la visualización
    df_combined = pd.merge(df_obesidad[[x, 'Porcentaje']], df_sobrepeso[[x, 'Porcentaje']], on=x, suffixes=('_Obesidad', '_Sobrepeso'))
    df_combined = pd.merge(df_combined, df_desnutricion[[x, 'Porcentaje']], on=x)
    df_combined.columns = [x, 'Obesidad', 'Sobrepeso', 'Desnutrición']

    # Crear el DataFrame para la visualización
    df_plot = df_combined.set_index(x)

    # Crear el gráfico de barras apiladas con Plotly
    fig = go.Figure()

    # Agregar cada categoría como una traza en el gráfico con los colores deseados
    fig.add_trace(go.Bar(
        x=df_plot.index,
        y=df_plot['Obesidad'],
        name='Obesidad',
        marker_color='#011CA2'  # Color para Obesidad
    ))

    fig.add_trace(go.Bar(
        x=df_plot.index,
        y=df_plot['Sobrepeso'],
        name='Sobrepeso',
        marker_color='#0B2FE4'  # Color para Sobrepeso
    ))

    fig.add_trace(go.Bar(
        x=df_plot.index,
        y=df_plot['Desnutrición'],
        name='Desnutrición',
        marker_color='#2146FF'  # Color para Desnutrición
    ))

    # Actualizar el diseño del gráfico
    fig.update_layout(
        title={
            'text': title + ' (' + labels + ')', 
            'x': 0.5, 
            'xanchor': 'center',
            'font': {'size': size_title}
        },
        xaxis_title=x,
        yaxis_title='Porcentaje (%)',
        barmode='stack',
        legend=dict(font=dict(size=size_legend)),
        plot_bgcolor='white',  # Fondo del área de trazado blanco
        paper_bgcolor='white',  # Fondo del papel blanco
        width=1000,  # Ancho del gráfico
        height=600,  # Altura del gráfico
        xaxis=dict(
            type='category',  # Asegura que el eje X trate los valores como categorías
            tickmode='array',  # Usa una lista explícita de valores para las etiquetas del eje X
            tickvals=df_plot.index,  # Etiquetas de los años
            ticktext=df_plot.index.astype(str)  # Convierte los años a texto
        )
    )

    # Añadir porcentajes dentro de las barras con texto blanco centrado
    fig.update_traces(
        texttemplate='%{y:.1f}%',
        textposition='inside',
        textfont=dict(color='white', size=size_graph),  # Color del texto blanco y tamaño ajustado
        insidetextanchor='middle'  # Centra el texto dentro de las barras
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_x))
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_y))

    # Mostrar el gráfico
    return fig

def generate_population_pyramid(df, region):
    # Filtrar los datos por la región especificada
    df_region = df[df['DEP_PROV_MUN'] == region]
    
    # Definir los grupos etarios en orden inverso
    grupos_etarios = ['80 o más', '70-79 años', '60-69 años', '50-59 años', '40-49 años', '30-39 años', '20-29 años', '10-19 años', '0-9 años']

    # Filtrar los datos por sexo
    df_male = df_region[df_region['Sexo'] == 'Hombre']
    df_female = df_region[df_region['Sexo'] == 'Mujer']

    # Sumar los datos por grupos etarios
    x_male = df_male[grupos_etarios].sum()
    x_female = df_female[grupos_etarios].sum()

    # Encontrar el valor máximo en los datos
    max_value = max(x_male.max(), x_female.max())

    # Definir intervalos de ticks para diferentes rangos de datos
    intervals = {
        (0, 500): 50,
        (501, 1000): 100,
        (1001, 5000): 500,
        (5001, 10000): 1000,
        (10001, 50000): 5000,
        (50001, 100000): 10000
    }

    # Encontrar el intervalo de ticks adecuado para el valor máximo
    tick_interval = next(interval for range, interval in intervals.items() if range[0] <= max_value <= range[1])

    # Definir los colores por grupos etarios
    colors_male = ['#0B34FE', '#0B34FE', '#0B34FE', '#0126DF', '#0126DF', '#011FB7', '#011FB7', '#01198D', '#01198D']
    colors_female = ['#FFAB5C', '#FFAB5C', '#FFAB5C', '#FF9633', '#FF9633', '#FF810A', '#FF810A', '#E06C00', '#E06C00']

    # Crear la figura con dos subgráficas
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Hombres', 'Mujeres'),
        shared_yaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.04
    )

    # Añadir el gráfico de hombres
    fig.add_trace(go.Bar(
        y=grupos_etarios,
        x=x_male,
        orientation='h',
        name='Hombres',
        marker=dict(color=colors_male, line=dict(color='black', width=1)),
        text=[f'{int(value)}' for value in x_male],
        textposition='inside',
        texttemplate='%{text}',
        textfont=dict(color='white', size=16),
        insidetextanchor='middle',
        # Ajustar el eje x para mostrar valores en orden descendente
        xaxis='x'
    ), row=1, col=1)

    # Añadir el gráfico de mujeres
    fig.add_trace(go.Bar(
        y=grupos_etarios,
        x=x_female,
        orientation='h',
        name='Mujeres',
        marker=dict(color=colors_female, line=dict(color='black', width=1)),
        text=[f'{int(value)}' for value in x_female],
        textposition='inside',
        texttemplate='%{text}',
        textfont=dict(color='white', size=16),
        insidetextanchor='middle'
    ), row=1, col=2)

    # Configurar el diseño de la gráfica
    fig.update_layout(
        title=f'Pirámide de Población - {region}',
        title_x=0.5,  # Centrar el título
        title_font=dict(size=24),
        xaxis_title='Población',
        yaxis_title='Grupos Etarios',
        yaxis=dict(
            title='Grupos Etarios',
            autorange='reversed',
            gridcolor='gray',  # Color de las líneas de la cuadrícula en el eje Y
            gridwidth=0.5
        ),
        yaxis2=dict(
            gridcolor='gray',  # Color de las líneas de la cuadrícula en el eje Y
            gridwidth=0.5
        ),
        xaxis=dict(
            title='Población',
            tickvals=np.arange(0, max_value + tick_interval, tick_interval),  # Ajustar los ticks del eje X
            ticktext=[f'{i}' for i in np.arange(0, max_value + tick_interval, tick_interval)],
            tickformat=',',  # Formato de los ticks con comas
            range=[max_value, 0],  # Ajustar el rango del eje x para que vaya de max_value a 0
            gridcolor='gray',  # Color de las líneas de la cuadrícula en el eje Y
            gridwidth=0.5
        ),
        xaxis2=dict(
            title='Población',
            tickvals=np.arange(0, max_value + tick_interval, tick_interval),  # Ajustar los ticks del eje X para la segunda gráfica
            ticktext=[f'{i}' for i in np.arange(0, max_value + tick_interval, tick_interval)],
            tickformat=',',  # Formato de los ticks con comas
            gridcolor='gray',  # Color de las líneas de la cuadrícula en el eje Y
            gridwidth=0.5
        ),
        plot_bgcolor='white',  # Fondo del área de la gráfica
        paper_bgcolor='white',  # Fondo del área de la gráfica completa
        showlegend=True,
        legend=dict(x=0.8, y=1),
        autosize=True,
        width=1000,  # Ancho total de la gráfica
        height=600  # Alto de la gráfica
    )
    return fig

def generate_language_donut_chart(df, region, colors, footer_text="Fuente: Datos proporcionados por el Departamento de Estadísticas"):
    row = df[df["Dep_Prov_Mun"] == region].iloc[0]
    values = row[1:].values
    labels = df.columns[1:].values

    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.5, 
        marker=dict(colors=colors, line=dict(color='black', width=0.5)),
        #textinfo='label+percent',
        textfont=dict(size=15)
        )])

    fig.update_layout(
        title={
            'text': f'Porcentaje de Idiomas con el que aprendió a hablar la población de 4 años a más ({region})',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        annotations=[
            dict(text=region, x=0.5, y=0.5, font_size=20, showarrow=False),
            dict(text="Fuente: Datos del Censo 2012", x=0.5, y=-0.2, font_size=12, showarrow=False)
        ],
        legend=dict(
            font=dict(size=14),  # Tamaño de la letra de la leyenda
            orientation='h',  # Orientación horizontal de la leyenda
            x=0.5,  # Posición horizontal de la leyenda
            y=-0.05  # Posición vertical de la leyenda
        ),
        height=500,  # Ajustar el alto de la gráfica
        width=1000   # Ajustar el ancho de la gráfica
    )

    return fig

def generate_literacy_donut_chart(df, region):
    row = df[df["Dep_Prov_Mun"] == region].iloc[0]
    # Datos para hombres
    values_hombre = [row["Hombre"], row["Hombre_Illiterate"]]
    labels_hombre = ["Alfabetizados", "Analfabetos"]
    colors_hombre = ["#0B34FE", "#0126DF"]  # Colores para alfabetizados y analfabetos

    # Datos para mujeres
    values_mujer = [row["Mujer"], row["Mujer_Illiterate"]]
    labels_mujer = ["Alfabetizadas", "Analfabetas"]
    colors_mujer = ["#FFAB5C", "#FF9633"]  # Colores para alfabetizadas y analfabetas
    
    # Crear subgráficas
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])

    # Añadir trazas a las subgráficas
    fig.add_trace(
        go.Pie(
            labels=labels_hombre,
            values=values_hombre,
            name="Hombres",
            marker=dict(colors=colors_hombre, line=dict(color='black', width=0.5)),
            hole=.5,
            #textinfo='label+percent',
            textfont=dict(size=16)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Pie(
            labels=labels_mujer,
            values=values_mujer,
            name="Mujeres",
            marker=dict(colors=colors_mujer, line=dict(color='black', width=0.5)),
            hole=.5,
            #textinfo='label+percent',
            textfont=dict(size=16)
        ),
        row=1, col=2
    )

    # Configurar el diseño de la gráfica
    fig.update_layout(
        title={
            'text': f'Porcentaje de Alfabetismo por sexo ({region})',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        annotations=[
            dict(text='Hombres', x=0.165, y=0.5, font_size=20, showarrow=False),
            dict(text='Mujeres', x=0.83, y=0.5, font_size=20, showarrow=False),
            dict(text="Fuente: Datos del Censo 2012", x=0.5, y=-0.2, font_size=12, showarrow=False)
        ],
        legend=dict(
            font=dict(size=14),  # Tamaño de la letra de la leyenda
            orientation='h',  # Orientación horizontal de la leyenda
            x=0.5,  # Posición horizontal de la leyenda
            y=-0.05  # Posición vertical de la leyenda
        ),
        height=500,  # Ajustar el alto de la gráfica
        width=1000  # Ajustar el ancho de la gráfica
    )

    return fig

def generate_services_bar_chart(df, region):
    row = df[df["Dep_Prov_Mun"] == region].iloc[0]
    servicios = row.index[1:]
    porcentajes = row.values[1:]

    # Calcular los porcentajes restantes
    porcentajes_restantes = 100 - porcentajes

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=servicios,
        y=porcentajes,
        name='Con Servicio',
        marker_color='#636EFA'  # Color para "Con Servicio"
    ))

    fig.add_trace(go.Bar(
        x=servicios,
        y=porcentajes_restantes,
        name='Sin Servicio',
        marker_color='lightgrey'  # Color para "Sin Servicio"
    ))

    fig.update_layout(
        barmode='stack',
        title={
            'text': f'Acceso a Servicios Básicos ({region})',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        yaxis=dict(
            title='Porcentaje',
            ticksuffix='%',
            gridcolor='grey'
        ),
        xaxis=dict(
            title='Servicio Básico',
            gridcolor='grey',
            tickangle=-45
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True,
        height=500,  # Ajusta la altura del gráfico
        width=800    # Ajusta el ancho del gráfico
    )

    # Añadir porcentaje a la derecha de las barras
    for i, service in enumerate(servicios):
        fig.add_annotation(
            x=service,
            y=porcentajes[i] + porcentajes_restantes[i] / 2,
            text=f'{porcentajes_restantes[i]:.1f}%',
            showarrow=False,
            font=dict(size=16, color='black'),
            align='left',
            xanchor='center',
            yanchor='middle'
        )
        fig.add_annotation(
            x=service,
            y=porcentajes[i] / 2,
            text=f'{porcentajes[i]:.1f}%',
            showarrow=False,
            font=dict(size=16, color='black'),
            align='right',
            xanchor='center',
            yanchor='middle'
        )

    return fig

def generate_housing_pie_chart(df, region):
    row = df[df["Dep_Prov_Mun"] == region].iloc[0]

    # Datos para cada categoría de habitaciones
    values = row[1:-1]  # Excluye la columna 'Dep_Prov_Mun' y 'Total'
    labels = row.index[1:-1]  # Excluye la columna 'Dep_Prov_Mun' y 'Total'
    colors = ['#AB63FA', '#636EFA', '#FFA15A', '#EF553B', '#00CC96']  # Colores para cada categoría

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.5,
        marker=dict(colors=colors, line=dict(color='black', width=0.5)),
        #textinfo='label+percent',
        textfont=dict(size=15)
    )])

    fig.update_layout(
        title={
            'text': f'Porcentaje de Número de Dormitorios en Viviendas Particulares Ocupadas ({region})',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        annotations=[
            dict(text="Fuenta: Datos del Censo 2012", x=0.5, y=-0.2, font_size=12, showarrow=False)
        ],
        height=500,
        width=800,
        legend=dict(
            font=dict(size=14),
            orientation='h',
            x=0.5,
            y=-0.05
        )
    )

    return fig

def generate_ocupation_bar_chart(df, region):
    # Filtrar los datos por región
    df_region = df[df["Dep_Prov_Mun"] == region]
    
    df_percent = df_region.copy()

    # Calcular los porcentajes
    for col in df_region.columns[2:-1]:
        df_percent[col] = df_region[col] / df_region['Total'] * 100

    df_percent['Total'] = 100 
    
    # Crear gráfico de barras agrupadas
    fig = go.Figure()

    # Lista de ocupaciones
    occupations = df_percent.columns[2:-1]

    # Añadir trazas para cada sexo
    for gender in ['Hombre', 'Mujer']:
        subset = df_percent[df_percent['Sexo'] == gender]
        values = [subset[occupation].sum() for occupation in occupations]
        
        fig.add_trace(go.Bar(
            x=occupations,  # Eje x con ocupaciones
            y=values,       # Valores totales por ocupación
            name=gender,
            marker_color='#0126DF' if gender == 'Hombre' else '#FF9633',  # Colores
            text=[f'{v:.2f}%' for v in values],  # Valores sobre las barras
            textposition='auto',  # Posición del texto sobre las barras
            marker=dict(
                line=dict(
                    color='black',  # Color del borde
                    width=0.5       # Ancho del borde
                )
            )
        ))

    fig.update_layout(
        barmode='group',  # Modo de barras agrupadas
        title={
            'text': f'Comparación en Porcentajes de Ocupaciones por Sexo ({region})',
            'x': 0.5,  # Centrar título
            'xanchor': 'center'
        },
        xaxis_title='Ocupación',
        yaxis_title='Porcentaje (%)',
        legend_title='Sexo',
        plot_bgcolor='white',  # Fondo blanco de la gráfica
        paper_bgcolor='white',  # Fondo blanco del área de trabajo
        yaxis=dict(
            gridcolor='lightgray'  # Líneas del eje y grises
        ),
        annotations=[{
            'text': 'Fuente: Datos del Censo 2012',
            'xref': 'paper',
            'yref': 'paper',
            'x': 0.5,
            'y': -0.45,
            'showarrow': False,
            'font': dict(size=12, color='gray'),
            'xanchor': 'center'
        }],
        height=600,
        width=1000,
    )

    return fig

def generate_secondary_abandonment_trend(df, region):
    # Filtrar los datos para la tasa de abandono en secundaria
    df_secondary = df[['Año', 'Sexo', 'Secundaria']].copy()
    
    # Crear gráfico de líneas
    fig = go.Figure()

    # Añadir trazas para cada sexo
    for gender in ['Hombre', 'Mujer']:
        subset = df_secondary[df_secondary['Sexo'] == gender]
        color = '#0126DF' if gender == 'Hombre' else '#FF9633'
        
        fig.add_trace(go.Scatter(
            x=subset['Año'],  # Eje x con años
            y=subset['Secundaria'],  # Eje y con tasa de abandono en secundaria
            mode='lines+markers+text',  # Mostrar líneas y puntos
            name=gender,
            line=dict(color='#0126DF' if gender == 'Hombre' else '#FF9633', width=2),  # Colores y ancho de línea
            marker=dict(size=8),  # Tamaño de los puntos
            text=[f'{v:.2f}%' for v in subset['Secundaria']],  # Valores sobre los puntos
            textposition='top center',  # Posición del texto
            textfont=dict(
                color=color,  # Color del texto igual al de la línea
                size=14  # Tamaño del texto
            )
        ))

    fig.update_layout(
        title={
            'text': f'Tasa de Abandono en Nivel Secundario por Género (2013-2023) ({region})',
            'x': 0.5,  # Centrar título
            'xanchor': 'center'
        },
        xaxis_title='Año',
        yaxis_title='Tasa de Abandono en Secundaria (%)',
        legend_title='Sexo',
        plot_bgcolor='white',  # Fondo blanco de la gráfica
        paper_bgcolor='white',  # Fondo blanco del área de trabajo
        xaxis=dict(
            tickmode='linear',  # Mostrar todos los años
            tick0=df_secondary['Año'].min(),  # Iniciar desde el año mínimo
            dtick=1,  # Intervalo de ticks en el eje X
        ),
        yaxis=dict(
            gridcolor='lightgray'  # Líneas del eje y grises
        ),
        annotations=[{
            'text': 'Fuente: Ministerio de Educación',
            'xref': 'paper',
            'yref': 'paper',
            'x': 0.5,
            'y': -0.15,
            'showarrow': False,
            'font': dict(size=12, color='gray'),
            'xanchor': 'center'
        }],
        height=600,
        width=1000,
    )

    return fig

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div([
        html.H2("Menú"),
        html.Ul([
            html.Li(dcc.Link('Datos Salud', href='/salud')),
            html.Li(dcc.Link('Cancer', href='/cancer')),
            html.Li(dcc.Link('Diabetes', href='/diabetes')),
            html.Li(dcc.Link('Hipertensión Arterial', href='/hipertension')),
            html.Li(dcc.Link('Obesidad', href='/obesidad')),
            html.Li(dcc.Link('Neumonia', href='/neumonia')),
            html.Li(dcc.Link('Chagas', href='/chagas')),
            html.Li(dcc.Link('VIH', href='/vih')),
            html.Li(dcc.Link('Nutrición', href='/nutricion')),
            html.Li(dcc.Link('Embarazo Adolescente', href='/embarazo')),
            html.Li(dcc.Link('Consultas Externas', href='/consultas')),
        ], className='menu')
    ], className='menu-column'),
    html.Div([
        html.Div(id='page-content')
    ], className='content-column'),
    html.Div(id='btn-calcular', style={'display': 'none'}),  # Div oculto para generar el botón
], className='container')

def generate_calculo_layout_salud(title):
    return html.Div([
        html.Label('Gráfica a mostrar:'),
        dcc.Dropdown(
            id='dropdown-graphic-type',
            options=[
                {'label': 'Piramides Poblacionales', 'value': 'p'},
                {'label': 'Etnicidad', 'value': 'e'},
                {'label': 'Alfabetismo', 'value': 'a'},
                {'label': 'Servicios Basicos', 'value': 'sb'},
                {'label': 'Hacinamiento', 'value': 'h'},
                {'label': 'Ocupacion', 'value': 'o'},
                {'label': 'Abandono Secundaria', 'value': 'as'},
                {'label': 'Mortalidad Infantil', 'value': 'mi'},
                {'label': 'Mortalidad', 'value': 'm'},
                {'label': 'Fertilidad', 'value': 'f'}
            ],
            value='p'
        ),
        html.Div([
            html.Label('Título del gráfico: '),
            dcc.Input(
                id='input-titulo',
                type='text',
                value=title
            ),
            html.Label("Tamaño de letra título: "),
            dcc.Input(
                id='input-tamaño-titulo',
                type='number',
                value=16
            )
        ]),
        html.Div([
            html.Label('Pie de Página: '),
            dcc.Input(
                id='input-pie',
                type='text',
                value='Datos obtenidos de la página del SNIS'
            ),
            html.Label("Tamaño de letra pie: "),
            dcc.Input(
                id='input-tamaño-pie',
                type='number',
                value=11
            )
        ]),
        html.Div([
            html.Label('Eje X: '),
            dcc.Input(
                id='input-eje-x',
                type='text',
                value='Año'
            ),
            html.Label("Tamaño Eje X: "),
            dcc.Input(
                id='input-tamaño-eje-x',
                type='number',
                value=14
            )
        ]),
        html.Div([
            html.Label('Eje Y: '),
            dcc.Input(
                id='input-eje-y',
                type='text',
                value='Total'
            ),
            html.Label("Tamaño Eje Y: "),
            dcc.Input(
                id='input-tamaño-eje-y',
                type='number',
                value=14
            )
        ]),
        html.Div([
            html.Label('Tamaño de letra leyenda: '),
            dcc.Input(
                id='input-tamaño-leyenda',
                type='number',
                value=9,
                style={'width': '80px'}
            ),
            html.Label("Tamaño de letra de Números Gráficas: "),
            dcc.Input(
                id='input-tamaño-num-grafica',
                type='number',
                value=15,
                style={'width': '80px'}
            )
        ]),
        html.Div(id='output-data-salud'),
        html.Button('Generar Gráfico', id='btn-calcular-salud'),
        html.Div(id='output-data-salud')
    ])

def generate_calculo_layout(title):
    opciones_dataframes_departamento = [
        {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
    ]
    opciones_dataframes_provincia = [
        {'label': 'Cordillera', 'value': 'Cordillera'},
    ]
    opciones_dataframes_municipio = [
        {'label': 'Camiri', 'value': 'Camiri'},
        {'label': 'Gutierrez', 'value': 'Gutierrez'},
        {'label': 'Lagunillas', 'value': 'Lagunillas'}
    ]
    
    return html.Div([
        html.H1("Gráficos de Tendencia"),
        html.Div([
            html.Span('Factor'),
            dcc.Input(id='input-factor', type='number', value=10000, style={'width': '80px'})
        ]),
        html.Label('Gráfica a mostrar:'),
        dcc.Dropdown(
            id='dropdown-graphic-type',
            options=[
                {'label': 'Totales', 'value': 't'},
                {'label': 'Por sexo (Comparación Municipio, Provincia, Departamento)', 'value': 's1'},
                {'label': 'Por sexo (Comparación entre Mujeres y Hombres)', 'value': 's2'},
                {'label': 'Por edad (Separadas por año)', 'value': 'e1'},
                {'label': 'Por edad (Sumatoria por año)', 'value': 'e2'},
            ],
            value='t'
        ),
        html.Label('Porcentaje o Incidencias:'),
        dcc.Dropdown(
            id='dropdown-type-percent',
            options=[
                {'label': 'Incidencias', 'value': 'Incidencia'},
                {'label': 'Total', 'value': 'Total'},
                {'label': 'Porcentajes', 'value': 'Porcentaje'},
            ],
            value='Incidencia'
        ),
        html.H3("Departamento"),
        html.Div([
            html.Div([
                html.Label('Gráfico de Barras:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-dep',
                    options=opciones_dataframes_departamento,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-dep',
                    options=opciones_dataframes_departamento,
                    multi=True,
                    value=['Santa Cruz']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'margin-left': '5%'})
        ]),
        
        html.H3("Provincia"),
        html.Div([
            html.Div([
                html.Label('Gráfico de Barras:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-prov',
                    options=opciones_dataframes_provincia,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-prov',
                    options=opciones_dataframes_provincia,
                    multi=True,
                    value=['Cordillera']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'margin-left': '5%'})
        ]),
        html.H3("Municipios"),
        html.Div([
            html.Div([
                html.Label('Gráfico de Barras:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-mun',
                    options=opciones_dataframes_municipio,
                    multi=True,
                    value=['Camiri', 'Gutierrez', 'Lagunillas']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-mun',
                    options=opciones_dataframes_municipio,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'margin-left': '5%'})
        ]),
        html.Div([
            html.Label('Título del gráfico: '),
            dcc.Input(
                id='input-titulo',
                type='text',
                value=title
            ),
            html.Label("Tamaño de letra título: "),
            dcc.Input(
                id='input-tamaño-titulo',
                type='number',
                value=16
            )
        ]),
        html.Div([
            html.Label('Pie de Página: '),
            dcc.Input(
                id='input-pie',
                type='text',
                value='Datos obtenidos de la página del SNIS'
            ),
            html.Label("Tamaño de letra pie: "),
            dcc.Input(
                id='input-tamaño-pie',
                type='number',
                value=11
            )
        ]),
        html.Div([
            html.Label('Eje X: '),
            dcc.Input(
                id='input-eje-x',
                type='text',
                value='Año'
            ),
            html.Label("Tamaño Eje X: "),
            dcc.Input(
                id='input-tamaño-eje-x',
                type='number',
                value=14
            )
        ]),
        html.Div([
            html.Label('Eje Y: '),
            dcc.Input(
                id='input-eje-y',
                type='text',
                value='Total'
            ),
            html.Label("Tamaño Eje Y: "),
            dcc.Input(
                id='input-tamaño-eje-y',
                type='number',
                value=14
            )
        ]),
        html.Div([
            html.Label('Tamaño de letra leyenda: '),
            dcc.Input(
                id='input-tamaño-leyenda',
                type='number',
                value=9,
                style={'width': '80px'}
            ),
            html.Label("Tamaño de letra de Números Gráficas: "),
            dcc.Input(
                id='input-tamaño-num-grafica',
                type='number',
                value=15,
                style={'width': '80px'}
            )
        ]),
        html.Button('Generar Gráfico', id='btn-calcular'),
        html.Div(id='output-data')
    ])

def generate_calculo_layout_nutricion(title):
    opciones_dataframes_departamento = [
        {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
    ]
    opciones_dataframes_provincia = [
        {'label': 'Cordillera', 'value': 'Cordillera'},
    ]
    opciones_dataframes_municipio = [
        {'label': 'Camiri', 'value': 'Camiri'},
        {'label': 'Gutierrez', 'value': 'Gutierrez'},
        {'label': 'Lagunillas', 'value': 'Lagunillas'}
    ]
    
    return html.Div([
        html.H1("Gráficos de Tendencia"),
        html.Div([
            html.Span('Factor'),
            dcc.Input(id='input-factor', type='number', value=10000, style={'width': '80px'})
        ]),
        html.Label('Estado Nutricional:'),
        dcc.Dropdown(
            id='dropdown-type-nutrition',
            options=[
                {'label': 'Obesidad', 'value': 'o'},
                {'label': 'Sobrepeso', 'value': 's'},
                {'label': 'Desnutricion', 'value': 'd'},
            ],
            value='o'
        ),
        html.Label('Porcentaje o Incidencias:'),
        dcc.Dropdown(
            id='dropdown-type-percent',
            options=[
                {'label': 'Incidencias', 'value': 'Incidencia'},
                {'label': 'Embarazadas', 'value': 'Embarazadas'},
                {'label': 'Porcentajes', 'value': 'Porcentaje'},
            ],
            value='Incidencia'
        ),
        html.H3("Departamento"),
        html.Div([
            html.Div([
                html.Label('Gráfico de Barras:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-dep',
                    options=opciones_dataframes_departamento,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-dep',
                    options=opciones_dataframes_departamento,
                    multi=True,
                    value=['Santa Cruz']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'margin-left': '5%'})
        ]),
        
        html.H3("Provincia"),
        html.Div([
            html.Div([
                html.Label('Gráfico de Barras:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-prov',
                    options=opciones_dataframes_provincia,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-prov',
                    options=opciones_dataframes_provincia,
                    multi=True,
                    value=['Cordillera']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'margin-left': '5%'})
        ]),
        html.H3("Municipios"),
        html.Div([
            html.Div([
                html.Label('Gráfico de Barras:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-mun',
                    options=opciones_dataframes_municipio,
                    multi=True,
                    value=['Camiri', 'Gutierrez', 'Lagunillas']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-mun',
                    options=opciones_dataframes_municipio,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'margin-left': '5%'})
        ]),
        html.Div([
            html.Label('Título del gráfico: '),
            dcc.Input(
                id='input-titulo',
                type='text',
                value=title
            ),
            html.Label("Tamaño de letra título: "),
            dcc.Input(
                id='input-tamaño-titulo',
                type='number',
                value=16
            )
        ]),
        html.Div([
            html.Label('Pie de Página: '),
            dcc.Input(
                id='input-pie',
                type='text',
                value='Datos obtenidos de la página del SNIS'
            ),
            html.Label("Tamaño de letra pie: "),
            dcc.Input(
                id='input-tamaño-pie',
                type='number',
                value=11
            )
        ]),
        html.Div([
            html.Label('Eje X: '),
            dcc.Input(
                id='input-eje-x',
                type='text',
                value='Año'
            ),
            html.Label("Tamaño Eje X: "),
            dcc.Input(
                id='input-tamaño-eje-x',
                type='number',
                value=14
            )
        ]),
        html.Div([
            html.Label('Eje Y: '),
            dcc.Input(
                id='input-eje-y',
                type='text',
                value='Total'
            ),
            html.Label("Tamaño Eje Y: "),
            dcc.Input(
                id='input-tamaño-eje-y',
                type='number',
                value=14
            )
        ]),
        html.Div([
            html.Label('Tamaño de letra leyenda: '),
            dcc.Input(
                id='input-tamaño-leyenda',
                type='number',
                value=9,
                style={'width': '80px'}
            ),
            html.Label("Tamaño de letra de Números Gráficas: "),
            dcc.Input(
                id='input-tamaño-num-grafica',
                type='number',
                value=15,
                style={'width': '80px'}
            )
        ]),
        html.Button('Generar Gráfico', id='btn-calcular-nutricion'),
        html.Div(id='output-data-nutricion')
    ])

def generate_calculo_layout_embarazo(title):
    opciones_dataframes_departamento = [
        {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
    ]
    opciones_dataframes_provincia = [
        {'label': 'Cordillera', 'value': 'Cordillera'},
    ]
    opciones_dataframes_municipio = [
        {'label': 'Camiri', 'value': 'Camiri'},
        {'label': 'Gutierrez', 'value': 'Gutierrez'},
        {'label': 'Lagunillas', 'value': 'Lagunillas'}
    ]
    return html.Div([
        html.H1("Gráficos de Tendencia"),
        html.Div([
            html.Span('Factor'),
            dcc.Input(id='input-factor', type='number', value=10000, style={'width': '80px'})
        ]),
        html.Label('Rango de edad:'),
        dcc.Dropdown(
            id='dropdown-type-age',
            options=[
                {'label': '< 15', 'value': 'r1'},
                {'label': '15 - 19', 'value': 'r2'},
                {'label': '< 19', 'value': 'r3'},
            ],
            value='r3'  # Valor inicial seleccionado
        ),
        html.Label('Meses de Embarazo:'),
        dcc.Dropdown(
            id='dropdown-type-mounth',
            options=[
                {'label': '< 5to mes + > 5to mes', 'value': 'm1'},
                {'label': '< 5to mes', 'value': 'm2'},
                {'label': '> 5to mes', 'value': 'm3'},
            ],
            value='m1'  # Valor inicial seleccionado
        ),
        html.Label('Porcentaje o Incidencias:'),
        dcc.Dropdown(
            id='dropdown-type-percent',
            options=[
                {'label': 'Incidencias', 'value': 'Incidencia'},
                {'label': 'Total', 'value': 'Total'},
                {'label': 'Porcentajes', 'value': 'Porcentaje'},
            ],
            value='Incidencia'  # Valor inicial seleccionado
        ),
        html.H3("Departamento"),
        html.Div([
            html.Div([
                html.Label('Gráfico de Barras:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-dep',
                    options=opciones_dataframes_departamento,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-dep',
                    options=opciones_dataframes_departamento,
                    multi=True,
                    value=['Santa Cruz']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'margin-left': '5%'})
        ]),
        
        html.H3("Provincia"),
        html.Div([
            html.Div([
                html.Label('Gráfico de Barras:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-prov',
                    options=opciones_dataframes_provincia,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-prov',
                    options=opciones_dataframes_provincia,
                    multi=True,
                    value=['Cordillera']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'margin-left': '5%'})
        ]),
        html.H3("Municipios"),
        html.Div([
            html.Div([
                html.Label('Gráfico de Barras:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-mun',
                    options=opciones_dataframes_municipio,
                    multi=True,
                    value=['Camiri', 'Gutierrez', 'Lagunillas']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-mun',
                    options=opciones_dataframes_municipio,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'margin-left': '5%'})
        ]),
        html.Div([
            html.Label('Título del gráfico: '),
            dcc.Input(
                id='input-titulo',
                type='text',
                value=title
            ),
            html.Label("Tamaño de letra título: "),
            dcc.Input(
                id='input-tamaño-titulo',
                type='number',
                value=16
            )
        ]),
        html.Div([
            html.Label('Pie de Página: '),
            dcc.Input(
                id='input-pie',
                type='text',
                value='Datos obtenidos de la página del SNIS'
            ),
            html.Label("Tamaño de letra pie: "),
            dcc.Input(
                id='input-tamaño-pie',
                type='number',
                value=11
            )
        ]),
        html.Div([
            html.Label('Eje X: '),
            dcc.Input(
                id='input-eje-x',
                type='text',
                value='Año'
            ),
            html.Label("Tamaño Eje X: "),
            dcc.Input(
                id='input-tamaño-eje-x',
                type='number',
                value=14
            )
        ]),
        html.Div([
            html.Label('Eje Y: '),
            dcc.Input(
                id='input-eje-y',
                type='text',
                value='Total'
            ),
            html.Label("Tamaño Eje Y: "),
            dcc.Input(
                id='input-tamaño-eje-y',
                type='number',
                value=14
            )
        ]),
        html.Div([
            html.Label('Tamaño de letra leyenda: '),
            dcc.Input(
                id='input-tamaño-leyenda',
                type='number',
                value=9,
                style={'width': '80px'}
            ),
            html.Label("Tamaño de letra de Números Gráficas: "),
            dcc.Input(
                id='input-tamaño-num-grafica',
                type='number',
                value=15,
                style={'width': '80px'}
            )
        ]),
        html.Button('Generar Gráfico', id='btn-calcular-embarazo'),
        html.Div(id='output-data-embarazo')
    ])
def generate_calculo_layout_consultas(title):
    opciones_dataframes_departamento = [
        {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
    ]
    opciones_dataframes_provincia = [
        {'label': 'Cordillera', 'value': 'Cordillera'},
    ]
    opciones_dataframes_municipio = [
        {'label': 'Camiri', 'value': 'Camiri'},
        {'label': 'Gutierrez', 'value': 'Gutierrez'},
        {'label': 'Lagunillas', 'value': 'Lagunillas'}
    ]
    
    return html.Div([
        html.H1("Gráficos de Tendencia"),
        html.Div([
            html.Span('Factor'),
            dcc.Input(id='input-factor', type='number', value=10000, style={'width': '80px'})
        ]),
        html.Label('Gráfica a mostrar:'),
        dcc.Dropdown(
            id='dropdown-graphic-type',
            options=[
                {'label': 'Totales', 'value': 't'},
                {'label': 'Por edad (Separadas por año)', 'value': 'e1'},
                {'label': 'Por edad (Sumatoria por año)', 'value': 'e2'},
            ],
            value='t'
        ),
        
        html.H3("Departamento"),
        html.Div([
            html.Div([
                html.Label('Gráfico de Barras:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-dep',
                    options=opciones_dataframes_departamento,
                    multi=True,
                    value=['Santa Cruz']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-dep',
                    options=opciones_dataframes_departamento,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'visibility': 'hidden'})
        ]),
        
        html.H3("Provincia"),
        html.Div([
            html.Div([
                html.Label('Gráfico de Barras:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-prov',
                    options=opciones_dataframes_provincia,
                    multi=True,
                    value=['Cordillera']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-prov',
                    options=opciones_dataframes_provincia,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'visibility': 'hidden'})
        ]),
        html.H3("Municipios"),
        html.Div([
            html.Div([
                html.Label('Gráfico de Barras:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-mun',
                    options=opciones_dataframes_municipio,
                    multi=True,
                    value=['Camiri', 'Gutierrez', 'Lagunillas']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-mun',
                    options=opciones_dataframes_municipio,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'visibility': 'hidden'})
        ]),
        html.Div([
            html.Label('Título del gráfico: '),
            dcc.Input(
                id='input-titulo',
                type='text',
                value=title
            ),
            html.Label("Tamaño de letra título: "),
            dcc.Input(
                id='input-tamaño-titulo',
                type='number',
                value=16
            )
        ]),
        html.Div([
            html.Label('Pie de Página: '),
            dcc.Input(
                id='input-pie',
                type='text',
                value='Datos obtenidos de la página del SNIS'
            ),
            html.Label("Tamaño de letra pie: "),
            dcc.Input(
                id='input-tamaño-pie',
                type='number',
                value=11
            )
        ]),
        html.Div([
            html.Label('Eje X: '),
            dcc.Input(
                id='input-eje-x',
                type='text',
                value='Año'
            ),
            html.Label("Tamaño Eje X: "),
            dcc.Input(
                id='input-tamaño-eje-x',
                type='number',
                value=14
            )
        ]),
        html.Div([
            html.Label('Eje Y: '),
            dcc.Input(
                id='input-eje-y',
                type='text',
                value='Total'
            ),
            html.Label("Tamaño Eje Y: "),
            dcc.Input(
                id='input-tamaño-eje-y',
                type='number',
                value=14
            )
        ]),
        html.Div([
            html.Label('Tamaño de letra leyenda: '),
            dcc.Input(
                id='input-tamaño-leyenda',
                type='number',
                value=9,
                style={'width': '80px'}
            ),
            html.Label("Tamaño de letra de Números Gráficas: "),
            dcc.Input(
                id='input-tamaño-num-grafica',
                type='number',
                value=15,
                style={'width': '80px'}
            )
        ]),
        html.Button('Generar Gráfico', id='btn-calcular-consultas'),
        html.Div(id='output-data-consultas')
    ])

# Callback para actualizar el contenido según la URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/cancer':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Cancer'),
            generate_calculo_layout('Casos nuevos de Cáncer por Año en Departamentos, Provincias y Municipios')
        ])
    if pathname == '/diabetes':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Diabetes'),
            generate_calculo_layout('Casos nuevos de Diabetes por año a nivel Departamental, Provincial y Municipal')
        ])
    if pathname == '/hipertension':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Hipertensión Arterial'),
            generate_calculo_layout('Casos nuevos de Hipertension por año a nivel Departamental, Provincial y Municipal')
        ])
    if pathname == '/obesidad':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Obesidad'),
            generate_calculo_layout('Casos nuevos de Obesidad por año a nivel Departamental, Provincial y Municipal')
        ])
    if pathname == '/neumonia':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Neumonía'),
            generate_calculo_layout('Casos nuevos de Neumonia por año a nivel Departamental, Provincial y Municipal')
        ])
    if pathname == '/chagas':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Chagas'),
            generate_calculo_layout('Casos nuevos de Chagas por año a nivel Departamental, Provincial y Municipal')
        ])
    if pathname == '/vih':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos VIH'),
            generate_calculo_layout('Casos nuevos de VIH por año a nivel Departamental, Provincial y Municipal')
        ])
    elif pathname == '/nutricion':        
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Nutricion'),
            generate_calculo_layout_nutricion('Casos nuevos de Estado Nutricional en Embarazadas por año a nivel Departamental, Provincial y Municipal')
        ])
    elif pathname == '/embarazo':
        #df_c_embarazo, df_g_embarazo, d1, d2 = get_casos_embarazo()
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Embarazo Adolescente'),
            generate_calculo_layout_embarazo('Casos de Embarazadas Adolescentes por año a nivel Departamental, Provincial y Municipal')
        ])
    elif pathname == '/consultas':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Consultas Externas'),
            generate_calculo_layout_consultas('Total de consultas por sexo y año a nivel Departamental, Provincial y Municipal')
        ])
    elif pathname == '/salud':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Salud'),
            generate_calculo_layout_salud('Datos de salud')
        ])
    else:
        return html.Div([
            html.H1('Mi primera aplicación Dash en Heroku'),
            html.P('Hola mundo' + pathname),
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': [1, 2, 3], 'y': [8, 5, 2], 'type': 'bar', 'name': 'SF'},
                        {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'Montreal'},
                    ],
                    'layout': {
                        'title': 'ACTUALIZACION PAGINA'
                    }
                }
            )
        ])

@app.callback(
    [Output('dropdown-dataframes-bar-dep', 'value'),
     Output('dropdown-dataframes-ten-dep', 'value')],
    [Input('dropdown-dataframes-bar-dep', 'value'),
     Input('dropdown-dataframes-ten-dep', 'value')]
)
def sync_departamento_dropdowns(bar_value, ten_value):
    # Obtener el ID del componente que activó la callback
    ctx = callback_context
    if not ctx.triggered:
        return bar_value, ten_value  # Sin cambios si no hay activación

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'dropdown-dataframes-bar-dep':
        ten_value = [item for item in ten_value if item not in bar_value]
    elif triggered_id == 'dropdown-dataframes-ten-dep':
        bar_value = [item for item in bar_value if item not in ten_value]
    return bar_value, ten_value

# Callback para realizar el cálculo de incidencias y porcentajes
@app.callback(
    Output('output-data', 'children'),
    [
        Input('btn-calcular', 'n_clicks'),
        Input('dropdown-graphic-type', 'value'),
        Input('dropdown-type-percent', 'value'),
        Input('dropdown-dataframes-bar-dep', 'value'),
        Input('dropdown-dataframes-ten-dep', 'value'),
        Input('dropdown-dataframes-bar-prov', 'value'),
        Input('dropdown-dataframes-ten-prov', 'value'),
        Input('dropdown-dataframes-bar-mun', 'value'),
        Input('dropdown-dataframes-ten-mun', 'value'),
        Input('input-titulo', 'value'),
        Input('input-tamaño-titulo', 'value'),
        Input('input-eje-x', 'value'),
        Input('input-tamaño-eje-x', 'value'),
        Input('input-eje-y', 'value'),
        Input('input-tamaño-eje-y', 'value'),
        Input('input-pie', 'value'),
        Input('input-tamaño-pie', 'value'),
        Input('input-tamaño-leyenda', 'value'),
        Input('input-tamaño-num-grafica', 'value'),
    ],
    [State('input-factor', 'value'),
     State('url', 'pathname')]  # Capturar el pathname actual
)

def update_output(n_clicks, graphic_type, type_percent, 
                  dataframes_bar_dep, dataframes_ten_dep, 
                  dataframes_bar_prov, dataframes_ten_prov,
                  dataframes_bar_mun, dataframes_ten_mun,
                  titulo, tamanio_titulo,
                  eje_x, tamanio_eje_x, 
                  eje_y, tamanio_eje_y, 
                  pie, tamanio_pie, 
                  tamanio_leyenda, tamanio_num_grafica, 
                  factor, pathname):
    if n_clicks:
        try: 
            tamanio_titulo = int(tamanio_titulo) if tamanio_titulo is not None else 12
            tamanio_eje_x = int(tamanio_eje_x) if tamanio_eje_x is not None else 10
            tamanio_eje_y = int(tamanio_eje_y) if tamanio_eje_y is not None else 10
            tamanio_pie = int(tamanio_pie) if tamanio_pie is not None else 10
            tamanio_leyenda = int(tamanio_leyenda) if tamanio_leyenda is not None else 8
            tamanio_num_grafica = int(tamanio_num_grafica) if tamanio_num_grafica is not None else 10

            dfl_barras = dataframes_bar_dep + dataframes_bar_prov + dataframes_bar_mun
            dfl_tendencias = dataframes_ten_dep + dataframes_ten_prov + dataframes_ten_mun
            dfl_total = dataframes_bar_dep + dataframes_ten_dep + dataframes_bar_prov + dataframes_ten_prov + dataframes_bar_mun + dataframes_ten_mun
            dfl_barras = [item for item in dfl_barras if item]
            dfl_tendencias = [item for item in dfl_tendencias if item]
            dfl_total = [item for item in dfl_total if item]

            resultados = []
            partes = pathname.split('/')
            resultados.append(html.H2(f'Gráfico '+partes[1]))
            if 'Santa Cruz' in dfl_barras or 'Santa Cruz' in dfl_tendencias:
                df = get_casos(partes[1], 'guarani')
                df_names = ['df_c', 'df_g', 'df_l', 'df_pc', 'df_sc']
                df_c = df[0]
                df_g = df[1]
                df_l = df[2]
                df_pc = df[3]
                df_sc = df[4]

                p_c, p_g, p_l, p_pc, p_sc = get_casos('poblacion', 'guarani')
                m = p_c[p_c['Sexo'] == 'Mujer']['Total'].tolist()
                h = p_c[p_c['Sexo'] == 'Hombre']['Total'].tolist()
                p = p_c.groupby('Año')['Total'].sum().tolist()
                m_2 = p_g[p_g['Sexo'] == 'Mujer']['Total'].tolist()
                h_2 = p_g[p_g['Sexo'] == 'Hombre']['Total'].tolist()
                p_2 = p_g.groupby('Año')['Total'].sum().tolist()
                m_3 = p_l[p_l['Sexo'] == 'Mujer']['Total'].tolist()
                h_3 = p_l[p_l['Sexo'] == 'Hombre']['Total'].tolist()
                p_3 = p_l.groupby('Año')['Total'].sum().tolist()
                m_4 = p_pc[p_pc['Sexo'] == 'Mujer']['Total'].tolist()
                h_4 = p_pc[p_pc['Sexo'] == 'Hombre']['Total'].tolist()
                p_4 = p_pc.groupby('Año')['Total'].sum().tolist()
                m_5 = p_sc[p_sc['Sexo'] == 'Mujer']['Total'].tolist()
                h_5 = p_sc[p_sc['Sexo'] == 'Hombre']['Total'].tolist()
                p_5 = p_sc.groupby('Año')['Total'].sum().tolist()

                df_c_t = generate_total(df_c)
                df_g_t = generate_total(df_g)
                df_l_t = generate_total(df_l)
                df_pc_t = generate_total(df_pc)
                df_sc_t = generate_total(df_sc)
                
                df_c_t = calculate_total(df_c_t, factor, p, 'Total')
                df_g_t = calculate_total(df_g_t, factor, p_2, 'Total')
                df_l_t = calculate_total(df_l_t, factor, p_3, 'Total')
                df_pc_t = calculate_total(df_pc_t, factor, p_4, 'Total')
                df_sc_t = calculate_total(df_sc_t, factor, p_5, 'Total')

                if partes[1] != 'vih':
                    df_c_y = calculate_age(df_c, p_c, partes[1])
                    df_g_y = calculate_age(df_g, p_g, partes[1])
                    df_l_y = calculate_age(df_l, p_l, partes[1])
                    df_pc_y = calculate_age(df_pc, p_pc, partes[1])
                    df_sc_y = calculate_age(df_sc, p_sc, partes[1])

                    df_c_y_t = calculate_age_total(df_c_y, partes[1])
                    df_g_y_t = calculate_age_total(df_g_y, partes[1])
                    df_l_y_t = calculate_age_total(df_l_y, partes[1])
                    df_pc_y_t = calculate_age_total(df_pc_y, partes[1])
                    df_sc_y_t = calculate_age_total(df_sc_y, partes[1])

                df_c = calculate_gender(df_c, factor, m, h)
                df_g = calculate_gender(df_g, factor, m_2, h_2)
                df_l = calculate_gender(df_l, factor, m_3, h_3)
                df_pc = calculate_gender(df_pc, factor, m_4, h_4)
                df_sc = calculate_gender(df_sc, factor, m_5, h_5)

                dataframes = {
                    'Santa Cruz': df_sc,
                    'Cordillera': df_pc,
                    'Camiri': df_c,
                    'Gutierrez': df_g,
                    'Lagunillas': df_l
                }
                dataframes_total = {
                    'Santa Cruz': df_sc_t,
                    'Cordillera': df_pc_t,
                    'Camiri': df_c_t,
                    'Gutierrez': df_g_t,
                    'Lagunillas': df_l_t
                }
                if partes[1] != 'vih':
                    dataframes_age = {
                        'Santa Cruz': df_sc_y,
                        'Cordillera': df_pc_y,
                        'Camiri': df_c_y,
                        'Gutierrez': df_g_y,
                        'Lagunillas': df_l_y
                    }
                    dataframes_age_total = {
                        'Santa Cruz': df_sc_y_t,
                        'Cordillera': df_pc_y_t,
                        'Camiri': df_c_y_t,
                        'Gutierrez': df_g_y_t,
                        'Lagunillas': df_l_y_t
                    }
            
            df_barras_total = [dataframes_total[nombre] for nombre in dfl_barras if nombre in dataframes_total]
            df_tendencias_total = [dataframes_total[nombre] for nombre in dfl_tendencias if nombre in dataframes_total]

            df_barras = [dataframes[nombre] for nombre in dfl_barras if nombre in dataframes]
            df_tendencias = [dataframes[nombre] for nombre in dfl_tendencias if nombre in dataframes]

            df_total = [dataframes[nombre] for nombre in dfl_total if nombre in dataframes]
            if partes[1] != 'vih':
                df_total_year = [dataframes_age[nombre] for nombre in dfl_total if nombre in dataframes_age]
                df_total_year_t = [dataframes_age_total[nombre] for nombre in dfl_total if nombre in dataframes_age_total]

            if partes[1] == 'obesidad' or partes[1] == 'diabetes' or partes[1] == 'hipertension':
                df_r_c = df[5]
                df_r_g = df[6]
                df_r_l = df[7]
                df_r_pc = df[8]
                df_r_sc = df[9]

                df_r_c_t = generate_total(df_r_c)
                df_r_g_t = generate_total(df_r_g)
                df_r_l_t = generate_total(df_r_l)
                df_r_pc_t = generate_total(df_r_pc)
                df_r_sc_t = generate_total(df_r_sc)
                
                df_r_c_t = calculate_total(df_r_c_t, factor, p, 'Total')
                df_r_g_t = calculate_total(df_r_g_t, factor, p_2, 'Total')
                df_r_l_t = calculate_total(df_r_l_t, factor, p_3, 'Total')
                df_r_pc_t = calculate_total(df_r_pc_t, factor, p_4, 'Total')
                df_r_sc_t = calculate_total(df_r_sc_t, factor, p_5, 'Total')

                df_r_c_y = calculate_age(df_r_c, p_c, partes[1])
                df_r_g_y = calculate_age(df_r_g, p_g, partes[1])
                df_r_l_y = calculate_age(df_r_l, p_l, partes[1])
                df_r_pc_y = calculate_age(df_r_pc, p_pc, partes[1])
                df_r_sc_y = calculate_age(df_r_sc, p_sc, partes[1])

                df_r_c_y_t = calculate_age_total(df_r_c_y, partes[1])
                df_r_g_y_t = calculate_age_total(df_r_g_y, partes[1])
                df_r_l_y_t = calculate_age_total(df_r_l_y, partes[1])
                df_r_pc_y_t = calculate_age_total(df_r_pc_y, partes[1])
                df_r_sc_y_t = calculate_age_total(df_r_sc_y, partes[1])

                df_r_c = calculate_gender(df_r_c, factor, m, h)
                df_r_g = calculate_gender(df_r_g, factor, m_2, h_2)
                df_r_l = calculate_gender(df_r_l, factor, m_3, h_3)
                df_r_pc = calculate_gender(df_r_pc, factor, m_4, h_4)
                df_r_sc = calculate_gender(df_r_sc, factor, m_5, h_5)
                
                dataframes_r = {
                    'Santa Cruz': df_r_sc,
                    'Cordillera': df_r_pc,
                    'Camiri': df_r_c,
                    'Gutierrez': df_r_g,
                    'Lagunillas': df_r_l
                }
                dataframes_r_total = {
                    'Santa Cruz': df_r_sc_t,
                    'Cordillera': df_r_pc_t,
                    'Camiri': df_r_c_t,
                    'Gutierrez': df_r_g_t,
                    'Lagunillas': df_r_l_t
                }
                dataframes_r_age = {
                    'Santa Cruz': df_r_sc_y,
                    'Cordillera': df_r_pc_y,
                    'Camiri': df_r_c_y,
                    'Gutierrez': df_r_g_y,
                    'Lagunillas': df_r_l_y
                }
                dataframes_r_age_total = {
                    'Santa Cruz': df_r_sc_y_t,
                    'Cordillera': df_r_pc_y_t,
                    'Camiri': df_r_c_y_t,
                    'Gutierrez': df_r_g_y_t,
                    'Lagunillas': df_r_l_y_t
                }
                df_r_barras_total = [dataframes_r_total[nombre] for nombre in dfl_barras if nombre in dataframes_r_total]
                df_r_tendencias_total = [dataframes_r_total[nombre] for nombre in dfl_tendencias if nombre in dataframes_r_total]

                df_r_barras = [dataframes_r[nombre] for nombre in dfl_barras if nombre in dataframes_r]
                df_r_tendencias = [dataframes_r[nombre] for nombre in dfl_tendencias if nombre in dataframes_r]

                df_r_total = [dataframes_r[nombre] for nombre in dfl_total if nombre in dataframes_r]

                df_r_total_year = [dataframes_r_age[nombre] for nombre in dfl_total if nombre in dataframes_r_age]
                df_r_total_year_t = [dataframes_r_age_total[nombre] for nombre in dfl_total if nombre in dataframes_r_age_total]

            if n_clicks > 0:
                if graphic_type == 't':
                    fig=generate_graph_total(df_barras_total, df_tendencias_total, dfl_barras, dfl_tendencias,
                                                titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    resultados.append(dcc.Graph(id='mi-grafico-total', figure=fig))
                    if partes[1] == 'obesidad' or partes[1] == 'diabetes' or partes[1] == 'hipertension':
                        fig=generate_graph_total(df_r_barras_total, df_r_tendencias_total, dfl_barras, dfl_tendencias,
                                                    "Casos repetidos diagnosticados de "+ partes[1] +" por año a nivel Departamental, Provincial y Municipal", 
                                                    tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                    pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                        resultados.append(dcc.Graph(id='mi-grafico-rep-total', figure=fig))
                elif graphic_type == 's1':
                    fig=generate_graph_join_gender(df_barras, df_tendencias, dfl_barras, dfl_tendencias,
                                                titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    resultados.append(dcc.Graph(id='mi-grafico-sexo-1', figure=fig))
                    if partes[1] == 'obesidad' or partes[1] == 'diabetes' or partes[1] == 'hipertension':
                        fig=generate_graph_join_gender(df_r_barras, df_r_tendencias, dfl_barras, dfl_tendencias,
                                                    "Casos repetidos diagnosticados de "+ partes[1] +" por año y sexo a nivel Departamental, Provincial y Municipal",
                                                    tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                    pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                        resultados.append(dcc.Graph(id='mi-grafico-rep-sexo-1', figure=fig))
                elif graphic_type == 's2':
                    if len(df_tendencias) >= len(df_barras):
                        fig = generate_graph_separate_gender(df_total, "tendencias", dfl_total, 
                                    titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                        if partes[1] == 'obesidad' or partes[1] == 'diabetes' or partes[1] == 'hipertension':
                            fig2 = generate_graph_separate_gender(df_r_total, "tendencias", dfl_total,
                                            "Casos repetidos diagnosticados de "+ partes[1] +" por año y sexo a nivel Departamental, Provincial y Municipal",
                                            tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                    pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    else:
                        fig = generate_graph_separate_gender(df_total, "barras", dfl_total, 
                                    titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                        if partes[1] == 'obesidad' or partes[1] == 'diabetes' or partes[1] == 'hipertension':
                            fig2 = generate_graph_separate_gender(df_r_total, "barras", dfl_total, 
                                            "Casos repetidos diagnosticados de "+ partes[1] +" por año y sexo a nivel Departamental, Provincial y Municipal", 
                                            tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                    pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    resultados.append(dcc.Graph(id='mi-grafico-sexo-2', figure=fig))
                    if partes[1] == 'obesidad' or partes[1] == 'diabetes' or partes[1] == 'hipertension':
                        resultados.append(dcc.Graph(id='mi-grafico-rep-sexo-2', figure=fig2))
                elif graphic_type == 'e1':
                    for i, df in enumerate(df_total_year):
                        fig = generate_graph_separate_age(df, partes[1], dfl_total[i], 
                                                        titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                        pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                        resultados.append(dcc.Graph(id=f'mi-grafico-edad-{i}', figure=fig))
                    
                    if partes[1] == 'obesidad' or partes[1] == 'diabetes' or partes[1] == 'hipertension':
                        for i, df in enumerate(df_r_total_year):
                            fig = generate_graph_separate_age(df, partes[1], dfl_total[i], 
                                                        "Casos repetidos diagnosticados de "+ partes[1] +" por grupo etario, sexo y año a nivel Departamental, Provincial y Municipal", 
                                                        tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                        pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                            resultados.append(dcc.Graph(id=f'mi-rep-grafico-edad-{i}', figure=fig))
                elif graphic_type == 'e2':
                    for i, df in enumerate(df_total_year_t):
                        fig = generate_graph_join_age(df, partes[1], dfl_total[i], 
                                                        titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                        pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                        resultados.append(dcc.Graph(id=f'mi-grafico-edad-total-{i}', figure=fig))
                    if partes[1] == 'obesidad' or partes[1] == 'diabetes' or partes[1] == 'hipertension':
                        for i, df in enumerate(df_r_total_year_t):
                            fig = generate_graph_join_age(df, partes[1], dfl_total[i], 
                                                            "Casos repetidos diagnosticados de "+ partes[1] +" por grupo etario y sexo a nivel Departamental, Provincial y Municipal",  
                                                            tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                            pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                            resultados.append(dcc.Graph(id=f'mi-rep-grafico-edad-total-{i}', figure=fig))
                else:
                    return html.Div("")

                return resultados
            
            return html.Div("")

        except Exception as e:
            return html.Div(f'Error: {e}')

# Callback para realizar el cálculo de incidencias y porcentajes
@app.callback(
    Output('output-data-nutricion', 'children'),
    [
        Input('btn-calcular-nutricion', 'n_clicks'),
        Input('dropdown-type-nutrition', 'value'),
        Input('dropdown-type-percent', 'value'),
        Input('dropdown-dataframes-bar-dep', 'value'),
        Input('dropdown-dataframes-ten-dep', 'value'),
        Input('dropdown-dataframes-bar-prov', 'value'),
        Input('dropdown-dataframes-ten-prov', 'value'),
        Input('dropdown-dataframes-bar-mun', 'value'),
        Input('dropdown-dataframes-ten-mun', 'value'),
        Input('input-titulo', 'value'),
        Input('input-tamaño-titulo', 'value'),
        Input('input-eje-x', 'value'),
        Input('input-tamaño-eje-x', 'value'),
        Input('input-eje-y', 'value'),
        Input('input-tamaño-eje-y', 'value'),
        Input('input-pie', 'value'),
        Input('input-tamaño-pie', 'value'),
        Input('input-tamaño-leyenda', 'value'),
        Input('input-tamaño-num-grafica', 'value'),
    ],
    [State('input-factor', 'value'),
     State('url', 'pathname')]  # Capturar el pathname actual
)
def update_output_nutricion(n_clicks, type_nutrition, type_percent,
                  dataframes_bar_dep, dataframes_ten_dep, 
                  dataframes_bar_prov, dataframes_ten_prov,
                  dataframes_bar_mun, dataframes_ten_mun, 
                  titulo, tamanio_titulo, 
                  eje_x, tamanio_eje_x, 
                  eje_y, tamanio_eje_y, 
                  pie, tamanio_pie, 
                  tamanio_leyenda, tamanio_num_grafica, 
                  factor, pathname):
    if n_clicks:
        try: 
            tamanio_titulo = int(tamanio_titulo) if tamanio_titulo is not None else 12
            tamanio_eje_x = int(tamanio_eje_x) if tamanio_eje_x is not None else 10
            tamanio_eje_y = int(tamanio_eje_y) if tamanio_eje_y is not None else 10
            tamanio_pie = int(tamanio_pie) if tamanio_pie is not None else 10
            tamanio_leyenda = int(tamanio_leyenda) if tamanio_leyenda is not None else 8
            tamanio_num_grafica = int(tamanio_num_grafica) if tamanio_num_grafica is not None else 10

            dfl_barras = dataframes_bar_dep + dataframes_bar_prov + dataframes_bar_mun
            dfl_tendencias = dataframes_ten_dep + dataframes_ten_prov + dataframes_ten_mun
            dfl_total = dataframes_bar_dep + dataframes_ten_dep + dataframes_bar_prov + dataframes_ten_prov + dataframes_bar_mun + dataframes_ten_mun

            dfl_barras = [item for item in dfl_barras if item]
            dfl_tendencias = [item for item in dfl_tendencias if item]
            dfl_total = [item for item in dfl_total if item]

            resultados = []
            partes = pathname.split('/')
            resultados.append(html.H2(f'Gráfico '+partes[1]))
            if 'Santa Cruz' in dfl_barras or 'Santa Cruz' in dfl_tendencias:
                df = get_casos(partes[1], 'guarani')
                df_c_o = df[0]
                df_g_o = df[1]
                df_l_o = df[2]
                df_pc_o = df[3]
                df_sc_o = df[4]
                df_c_s = df[5]
                df_g_s = df[6]
                df_l_s = df[7]
                df_pc_s = df[8]
                df_sc_s = df[9]
                df_c_d = df[10]
                df_g_d = df[11]
                df_l_d = df[12]
                df_pc_d = df[13]
                df_sc_d = df[14]

                # Convertir a listas de poblaciones
                p = get_casos('poblacion-especial', 'guarani')
                p_c = p[0] 
                p_g = p[1]
                p_l = p[2]
                p_pc = p[3] 
                p_sc = p[4]
                p = p_c.groupby('Año')['Embarazos'].sum().tolist()
                p_2 = p_g.groupby('Año')['Embarazos'].sum().tolist()
                p_3 = p_l.groupby('Año')['Embarazos'].sum().tolist()
                p_4 = p_pc.groupby('Año')['Embarazos'].sum().tolist()
                p_5 = p_sc.groupby('Año')['Embarazos'].sum().tolist()

                if type_nutrition == 'o':
                    df_c = calculate_total(df_c_o, factor, p, 'Embarazadas')
                    df_g = calculate_total(df_g_o, factor, p_2, 'Embarazadas')
                    df_l = calculate_total(df_l_o, factor, p_3, 'Embarazadas')
                    df_pc = calculate_total(df_pc_o, factor, p_4, 'Embarazadas')
                    df_sc = calculate_total(df_sc_o, factor, p_5, 'Embarazadas')
                    df_r_c = calculate_total(df_c_o, factor, p, 'Repetidos')
                    df_r_g = calculate_total(df_g_o, factor, p_2, 'Repetidos')
                    df_r_l = calculate_total(df_l_o, factor, p_3, 'Repetidos')
                    df_r_pc = calculate_total(df_pc_o, factor, p_4, 'Repetidos')
                    df_r_sc = calculate_total(df_sc_o, factor, p_5, 'Repetidos')
                elif type_nutrition == 's':
                    df_c = calculate_total(df_c_s, factor, p, 'Embarazadas')
                    df_g = calculate_total(df_g_s, factor, p_2, 'Embarazadas')
                    df_l = calculate_total(df_l_s, factor, p_3, 'Embarazadas')
                    df_pc = calculate_total(df_pc_s, factor, p_4, 'Embarazadas')
                    df_sc = calculate_total(df_sc_s, factor, p_5, 'Embarazadas')
                    df_r_c = calculate_total(df_c_s, factor, p, 'Repetidos')
                    df_r_g = calculate_total(df_g_s, factor, p_2, 'Repetidos')
                    df_r_l = calculate_total(df_l_s, factor, p_3, 'Repetidos')
                    df_r_pc = calculate_total(df_pc_s, factor, p_4, 'Repetidos')
                    df_r_sc = calculate_total(df_sc_s, factor, p_5, 'Repetidos')
                elif type_nutrition == 'd':
                    df_c = calculate_total(df_c_d, factor, p, 'Embarazadas')
                    df_g = calculate_total(df_g_d, factor, p_2, 'Embarazadas')
                    df_l = calculate_total(df_l_d, factor, p_3, 'Embarazadas')
                    df_pc = calculate_total(df_pc_d, factor, p_4, 'Embarazadas')
                    df_sc = calculate_total(df_sc_d, factor, p_5, 'Embarazadas')
                    df_r_c = calculate_total(df_c_d, factor, p, 'Repetidos')
                    df_r_g = calculate_total(df_g_d, factor, p_2, 'Repetidos')
                    df_r_l = calculate_total(df_l_d, factor, p_3, 'Repetidos')
                    df_r_pc = calculate_total(df_pc_d, factor, p_4, 'Repetidos')
                    df_r_sc = calculate_total(df_sc_d, factor, p_5, 'Repetidos')

                
                # Seleccionar los dataframes según la selección del usuario
                df_c.sort_values(by='Año', inplace=True)
                df_g.sort_values(by='Año', inplace=True)
                df_l.sort_values(by='Año', inplace=True)
                df_pc.sort_values(by='Año', inplace=True)
                df_sc.sort_values(by='Año', inplace=True)
                df_r_c.sort_values(by='Año', inplace=True)
                df_r_g.sort_values(by='Año', inplace=True)
                df_r_l.sort_values(by='Año', inplace=True)
                df_r_pc.sort_values(by='Año', inplace=True)
                df_r_sc.sort_values(by='Año', inplace=True)
                
                dataframes = {
                    'Santa Cruz': df_sc,
                    'Cordillera': df_pc,
                    'Camiri': df_c,
                    'Gutierrez': df_g,
                    'Lagunillas': df_l
                }
                dataframes_r = {
                    'Santa Cruz': df_r_sc,
                    'Cordillera': df_r_pc,
                    'Camiri': df_r_c,
                    'Gutierrez': df_r_g,
                    'Lagunillas': df_r_l
                }
                dataframes_t = {
                    'Santa Cruz': [df_sc_o, df_sc_s, df_sc_d],
                    'Cordillera': [df_pc_o, df_pc_s, df_pc_d],
                    'Camiri': [df_c_o, df_c_s, df_c_d],
                    'Gutierrez': [df_g_o, df_g_s, df_g_d],
                    'Lagunillas': [df_l_o, df_l_s, df_l_d]
                }

                df_barras = [dataframes[nombre] for nombre in dfl_barras if nombre in dataframes]
                df_tendencias = [dataframes[nombre] for nombre in dfl_tendencias if nombre in dataframes]
                df_r_barras = [dataframes_r[nombre] for nombre in dfl_barras if nombre in dataframes_r]
                df_r_tendencias = [dataframes_r[nombre] for nombre in dfl_tendencias if nombre in dataframes_r]
                df_total = [dataframes_t[nombre] for nombre in dfl_total if nombre in dataframes_t]

                resultados = []
                if n_clicks > 0:                
                    fig=generate_graph_total(df_barras, df_tendencias, dfl_barras, dfl_tendencias,
                                                titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    resultados.append(dcc.Graph(id='mi-grafico-total-nutricion', figure=fig))

                    fig=generate_graph_total(df_r_barras, df_r_tendencias, dfl_barras, dfl_tendencias,
                                                'Casos repetidos de Estado Nutricional en Embarazadas por año a nivel Departamental, Provincial y Municipal', 
                                                tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    resultados.append(dcc.Graph(id='mi-grafico-rep-total-nutricion', figure=fig))

                    for i, df in enumerate(df_total):
                        fig = generate_stacked_bar_chart(df, dfl_total[i], titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                    pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                        resultados.append(dcc.Graph(id=f'mi-grafico-total-nutricion-{i}', figure=fig))
                    
                    return resultados
            
                return html.Div("")

        except Exception as e:
            return html.Div(f'Error: {e}')
        
# Callback para realizar el cálculo de incidencias y porcentajes
@app.callback(
    Output('output-data-embarazo', 'children'),
    [
        Input('btn-calcular-embarazo', 'n_clicks'),
        Input('dropdown-type-age', 'value'),
        Input('dropdown-type-mounth', 'value'),
        Input('dropdown-type-percent', 'value'),
        Input('dropdown-dataframes-bar-dep', 'value'),
        Input('dropdown-dataframes-ten-dep', 'value'),
        Input('dropdown-dataframes-bar-prov', 'value'),
        Input('dropdown-dataframes-ten-prov', 'value'),
        Input('dropdown-dataframes-bar-mun', 'value'),
        Input('dropdown-dataframes-ten-mun', 'value'),
        Input('input-titulo', 'value'),
        Input('input-tamaño-titulo', 'value'),
        Input('input-eje-x', 'value'),
        Input('input-tamaño-eje-x', 'value'),
        Input('input-eje-y', 'value'),
        Input('input-tamaño-eje-y', 'value'),
        Input('input-pie', 'value'),
        Input('input-tamaño-pie', 'value'),
        Input('input-tamaño-leyenda', 'value'),
        Input('input-tamaño-num-grafica', 'value'),
    ],
    [State('input-factor', 'value'),
     State('url', 'pathname')]  # Capturar el pathname actual
)
def update_output_embarazo(n_clicks, type_age, type_mounth, type_percent,
                  dataframes_bar_dep, dataframes_ten_dep, 
                  dataframes_bar_prov, dataframes_ten_prov,
                  dataframes_bar_mun, dataframes_ten_mun, 
                  titulo, tamanio_titulo, 
                  eje_x, tamanio_eje_x, 
                  eje_y, tamanio_eje_y, 
                  pie, tamanio_pie, 
                  tamanio_leyenda, tamanio_num_grafica, 
                  factor, pathname):
    if n_clicks:
        try: 
            tamanio_titulo = int(tamanio_titulo) if tamanio_titulo is not None else 12
            tamanio_eje_x = int(tamanio_eje_x) if tamanio_eje_x is not None else 10
            tamanio_eje_y = int(tamanio_eje_y) if tamanio_eje_y is not None else 10
            tamanio_pie = int(tamanio_pie) if tamanio_pie is not None else 10
            tamanio_leyenda = int(tamanio_leyenda) if tamanio_leyenda is not None else 8
            tamanio_num_grafica = int(tamanio_num_grafica) if tamanio_num_grafica is not None else 10

            dfl_barras = dataframes_bar_dep + dataframes_bar_prov + dataframes_bar_mun
            dfl_tendencias = dataframes_ten_dep + dataframes_ten_prov + dataframes_ten_mun
            dfl_total = dataframes_bar_dep + dataframes_ten_dep + dataframes_bar_prov + dataframes_ten_prov + dataframes_bar_mun + dataframes_ten_mun
            dfl_barras = [item for item in dfl_barras if item]
            dfl_tendencias = [item for item in dfl_tendencias if item]
            dfl_total = [item for item in dfl_total if item]

            resultados = []
            partes = pathname.split('/')
            resultados.append(html.H2(f'Gráfico '+partes[1]))
            if 'Santa Cruz' in dfl_barras or 'Santa Cruz' in dfl_tendencias:
                df = get_casos(partes[1], 'guarani')
                df_c = df[0]
                df_g = df[1]
                df_l = df[2]
                df_pc = df[3]
                df_sc = df[4]

                # Convertir a listas de poblaciones
                p = get_casos('poblacion-especial', 'guarani')
                p_c = p[0] 
                p_g = p[1]
                p_l = p[2]
                p_pc = p[3] 
                p_sc = p[4]

                if type_mounth == 'm1':
                    df_c = df_c.groupby('Año').sum().reset_index()
                    df_g = df_g.groupby('Año').sum().reset_index()
                    df_l = df_l.groupby('Año').sum().reset_index()
                    df_pc = df_pc.groupby('Año').sum().reset_index()
                    df_sc = df_sc.groupby('Año').sum().reset_index()
                elif type_mounth == 'm2':
                    #df_c = df_c[df_c['Tipo'] == 'Nuevo < 5'].drop(columns=['Tipo']).reset_index(drop=True)
                    df_c = df_c[df_c['Tipo'] == 'Nuevo < 5']
                    df_g = df_g[df_g['Tipo'] == 'Nuevo < 5']
                    df_l = df_l[df_l['Tipo'] == 'Nuevo < 5']
                    df_pc = df_pc[df_pc['Tipo'] == 'Nuevo < 5']
                    df_sc = df_sc[df_sc['Tipo'] == 'Nuevo < 5']
                else:
                    df_c = df_c[df_c['Tipo'] == 'Nuevo > 5']
                    df_g = df_g[df_g['Tipo'] == 'Nuevo > 5']
                    df_l = df_l[df_l['Tipo'] == 'Nuevo > 5']
                    df_pc = df_pc[df_pc['Tipo'] == 'Nuevo > 5']
                    df_sc = df_sc[df_sc['Tipo'] == 'Nuevo > 5']
                
                if type_age == 'r1':
                    p = p_c.groupby('Año')['10-14'].sum().tolist()
                    p_2 = p_g.groupby('Año')['10-14'].sum().tolist()
                    p_3 = p_l.groupby('Año')['10-14'].sum().tolist()
                    p_4 = p_pc.groupby('Año')['10-14'].sum().tolist()
                    p_5 = p_sc.groupby('Año')['10-14'].sum().tolist()

                    df_c = calculate_total(df_c, factor, p, '< 15')
                    df_g = calculate_total(df_g, factor, p_2, '< 15')
                    df_l = calculate_total(df_l, factor, p_3, '< 15')
                    df_pc = calculate_total(df_pc, factor, p_4, '< 15')
                    df_sc = calculate_total(df_sc, factor, p_5, '< 15')
                elif type_age == 'r2':
                    p = p_c.groupby('Año')['15-19'].sum().tolist()
                    p_2 = p_g.groupby('Año')['15-19'].sum().tolist()
                    p_3 = p_l.groupby('Año')['15-19'].sum().tolist()
                    p_4 = p_pc.groupby('Año')['15-19'].sum().tolist()
                    p_5 = p_sc.groupby('Año')['15-19'].sum().tolist()

                    df_c = calculate_total(df_c, factor, p, '15-19')
                    df_g = calculate_total(df_g, factor, p_2, '15-19')
                    df_l = calculate_total(df_l, factor, p_3, '15-19')
                    df_pc = calculate_total(df_pc, factor, p_4, '15-19')
                    df_sc = calculate_total(df_sc, factor, p_5, '15-19')
                else:
                    p = p_c.groupby('Año')['Adolescentes'].sum().tolist()
                    p_2 = p_g.groupby('Año')['Adolescentes'].sum().tolist()
                    p_3 = p_l.groupby('Año')['Adolescentes'].sum().tolist()
                    p_4 = p_pc.groupby('Año')['Adolescentes'].sum().tolist()
                    p_5 = p_sc.groupby('Año')['Adolescentes'].sum().tolist()

                    df_c = calculate_total(df_c, factor, p, '< 19')
                    df_g = calculate_total(df_g, factor, p_2, '< 19')
                    df_l = calculate_total(df_l, factor, p_3, '< 19')
                    df_pc = calculate_total(df_pc, factor, p_4, '< 19')
                    df_sc = calculate_total(df_sc, factor, p_5, '< 19')

                # Seleccionar los dataframes según la selección del usuario
                df_c.sort_values(by='Año', inplace=True)
                df_g.sort_values(by='Año', inplace=True)
                df_l.sort_values(by='Año', inplace=True)
                df_pc.sort_values(by='Año', inplace=True)
                df_sc.sort_values(by='Año', inplace=True)

                df_c_y = calculate_age(df_c, p_c, partes[1])
                df_g_y = calculate_age(df_g, p_g, partes[1])
                df_l_y = calculate_age(df_l, p_l, partes[1])
                df_pc_y = calculate_age(df_pc, p_pc, partes[1])
                df_sc_y = calculate_age(df_sc, p_sc, partes[1])

                df_c_y = calculate_age_total(df_c_y, partes[1])
                df_g_y = calculate_age_total(df_g_y, partes[1])
                df_l_y = calculate_age_total(df_l_y, partes[1])
                df_pc_y = calculate_age_total(df_pc_y, partes[1])
                df_sc_y = calculate_age_total(df_sc_y, partes[1])
                
                dataframes = {
                    'Santa Cruz': df_sc,
                    'Cordillera': df_pc,
                    'Camiri': df_c,
                    'Gutierrez': df_g,
                    'Lagunillas': df_l
                }

                dataframes_age = {
                    'Santa Cruz': df_sc_y,
                    'Cordillera': df_pc_y,
                    'Camiri': df_c_y,
                    'Gutierrez': df_g_y,
                    'Lagunillas': df_l_y
                }

                df_barras = [dataframes[nombre] for nombre in dfl_barras if nombre in dataframes]
                df_tendencias = [dataframes[nombre] for nombre in dfl_tendencias if nombre in dataframes]

                df_total = [dataframes_age[nombre] for nombre in dfl_total if nombre in dataframes_age]
                resultados = []
                if n_clicks > 0:                
                    fig=generate_graph_total(df_barras, df_tendencias, dfl_barras, dfl_tendencias,
                                                titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    resultados.append(dcc.Graph(id='mi-grafico-total-nutricion', figure=fig))
                    
                    for i, df in enumerate(df_total):
                        fig = generate_graph_age_pregnans(df, partes[1], dfl_total[i], 
                                                        titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                        pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                        resultados.append(dcc.Graph(id=f'mi-grafico-edad-embarazo-{i}', figure=fig))

                    return resultados
            
                return html.Div("")

        except Exception as e:
            return html.Div(f'Error: {e}')

# Callback para realizar el cálculo de incidencias y porcentajes
@app.callback(
    Output('output-data-consultas', 'children'),
    [
        Input('btn-calcular-consultas', 'n_clicks'),
        Input('dropdown-graphic-type', 'value'),
        Input('dropdown-dataframes-bar-dep', 'value'),
        Input('dropdown-dataframes-ten-dep', 'value'),
        Input('dropdown-dataframes-bar-prov', 'value'),
        Input('dropdown-dataframes-ten-prov', 'value'),
        Input('dropdown-dataframes-bar-mun', 'value'),
        Input('dropdown-dataframes-ten-mun', 'value'),
        Input('input-titulo', 'value'),
        Input('input-tamaño-titulo', 'value'),
        Input('input-eje-x', 'value'),
        Input('input-tamaño-eje-x', 'value'),
        Input('input-eje-y', 'value'),
        Input('input-tamaño-eje-y', 'value'),
        Input('input-pie', 'value'),
        Input('input-tamaño-pie', 'value'),
        Input('input-tamaño-leyenda', 'value'),
        Input('input-tamaño-num-grafica', 'value'),
    ],
    [State('input-factor', 'value'),
     State('url', 'pathname')]  # Capturar el pathname actual
)
def update_output_consultas(n_clicks, graphic_type,
                  dataframes_bar_dep, dataframes_ten_dep, 
                  dataframes_bar_prov, dataframes_ten_prov,
                  dataframes_bar_mun, dataframes_ten_mun,
                  titulo, tamanio_titulo, 
                  eje_x, tamanio_eje_x, 
                  eje_y, tamanio_eje_y, 
                  pie, tamanio_pie, 
                  tamanio_leyenda, tamanio_num_grafica, 
                  factor, pathname):
    if n_clicks:
        try: 
            tamanio_titulo = int(tamanio_titulo) if tamanio_titulo is not None else 12
            tamanio_eje_x = int(tamanio_eje_x) if tamanio_eje_x is not None else 10
            tamanio_eje_y = int(tamanio_eje_y) if tamanio_eje_y is not None else 10
            tamanio_pie = int(tamanio_pie) if tamanio_pie is not None else 10
            tamanio_leyenda = int(tamanio_leyenda) if tamanio_leyenda is not None else 8
            tamanio_num_grafica = int(tamanio_num_grafica) if tamanio_num_grafica is not None else 10

            dfl = dataframes_bar_dep + dataframes_bar_prov + dataframes_bar_mun

            dfl = [item for item in dfl if item]

            resultados = []
            partes = pathname.split('/')
            resultados.append(html.H2(f'Gráfico '+partes[1]))
            if 'Santa Cruz' in dfl:
                df = get_casos(partes[1], 'guarani')
                df_c = df[0]
                df_g = df[1]
                df_l = df[2]
                df_pc = df[3]
                df_sc = df[4]

                p_c, p_g, p_l, p_pc, p_sc = get_casos('poblacion', 'guarani')
                
                df_c = calculate_age(df_c, p_c, partes[1])
                df_g = calculate_age(df_g, p_g, partes[1])
                df_l = calculate_age(df_l, p_l, partes[1])
                df_pc = calculate_age(df_pc, p_pc, partes[1])
                df_sc = calculate_age(df_sc, p_sc, partes[1])

                df_c_y = calculate_age_total(df_c, partes[1])
                df_g_y = calculate_age_total(df_g, partes[1])
                df_l_y = calculate_age_total(df_l, partes[1])
                df_pc_y = calculate_age_total(df_pc, partes[1])
                df_sc_y = calculate_age_total(df_sc, partes[1])

                df_c.sort_values(by='Año', inplace=True)
                df_g.sort_values(by='Año', inplace=True)
                df_l.sort_values(by='Año', inplace=True)
                df_pc.sort_values(by='Año', inplace=True)
                df_sc.sort_values(by='Año', inplace=True)

                dataframes = {
                    'Santa Cruz': df_sc,
                    'Cordillera': df_pc,
                    'Camiri': df_c,
                    'Gutierrez': df_g,
                    'Lagunillas': df_l
                }
                dataframes_y = {
                    'Santa Cruz': df_sc_y,
                    'Cordillera': df_pc_y,
                    'Camiri': df_c_y,
                    'Gutierrez': df_g_y,
                    'Lagunillas': df_l_y
                }
                
                df_total = [dataframes[nombre] for nombre in dfl if nombre in dataframes]
                df_y = [dataframes_y[nombre] for nombre in dfl if nombre in dataframes_y]

                resultados = []
                if n_clicks > 0:                
                    if graphic_type == 't':
                        for i, df in enumerate(df_total):
                            fig = generate_comparison_graph_by_year(df, dfl[i], 
                                                                titulo, tamanio_titulo, 'Año', tamanio_eje_x, 'Total', tamanio_eje_y,
                                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                            resultados.append(dcc.Graph(id=f'mi-grafico-total-consulta-{i}', figure=fig))
                    elif graphic_type == 'e1':
                        for i, df in enumerate(df_total):
                            fig = generate_graph_separate_age(df, partes[1], dfl[i], 
                                                            titulo, tamanio_titulo, 'Año', tamanio_eje_x, 'Porcentaje', tamanio_eje_y,
                                                            pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                            resultados.append(dcc.Graph(id=f'mi-grafico-edad-{i}', figure=fig))
                    elif graphic_type == 'e2':
                        for i, df in enumerate(df_total):
                            fig = generate_graph_join_age(df, partes[1], dfl[i], 
                                                            titulo, tamanio_titulo, 'Año', tamanio_eje_x, 'Porcentaje', tamanio_eje_y,
                                                            pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                            resultados.append(dcc.Graph(id=f'mi-grafico-edad-{i}', figure=fig))
                    return resultados
            
                return html.Div("")

        except Exception as e:
            return html.Div(f'Error: {e}')   

# Callback para realizar el cálculo de incidencias y porcentajes
@app.callback(
    Output('output-data-salud', 'children'),
    [
        Input('btn-calcular-salud', 'n_clicks'),
        Input('dropdown-graphic-type', 'value'),
        Input('input-titulo', 'value'),
        Input('input-tamaño-titulo', 'value'),
        Input('input-eje-x', 'value'),
        Input('input-tamaño-eje-x', 'value'),
        Input('input-eje-y', 'value'),
        Input('input-tamaño-eje-y', 'value'),
        Input('input-pie', 'value'),
        Input('input-tamaño-pie', 'value'),
        Input('input-tamaño-leyenda', 'value'),
        Input('input-tamaño-num-grafica', 'value'),
    ],
    [State('url', 'pathname')]
)
def update_output_salud(n_clicks, graphic_type,
                        titulo, tamanio_titulo, 
                        eje_x, tamanio_eje_x, 
                        eje_y, tamanio_eje_y, 
                        pie, tamanio_pie, 
                        tamanio_leyenda, tamanio_num_grafica, 
                        pathname):
    if n_clicks:
        try: 
            tamanio_titulo = int(tamanio_titulo) if tamanio_titulo is not None else 12
            tamanio_eje_x = int(tamanio_eje_x) if tamanio_eje_x is not None else 10
            tamanio_eje_y = int(tamanio_eje_y) if tamanio_eje_y is not None else 10
            tamanio_pie = int(tamanio_pie) if tamanio_pie is not None else 10
            tamanio_leyenda = int(tamanio_leyenda) if tamanio_leyenda is not None else 8
            tamanio_num_grafica = int(tamanio_num_grafica) if tamanio_num_grafica is not None else 10

            factor = None
            try:
                factor = int(dcc.get_element('input-factor').value)
            except:
                factor = 1  # Valor por defecto si no se encuentra o es inválido

            resultados = []
            partes = pathname.split('/')
            resultados.append(html.H2(f'Gráfico '+partes[1]))
            df = get_casos(partes[1], 'guarani')
            df_poblacion = df[0]
            df_etnicidad = df[1]
            df_alfabetismo = df[2]
            df_servicios_basicos = df[3]
            df_dormitorios = df[4]
            df_ocupacion = df[5]
            df_c_abandono = df[6]
            df_g_abandono = df[7]
            df_l_abandono = df[8]

            df_poblacion = calculate_population_group(df_poblacion)
            colors_language = ['#636EFA', '#EF553B', '#FFA15A', '#AB63FA', '#00CC96']  # Colores personalizados para los trozos de idiomas
            # Agregar columnas para el porcentaje de analfabetos
            df_alfabetismo["Hombre_Illiterate"] = 100 - df_alfabetismo["Hombre"]
            df_alfabetismo["Mujer_Illiterate"] = 100 - df_alfabetismo["Mujer"]
            df_alfabetismo["Total_Illiterate"] = 100 - df_alfabetismo["Total"]

            df_dormitorios['cuatro o más'] = df_dormitorios[['cuatro', 'cinco', 'seis', 'siete', 'ocho o más']].sum(axis=1)
            # Eliminar las columnas originales
            df_dormitorios = df_dormitorios.drop(columns=['cuatro', 'cinco', 'seis', 'siete', 'ocho o más'])
            df_dormitorios = df_dormitorios[['Dep_Prov_Mun', 's/n', 'un', 'dos', 'tres', 'cuatro o más', 'Total']]
            
            """graphic_config = {
                'ism': {'label': 'Indice de Salud', 'ids': ['mi-grafico-ism-1', 'mi-grafico-ism-2']},
                'mortalidad_infantil': {'label': 'Tasa de Mortalidad Infantil', 'ids': ['mi-grafico-tmi-1', 'mi-grafico-tmi-2']},
                'ingreso': {'label': 'Ingreso Municipal', 'ids': ['mi-grafico-im-1', 'mi-grafico-im-2']},
                'asistencia_escolar': {'label': 'Asistencia Escolar', 'ids': ['mi-grafico-ae-1', 'mi-grafico-ae-2']},
                'anio_estudio_promedio': {'label': 'Años Promedio de Estudio', 'ids': ['mi-grafico-aep-1', 'mi-grafico-aep-2']},
                'hogares_electricidad': {'label': 'Hogares con Electricidad', 'ids': ['mi-grafico-he-1', 'mi-grafico-he-2']},
                'hogares_saneamiento': {'label': 'Hogares con Saneamiento básico', 'ids': ['mi-grafico-hs-1', 'mi-grafico-hs-2']},
                'hogares_agua_potable': {'label': 'Hogares con Agua potable', 'ids': ['mi-grafico-hap-1', 'mi-grafico-hap-2']}
            }"""

            if n_clicks > 0:
                if graphic_type == 'p':
                    fig = generate_population_pyramid(df_poblacion, "Cordillera")
                    resultados.append(dcc.Graph(id='mi-piramide-poblacion-cordillera', figure=fig))
                    fig = generate_population_pyramid(df_poblacion, "Camiri")
                    resultados.append(dcc.Graph(id='mi-piramide-poblacion-camiri', figure=fig))
                    fig = generate_population_pyramid(df_poblacion, "Gutierrez")
                    resultados.append(dcc.Graph(id='mi-piramide-poblacion-gutierrez', figure=fig))
                    fig = generate_population_pyramid(df_poblacion, "Lagunillas")
                    resultados.append(dcc.Graph(id='mi-piramide-poblacion-lagunillas', figure=fig))
                elif graphic_type == 'e':
                    fig = generate_language_donut_chart(df_etnicidad, "Cordillera", colors_language)
                    resultados.append(dcc.Graph(id='mi-dona-etnica-cordillera', figure=fig))
                    fig = generate_language_donut_chart(df_etnicidad, "Camiri", colors_language)
                    resultados.append(dcc.Graph(id='mi-dona-etnica-camiri', figure=fig))
                    fig = generate_language_donut_chart(df_etnicidad, "Gutierrez", colors_language)
                    resultados.append(dcc.Graph(id='mi-dona-etnica-gutierrez', figure=fig))
                    fig = generate_language_donut_chart(df_etnicidad, "Lagunillas", colors_language)
                    resultados.append(dcc.Graph(id='mi-dona-etnica-lagunillas', figure=fig))
                elif graphic_type == 'a':
                    fig = generate_literacy_donut_chart(df_alfabetismo, "Cordillera")
                    resultados.append(dcc.Graph(id="mi-dona-analfabeta-cordillera", figure=fig))
                    fig = generate_literacy_donut_chart(df_alfabetismo, "Camiri")
                    resultados.append(dcc.Graph(id="mi-dona-analfabeta-camiri", figure=fig))
                    fig = generate_literacy_donut_chart(df_alfabetismo, "Gutierrez")
                    resultados.append(dcc.Graph(id="mi-dona-analfabeta-gutierrez", figure=fig))
                    fig = generate_literacy_donut_chart(df_alfabetismo, "Lagunillas")
                    resultados.append(dcc.Graph(id="mi-dona-analfabeta-lagunillas", figure=fig))
                elif graphic_type == 'sb':
                    fig = generate_services_bar_chart(df_servicios_basicos, "Cordillera")
                    resultados.append(dcc.Graph(id="mi-grafica-basicos-cordillera", figure=fig))
                    fig = generate_services_bar_chart(df_servicios_basicos, "Camiri")
                    resultados.append(dcc.Graph(id="mi-grafica-basicos-camiri", figure=fig))
                    fig = generate_services_bar_chart(df_servicios_basicos, "Gutierrez")
                    resultados.append(dcc.Graph(id="mi-grafica-basicos-gutierrez", figure=fig))
                    fig = generate_services_bar_chart(df_servicios_basicos, "Lagunillas")
                    resultados.append(dcc.Graph(id="mi-grafica-basicos-lagunillas", figure=fig))
                elif graphic_type == 'h':
                    fig = generate_housing_pie_chart(df_dormitorios, "Cordillera")
                    resultados.append(dcc.Graph(id="mi-dona-hacinamiento-cordillera", figure=fig))
                    fig = generate_housing_pie_chart(df_dormitorios, "Camiri")
                    resultados.append(dcc.Graph(id="mi-dona-hacinamiento-camiri", figure=fig))
                    fig = generate_housing_pie_chart(df_dormitorios, "Gutierrez")
                    resultados.append(dcc.Graph(id="mi-dona-hacinamiento-gutierrez", figure=fig))
                    fig = generate_housing_pie_chart(df_dormitorios, "Lagunillas")
                    resultados.append(dcc.Graph(id="mi-dona-hacinamiento-lagunillas", figure=fig))
                elif graphic_type == 'o':
                    fig = generate_ocupation_bar_chart(df_ocupacion, "Camiri")
                    resultados.append(dcc.Graph(id="mi-bar-ocupacion-camiri", figure=fig))
                    fig = generate_ocupation_bar_chart(df_ocupacion, "Gutierrez")
                    resultados.append(dcc.Graph(id="mi-bar-ocupacion-gutierrez", figure=fig))
                    fig = generate_ocupation_bar_chart(df_ocupacion, "Lagunillas")
                    resultados.append(dcc.Graph(id="mi-bar-ocupacion-lagunillas", figure=fig))
                elif graphic_type == 'as':
                    fig = generate_secondary_abandonment_trend(df_c_abandono, "Camiri")
                    resultados.append(dcc.Graph(id="mi-bar-abandono-camiri", figure=fig))
                    fig = generate_secondary_abandonment_trend(df_g_abandono, "Gutierrez")
                    resultados.append(dcc.Graph(id="mi-bar-abandono-gutierrez", figure=fig))
                    fig = generate_secondary_abandonment_trend(df_l_abandono, "Lagunillas")
                    resultados.append(dcc.Graph(id="mi-bar-abandono-lagunillas", figure=fig))
                
                return resultados
        
            return html.Div("")

        except Exception as e:
            return html.Div(f'Error: {e}')    

@app.callback(
    [Output('dropdown-dataframes-bar-prov', 'value'),
     Output('dropdown-dataframes-ten-prov', 'value')],
    [Input('dropdown-dataframes-bar-prov', 'value'),
     Input('dropdown-dataframes-ten-prov', 'value')]
)
def sync_provincia_dropdowns(bar_value, ten_value):
    # Obtener el ID del componente que activó la callback
    ctx = callback_context
    if not ctx.triggered:
        return bar_value, ten_value  # Sin cambios si no hay activación

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'dropdown-dataframes-bar-prov':
        ten_value = [item for item in ten_value if item not in bar_value]
    elif triggered_id == 'dropdown-dataframes-ten-prov':
        bar_value = [item for item in bar_value if item not in ten_value]
    
    return bar_value, ten_value

@app.callback(
    [Output('dropdown-dataframes-bar-mun', 'value'),
     Output('dropdown-dataframes-ten-mun', 'value')],
    [Input('dropdown-dataframes-bar-mun', 'value'),
     Input('dropdown-dataframes-ten-mun', 'value')]
)
def sync_municipio_dropdowns(bar_value, ten_value):
    # Obtener el ID del componente que activó la callback
    ctx = callback_context
    if not ctx.triggered:
        return bar_value, ten_value  # Sin cambios si no hay activación

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'dropdown-dataframes-bar-mun':
        ten_value = [item for item in ten_value if item not in bar_value]
    elif triggered_id == 'dropdown-dataframes-ten-mun':
        bar_value = [item for item in bar_value if item not in ten_value]
    
    return bar_value, ten_value


if __name__ == '__main__':
    app.run_server(debug=True)
