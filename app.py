import dash
import os
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
import math

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
    ('CasosPartoProfecional-Guaranis.xlsx', '19xp7w_EzjMDJobYsfHnNpCxZeHoOkLUy'),
    ('CasosVacunaPentavalente-Guaranis.xlsx', '1dtE5mD8YIkN4b-W-NABk66XQNhhaaxBd'),
    ('DatosPoblaciones-Guaranis.xlsx', '1Tkr9PBQJHAb5m8zq8k-EFl1bvUMvq-5B'),
    ('DatosEspeciales-Guaranis.xlsx', '1yvLMWWwlDVwKLhVqQO8iDxlVrI-cOOV7'),
    ('CaracteristicasSocieconomicas-Guaranis.xlsx', '1WTJkEGpCEkVoDmkGc_OoRfHpBvgDsUP1'),
    ('CasosCancer-Afrobolivianos.xlsx', '16lfz2GhZyPoKyeqTkHBYy2-wbnSZUkU2'),
    ('CasosDiabetes-Afrobolivianos.xlsx', '1x8BOhJyTEzyECk7TcvqNRV4HgaI3ShsB'),
    ('CasosHipertensionArterial-Afrobolivianos.xlsx', '1uyW2BXTycpc4Ewpjfwka7vLHnhVRqv4t'),
    ('CasosObesidad-Afrobolivianos.xlsx', '1ejKBMKoC--yLlMvsd_mIOOlQFq4X5NRL'),
    ('CasosNeumonia-Afrobolivianos.xlsx', '1J_pItHbEFSvv5pdkVJLnN4FUa_IWRLY4'),
    ('CasosChagas-Afrobolivianos.xlsx', '1UzxFdVyyXpjo_Kq8cxmBTX4o3WF4xZei'),
    ('CasosVIH-Afrobolivianos.xlsx', '1Ti2yHC4TT0BN8HFvBC58jSNdvwrAaZWb'),   
    ('CasosEstadoNutricional-Afrobolivianos.xlsx', '10d1hzer4VjByN6rx9wsgWNc6GEF6rVmI'),
    ('CasosEmbarazoAdolescente-Afrobolivianos.xlsx', '18qsKl9sDg6yL9Zt8-WwuInngsu9jCttQ'),
    ('CasosConsultaExterna-Afrobolivianos.xlsx', '1JaNQU--kM9s9_f7sZ7T7h0ukdJncc3vg'),
    ('CasosPartoProfecional-Afrobolivianos.xlsx', '1ZWIT9Bd8HvnJwwUyadCH2NgPlEVE-g-8'),
    ('CasosVacunaPentavalente-Afrobolivianos.xlsx', '1sU0hw0vWaPYwe2OsQpyWonuIIGr87yyq'),
    ('DatosPoblaciones-Afrobolivianos.xlsx', '1lXXK6zfDSs31D8ORxKfRYiUhoyRqZhcL'),
    ('DatosEspeciales-Afrobolivianos.xlsx', '1qQitkM27dpYI8LfQ6lJZ8rJoHKcHwWEo'),
    ('CaracteristicasSocieconomicas-Afrobolivianos.xlsx', '1PjwGFwWqgaIQ3O3Nj6JoGSHrHT08_EMP')
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
        'pentavalentes': {
            'guarani': 'CasosVacunaPentavalente-Guaranis.xlsx',
            'afroboliviano': 'CasosVacunaPentavalente-Afrobolivianos.xlsx'
        },
        'partos': {
            'guarani': 'CasosPartoProfecional-Guaranis.xlsx',
            'afroboliviano': 'CasosPartoProfecional-Afrobolivianos.xlsx'
        },
        'consultas': {
            'guarani': 'CasosConsultaExterna-Guaranis.xlsx',
            'afroboliviano': 'CasosConsultaExterna-Afrobolivianos.xlsx'
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
            'guarani': 'CaracteristicasSocieconomicas-Guaranis.xlsx',
            'afroboliviano': 'CaracteristicasSocieconomicas-Afrobolivianos.xlsx'
        }
    }
    
    sheets_map = {
        'cancer': {
            'guarani': ["CANCER-C", "CANCER-G", "CANCER-L", "CANCER-CV", "CANCER-PC", "CANCER-SC"],
            'afroboliviano': ["CANCER-A", "CANCER-B", "CANCER-C", "CANCER-I", "CANCER-Y", "CANCER-SY", "CANCER-LP"]
        },
        'diabetes': {
            'guarani': ["DIABETES-C", "DIABETES-G", "DIABETES-L", "DIABETES-CV", "DIABETES-PC", "DIABETES-SC"],
            'afroboliviano': ["DIABETES-A", "DIABETES-B", "DIABETES-C", "DIABETES-I", "DIABETES-Y", "DIABETES-SY", "DIABETES-LP"]
        },
        'hipertension': {
            'guarani': ["HIPERTENSION-C", "HIPERTENSION-G", "HIPERTENSION-L", "HIPERTENSION-CV", "HIPERTENSION-PC", "HIPERTENSION-SC"],
            'afroboliviano': ["HIPERTENSION-A", "HIPERTENSION-B", "HIPERTENSION-C", "HIPERTENSION-I", "HIPERTENSION-Y", "HIPERTENSION-SY", "HIPERTENSION-LP"]
        },
        'obesidad': {
            'guarani': ["OBESIDAD-C", "OBESIDAD-G", "OBESIDAD-L", "OBESIDAD-CV", "OBESIDAD-PC", "OBESIDAD-SC"],
            'afroboliviano': ["OBESIDAD-A", "OBESIDAD-B", "OBESIDAD-C", "OBESIDAD-I", "OBESIDAD-Y", "OBESIDAD-SY", "OBESIDAD-LP"]
        },
        'neumonia': {
            'guarani': ["NEUMONIA-C", "NEUMONIA-G", "NEUMONIA-L", "NEUMONIA-CV", "NEUMONIA-PC", "NEUMONIA-SC"],
            'afroboliviano': ["NEUMONIA-A", "NEUMONIA-B", "NEUMONIA-C", "NEUMONIA-I", "NEUMONIA-Y", "NEUMONIA-SY", "NEUMONIA-LP"]
        },
        'chagas': {
            'guarani': ["CHAGAS-C", "CHAGAS-G", "CHAGAS-L", "CHAGAS-CV", "CHAGAS-PC", "CHAGAS-SC"],
            'afroboliviano': ["CHAGAS-A", "CHAGAS-B", "CHAGAS-C", "CHAGAS-I", "CHAGAS-Y", "CHAGAS-SY", "CHAGAS-LP"]
        },
        'vih': {
            'guarani': ["VIH-C", "VIH-G", "VIH-L", "VIH-CV", "VIH-PC", "VIH-SC"],
            'afroboliviano': ["VIH-A", "VIH-B", "VIH-C", "VIH-I", "VIH-Y", "VIH-SY", "VIH-LP"]
        },
        'pentavalentes': {
            'guarani': ["PENTA-C", "PENTA-G", "PENTA-L", "PENTA-CV", "PENTA-PC", "PENTA-SC"],
            'afroboliviano': ["PENTA-A", "PENTA-B", "PENTA-C", "PENTA-I", "PENTA-Y", "PENTA-SY", "PENTA-LP"]
        },
        'nutricion': {
            'guarani': ["NUTRICION-C", "NUTRICION-G", "NUTRICION-L", "NUTRICION-CV", "NUTRICION-PC", "NUTRICION-SC"],
            'afroboliviano': ["NUTRICION-A", "NUTRICION-B", "NUTRICION-C", "NUTRICION-I", "NUTRICION-Y", "NUTRICION-SY", "NUTRICION-LP"],
        },
        'embarazo': {
            'guarani': ["EMBARAZO-C", "EMBARAZO-G", "EMBARAZO-L", "EMBARAZO-CV", "EMBARAZO-PC", "EMBARAZO-SC"],
            'afroboliviano': ["EMBARAZO-A", "EMBARAZO-B", "EMBARAZO-C", "EMBARAZO-I", "EMBARAZO-Y", "EMBARAZO-SY", "EMBARAZO-LP"]
        },
        'consultas': {
            'guarani': ["CONSULTAS-C", "CONSULTAS-G", "CONSULTAS-L", "CONSULTAS-CV", "CONSULTAS-PC", "CONSULTAS-SC"],
            'afroboliviano': ["CONSULTAS-A", "CONSULTAS-B", "CONSULTAS-C", "CONSULTAS-I", "CONSULTAS-Y", "CONSULTAS-SY", "CONSULTAS-LP"]
        },
        'partos': {
            'guarani': ["PARTO-C", "PARTO-G", "PARTO-L", "PARTO-CV", "PARTO-PC", "PARTO-SC"],
            'afroboliviano': ["PARTO-A", "PARTO-B", "PARTO-C", "PARTO-I", "PARTO-Y", "PARTO-SY", "PARTO-LP"]
        },
        'poblacion': {
            'guarani': ["POBLACION-C", "POBLACION-G", "POBLACION-L", "POBLACION-CV", "POBLACION-PC", "POBLACION-SC"],
            'afroboliviano': ["POBLACION-A", "POBLACION-B", "POBLACION-C", "POBLACION-I", "POBLACION-Y", "POBLACION-SY", "POBLACION-LP"]
        },
        'poblacion-especial': {
            'guarani': ["ESPECIALES-C", "ESPECIALES-G", "ESPECIALES-L", "ESPECIALES-CV", "ESPECIALES-PC", "ESPECIALES-SC"],
            'afroboliviano': ["ESPECIALES-A", "ESPECIALES-B", "ESPECIALES-C", "ESPECIALES-I", "ESPECIALES-Y", "ESPECIALES-SY", "ESPECIALES-LP"]
        },
        'salud': {
            'guarani': ["POBLACION-SC", "ETNICIDAD-SC", "ALFABETISMO-SC", "SERVICIOS-BASICOS-SC", "NUMERO-DORMITORIOS-SC", "CATEGORIA-OCUPACIONAL-SC", "ABAN-INTRA-ESC-C", "ABAN-INTRA-ESC-G", "ABAN-INTRA-ESC-L", "ABAN-INTRA-ESC-CV"],
            'afroboliviano': ["POBLACION-LP", "ETNICIDAD-LP", "ALFABETISMO-LP", "SERVICIOS-BASICOS-LP", "NUMERO-DORMITORIOS-LP", "CATEGORIA-OCUPACIONAL-LP", "ABAN-INTRA-ESC-A", "ABAN-INTRA-ESC-B", "ABAN-INTRA-ESC-C", "ABAN-INTRA-ESC-I", "ABAN-INTRA-ESC-Y"]
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

def calculate_gender(df_o, factor, m, h, column):
    df = df_o.copy()
    # Población estimada
    total_mujeres = {2019: m[0], 2020: m[1], 2021: m[2], 2022: m[3], 2023: m[4]}
    total_hombres = {2019: h[0], 2020: h[1], 2021: h[2], 2022: h[3], 2023: h[4]}

    # Calcular incidencias
    df['Incidencia'] = df.apply(
        lambda row: (
            (row[column] / max((total_hombres[row['Año']] - row[column]), 1) * factor)
            if row['Sexo'] == 'Hombre'
            else (row[column] / max((total_mujeres[row['Año']] - row[column]), 1) * factor)
        ) if (row['Sexo'] == 'Hombre' and total_hombres[row['Año']] - row[column] > 0) or (row['Sexo'] == 'Mujer' and total_mujeres[row['Año']] - row[column] > 0) else 0,
        axis=1
    ).round().astype(int)
    
    # Calcular los totales para hombres y mujeres
    total_hombres = df[df['Sexo'] == 'Hombre'][column].sum()
    total_mujeres = df[df['Sexo'] == 'Mujer'][column].sum()

    # Asegurarse de que los totales no sean cero
    total_hombres = total_hombres if total_hombres != 0 else 1
    total_mujeres = total_mujeres if total_mujeres != 0 else 1

    # Calcular el porcentaje y redondear a 2 decimales
    df['Porcentaje'] = df.apply(
        lambda row: (row[column] / total_hombres * 100) if row['Sexo'] == 'Hombre' else (row[column] / total_mujeres * 100),
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
    color_barras = ['#00188D', '#0123CB', '#3859FF', '#006EF4', '#424CD3', '#0278AA']  # Puedes agregar más colores
    color_tendencias = ['#DB4D00', '#F87F14', '#FF9500', '#FF6200', '#E06C00', '#E05B2A']  # Puedes agregar más colores

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
            line=dict(color=tend_color, width=3),
            marker=dict(size=10)  # Aquí ajustas el tamaño de los puntos
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

    y_max = max(df[y].max() for df in df_barras + df_tendencias)  # Encuentra el valor máximo en todos los DataFrames
    y_range = [0, y_max * 1.1]

    if y == "Incidencia":
        y = "Incidencia x 10000 personas"
    # Ajustes de layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title, weight='bold')
        },
        xaxis_title=x,
        yaxis=dict(
            title=y,
            titlefont_size=size_y,
            range=y_range,
        ),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(font=dict(size=size_legend)),
        #annotations=[
        #    dict(
        #        text=footer,
        #        xref="paper", yref="paper",
        #        x=0.5, y=-0.2,  # Ajustar posición según sea necesario
        #        showarrow=False,
        #        font=dict(size=size_footer)
        #    )
        #],
        width=1000,  # Ancho de la gráfica
        height=550  # Alto de la gráfica
    )

    # Habilitar líneas de cuadrícula
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_x))
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_y))

    return fig

def generate_graph_join_gender(df_barras, df_tendencias, labels_barras, labels_tendencias, 
                         title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=("Hombres", "Mujeres"),
        horizontal_spacing=0.04
    )

    texttemplate = "%{text:.2f}%" if y == 'Porcentajes' else None
    # Lista de colores
    color_barras_male = ['#00188D', '#0123CB', '#3859FF', '#006EF4', '#424CD3', '#0278AA']  # Puedes agregar más colores
    color_tendencias_male = ['#0278AA', '#424CD3', '#006EF4', '#3859FF', '#0123CB', '#00188D']
    color_barras_female = ['#E05B2A', '#E06C00', '#FF6200', '#FF9500', '#F87F14', '#DB4D00']
    color_tendencias_female = ['#DB4D00', '#F87F14', '#FF9500', '#FF6200', '#E06C00', '#E05B2A']
    #color_barras_female = ['#E95D0C', '#F88011', '#FF932E', '#FFAB5C', '#FFCDA6']  # Puedes agregar más colores
    #color_tendencias_female = ['#FFCDA6', '#FFAB5C', '#FF932E', '#F88011', '#E95D0C']

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
            line=dict(color=tend_color, width=3),
            marker=dict(size=10)  
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
            line=dict(color=tend_color, width=3),
            marker=dict(size=10) 
        ), row=1, col=2)

    y_max = max(df[y].max() for df in df_barras + df_tendencias)  # Encuentra el valor máximo en todos los DataFrames
    y_range = [0, y_max * 1.1]
    
    if y == "Incidencia":
        y = "Incidencia x 10000 personas"
    # Ajustes de layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title, weight='bold')
        },
        xaxis_title=x,
        yaxis=dict(
            title=y,
            titlefont_size=size_y,
            range=y_range,
        ),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(font=dict(size=size_legend)),  # Eliminar 'weight'
        #annotations=[
        #    dict(
        #        text=footer,
        #        xref="paper", yref="paper",
        #        x=0.5, y=-0.2,  # Ajustar posición según sea necesario
        #        showarrow=False,
        #        font=dict(size=size_footer)
        #    )
        #]

    )

    # Habilitar líneas de cuadrícula
    fig.update_xaxes(title_font=dict(size=size_x), title_text=x, row=1, col=1)
    fig.update_xaxes(title_font=dict(size=size_x), title_text=x, row=1, col=2)
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_y), matches='y')

    return fig


def generate_graph_separate_gender(dfs, graph_type, labels, 
                                    title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    num_dfs = len(dfs)
    num_cols = 2
    num_rows = (num_dfs + 1) // 2  # Calcula el número de filas necesarias

    fig = make_subplots(
        rows=num_rows, 
        cols=num_cols, 
        subplot_titles=labels, 
        vertical_spacing=0.08, 
        horizontal_spacing=0.1
    )

    texttemplate = "%{text:.2f}%" if y == 'Porcentajes' else None

    # Lista de colores
    color_1_male = ['#00188D', '#0123CB', '#3859FF', '#006EF4', '#424CD3', '#0278AA']
    color_2_male = ['#00188D', '#0123CB', '#3859FF', '#006EF4', '#424CD3', '#0278AA']
    color_1_female = ['#E05B2A', '#E06C00', '#FF6200', '#FF9500', '#F87F14', '#DB4D00']
    color_2_female = ['#E05B2A', '#E06C00', '#FF6200', '#FF9500', '#F87F14', '#DB4D00']

    #color_1_male = ['#011CA2', '#0B2FE4', '#2146FF', '#6E85F5', '#99AAFF']  # Puedes agregar más colores
    #color_2_male = ['#99AAFF', '#6E85F5', '#2146FF', '#0B2FE4', '#011CA2']
    #color_1_female = ['#E06C00', '#EA8A00', '#F88011', '#FFAB5C', '#FFAB5C']
    #color_2_female = ['#FFAB5C', '#FFAB5C', '#F88011', '#EA8A00', '#E06C00']

    #color_tendencias_female = ['#FFCDA6', '#FFAB5C', '#F88011', '#EA8A00', '#E06C00']
    #['#FFAB5C', '#FF932E', '#F88011', '#EA8A00', '#E06C00']

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
                line=dict(color=tend_color, width=3),
                marker=dict(size=10),
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
                line=dict(color=tend_color, width=3),
                marker=dict(size=10),
                showlegend=False
            ), row=row, col=col)

    y_max = max(df[y].max() for df in dfs)  # Encuentra el valor máximo en todos los DataFrames
    y_range = [0, y_max * 1.2]

    if y == "Incidencia":
        y = "Incidencia x 10000 personas"
    # Ajustes de layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title, weight='bold')
        },
        xaxis_title=x,
        yaxis=dict(
            title=y,
            titlefont_size=size_y,
            range=y_range,
        ),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500 * num_rows,  # Ajustar la altura de las subplots
        width=1200,  # Ajustar la anchura de las subplots
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
    colors_male = ['#00188D', '#0123CB', '#3859FF', '#006EF4', '#424CD3', '#0278AA']
    colors_female = ['#E05B2A', '#E06C00', '#FF6200', '#FF9500', '#F87F14', '#DB4D00']
    
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

    def convert_labels(labels):
        conversion = {
            '% 0-1': '0-1', '% 0-9': '0-9', '% 0-19': '0-19', '% 1-4': '1-4', 
            '% 5-9': '5-9', '% 10-19': '10-19',
            '% 20-39': '20-39', '% 40-49': '40-49', '% 50-59': '50-59', '% 60+': '60+',
            'I_0-1': '0-1', 'I_0-9': '0-9', 'I_0-19': '0-19', 'I_1-4': '1-4', 
            'I_5-9': '5-9', 'I_10-19': '10-19',
            'I_20-39': '20-39', 'I_40-49': '40-49', 'I_50-59': '50-59', 'I_60+': '60+'
        }
        return [conversion.get(label, label) for label in labels]

    # Añadir barras para hombres
    for i, year in enumerate(df[x].unique()):
        fig.add_trace(go.Bar(
            x=convert_labels(age_columns),  # Columnas desde < 6 en adelante son edades
            y=df_male[df_male[x] == year][age_columns].iloc[0],  # Fila correspondiente al año y hombres
            name=str(year),
            marker=dict(color=colors_male[i], line=dict(color='black', width=1)),
            legendgroup='hombres',
            texttemplate='%{y}',
            textposition='outside', 
            textfont=dict(color=colors_male[i], size=size_graph),
        ), row=1, col=1)

    # Añadir barras para mujeres
    for i, year in enumerate(df[x].unique()):
        fig.add_trace(go.Bar(
            x=convert_labels(age_columns),  # Columnas desde < 6 en adelante son edades
            y=df_female[df_female[x] == year][age_columns].iloc[0],  # Fila correspondiente al año y mujeres
            name=str(year),
            marker=dict(color=colors_female[i], line=dict(color='black', width=1)),
            legendgroup='mujeres',
            texttemplate='%{y}',
            textposition='outside', 
            textfont=dict(color=colors_female[i], size=size_graph),
        ), row=1, col=2)

    if y == "Incidencia":
        y = "Incidencia x 10000 personas"

    # Ajustes de layout
    fig.update_layout(
        title={
            'text': title + ' ' + labels,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title, weight='bold')
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

def generate_graph_join_age(df, age_columns, labels, 
                                   title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph, y_max=None):
    # Filtrar datos por sexo
    df_male = df[df['Sexo'] == 'Hombre']
    df_female = df[df['Sexo'] == 'Mujer']

    # Crear figura con subplots
    fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=['Hombres', 'Mujeres'],
        horizontal_spacing=0.04
    )

    # Configuración de colores
    color_male = '#0123CB'
    color_female = '#E06C00'
    
    def convert_labels(labels):
        conversion = {
            '% 0-1': '0-1', '% 0-9': '0-9', '% 0-19': '0-19', '% 1-4': '1-4', 
            '% 5-9': '5-9', '% 10-19': '10-19',
            '% 20-39': '20-39', '% 40-49': '40-49', '% 50-59': '50-59', '% 60+': '60+',
            'I_0-1': '0-1', 'I_0-9': '0-9', 'I_0-19': '0-19', 'I_1-4': '1-4', 
            'I_5-9': '5-9', 'I_10-19': '10-19',
            'I_20-39': '20-39', 'I_40-49': '40-49', 'I_50-59': '50-59', 'I_60+': '60+'
        }
        return [conversion.get(label, label) for label in labels]

    # Añadir barras para hombres
    fig.add_trace(go.Bar(
        x=convert_labels(age_columns),
        y=df_male[age_columns].iloc[0],
        name='Hombres',
        marker=dict(color=color_male, line=dict(color='black', width=1)),
        texttemplate='%{y}',
        textposition='outside',
        textfont=dict(color=color_male, size=size_graph),
    ), row=1, col=1)

    # Añadir barras para mujeres
    fig.add_trace(go.Bar(
        x=convert_labels(age_columns),
        y=df_female[age_columns].iloc[0],
        name='Mujeres',
        marker=dict(color=color_female, line=dict(color='black', width=1)),
        texttemplate='%{y}', 
        textposition='outside',
        textfont=dict(color=color_female, size=size_graph),
    ), row=1, col=2)
    
    if y == "Incidencia":
        y = "Incidencia x 10000 personas"
    
    if y_max is not None:
        yaxis_range = [0, y_max + y_max * 0.09]
    else:
        yaxis_range = [0, df[age_columns].max().max() + df[age_columns].max().max() * 0.09]
    # Ajustes de layout
    fig.update_layout(
        title={
            'text': title + ' ' + labels,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title, weight='bold')
        },
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            title=y,
            titlefont_size=size_y,
            range = yaxis_range,
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
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_y), row=1, col=2, matches='y')

    return fig

def generate_graph_age_pregnans(df, graph_type, labels, 
                           title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    # Crear figura
    fig = go.Figure()

    df_female = df.copy()
    colors_female = ['#E05B2A', '#E06C00', '#FF6200', '#FF9500', '#F87F14', '#DB4D00']
    
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
                'text': title + ' ' + labels,
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=size_title, weight='bold')
            },
            barmode='stack',  # Apilar barras
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(
                title=y,
                titlefont_size=size_y,
                showgrid=True,  # Mostrar cuadrícula en el eje y
            ),
            #annotations=[
            #    dict(
            #        text=footer,
            #        xref="paper", yref="paper",
            #        x=0.5, y=-0.2,  # Ajustar posición según sea necesario
            #        showarrow=False,
            #        font=dict(size=size_footer)
            #    )
            #],
            width=800,  # Ancho de la gráfica
            height=500  # Alto de la gráfica
        )

        # Ajustar título del eje x
        fig.update_xaxes(title_font=dict(size=size_x), title_text="Edad")
        fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1, title_font=dict(size=size_y))

    return fig

def generate_comparison_graph_by_year(df, labels, 
                                      title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    # Filtrar datos por sexo
    df_male = df[df['Sexo'] == 'Hombre']
    df_female = df[df['Sexo'] == 'Mujer']

    # Configuración de colores
    color_male = '#0123CB'
    color_female = '#E05B2A'
    
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
            'text': title + ' ' + labels,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=size_title, weight='bold')
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
        #annotations=[
        #    dict(
        #        text=footer,
        #        xref="paper", yref="paper",
        #        x=0.5, y=-0.2,  # Ajustar posición según sea necesario
        #        showarrow=False,
        #        font=dict(size=size_footer)
        #    )
        #],
        width=800,  # Ancho de la gráfica
        height=500
    )

    return fig

def generate_total_area(df, labels, y_max=None):
    # Crear el gráfico de áreas
    fig = go.Figure()

    # Definir colores para las áreas
    colors = {"Hombre": "#0123CB", "Mujer": "#E05B2A"}

    for sexo in df["Sexo"].unique():
        df_sexo = df[df["Sexo"] == sexo]

        fig.add_trace(go.Scatter(
            x=df_sexo["Año"],
            y=df_sexo["Total"],
            mode='lines+markers+text',
            #fill='tozeroy' if sexo == "Hombre" else 'tonexty',  # Ajusta el relleno
            name=f'Total {sexo}',
            text=df_sexo["Total"],  # Usa los valores de 'Total' como texto
            textposition='top center',
            textfont=dict(color="black", size=18),
            line=dict(color=colors[sexo]),
            marker=dict(size=10)  # Tamaño de los marcadores
        ))

    #if y_max != None:
    #    yaxis_range = [0, y_max + y_max * 0.09]
    #else:
    yaxis_range = [0, df["Total"].max() + df["Total"].max() * 0.09]

    if labels in ['Cordillera', 'Sud Yungas']:
        title_text = 'Número total de consultas externas por sexo y año según nivel Provincial de'
    elif labels in ['Santa Cruz', 'La Paz']:
        title_text = 'Número total de consultas externas por sexo y año según nivel Departamental de'
    else:
        title_text = 'Número total de consultas externas por sexo y año según nivel Municipal de'

    fig.update_layout(
        title={
            'text': f"{title_text} {labels}",
            'x': 0.5,  # Centrar el título horizontalmente
            'xanchor': 'center',  # Alinear el título al centro
            'font': dict(size=18, weight='bold')  # Tamaño del título
        },
        xaxis_title="Año",
        yaxis_title="Número de Personas",
        xaxis=dict(
            tickvals=df["Año"].unique(),  # Mostrar años como enteros
            ticktext=[str(year) for year in df["Año"].unique()],
            range=[df["Año"].min() - 0.2, df["Año"].max() + 0.2],  # Extender el rango del eje X
        ),
        yaxis=dict(
            showgrid=True,  # Mostrar las líneas de la cuadrícula en el eje Y
            zeroline=False,  # No mostrar la línea del eje Y en cero
            range=yaxis_range,
            #tickformat=',',  # Mostrar números sin abreviaturas
        ),
        #annotations=[
            #dict(
            #    text="Datos obtenidos de la página del SNIS",
            #    xref="paper", yref="paper",
            #    x=0.5, y=-0.2,  # Ajustar posición según sea necesario
            #    showarrow=False,
            #    font=dict(size=11)
            #)
        #],
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title="Sexo",
        height=480,  # Ajustar la altura del gráfico
        width=900,  # Ajustar el ancho del gráfico para mayor visibilidad
    )
    # Habilitar líneas de cuadrícula
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1)

    return fig

def generate_graph_by_years(df, label, 
                                      title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    # Configuración de colores
    color = '#0123CB'
    
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
            'font': dict(size=size_title, weight='bold')
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
        #annotations=[
        #    dict(
        #        text=footer,
        #        xref="paper", yref="paper",
        #        x=0.5, y=-0.2,  # Ajustar posición según sea necesario
        #        showarrow=False,
        #        font=dict(size=size_footer)
        #    )
        #],
        width=800,  # Ancho de la gráfica
        height=500
    )

    return fig

def generate_stacked_bar_chart(df, labels, title, size_title, x, size_x, y, size_y, footer, size_footer, size_legend, size_graph):
    df_obesidad = df[0]
    df_sobrepeso = df[1]
    df_desnutricion = df[2]

    # Calcular el total de embarazadas por año
    df_obesidad['Total'] = df_obesidad['Obesidad'] + df_sobrepeso['Sobrepeso'] + df_desnutricion['Desnutricion']
    df_sobrepeso['Total'] = df_obesidad['Total']
    df_desnutricion['Total'] = df_obesidad['Total']

    # Calcular el porcentaje de cada categoría respecto al total del año
    df_obesidad['Porcentaje'] = (df_obesidad['Obesidad'] / df_obesidad['Total']) * 100
    df_sobrepeso['Porcentaje'] = (df_sobrepeso['Sobrepeso'] / df_sobrepeso['Total']) * 100
    df_desnutricion['Porcentaje'] = (df_desnutricion['Desnutricion'] / df_desnutricion['Total']) * 100

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
        marker_color='#00188D'  # Color para Obesidad
    ))

    fig.add_trace(go.Bar(
        x=df_plot.index,
        y=df_plot['Sobrepeso'],
        name='Sobrepeso',
        marker_color='#0123CB'  # Color para Sobrepeso
    ))

    fig.add_trace(go.Bar(
        x=df_plot.index,
        y=df_plot['Desnutrición'],
        name='Desnutrición',
        marker_color='#3859FF'  # Color para Desnutrición
    ))

    # Actualizar el diseño del gráfico
    fig.update_layout(
        title={
            'text': title + ' ' + labels, 
            'x': 0.5, 
            'xanchor': 'center',
            'font': dict(size=size_title, weight='bold')
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

'''
def generate_population_pyramid(df, region):
    # Filtrar los datos por la región especificada
    df_region = df[df['DEP_PROV_MUN'] == region]
    
    # Definir los grupos etarios en orden inverso
    grupos_etarios = ['80 o más', '70-79 años', '60-69 años', '50-59 años', '40-49 años', '30-39 años', '20-29 años', '10-19 años', '0-9 años']

    # Filtrar los datos por sexo
    df_male = df_region[df_region['Sexo'] == 'Hombre']
    df_female = df_region[df_region['Sexo'] == 'Mujer']

    # Calcular el total de la población por sexo
    total_male = df_male[grupos_etarios].sum().sum()
    total_female = df_female[grupos_etarios].sum().sum()

    # Calcular los porcentajes por grupo etario
    x_male = (df_male[grupos_etarios].sum() / total_male) * 100
    x_female = (df_female[grupos_etarios].sum() / total_female) * 100

    # Definir el valor máximo para los ticks del eje X
    max_value = max(x_male.max(), x_female.max())

    # Definir intervalos de ticks para diferentes rangos de datos
    tick_interval = 10

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
        text=[f'{value:.1f}%' for value in x_male],
        textposition='inside',
        texttemplate='%{text}',
        textfont=dict(color='white', size=18),
        insidetextanchor='middle',
        xaxis='x'
    ), row=1, col=1)

    # Añadir el gráfico de mujeres
    fig.add_trace(go.Bar(
        y=grupos_etarios,
        x=x_female,
        orientation='h',
        name='Mujeres',
        marker=dict(color=colors_female, line=dict(color='black', width=1)),
        text=[f'{value:.1f}%' for value in x_female],
        textposition='inside',
        texttemplate='%{text}',
        textfont=dict(size=20),
        insidetextanchor='middle'
    ), row=1, col=2)

    if region == 'Cordillera':
        title_text = 'Pirámide de Población por sexo según la provincia de'
    elif region == 'Santa Cruz':
        title_text = 'Pirámide de Población por sexo según el departamento de'
    else:
        title_text = 'Pirámide de Población por sexo según el municipio de'

    # Configurar el diseño de la gráfica
    fig.update_layout(
        title={
            'text': f'{title_text} {region}',  # Combinación de texto con negrita
            'x': 0.5,  # Centrar título
            'xanchor': 'center',
            'font': dict(size=20, weight='bold')  # Tamaño del título
        },
        xaxis_title='Porcentaje',
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
            title='Porcentaje',
            tickvals=np.arange(0, max_value + tick_interval, tick_interval),  # Ajustar los ticks del eje X
            ticktext=[f'{i}%' for i in np.arange(0, max_value + tick_interval, tick_interval)],
            range=[max_value, 0],  # Ajustar el rango del eje x para que vaya de max_value a 0
            gridcolor='gray',  # Color de las líneas de la cuadrícula en el eje Y
            gridwidth=0.5
        ),
        xaxis2=dict(
            title='Porcentaje',
            tickvals=np.arange(0, max_value + tick_interval, tick_interval),  # Ajustar los ticks del eje X para la segunda gráfica
            ticktext=[f'{i}%' for i in np.arange(0, max_value + tick_interval, tick_interval)],
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
'''

def generate_population_pyramid(df, region, title):
    # Filtrar los datos por la región especificada
    df_region = df[df['DEP_PROV_MUN'] == region]
    
    # Definir los grupos etarios en orden inverso
    grupos_etarios = ['80 o más', '70-79 años', '60-69 años', '50-59 años', '40-49 años', '30-39 años', '20-29 años', '10-19 años', '0-9 años']

    # Filtrar los datos por sexo
    df_male = df_region[df_region['Sexo'] == 'Hombre']
    df_female = df_region[df_region['Sexo'] == 'Mujer']

    # Calcular el total de la población por sexo
    total_male = df_male[grupos_etarios].sum().sum()
    total_female = df_female[grupos_etarios].sum().sum()

    # Calcular los porcentajes por grupo etario
    x_male = (df_male[grupos_etarios].sum() / total_male) * 100
    x_female = (df_female[grupos_etarios].sum() / total_female) * 100

    # Definir el valor máximo para los ticks del eje X
    max_value = max(x_male.max(), x_female.max())

    # Definir intervalos de ticks para diferentes rangos de datos
    tick_interval = 10

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
        text=[f'{value:.1f}%' for value in x_male],
        textposition='inside',
        texttemplate='%{text}',
        textfont=dict(color='white', size=18),
        insidetextanchor='middle',
        xaxis='x'
    ), row=1, col=1)

    # Añadir el gráfico de mujeres
    fig.add_trace(go.Bar(
        y=grupos_etarios,
        x=x_female,
        orientation='h',
        name='Mujeres',
        marker=dict(color=colors_female, line=dict(color='black', width=1)),
        text=[f'{value:.1f}%' for value in x_female],
        textposition='inside',
        texttemplate='%{text}',
        textfont=dict(size=20),
        insidetextanchor='middle'
    ), row=1, col=2)

    if region == 'Cordillera':
        title_text = 'Pirámide de Población por sexo según la provincia de'
    elif region == 'Santa Cruz':
        title_text = 'Pirámide de Población por sexo según el departamento de'
    else:
        title_text = 'Pirámide de Población por sexo según el municipio de'

    # Configurar el diseño de la gráfica
    fig.update_layout(
        title={
            'text': f'{title_text} {region}',  # Combinación de texto con negrita
            'x': 0.5,  # Centrar título
            'xanchor': 'center',
            'font': dict(size=20, weight='bold')  # Tamaño del título
        },
        xaxis_title='Porcentaje',
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
            title='Porcentaje',
            tickvals=np.arange(0, max_value + tick_interval, tick_interval),  # Ajustar los ticks del eje X
            ticktext=[f'{i}%' for i in np.arange(0, max_value + tick_interval, tick_interval)],
            range=[max_value, 0],  # Ajustar el rango del eje x para que vaya de max_value a 0
            gridcolor='gray',  # Color de las líneas de la cuadrícula en el eje Y
            gridwidth=0.5
        ),
        xaxis2=dict(
            title='Porcentaje',
            tickvals=np.arange(0, max_value + tick_interval, tick_interval),  # Ajustar los ticks del eje X para la segunda gráfica
            ticktext=[f'{i}%' for i in np.arange(0, max_value + tick_interval, tick_interval)],
            gridcolor='gray',  # Color de las líneas de la cuadrícula en el eje Y
            gridwidth=0.5
        ),
        plot_bgcolor='white',  # Fondo del área de la gráfica
        paper_bgcolor='white',  # Fondo del área de la gráfica completa
        showlegend=True,
        legend=dict(x=0.8, y=1),
        autosize=True,
        width=1200,  # Ancho total de la gráfica
        height=650  # Alto de la gráfica
    )
    return fig


def generate_language_donut_chart(df, regions, colors):
    num_charts = len(regions)
    
    # Determinar el número de columnas según el número de regiones
    if num_charts == 1:
        num_columns = 1
    elif 2 <= num_charts <= 4:
        num_columns = 2
    elif 5 <= num_charts <= 9:
        num_columns = 3
    else:
        num_columns = 4  # Puedes ajustar este valor si necesitas más columnas

    num_rows = math.ceil(num_charts / num_columns)  # Calcular el número de filas necesarias

    # Determinar los subtítulos basados en la región
    subtitles = []
    for region in regions:
        if region in ['Santa Cruz', 'La Paz']:
            subtitle = 'DEPARTAMENTO'
        elif region in ['Cordillera', 'Sud Yungas']:
            subtitle = 'PROVINCIA'
        else:
            subtitle = 'MUNICIPIO'
        subtitles.append(subtitle)

    # Crear la figura con subplots
    fig = make_subplots(
        rows=num_rows, cols=num_columns,
        subplot_titles=subtitles,
        specs=[[{'type': 'domain'}] * num_columns for _ in range(num_rows)],  # Tipo de gráfico para donuts
        vertical_spacing=(1/num_rows)*0.1,  # Reducir el espacio vertical entre las filas
        horizontal_spacing=0.06
    )
    
    for i, region in enumerate(regions):
        row = df[df["Dep_Prov_Mun"] == region].iloc[0]
        values = row[1:].values
        labels = df.columns[1:].values
        
        # Crear gráfico de donut para la región actual
        donut_chart = go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors, line=dict(color='black', width=0.5)),
            textinfo='percent',
            textfont=dict(size=24),
            name=region,
            hole=.5
        )
    
        # Determinar la fila y columna para la región actual
        row_num = i // num_columns + 1
        col_num = i % num_columns + 1
        
        # Añadir el gráfico a la figura
        fig.add_trace(donut_chart, row=row_num, col=col_num)

        # Crear gráfico adicional con el nombre de la región en el centro
        text_chart = go.Pie(
            labels=[''],  # Vacío para que solo se muestre el texto
            values=[1],  # Valor arbitrario para mostrar el texto
            text=[region],
            textinfo='text',  # Mostrar solo el texto
            textfont=dict(size=25, color='black'),
            marker=dict(colors=['rgba(0,0,0,0)']),  # Hacer el gráfico transparente
            showlegend=False,  # No mostrar leyenda
            name=''
        )
    
        # Añadir el gráfico de texto a la figura
        fig.add_trace(text_chart, row=row_num, col=col_num)

    # Configurar el diseño de la figura
    fig.update_layout(
        title={
            'text': 'Porcentaje de Idiomas Principal que habla la población de 6 años a más',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=21, weight='bold')  # Ajustar el tamaño del título aquí
        },
        legend=dict(
            font=dict(size=11),
            orientation='v',
            x=1.10,
            y=0.9
        ),
        height=500 * num_rows,  # Ajustar el alto según el número de filas
        width=1250,  # Ajustar el ancho de la figura para dos columnas
    )
    
    # Ajustar el tamaño de los subtítulos de los subplots
    for i, region in enumerate(regions):
        row_num = i // num_columns + 1
        col_num = i % num_columns + 1
        fig.layout.annotations[i].update(
            font=dict(size=20)  # Ajustar el tamaño del texto de los subtítulos aquí
        )

    return fig

def generate_literacy_donut_chart(df, regions):
    # Definir número de columnas y filas necesarias
    num_columns = 4
    num_rows = math.ceil(len(regions) / 2)  # Dos pares de gráficos por fila

    # Crear la figura con subplots
    fig = make_subplots(
        rows=num_rows, cols=num_columns,
        specs=[
            [{'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}] 
            for _ in range(num_rows)
        ],
        subplot_titles=[None] * num_columns * num_rows,
        vertical_spacing=0,  # Reducir el espacio vertical entre las filas
        horizontal_spacing=0.02
    )

    for i, region in enumerate(regions):
        row = df[df["Dep_Prov_Mun"] == region].iloc[0]
        
        # Datos para hombres
        values_hombre = [row["Hombre"], row["Hombre_Illiterate"]]
        labels_hombre = ["Alfabetizados", "Analfabetos"]
        colors_hombre = ["#99AAFF", "#011CA2"]  # Colores para alfabetizados y analfabetos

        # Datos para mujeres
        values_mujer = [row["Mujer"], row["Mujer_Illiterate"]]
        labels_mujer = ["Alfabetizadas", "Analfabetas"]
        colors_mujer = ["#FFCDA6", "#E05B2A"]  # Colores para alfabetizadas y analfabetas

        # Determinar fila y columna para los gráficos
        row_num = i // 2 + 1  # Una fila por par de gráficos
        col_num_hombre = (i % 2) * 2 + 1  # Primera columna para hombres
        col_num_mujer = (i % 2) * 2 + 2  # Segunda columna para mujeres

        # Añadir gráficos de hombres
        fig.add_trace(
            go.Pie(
                labels=labels_hombre,
                values=values_hombre,
                name=f"Hombres - {region}",
                marker=dict(colors=colors_hombre, line=dict(color='black', width=0.5)),
                hole=.5,
                textinfo='percent',
                textfont=dict(size=22),
            ),
            row=row_num, col=col_num_hombre
        )

        # Crear gráfico adicional con el nombre "Hombre" en el centro
        fig.add_trace(
            go.Pie(
                labels=[''],  # Vacío para que solo se muestre el texto
                values=[1],  # Valor arbitrario para mostrar el texto
                text=['Hombres'],
                textinfo='text',
                textfont=dict(size=20, color='black'),
                #hole=.5,
                marker=dict(colors=['rgba(0,0,0,0)']),  # Hacer el gráfico transparente
                showlegend=False,  # No mostrar leyenda
                name=''
            ),
            row=row_num, col=col_num_hombre
        )

        # Añadir gráficos de mujeres
        fig.add_trace(
            go.Pie(
                labels=labels_mujer,
                values=values_mujer,
                name=f"Mujeres - {region}",
                marker=dict(colors=colors_mujer, line=dict(color='black', width=0.5)),
                hole=.5,
                textinfo='percent',
                textfont=dict(size=22),
            ),
            row=row_num, col=col_num_mujer
        )

        # Crear gráfico adicional con el nombre "Mujer" en el centro
        fig.add_trace(
            go.Pie(
                labels=[''],  # Vacío para que solo se muestre el texto
                values=[1],  # Valor arbitrario para mostrar el texto
                text=['Mujeres'],
                textinfo='text',
                textfont=dict(size=20, color='black'),
                #hole=.5,
                marker=dict(colors=['rgba(0,0,0,0)']),  # Hacer el gráfico transparente
                showlegend=False,  # No mostrar leyenda
                name=''
            ),
            row=row_num, col=col_num_mujer
        )

        if region in ['Santa Cruz', 'La Paz']:
            sub_title = 'DEPARTAMENTO'
        elif region in ['Cordillera', 'Sud Yungas']:
            sub_title = 'PROVINCIA'
        else:
            sub_title = 'MUNICIPIO'

        if col_num_hombre == 1 or col_num_mujer == 2:
            dato=(((col_num_hombre + col_num_mujer)) / 8) - 0.25
        else:
            dato=(((col_num_hombre + col_num_mujer)) / 8) - 0.02
        fig.add_annotation(
            text=f'{sub_title} - {region}',
            x=dato,  # Centrar entre col_num_hombre y col_num_mujer
            y=1-((row_num / num_rows) - (1/num_rows)),  # Justo encima de los gráficos
            showarrow=False,
            font=dict(size=22, color='black'),
            xref="paper",
            yref="paper",
            align="center"
        )

    # Configurar el diseño de la gráfica
    fig.update_layout(
        title={
            'text': 'Porcentaje de Analfabetismo por Sexo',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=30, weight='bold')
        },
        legend=dict(
            font=dict(size=15),  # Tamaño de la letra de la leyenda
            orientation='v',  # Orientación horizontal de la leyenda
            x=1.05,  # Posición horizontal de la leyenda (a la derecha del gráfico)
            y=0.9  # Posición vertical de la leyenda (centrado verticalmente)
        ),
        height=400 * num_rows,  # Ajustar el alto de la gráfica
        width=1200  # Ajustar el ancho de la gráfica
    )

    return fig

def generate_services_bar_chart(df, regions):
    # Fijar el número de columnas a 2
    num_cols = 2
    # Calcular el número de filas necesarias
    num_rows = math.ceil(len(regions) / num_cols)
    
    # Crear subplots dinámicos
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[f'{region}' for region in regions],  # Títulos para cada gráfico
        vertical_spacing=1/(num_rows**2),
    )

    for i, region in enumerate(regions):
        row = df[df["Dep_Prov_Mun"] == region].iloc[0]
        servicios = row.index[1:]
        porcentajes = row.values[1:]

        # Calcular los porcentajes restantes
        porcentajes_restantes = 100 - porcentajes

        # Determinar la posición del gráfico en los subplots
        row_num = (i // num_cols) + 1
        col_num = (i % num_cols) + 1

        # Añadir las barras de "Sin Servicio"
        fig.add_trace(go.Bar(
            x=servicios,
            y=porcentajes_restantes,
            name='Sin Servicio',
            marker_color='#0123CB',
            showlegend=(i == 0)  # Mostrar leyenda solo en la primera gráfica
        ), row=row_num, col=col_num)

        # Añadir las barras de "Con Servicio"
        fig.add_trace(go.Bar(
            x=servicios,
            y=porcentajes,
            name='Con Servicio',
            marker_color='lightgrey',
            showlegend=(i == 0)  # Mostrar leyenda solo en la primera gráfica
        ), row=row_num, col=col_num)

        # Añadir los porcentajes en las barras
        for j, service in enumerate(servicios):
            fig.add_annotation(
                x=service,
                y=porcentajes_restantes[j] / 2,
                text=f'{porcentajes_restantes[j]:.1f}%',
                showarrow=False,
                font=dict(size=18, color='white'),
                align='right',
                xanchor='center',
                yanchor='middle',
                row=row_num,
                col=col_num
            )
            fig.add_annotation(
                x=service,
                y=porcentajes_restantes[j] + porcentajes[j] / 2,
                text=f'{porcentajes[j]:.1f}%',
                showarrow=False,
                font=dict(size=18, color='black'),
                align='left',
                xanchor='center',
                yanchor='middle',
                row=row_num,
                col=col_num
            )

    # Configuración del layout
    fig.update_layout(
        title={
            'text': 'Acceso a servicios básicos según provincia o municipios',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, weight='bold')
        },
        barmode='stack',
        height=400 * num_rows,  # Ajustar la altura en función del número de filas
        width=1200,  # Ancho total del gráfico
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True,
    )

    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))  # Rotar y ajustar tamaño de etiquetas en eje X

    return fig

def generate_housing_pie_chart(df, regions):
    num_charts = len(regions)
    
    # Determine the number of columns based on the number of charts
    if num_charts == 1:
        num_columns = 1
    elif 2 <= num_charts <= 4:
        num_columns = 2
    elif 5 <= num_charts <= 9:
        num_columns = 3
    else:
        num_columns = 1  # Fallback for unexpected counts
    
    num_rows = math.ceil(num_charts / num_columns)  # Calculate the number of rows needed
    colors = ['#AB63FA', '#636EFA', '#FFA15A', '#EF553B', '#00CC96']  # Colors for each category

    # Determine subtitles based on the region
    subtitles = []
    for region in regions:
        if region in ['Santa Cruz', 'La Paz']:
            subtitle = 'DEPARTAMENTO'
        elif region in ['Cordillera', 'Sud Yungas']:
            subtitle = 'PROVINCIA'
        else:
            subtitle = 'MUNICIPIO'
        subtitles.append(subtitle)

    # Create the figure with subplots
    fig = make_subplots(
        rows=num_rows, cols=num_columns,
        subplot_titles=subtitles,
        specs=[[{'type': 'domain'}] * num_columns for _ in range(num_rows)],  # Type of chart for donuts
        vertical_spacing=(1/num_rows)*0.1,  # Reduce vertical space between rows
        horizontal_spacing=0.03
    )
    
    for i, region in enumerate(regions):
        # Check if the region is in the DataFrame
        if region in df["Dep_Prov_Mun"].values:
            # Extract the corresponding row for the region
            row = df[df["Dep_Prov_Mun"] == region].iloc[0]
            values = row[1:-1].values  # Values from columns of interest (excluding Dep_Prov_Mun and Total)
            labels = df.columns[1:-1]  # Names of the columns of interest
            
            # Create donut chart for the current region
            donut_chart = go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors, line=dict(color='black', width=0.5)),
                textinfo='percent',
                texttemplate='%{percent:.2%}',  # Format percentages with two decimal places
                textfont=dict(size=25),
                name=region,
                hole=.5
            )
        
            # Determine the row and column for the current region
            row_num = i // num_columns + 1
            col_num = i % num_columns + 1
            
            # Add the donut chart to the figure
            fig.add_trace(donut_chart, row=row_num, col=col_num)
            
            # Create additional chart with the region name in the center
            text_chart = go.Pie(
                labels=[''],  # Empty to only show text
                values=[1],  # Arbitrary value to show the text
                text=[region],
                textinfo='text',  # Show only the text
                textfont=dict(size=20, color='black'),
                marker=dict(colors=['rgba(0,0,0,0)']),  # Make the chart transparent
                showlegend=False,  # Don't show legend
                name=''
            )
        
            # Add the text chart to the figure
            fig.add_trace(text_chart, row=row_num, col=col_num)
        else:
            # Show a message if the region is not in the DataFrame
            print(f"Región '{region}' no encontrada en el DataFrame.")
    
    # Configure the layout of the figure
    fig.update_layout(
        title={
            'text': 'Porcentaje de Número de Dormitorios en Viviendas Particulares Ocupadas',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=21, weight='bold')
        },
        legend=dict(
            font=dict(size=14),
            orientation='v',
            x=1.05,
            y=0.5
        ),
        height=500 * num_rows,  # Adjust height based on the number of rows
        width=1250,  # Adjust width for two columns
    )
    
    # Adjust the subtitle size of the subplots
    for i in range(num_charts):
        fig.layout.annotations[i].update(
            font=dict(size=15)  # Adjust the size of the subtitle text here
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
            marker_color='#0123CB' if gender == 'Hombre' else '#E06C00',  # Colores
            text=[f'{v:.2f}%' for v in values],  # Valores sobre las barras
            textposition='auto',  # Posición del texto sobre las barras
            textfont=dict(size=20),  # Tamaño y color del texto sobre las barras
            marker=dict(
                line=dict(
                    color='black',  # Color del borde
                    width=0.5       # Ancho del borde
                )
            )
        ))

    if region == 'Cordillera':
        title_text = 'Porcentaje de categoría ocupacional por municipio por sexo según provincia de'
    elif region == 'Santa Cruz':
        title_text = 'Porcentaje de categoría ocupacional por municipio por sexo según departamento de'
    else:
        title_text = 'Porcentaje de categoría ocupacional por municipio por sexo según municipio de'

    fig.update_layout(
        barmode='group',  # Modo de barras agrupadas
        title={
            'text': f'{title_text} {region}',
            'x': 0.5,  # Centrar título
            'xanchor': 'center',
            'font': dict(size=18, weight='bold')
        },
        xaxis_title='Ocupación',
        yaxis_title='Porcentaje (%)',
        legend_title='Sexo',
        plot_bgcolor='white',  # Fondo blanco de la gráfica
        paper_bgcolor='white',  # Fondo blanco del área de trabajo
        yaxis=dict(
            gridcolor='lightgray'  # Líneas del eje y grises
        ),
        #annotations=[{
            #'text': 'Fuente: Datos del Censo 2012',
            #'xref': 'paper',
            #'yref': 'paper',
            #'x': 0.5,
            #'y': -0.45,
            #'showarrow': False,
            #'font': dict(size=12, color='gray'),
            #'xanchor': 'center'
        #}],
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
        color = '#0123CB' if gender == 'Hombre' else '#E05B2A'
        
        fig.add_trace(go.Scatter(
            x=subset['Año'],  # Eje x con años
            y=subset['Secundaria'],  # Eje y con tasa de abandono en secundaria
            mode='lines+markers+text',  # Mostrar líneas y puntos
            name=gender,
            line=dict(color=color, width=2),  # Colores y ancho de línea
            marker=dict(size=8),  # Tamaño de los puntos
            text=[f'{v:.2f}%' for v in subset['Secundaria']],  # Valores sobre los puntos
            textposition='top center',  # Posición del texto
            textfont=dict(
                color=color,  # Color del texto igual al de la línea
                size=20  # Tamaño del texto
            )
        ))

    if region == 'Cordillera':
        title_text = 'Tasa de Abandono a Nivel Secundario por sexo según provincia de'
    elif region == 'Santa Cruz':
        title_text = 'Tasa de Abandono a Nivel Secundario por sexo según departamento de'
    else:
        title_text = 'Tasa de Abandono a Nivel Secundario por sexo según municipio de'

    fig.update_layout(
        title={
            'text': f'{title_text} {region}',
            'x': 0.5,  # Centrar título
            'xanchor': 'center',
            'font': dict(size=18, weight='bold')
        },
        xaxis_title='Año',
        yaxis_title='Tasa de Abandono en Secundaria',
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
            html.Li(dcc.Link('Pentavalentes', href='/pentavalentes')),
            html.Li(dcc.Link('Nutrición', href='/nutricion')),
            html.Li(dcc.Link('Embarazo Adolescente', href='/embarazo')),
            html.Li(dcc.Link('Partos', href='/partos')),
            html.Li(dcc.Link('Consultas Externas', href='/consultas')),
        ], className='menu')
    ], className='menu-column'),
    html.Div([
        html.Div(id='page-content')
    ], className='content-column'),
    html.Div(id='btn-calcular', style={'display': 'none'}),  # Div oculto para generar el botón
], className='container')

# Diccionario de opciones de departamentos y municipios
opciones_dataframes = {
    'Santa Cruz': {'provincias': {'Cordillera': ['Camiri', 'Gutierrez', 'Lagunillas', 'Cuevo']}},
    'La Paz': {'provincias': {'Sud Yungas': ['La Asunta', 'Palos Blancos', 'Chulumani', 'Irupana', 'Yanacachi']}},
}

def generate_calculo_layout_salud(title):
    opciones_dataframes_departamento = [{'label': k, 'value': k} for k in opciones_dataframes.keys()]

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
        html.H3("Departamento"),
        html.Div([
            html.Div([
                html.Label('Gráfico:'),
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
                html.Label('Gráfico:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-prov',
                    #options=opciones_dataframes_provincia,
                    multi=True,
                    value=['Cordillera']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-prov',
                    #options=opciones_dataframes_provincia,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'visibility': 'hidden'})
        ]),
        html.H3("Municipios"),
        html.Div([
            html.Div([
                html.Label('Gráfico:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-mun',
                    #options=opciones_dataframes_municipio,
                    multi=True,
                    value=['Camiri', 'Gutierrez', 'Lagunillas']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-mun',
                    #options=opciones_dataframes_municipio,
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
        html.Div(id='output-data-salud'),
        html.Button('Generar Gráfico', id='btn-calcular-salud'),
        html.Div(id='output-data-salud')
    ])

def generate_calculo_layout(title):
    '''opciones_dataframes_departamento = [
        {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
    ]
    opciones_dataframes_provincia = [
        {'label': 'Cordillera', 'value': 'Cordillera'},
    ]
    opciones_dataframes_municipio = [
        {'label': 'Camiri', 'value': 'Camiri'},
        {'label': 'Gutierrez', 'value': 'Gutierrez'},
        {'label': 'Lagunillas', 'value': 'Lagunillas'},
        {'label': 'Cuevo', 'value': 'Cuevo'},
    ]'''
    opciones_dataframes_departamento = [{'label': k, 'value': k} for k in opciones_dataframes.keys()]
    
    # Crear las opciones de departamentos
    #opciones_dataframes_departamento = [{'label': k, 'value': k} for k in opciones_dataframes.keys()]
    # Opciones predeterminadas para Provincia y Municipio
    #opciones_dataframes_provincia = [{'label': prov, 'value': prov} for prov in opciones_dataframes['Santa Cruz']['provincias'].keys()]
    #opciones_dataframes_municipio = [{'label': mun, 'value': mun} for mun in opciones_dataframes['Santa Cruz']['provincias']['Cordillera']]
    
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
                    #options=opciones_dataframes_provincia,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-prov',
                    #options=opciones_dataframes_provincia,
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
                    #options=opciones_dataframes_municipio,
                    multi=True,
                    value=['Camiri', 'Gutierrez', 'Lagunillas']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-mun',
                    #options=opciones_dataframes_municipio,
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
    #opciones_dataframes_departamento = [
    #    {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
    #]
    #opciones_dataframes_provincia = [
    #    {'label': 'Cordillera', 'value': 'Cordillera'},
    #]
    #opciones_dataframes_municipio = [
    #    {'label': 'Camiri', 'value': 'Camiri'},
    #    {'label': 'Gutierrez', 'value': 'Gutierrez'},
    #    {'label': 'Lagunillas', 'value': 'Lagunillas'},
    #    {'label': 'Cuevo', 'value': 'Cuevo'}
    #]
    opciones_dataframes_departamento = [{'label': k, 'value': k} for k in opciones_dataframes.keys()]
    
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
                    #options=opciones_dataframes_provincia,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-prov',
                    #options=opciones_dataframes_provincia,
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
                    #options=opciones_dataframes_municipio,
                    multi=True,
                    value=['Camiri', 'Gutierrez', 'Lagunillas']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-mun',
                    #options=opciones_dataframes_municipio,
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
    '''
    opciones_dataframes_departamento = [
        {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
    ]
    opciones_dataframes_provincia = [
        {'label': 'Cordillera', 'value': 'Cordillera'},
    ]
    opciones_dataframes_municipio = [
        {'label': 'Camiri', 'value': 'Camiri'},
        {'label': 'Gutierrez', 'value': 'Gutierrez'},
        {'label': 'Lagunillas', 'value': 'Lagunillas'},
        {'label': 'Cuevo', 'value': 'Cuevo'}
    ]
    '''
    opciones_dataframes_departamento = [{'label': k, 'value': k} for k in opciones_dataframes.keys()]
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
                    #options=opciones_dataframes_provincia,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-prov',
                    #options=opciones_dataframes_provincia,
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
                    #options=opciones_dataframes_municipio,
                    multi=True,
                    value=['Camiri', 'Gutierrez', 'Lagunillas']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-mun',
                    #options=opciones_dataframes_municipio,
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
    '''
    opciones_dataframes_departamento = [
        {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
    ]
    opciones_dataframes_provincia = [
        {'label': 'Cordillera', 'value': 'Cordillera'},
    ]
    opciones_dataframes_municipio = [
        {'label': 'Camiri', 'value': 'Camiri'},
        {'label': 'Gutierrez', 'value': 'Gutierrez'},
        {'label': 'Lagunillas', 'value': 'Lagunillas'},
        {'label': 'Cuevo', 'value': 'Cuevo'},
    ]
    '''
    opciones_dataframes_departamento = [{'label': k, 'value': k} for k in opciones_dataframes.keys()]
    
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
                html.Label('Gráfico:'),
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
                html.Label('Gráfico:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-prov',
                    #options=opciones_dataframes_provincia,
                    multi=True,
                    value=['Cordillera']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-prov',
                    #options=opciones_dataframes_provincia,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%', 'visibility': 'hidden'})
        ]),
        html.H3("Municipios"),
        html.Div([
            html.Div([
                html.Label('Gráfico:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-bar-mun',
                    #options=opciones_dataframes_municipio,
                    multi=True,
                    value=['Camiri', 'Gutierrez', 'Lagunillas']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-mun',
                    #options=opciones_dataframes_municipio,
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

def generate_calculo_layout_partos(title):
    '''
    opciones_dataframes_departamento = [
        {'label': 'Santa Cruz', 'value': 'Santa Cruz'}
    ]
    opciones_dataframes_provincia = [
        {'label': 'Cordillera', 'value': 'Cordillera'},
    ]
    opciones_dataframes_municipio = [
        {'label': 'Camiri', 'value': 'Camiri'},
        {'label': 'Gutierrez', 'value': 'Gutierrez'},
        {'label': 'Lagunillas', 'value': 'Lagunillas'},
        {'label': 'Cuevo', 'value': 'Cuevo'}
    ]'''
    opciones_dataframes_departamento = [{'label': k, 'value': k} for k in opciones_dataframes.keys()]
    return html.Div([
        html.H1("Gráficos de Tendencia"),
        html.Div([
            html.Span('Factor'),
            dcc.Input(id='input-factor', type='number', value=10000, style={'width': '80px'})
        ]),
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
                    #options=opciones_dataframes_provincia,
                    multi=True,
                    value=['']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-prov',
                    #options=opciones_dataframes_provincia,
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
                    #options=opciones_dataframes_municipio,
                    multi=True,
                    value=['Camiri', 'Gutierrez', 'Lagunillas']
                )
            ], style={'display': 'inline-block', 'width': '45%'}),
            html.Div([
                html.Label('Gráfico de Tendencias:'),
                dcc.Dropdown(
                    id='dropdown-dataframes-ten-mun',
                    #options=opciones_dataframes_municipio,
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
        html.Button('Generar Gráfico', id='btn-calcular-partos'),
        html.Div(id='output-data-partos')
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
            generate_calculo_layout('Casos nuevos de Diabetes por año según nivel Departamental, Provincial y Municipal')
        ])
    if pathname == '/hipertension':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Hipertensión Arterial'),
            generate_calculo_layout('Casos nuevos de Hipertension por año según nivel Departamental, Provincial y Municipal')
        ])
    if pathname == '/obesidad':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Obesidad'),
            generate_calculo_layout('Casos nuevos de Obesidad por año según nivel Departamental, Provincial y Municipal')
        ])
    if pathname == '/neumonia':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Neumonía'),
            generate_calculo_layout('Casos nuevos de Neumonia por año según nivel Departamental, Provincial y Municipal')
        ])
    if pathname == '/chagas':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Chagas'),
            generate_calculo_layout('Casos nuevos de Chagas por año según nivel Departamental, Provincial y Municipal')
        ])
    if pathname == '/vih':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos VIH'),
            generate_calculo_layout('Casos nuevos de VIH por año según nivel Departamental, Provincial y Municipal')
        ])
    elif pathname == '/nutricion':        
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Nutricion'),
            generate_calculo_layout_nutricion('Casos nuevos de Estado Nutricional en Embarazadas por año según nivel Departamental, Provincial y Municipal')
        ])
    elif pathname == '/embarazo':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Embarazo Adolescente'),
            generate_calculo_layout_embarazo('Casos de Embarazadas Adolescentes por año según nivel Departamental, Provincial y Municipal')
        ])
    elif pathname == '/consultas':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Datos Consultas Externas'),
            generate_calculo_layout_consultas('Total de consultas por sexo y año según nivel Departamental, Provincial y Municipal')
        ])
    elif pathname == '/salud':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Salud'),
            generate_calculo_layout_salud('Datos de salud')
        ])
    elif pathname == '/partos':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Partos'),
            generate_calculo_layout_partos('Datos de partos')
        ])
    elif pathname == '/pentavalentes':
        return html.Div([
            html.H1('Recolección de datos - Análisis de Vacunas Pentavalentes 3ra dosis'),
            generate_calculo_layout('Vacunas pentavalentes aplicadas a menores de 1 año por año según nivel Departamental, Provincial y Municipal (3ra Dosis)')
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

            if 'Santa Cruz' in dfl_total:
                municipios = ['Camiri', 'Gutierrez', 'Lagunillas', 'Cuevo', 'Cordillera', 'Santa Cruz']
                etnicidad = 'guarani'
            else:
                municipios = ['La Asunta', 'Palos Blancos', 'Chulumani', 'Irupana', 'Yanacachi', 'Sud Yungas', 'La Paz']
                etnicidad = 'afroboliviano'

            # Obtener datos de todos los municipios dinámicamente
            dataframes = {}
            dataframes_total = {}
            dataframes_age = {}
            dataframes_age_total = {}

            df = get_casos(partes[1], etnicidad)  # Supongo que la función devuelve una lista con múltiples DataFrames
            p_casos = get_casos('poblacion', etnicidad)  # Población para cálculo

            total = ('Total') if partes[1] != 'pentavalentes' else ('3ra Dosis')

            for idx, municipio in enumerate(municipios):
                # Asignar dinámicamente DataFrames y totales
                df_actual = df[idx]  # El índice coincide con el municipio en la lista
                p_municipio = p_casos[idx]
                m = p_municipio[p_municipio['Sexo'] == 'Mujer']['Total'].tolist()
                h = p_municipio[p_municipio['Sexo'] == 'Hombre']['Total'].tolist()
                p = p_municipio.groupby('Año')['Total'].sum().tolist()

                df_total = generate_total(df_actual)
                
                df_total = calculate_total(df_total, factor, p, total)

                # Calcular edad y género si aplica
                df_gender = calculate_gender(df_actual, factor, m, h, total)
                # Asignar al diccionario
                dataframes[municipio] = df_gender
                dataframes_total[municipio] = df_total

                if partes[1] != 'pentavalentes' and partes[1] != 'vih':
                    df_age = calculate_age(df_actual, p_municipio, partes[1])
                    df_age_total = calculate_age_total(df_age, partes[1])
                    dataframes_age[municipio] = df_age
                    dataframes_age_total[municipio] = df_age_total
            
            df_barras_total = [dataframes_total[nombre] for nombre in dfl_barras if nombre in dataframes_total]
            df_tendencias_total = [dataframes_total[nombre] for nombre in dfl_tendencias if nombre in dataframes_total]

            df_barras = [dataframes[nombre] for nombre in dfl_barras if nombre in dataframes]
            df_tendencias = [dataframes[nombre] for nombre in dfl_tendencias if nombre in dataframes]

            df_total = [dataframes[nombre] for nombre in dfl_total if nombre in dataframes]
            
            if partes[1] != 'vih' and partes[1] != 'pentavalentes':
                df_total_year = [dataframes_age[nombre] for nombre in dfl_total if nombre in dataframes_age]
                df_total_year_t = [dataframes_age_total[nombre] for nombre in dfl_total if nombre in dataframes_age_total]

            if n_clicks > 0:
                if partes[1] != 'vih' and partes[1] != 'pentavalentes':
                    age_columns = []
                    if type_percent == 'Total':
                        if partes[1] == "neumonia":
                            age_columns = ['0-1', '1-4', '5-9', '10-19', '20-39', '40-49', '50-59', '60+']
                        elif partes[1] == "diabetes" or partes[1] == "hipertension":
                            age_columns = ['0-19', '20-39', '40-49', '50-59', '60+']
                        else:
                            age_columns = ['0-9', '10-19', '20-39', '40-49', '50-59', '60+']
                    elif type_percent == 'Incidencia':
                        if partes[1] == "neumonia":
                            age_columns = ['I_0-1', 'I_1-4', 'I_5-9', 'I_10-19', 'I_20-39', 'I_40-49', 'I_50-59', 'I_60+']
                        elif partes[1] == "diabetes" or partes[1] == "hipertension":
                            age_columns = ['I_0-19', 'I_20-39', 'I_40-49', 'I_50-59', 'I_60+']
                        else:
                            age_columns = ['I_0-9', 'I_10-19', 'I_20-39', 'I_40-49', 'I_50-59', 'I_60+']
                    elif type_percent == 'Porcentaje':
                        if partes[1] == "neumonia":
                            age_columns = ['% 0-1', '% 1-4', '% 5-9', '% 10-19', '% 20-39', '% 40-49', '% 50-59', '% 60+']
                        elif partes[1] == "diabetes" or partes[1] == "hipertension":
                            age_columns = ['% 0-19', '% 20-39', '% 40-49', '% 50-59', '% 60+']
                        else:
                            age_columns = ['% 0-9', '% 10-19', '% 20-39', '% 40-49', '% 50-59', '% 60+']
                    
                    specific_labels = dataframes_bar_mun + dataframes_ten_mun

                    max_y = 0
                    for i, df in enumerate(df_total_year_t):
                        if dfl_total[i] in specific_labels:
                            max_y = max(max_y, df[age_columns].max().max())

                if graphic_type == 't':
                    fig=generate_graph_total(df_barras_total, df_tendencias_total, dfl_barras, dfl_tendencias,
                                                titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    resultados.append(dcc.Graph(id='mi-grafico-total', figure=fig))
                elif graphic_type == 's1':
                    fig=generate_graph_join_gender(df_barras, df_tendencias, dfl_barras, dfl_tendencias,
                                                titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    resultados.append(dcc.Graph(id='mi-grafico-sexo-1', figure=fig))
                elif graphic_type == 's2':
                    if len(df_tendencias) >= len(df_barras):
                        fig = generate_graph_separate_gender(df_total, "tendencias", dfl_total, 
                                    titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    else:
                        fig = generate_graph_separate_gender(df_total, "barras", dfl_total, 
                                    titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    resultados.append(dcc.Graph(id='mi-grafico-sexo-2', figure=fig))
                elif graphic_type == 'e1':
                    for i, df in enumerate(df_total_year):
                        fig = generate_graph_separate_age(df, partes[1], dfl_total[i], 
                                                        titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                        pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                        resultados.append(dcc.Graph(id=f'mi-grafico-edad-{i}', figure=fig))
                elif graphic_type == 'e2':
                    for i, df in enumerate(df_total_year_t):
                        if dfl_total[i] in specific_labels:
                            fig = generate_graph_join_age(df, age_columns, dfl_total[i], 
                                                        titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                        pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica, max_y)
                        else:                           
                            fig = generate_graph_join_age(df, age_columns, dfl_total[i], 
                                                        titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                        pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                        resultados.append(dcc.Graph(id=f'mi-grafico-edad-total-{i}', figure=fig))
                else:
                    return html.Div("")

                return resultados
            
            return html.Div("")

        except Exception as e:
            return html.Div(f'Error: {e}')
        

@app.callback(
    Output('output-data-partos', 'children'),
    [
        Input('btn-calcular-partos', 'n_clicks'),
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

def update_output_partos(n_clicks, type_percent, 
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
            tamanio_titulo = int(tamanio_titulo) if tamanio_titulo is not None else 0
            tamanio_eje_x = int(tamanio_eje_x) if tamanio_eje_x is not None else 0
            tamanio_eje_y = int(tamanio_eje_y) if tamanio_eje_y is not None else 0
            tamanio_pie = int(tamanio_pie) if tamanio_pie is not None else 0
            tamanio_leyenda = int(tamanio_leyenda) if tamanio_leyenda is not None else 0
            tamanio_num_grafica = int(tamanio_num_grafica) if tamanio_num_grafica is not None else 0

            dfl_barras = dataframes_bar_dep + dataframes_bar_prov + dataframes_bar_mun
            dfl_tendencias = dataframes_ten_dep + dataframes_ten_prov + dataframes_ten_mun
            dfl_total = dataframes_bar_dep + dataframes_ten_dep + dataframes_bar_prov + dataframes_ten_prov + dataframes_bar_mun + dataframes_ten_mun
            dfl_barras = [item for item in dfl_barras if item]
            dfl_tendencias = [item for item in dfl_tendencias if item]
            dfl_total = [item for item in dfl_total if item]

            resultados = []
            partes = pathname.split('/')
            resultados.append(html.H2(f'Gráfico '+partes[1]))

            # Municipios y nombres de los DataFrames asociados
            if 'Santa Cruz' in dfl_total:
                municipios = ['Camiri', 'Gutierrez', 'Lagunillas', 'Cuevo', 'Cordillera', 'Santa Cruz']
                etnicidad = 'guarani'
            else:
                municipios = ['La Asunta', 'Palos Blancos', 'Chulumani', 'Irupana', 'Yanacachi', 'Sud Yungas', 'La Paz']
                etnicidad = 'afroboliviano'

            dataframes = {}
            dataframes_total = {}

            # Obtener todos los DataFrames y poblaciones de una vez
            df = get_casos(partes[1], etnicidad)  # Asumiendo que devuelve una lista de DataFrames
            p_casos = get_casos('poblacion', etnicidad)  # Asumiendo que devuelve una lista de poblaciones
            
            if len(dfl_total) != 0:
                for idx, municipio in enumerate(municipios):
                    df_actual = df[idx]  # El índice coincide con el municipio en la lista
                    p_municipio = p_casos[idx]

                    # Calcular datos de población
                    p = p_municipio.groupby('Año')['Total'].sum().tolist()

                    # Generar total y calcular totales
                    df_total = calculate_total(df_actual, factor, p, 'Total')

                    # Asignar al diccionario
                    dataframes[municipio] = df
                    dataframes_total[municipio] = df_total
            
                df_barras_total = [dataframes_total[nombre] for nombre in dfl_barras if nombre in dataframes_total]
                df_tendencias_total = [dataframes_total[nombre] for nombre in dfl_tendencias if nombre in dataframes_total]

                if n_clicks > 0:
                    fig=generate_graph_total(df_barras_total, df_tendencias_total, dfl_barras, dfl_tendencias,
                                            titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                            pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    resultados.append(dcc.Graph(id='mi-grafico-total-nutricion', figure=fig))

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

            # Municipios y nombres de los DataFrames asociados
            if 'Santa Cruz' in dfl_total:
                municipios = ['Camiri', 'Gutierrez', 'Lagunillas', 'Cuevo', 'Cordillera', 'Santa Cruz']
                etnicidad = 'guarani'
            else:
                municipios = ['La Asunta', 'Palos Blancos', 'Chulumani', 'Irupana', 'Yanacachi', 'Sud Yungas', 'La Paz']
                etnicidad = 'afroboliviano'

            dataframes = {}
            dataframes_t = {}

            # Obtener todos los DataFrames y poblaciones de una vez
            df = get_casos(partes[1], etnicidad)  # Asumiendo que devuelve una lista de DataFrames
            num = len(municipios)

            p_casos = get_casos('poblacion-especial', etnicidad)  # Asumiendo que devuelve una lista de poblaciones
            
            if len(dfl_total) != 0:
                for idx, municipio in enumerate(municipios):
                    df_actual = df[idx]

                    p_municipio = p_casos[idx]
                    # Calcular datos de población
                    p = p_municipio.groupby('Año')['Embarazos'].sum().tolist()

                    # Generar total y calcular totales
                    df_total_o = calculate_total(df_actual, factor, p, 'Obesidad')
                    df_total_s = calculate_total(df_actual, factor, p, 'Sobrepeso')
                    df_total_d = calculate_total(df_actual, factor, p, 'Desnutricion')

                    # Asignar al diccionario
                    if type_nutrition == 'o':
                        dataframes[municipio] = df_total_o
                    elif type_nutrition == 's':
                        dataframes[municipio] = df_total_s
                    else:
                        dataframes[municipio] = df_total_d
                    dataframes_t[municipio] = [df_total_o, df_total_s, df_total_d]

                df_barras = [dataframes[nombre] for nombre in dfl_barras if nombre in dataframes]
                df_tendencias = [dataframes[nombre] for nombre in dfl_tendencias if nombre in dataframes]
                df_total = [dataframes_t[nombre] for nombre in dfl_total if nombre in dataframes_t]
                print(type_percent)
                resultados = []
                if n_clicks > 0:                
                    fig=generate_graph_total(df_barras, df_tendencias, dfl_barras, dfl_tendencias,
                                                titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    resultados.append(dcc.Graph(id='mi-grafico-total-nutricion', figure=fig))

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

            # Municipios y nombres de los DataFrames asociados
            if 'Santa Cruz' in dfl_total:
                municipios = ['Camiri', 'Gutierrez', 'Lagunillas', 'Cuevo', 'Cordillera', 'Santa Cruz']
                etnicidad = 'guarani'
            else:
                municipios = ['La Asunta', 'Palos Blancos', 'Chulumani', 'Irupana', 'Yanacachi', 'Sud Yungas', 'La Paz']
                etnicidad = 'afroboliviano'

            dataframes = {}
            dataframes_age = {}

            # Obtener todos los DataFrames y poblaciones de una vez
            df = get_casos(partes[1], etnicidad)  
            p_casos = get_casos('poblacion-especial', etnicidad)

            if len(dfl_total) != 0:
                for idx, municipio in enumerate(municipios):
                    # Obtener DataFrame actual y poblaciones correspondientes
                    df_actual = df[idx]
                    p_municipio = p_casos[idx]

                    # Agrupar según el tipo de mes
                    if type_mounth == 'm1':
                        df_actual = df_actual.groupby('Año').sum().reset_index()
                    elif type_mounth == 'm2':
                        df_actual = df_actual[df_actual['Tipo'] == 'Nuevo < 5']
                    else:
                        df_actual = df_actual[df_actual['Tipo'] == 'Nuevo > 5']

                    # Filtrar según el rango de edad
                    if type_age == 'r1':
                        p = p_municipio.groupby('Año')['10-14'].sum().tolist()
                        df_actual = calculate_total(df_actual, factor, p, '< 15')
                    elif type_age == 'r2':
                        p = p_municipio.groupby('Año')['15-19'].sum().tolist()
                        df_actual = calculate_total(df_actual, factor, p, '15-19')
                    else:
                        p = p_municipio.groupby('Año')['Adolescentes'].sum().tolist()
                        df_actual = calculate_total(df_actual, factor, p, '< 19')

                    # Calcular la edad ajustada del DataFrame
                    df_actual_y = calculate_age(df_actual, p_municipio, partes[1])
                    df_actual_y = calculate_age_total(df_actual_y, partes[1])

                    # Guardar resultados en los diccionarios
                    dataframes[municipio] = df_actual
                    dataframes_age[municipio] = df_actual_y

                df_barras = [dataframes[nombre] for nombre in dfl_barras if nombre in dataframes]
                df_tendencias = [dataframes[nombre] for nombre in dfl_tendencias if nombre in dataframes]

                df_total = [dataframes_age[nombre] for nombre in dfl_total if nombre in dataframes_age]
                resultados = []
                if n_clicks > 0:                
                    fig=generate_graph_total(df_barras, df_tendencias, dfl_barras, dfl_tendencias,
                                                titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                    resultados.append(dcc.Graph(id='mi-grafico-total-nutricion', figure=fig))
                    
                    '''for i, df in enumerate(df_total):
                        fig = generate_graph_age_pregnans(df, partes[1], dfl_total[i], 
                                                        titulo, tamanio_titulo, 'Año', tamanio_eje_x, type_percent, tamanio_eje_y,
                                                        pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                        resultados.append(dcc.Graph(id=f'mi-grafico-edad-embarazo-{i}', figure=fig))'''

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

            # Municipios y nombres de los DataFrames asociados
            if 'Santa Cruz' in dfl:
                municipios = ['Camiri', 'Gutierrez', 'Lagunillas', 'Cuevo', 'Cordillera', 'Santa Cruz']
                etnicidad = 'guarani'
                specific_labels = ['Camiri', 'Gutierrez', 'Lagunillas', 'Cuevo']
            else:
                municipios = ['La Asunta', 'Palos Blancos', 'Chulumani', 'Irupana', 'Yanacachi', 'Sud Yungas', 'La Paz']
                etnicidad = 'afroboliviano'
                specific_labels = ['La Asunta', 'Palos Blancos', 'Chulumani', 'Irupana', 'Yanacachi']

            dataframes = {}
            dataframes_y = {}

            # Obtener todos los DataFrames y poblaciones de una vez
            df = get_casos(partes[1], etnicidad)  
            p_casos = get_casos('poblacion', etnicidad)

            if len(dfl) != 0:
                for idx, municipio in enumerate(municipios):
                    # Obtener DataFrame actual y poblaciones correspondientes
                    df_actual = df[idx]
                    p = p_casos[idx]

                    df_actual = calculate_age(df_actual, p, partes[1])
                    df_actual_y = calculate_age_total(df_actual, partes[1])

                    # Guardar resultados en los diccionarios
                    dataframes[municipio] = df_actual
                    dataframes_y[municipio] = df_actual_y
                
                df_total = [dataframes[nombre] for nombre in dfl if nombre in dataframes]
                df_y = [dataframes_y[nombre] for nombre in dfl if nombre in dataframes_y]

                resultados = []
                if n_clicks > 0:    
                    max_y = 0
                    age_columns = ['% 0-9', '% 10-19', '% 20-39', '% 40-49', '% 50-59', '% 60+']
                    for i, df in enumerate(df_total):
                        if dfl[i] in specific_labels:
                            max_y = max(max_y, df["Total"].max())

                    if graphic_type == 't':
                        for i, df in enumerate(df_total):
                            #fig = generate_comparison_graph_by_year(df, dfl[i], 
                            #                                    titulo, tamanio_titulo, 'Año', tamanio_eje_x, 'Total', tamanio_eje_y,
                            #                                    pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                            #if dfl[i] in specific_labels:
                            #    fig = generate_total_area(df, dfl[i], max_y)
                            #else:
                            fig = generate_total_area(df, dfl[i])
                            resultados.append(dcc.Graph(id=f'mi-grafico-total-consulta-{i}', figure=fig))
                    elif graphic_type == 'e1':
                        for i, df in enumerate(df_total):
                            fig = generate_graph_separate_age(df, partes[1], dfl[i], 
                                                            titulo, tamanio_titulo, 'Año', tamanio_eje_x, 'Porcentaje', tamanio_eje_y,
                                                            pie, tamanio_pie, tamanio_leyenda, tamanio_num_grafica)
                            resultados.append(dcc.Graph(id=f'mi-grafico-edad-{i}', figure=fig))
                    elif graphic_type == 'e2':
                        for i, df in enumerate(df_y):

                            fig = generate_graph_join_age(df, age_columns, dfl[i], 
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
    [State('url', 'pathname')]
)
def update_output_salud(n_clicks, graphic_type,
                        dataframes_bar_dep, dataframes_ten_dep, 
                        dataframes_bar_prov, dataframes_ten_prov,
                        dataframes_bar_mun, dataframes_ten_mun,
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

            dfl = dataframes_bar_dep + dataframes_bar_prov + dataframes_bar_mun
            prov = dataframes_bar_prov + dataframes_bar_mun

            regions = [item for item in dfl if item]
            prov = [item for item in prov if item]
            municipios = [item for item in dataframes_bar_mun if item]

            resultados = []
            partes = pathname.split('/')
            resultados.append(html.H2(f'Gráfico '+partes[1]))

            if 'Santa Cruz' in dfl:
                mun = ['Camiri', 'Gutierrez', 'Lagunillas', 'Cuevo']
                etnicidad = 'guarani'
            else:
                mun = ['La Asunta', 'Palos Blancos', 'Chulumani', 'Irupana', 'Yanacachi']
                etnicidad = 'afroboliviano'

            df = get_casos(partes[1], etnicidad)
            df_poblacion = df[0]
            df_etnicidad = df[1]
            df_alfabetismo = df[2]
            df_servicios_basicos = df[3]
            df_dormitorios = df[4]
            df_ocupacion = df[5]

            # Crear el diccionario de DataFrames de abandono
            dataframes_abandono = {}
            for i, municipio in enumerate(mun):
                if 6 + i < len(df):  # Verificar si hay un DataFrame de abandono correspondiente
                    dataframes_abandono[municipio] = df[6 + i]    

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

            
            if n_clicks > 0:
                if graphic_type == 'p':
                    for region in regions:
                        fig = generate_population_pyramid(df_poblacion, region, titulo)
                        resultados.append(dcc.Graph(id=f'mi-piramide-poblacion-{region.lower()}', figure=fig))
                elif graphic_type == 'e':
                    fig = generate_language_donut_chart(df_etnicidad, prov, colors_language)
                    resultados.append(dcc.Graph(id='mi-dona-etnica-combinada', figure=fig))
                elif graphic_type == 'a':
                    fig = generate_literacy_donut_chart(df_alfabetismo, prov)
                    resultados = [dcc.Graph(id="mi-dona-analfabeta", figure=fig)]
                elif graphic_type == 'sb':
                    fig = generate_services_bar_chart(df_servicios_basicos, prov)
                    resultados = [dcc.Graph(id="mi-dona-sin-servicios-basicos", figure=fig)]
                elif graphic_type == 'h':
                    fig = generate_housing_pie_chart(df_dormitorios, prov)
                    resultados = [dcc.Graph(id="mi-dona-hacinada", figure=fig)]
                elif graphic_type == 'o':
                    for region in municipios:
                        fig = generate_ocupation_bar_chart(df_ocupacion, region)
                        resultados.append(dcc.Graph(id=f"mi-bar-ocupacion-{region.lower()}", figure=fig))
                elif graphic_type == 'as':
                    for municipio in municipios:
                        if municipio in dataframes_abandono:  # Verificar si el municipio tiene un DataFrame asociado
                            df_municipio = dataframes_abandono[municipio]
                            # Generar la figura para el municipio específico
                            fig = generate_secondary_abandonment_trend(df_municipio, municipio)
                            resultados.append(dcc.Graph(id=f"mi-bar-abandono-{municipio.lower()}", figure=fig))
                
                return resultados
        
            return html.Div("")

        except Exception as e:
            return html.Div(f'Error: {e}')  
        
@app.callback(
    [Output('dropdown-dataframes-bar-prov', 'options'),
     Output('dropdown-dataframes-ten-prov', 'options')],
    [Input('dropdown-dataframes-bar-dep', 'value'),
     Input('dropdown-dataframes-ten-dep', 'value')]
)
def update_provincia_options(selected_deps_bar, selected_deps_ten):
    if not selected_deps_bar and not selected_deps_ten:
        return [], []

    provincias = set()
    selected_deps = set(selected_deps_bar or []) | set(selected_deps_ten or [])

    for dep in selected_deps:
        if dep in opciones_dataframes:
            provincias.update(opciones_dataframes[dep]['provincias'].keys())

    options = [{'label': prov, 'value': prov} for prov in provincias]
    return options, options

@app.callback(
    [Output('dropdown-dataframes-bar-mun', 'options'),
     Output('dropdown-dataframes-ten-mun', 'options')],
    [Input('dropdown-dataframes-bar-prov', 'value'),
     Input('dropdown-dataframes-ten-prov', 'value')]
)
def update_municipio_options(selected_prov_bar, selected_prov_ten):
    if not selected_prov_bar and not selected_prov_ten:
        return [], []

    municipios = set()
    selected_provs = set(selected_prov_bar or []) | set(selected_prov_ten or [])

    for dep, data in opciones_dataframes.items():
        for prov in selected_provs:
            if prov in data['provincias']:
                municipios.update(data['provincias'][prov])

    options = [{'label': mun, 'value': mun} for mun in municipios]
    return options, options

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


if __name__ == "__main__":
    # Comprobar si estamos en un entorno de producción
    if os.environ.get("PORT"):
        port = int(os.environ["PORT"])  # Usa el puerto de la variable de entorno
        app.run_server(host='0.0.0.0', port=port)  # Para producción
    else:
        app.run_server(debug=True)  # Para desarrollo, usa el modo debug
