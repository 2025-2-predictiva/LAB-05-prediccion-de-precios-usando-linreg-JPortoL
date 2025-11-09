#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#


import pandas as pd


def load_data(dir_path):
    data = pd.read_csv(dir_path, compression='zip')
    return data

def preprocess_data(data):
    data['Age'] = 2021 - data['Year']
    data = data.drop(columns=['Year', 'Car_Name'])
    return data

def split_data(data):
    X = data.drop(columns=['Present_Price'])
    y = data['Present_Price']
    return X, y

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression

def create_pipeline():
    categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
    numerical_features = ['Selling_Price', 'Driven_kms', 'Owner', 'Age']

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = MinMaxScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression)),
        ('regressor', LinearRegression())
    ])

    return pipeline

from sklearn.model_selection import GridSearchCV

def optimize_hyperparameters(pipeline, X_train, y_train):
    param_grid = {
        'feature_selection__k':  range(1, 12)
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=10,
                                scoring='neg_mean_absolute_error',
                                n_jobs=-1,
                                verbose=2,
                                refit=True)

    return grid_search.fit(X_train, y_train)

import os
import gzip
import pickle
import json

def save_model(model):
    os.makedirs('files/models', exist_ok=True)
    with gzip.open('files/models/model.pkl.gz', 'wb') as file:
        pickle.dump(model, file)

from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

def save_metrics(model, X_train, y_train, X_test, y_test):
    metrics = []

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = {
        'type': 'metrics',
        'dataset': 'train',
        'r2': r2_score(y_train, y_train_pred),
        'mse': mean_squared_error(y_train, y_train_pred),
        'mad': median_absolute_error(y_train, y_train_pred)
    }
    metrics.append(train_metrics)

    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mad': median_absolute_error(y_test, y_test_pred)
    }
    metrics.append(test_metrics)

    metrics_path = "files/output/metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')




if __name__ == "__main__":
    data_train = pd.read_csv("files/input/train_data.csv.zip")
    data_test = pd.read_csv("files/input/test_data.csv.zip")

    print(data_train.head())

    data_train = preprocess_data(data_train)
    data_test = preprocess_data(data_test)

    print(data_train.head())

    X_train, y_train = split_data(data_train)
    X_test, y_test = split_data(data_test)

    pipeline = create_pipeline()

    model = optimize_hyperparameters(pipeline, X_train, y_train)

    print(model.best_params_)
    print(model.best_score_)

    save_model(model)

    save_metrics(model, X_train, y_train, X_test, y_test)





