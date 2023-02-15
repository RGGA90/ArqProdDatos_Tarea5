#Libreria para trabajar con dataframes
import pandas as pd
# Libreria para vectorizar 
import numpy as np
# Libreria para trabajar con paths relativos
import os
# Libreria para usar modelos de machine learning
from sklearn.linear_model import LinearRegression
# Libreria para estandarizar valores numericos, libreria para hacer one hot encoding
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Libreria para imputar valores
from sklearn.impute import SimpleImputer
# Libreria para hacer un pipeline de transformaciones
from sklearn.pipeline import Pipeline
# Libreria para juntar todos los pasos de preprocesamiento en una funcion de transformacion
from sklearn.compose import ColumnTransformer
# Importo una funcion que yo mismo cree en modulo de python que vive a la misma altura (en el mismo folder y mismo nivel) que mi notebook actual
from herramientas import borrar_columnas
#Importo la lilbreria yaml para poder usar mi archivo de configuracion
import yaml
#Importo la libreria logging para implementar monitoreo de mi script
import logging
#Importo la libreria pickle para guardar modelos y preprocesamiento
import pickle


# establezco una ruta para encontrar mi archivo config
CONFIG_PATH = "../"

with open(os.path.join(CONFIG_PATH, 'config.yaml'), encoding="utf-8") as conf:
    config = yaml.safe_load(conf)

    
logging.basicConfig(
    filename=os.path.join(config["directorio_logs"],config["archivo_logging"]),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

    
#1) Leer datos de un archivo csv, usamos la libreria de pandas(pd)
#uso la ruta parametrizada que viene del archivo de configuracion
train=pd.read_csv(os.path.join(config["directorio_datos"],config["datos_entrenamiento"]))

# 4) Separo mis variables independientes y mi variable de respuesta
#guardo una variable solo con mi variable dependiente
y=train[config["variable_dependiente"]]
#Tiro del dataframe mi variable dependiente y conservo las demas
train=train.drop(config["variable_dependiente"], axis=1)

try:
    train.drop(config["variables_eliminar"], axis=1)
except KeyError:
    logging.error("Hay una columna que intentaste borrar y que no pertenece al df")

# 6) Evito los errores ocasiondos por KeyError usando una funcion que almacene en el modulo herramientas
# Asigno el dataframe al resultado de invocar la funcion que defini en mi modulo anterior
train=borrar_columnas(train,config["variables_eliminar"])

#Preprocesamiento para variables numericas
#Uso la funcion Pipeline para juntar distintos preprocesamientos en un unico paso, en este caso aplico Standar Scaler que va a estandarizar mis variables numericas
# y simple imputer, que va a imputar valores faltantes con la media de las observaciones. Por ahora uso los parametros de default de las funciones
procesamiento_numericas = Pipeline(
    steps=[("scaler", StandardScaler()),
           ("imputer",SimpleImputer())])

# Uso la funcion pipeline para aplicar un preprocesamiento a mis variables categoricas, en este caso solo aplico One Hote Encoder y establezo que las variables que no pasen por
# este preprocesamiento deben ser ignoradas, esto es importante para evitar que el transformador tenga problemas al preprocesar datos que no ha visto, por ejemplo, los
# datos del set de test
procesamiento_categoricas = Pipeline(
    steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

#Junto los pasos del preprocesamiento
pasos_de_preprocesamiento = ColumnTransformer(
    transformers=[
        # "Nombre que asigno al paso de procesamiento", Nombre del pipeline, Lista de columnas a recibir ese preprocesamiento
        ("numericas", procesamiento_numericas, config["variables_numericas"]),
        # "Nombre que asigno al paso de procesamiento", Nombre del pipeline, Lista de columnas a recibir ese preprocesamiento
        ("categoricasl", procesamiento_categoricas, config["variables_categoricas"])
        ],
    # Comportamiento que defino para las variables que no recibieron ningun tipo de preprocesamiento, en este caso elijo tirarlas
    remainder='drop'
    )
## Aplico los pasos de preprocesamiento
pasos_de_preprocesamiento.fit(train)

#Abro el archivo en el que voy a escribir mi preprocesamiento para uso futuro
file = open(config["preprocesamiento"], 'wb')

### guardo mis pasos de preprocesamiento en un pickle file
pickle.dump(pasos_de_preprocesamiento,file)

Datos_preprocesados=pasos_de_preprocesamiento.transform(train)


# A continuacion empleo regresion linear con sklearn para demostrar los pasos basicos de esta libreria, en iteraciones posteriorres agregare: 1) gridsearch, 
#2) mas modelos, 3) pipeline de modelado
#Entreno mi regresor con los datos de entrenamiento
regresion_lineal=LinearRegression().fit(Datos_preprocesados,y)

# Hago predicciones sobre mi set de test
#1) aplico el preprocesamiento de datos, en este caso, no quiero volver a entrenar el preprocesamiento, por lo tanto solo aplico el metodo transform
test=pd.read_csv(os.path.join(config["directorio_datos"],config["datos_test"]))
Test_preprocesados=pasos_de_preprocesamiento.transform(test)


#2)Hago predicciones con mi modelo ya entrenado
predicciones=regresion_lineal.predict(Test_preprocesados)


#3) Mando mis predicciones a un archivo csv
predicciones.tofile(os.path.join(config["directorio_resultados"],config["predicciones"]), sep=",")











