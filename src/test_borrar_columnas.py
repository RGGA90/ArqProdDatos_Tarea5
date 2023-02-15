import pytest
import pandas as pd
import os,sys


CURRENT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(CURRENT)
sys.path.append(PARENT)

from herramientas import borrar_columnas



def test_borrar_columnas():
    """
    Test para verificar que se aplico correctamente la funcion de borrado de columnas
    """
    try:
        df=pd.read_csv('./Datos/train.csv')
        columnas_eliminar=["Id","LotShape","LandSlope","ExterQual","BsmtFinType1","BsmtFinType2","HeatingQC",
           "CentralAir","KitchenQual","Functional","FireplaceQu","GarageFinish","GarageQual",
           "GarageCond","PavedDrive","PoolQC","Fence","BsmtQual","BsmtCond","ExterCond","ExterQual","KitchenQual"]
        df2=borrar_columnas(df,columnas_eliminar)
    
    except FileNotFoundError as err:
        print("File not found")
        raise err
    try:
        #Verificar que el df original tiene mas columnas que el df despues de aplicar la funcino de borrar columnas
        assert df.shape[1] > df2.shape[1]
        #Verificar que el d2 depues del borrado aun tiene un numero positivo de columnas
        assert df2.shape[1]>0

    except AssertionError as err:
        ("La funcion de borrado no es correcta")
        raise err
