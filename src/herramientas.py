"""
En este modulo se definen funciones que se emplearan en distintas etapas del proceso de modelado de datos del proyecto de "Prediccion de precios de casas"
"""
def borrar_columnas(df, lista):
    """
    Argumentos: 
    df (pandas df): Data frame del cual se desean borrar columnas
    lista (list): Lista de columnas que se desean borrar
    Returns:
    df (pandas df): Data frame modificado sin las columnas que se deseaban borrar
    """
    #Hago un for loop para todos los elementos dentro de la lista que contiene las variables que quiero borrar
    for variable in lista:
        #reviso si la variable cumple con la condicion de estar en las columnas del df, asi evito que se desencadene un KeyError
        if variable in df.columns:
            df=df.drop([variable], axis=1)
        # si la variable que esta en la lista de variables a borrar, no se encuentra en la lista de columnas del df, no hago nada
        else:
            df=df
    return df
