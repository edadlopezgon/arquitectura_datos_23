import pandas as pd
import logging
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

logging.basicConfig(
    filename='./logs/results.log',
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def create_predictive_model(X, y, test_data, test_ids):
    candidate_max_leaf_nodes = [250]
    for node in candidate_max_leaf_nodes:
        model = RandomForestRegressor(max_leaf_nodes=node,)
        model.fit(X, y)
        score = cross_val_score(model, X, y, cv=10)
        logging.info(f"Score mean: {score.mean()}")
    price = model.predict(test_data)
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": price
    })
    return submission



def load_data(path):
    """
    Función que lee un pandas dataframe
    Params:
        path: Ruta del archivo
    Return:
        df: Dataframe pandas 
    """
    df = pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.error(f"Ha ocurrido un error en la lectura de información: {e}")
    return df

def create_heatmap_plot(df):
    """
    Función que crea un heatmap basado en las columnas de un dataframe
    Params:
        df: Dataframe
    
    """
    fig, ax = plt.subplots(figsize=(25,10))
    sns.heatmap(data=df.isnull(), yticklabels=False, ax=ax)

def create_plots(df):
    """
    Función que crea distintos tipos de gráficas
    Params:
        df: Dataframe
    
    """
    fig, ax = plt.subplots(figsize=(25,10))
    sns.countplot(x=df['SaleCondition'])
    sns.histplot(x=df['SaleType'], kde=True, ax=ax)
    sns.violinplot(x=df['HouseStyle'], y=df['SalePrice'],ax=ax)
    sns.scatterplot(x=df["Foundation"], y=df["SalePrice"], palette='deep', ax=ax)
    plt.grid()

def fill_all_missing_values(data, cols_ob ={} ):
    """
    Función que imputa valores faltantes
    Params:
        data: Dataframe con na's
        cols_ob: Diccionario con la relacion columna y valor a imputar
    Return:
        data: Dataframe con valores imputados
    """

    for col in data.columns:
        if((data[col].dtype == 'float64') or (data[col].dtype == 'int64')):
             data[col] = data[col].fillna(data[col].mean())
        elif data[col].dtype == 'object' and col in list(cols_ob.keys()):
             data[col] = data[col].fillna(cols_ob[col])
        else:
             data[col] = data[col].fillna(data[col].mode()[0])


    return data

def ordinal_encoder(train_data, test_data, key, value):
    """
    Función que genera un encoder basado en ciertas categorias
    Params:
        train_data: dataframe para detectar y transformar la columna
        test_data: dataframe para transformar la columna
        key: columna
        value: categorias

    """
    try:
        OE = OrdinalEncoder(categories=[value])
        train_data[key] = OE.fit_transform(train_data[[key]])
        test_data[key] = OE.transform(test_data[[key]])
    except Exception as e:
        logging.error(f"Ha ocurrido un error ordinal encoding en el campo {key}: {e}")
    return train_data, test_data

def encode_catagorical_columns(train, test, level_col):
    """
    Función que genera un encoder basado en ciertas categorias
    Params:
        train_data: dataframe para detectar y transformar la columna
        test_data: dataframe para transformar la columna
        level_col: columna
        
    """
    try:
        encoder = LabelEncoder()
        for col in level_col:
            train[col] = encoder.fit_transform(train[col])
            test[col]  = encoder.transform(test[col])
        
    except Exception as e:
        logging.error(f"Ha ocurrido un error categorical encoding en el campo {col}: {e}")
    return train, test