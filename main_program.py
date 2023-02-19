from src.houseprices import (load_data, create_heatmap_plot, create_plots, fill_all_missing_values,ordinal_encoder, encode_catagorical_columns, create_predictive_model )
import logging

def main():
    logging.info("Iniciando lectura de informacion")

    train_data = load_data("data/raw/train.csv")
    test_data = load_data("data/raw/test.csv")      
    test_ids = test_data['Id']
    logging.info(f"Train data Shape:, {train_data.shape}")
    logging.info(f"Train data Duplicated data :, { train_data.duplicated().sum()}")

    logging.info("Iniciando EDA")

    create_heatmap_plot(train_data)
    create_plots(train_data)

    logging.info("Iniciando Preprocessing")

    cols =  {'FireplaceQu': 'No',
            'BsmtQual':'No',
            'BsmtCond': 'No',
            'BsmtFinType1': 'No',
            'BsmtFinType2': "None"}

    logging.info("Iniciando imputación")
    train_data = fill_all_missing_values(train_data, cols)
    test_data = fill_all_missing_values(test_data, cols)#

    drop_col = ['Id', 'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'MoSold', 'YrSold', 'MSSubClass',
            'GarageType', 'GarageArea', 'GarageYrBlt', 'GarageFinish', 'YearRemodAdd', 'LandSlope',
            'BsmtUnfSF', 'BsmtExposure', '2ndFlrSF', 'LowQualFinSF', 'Condition1', 'Condition2', 'Heating',
             'Exterior1st', 'Exterior2nd', 'HouseStyle', 'LotShape', 'LandContour', 'LotConfig', 'Functional',
             'BsmtFinSF1', 'BsmtFinSF2', 'FireplaceQu', 'WoodDeckSF', 'GarageQual', 'GarageCond', 'OverallCond'
           ]

    train_data.drop(drop_col, axis=1, inplace=True)
    test_data.drop(drop_col, axis=1, inplace=True)


    dict_params = {'BsmtQual': ['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
               'BsmtCond': ['No', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
               'ExterQual':['Po', 'Fa', 'TA', 'Gd', 'Ex'],
               'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
               'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
               'PavedDrive':['N', 'P', 'Y'],
               'Electrical':['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'],
               'BsmtFinType1':['No', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
               'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
               'Utilities':['ELO', 'NoSeWa', 'NoSewr', 'AllPub'],
               'MSZoning':['C (all)', 'RH', 'RM', 'RL', 'FV'],
               'Foundation':['Slab', 'BrkTil', 'Stone', 'CBlock', 'Wood', 'PConc'],
               'Neighborhood':['MeadowV', 'IDOTRR', 'BrDale', 'Edwards', 'BrkSide', 'OldTown', 'NAmes', 'Sawyer', 'Mitchel', 'NPkVill', 'SWISU', 'Blueste', 'SawyerW', 'NWAmes', 'Gilbert', 'Blmngtn', 'ClearCr', 'Crawfor', 'CollgCr', 'Veenker', 'Timber', 'Somerst', 'NoRidge', 'StoneBr', 'NridgHt'],
               'MasVnrType':['None', 'BrkCmn', 'BrkFace', 'Stone'],
               'SaleCondition':['AdjLand', 'Abnorml','Alloca', 'Family', 'Normal', 'Partial'],
               'RoofStyle':['Gambrel', 'Gable','Hip', 'Mansard', 'Flat', 'Shed'],
               'RoofMatl':['ClyTile', 'CompShg', 'Roll','Metal', 'Tar&Grv','Membran', 'WdShake', 'WdShngl']
    }
    
    logging.info("Iniciando ordinal encoder")
    for k, v in dict_params.items():
        train_data, test_data = ordinal_encoder(train_data, test_data, k, v)


    logging.info("Iniciando  categorical encoder")
    level_col = ['Street' ,'BldgType', 'SaleType', 'CentralAir']
    train_data, test_data = encode_catagorical_columns(train_data, test_data, level_col)

    logging.info("Iniciando  calculo de columnas")
    train_data['BsmtRating'] = train_data['BsmtCond'] * train_data['BsmtQual']
    train_data['ExterRating'] = train_data['ExterCond'] * train_data['ExterQual']
    train_data['BsmtFinTypeRating'] = train_data['BsmtFinType1'] * train_data['BsmtFinType2']
    
    train_data['BsmtBath'] = train_data['BsmtFullBath'] + train_data['BsmtHalfBath']
    train_data['Bath'] = train_data['FullBath'] + train_data['HalfBath']
    train_data['PorchArea'] = train_data['OpenPorchSF'] + train_data['EnclosedPorch'] + train_data['3SsnPorch'] + train_data['ScreenPorch']
    
    test_data['BsmtRating'] = test_data['BsmtCond'] * test_data['BsmtQual']
    test_data['ExterRating'] = test_data['ExterCond'] * test_data['ExterQual']
    test_data['BsmtFinTypeRating'] = test_data['BsmtFinType1'] * test_data['BsmtFinType2']
    
    test_data['BsmtBath'] = test_data['BsmtFullBath'] + test_data['BsmtHalfBath']
    test_data['Bath'] = test_data['FullBath'] + test_data['HalfBath']
    test_data['PorchArea'] = test_data['OpenPorchSF'] + test_data['EnclosedPorch'] + test_data['3SsnPorch'] + test_data['ScreenPorch']

    logging.info("Drop columns from train and test table")
    drop_col = ['OverallQual', 
            'ExterCond', 'ExterQual',
            'BsmtCond', 'BsmtQual',
            'BsmtFinType1', 'BsmtFinType2',
            'HeatingQC',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
            'BsmtFullBath', 'BsmtHalfBath',
            'FullBath', 'HalfBath',
           ]
    train_data = train_data.drop(drop_col, axis=1)
    test_data = test_data.drop(drop_col, axis=1)

    logging.info("Inicializando modelo")
    y = train_data['SalePrice']
    X = train_data.drop(['SalePrice'], axis=1)

    
    
    submission= create_predictive_model(X, y, test_data, test_ids)
    logging.info("Escribiendo output")
    submission.to_csv("submission.csv", index=False)

    logging.info("Fin del proceso")
    
    



if __name__ == "__main__":
    main()