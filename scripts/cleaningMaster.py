import numpy as np
import pandas as pd
from scripts.importMaster import importData
from scripts.preparationMaster import traintest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def cleanData(importedData, strategy="median"):
    # select cols with nan values
    cols_to_clean = importedData.loc[:, importedData.isna().any().tolist()]
    # define imputer
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputer.fit(cols_to_clean)
    # impute values
    cleaned_cols = imputer.transform(cols_to_clean)
    importedData.loc[:, importedData.isna().any().tolist()] = cleaned_cols
    cleanedData = importedData

    # also return imputer so we can use it to impute nans in test sets
    return cleanedData, imputer


def scaleData(cleanedData, type="standard"):
    # select the cols which need scaling
    cols_to_scale = [col for col in cleanedData if not col.startswith(('label', 'User', 'discrete'))]
    cols_to_scale = cleanedData.loc[:, cols_to_scale]
    # build scaler
    if type == "standard":
        scaler = StandardScaler()
    elif type == "minmax":
        scaler = MinMaxScaler()
    scaler.fit(cols_to_scale)
    # scale values
    scaled_cols = scaler.transform(cols_to_scale)
    cleanedData.loc[:, [col for col in cleanedData if not col.startswith(('label', 'User', 'discrete'))]] = scaled_cols
    scaledData = cleanedData
    # also return scaler so we can use it to scale test set cols
    return scaledData, scaler


if __name__ == "__main__":
    dataPath = "../data/features_labels/"
    data = importData(dataPath)

    train_set, test_set = traintest(data, sampling="stratified", test_size=0.2)
    print("splitted")
    cleaned_train, imputer = cleanData(train_set, strategy="mean")
    print("cleaned")
    scaled_train, scaler = scaleData(cleaned_train, type="standard")
    print("scaled")





