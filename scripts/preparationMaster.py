import numpy as np
import pandas as pd
from scripts.importMaster import importData
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle


def traintest(importedData, sampling="stratified", test_size=0.2):
    if sampling == "stratified":
        # evenly distribute users among train and test set
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        for train_index, test_index in split.split(importedData, importedData["User"]):
            strat_train_set = importedData.iloc[train_index, ]
            strat_test_set = importedData.iloc[test_index, ]

        # remove user col, not needed anymore
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("User", axis=1, inplace=True)

        return strat_train_set, strat_test_set

    else:
        # sample sets so, that a user doesnt appear in both sets
        value_counts = np.cumsum(shuffle(importedData.User.value_counts()/len(importedData)))
        train_users = tuple(value_counts[value_counts <= 1-test_size].index)
        train_set = importedData.loc[importedData.User.isin(train_users), :].drop("User", axis=1)
        test_set = importedData.loc[~importedData.User.isin(train_users), :].drop("User", axis=1)

        return train_set, test_set


def prepareData(dataset):
    ## create sets for each sensor
    labels = [label for label in dataset.columns.values if label.startswith(('label'))]
    time = [t for t in dataset.columns.values if t.startswith(('discrete:time'))]

    # audio
    audio_features = [feature for feature in dataset.columns.values if feature.startswith(('audio'))]
    audio_set = dataset.loc[:, audio_features + time]

    # phonestate
    ps_features = [feature for feature in dataset.columns.values if feature.startswith(('lf_', 'discrete:'))]
    ps_set = dataset.loc[:, ps_features]

    # location
    loc_features = [feature for feature in dataset.columns.values if feature.startswith(('location'))]
    loc_set = dataset.loc[:, loc_features + time]

    # acc
    acc_features = [feature for feature in dataset.columns.values if feature.startswith(('proc_gyro',
                                                                                         'raw_acc', 'watch'))]
    acc_set = dataset.loc[:, acc_features + time]

    # magnet
    magnet_features = [feature for feature in dataset.columns.values if feature.startswith(('raw_magnet'))]
    magnet_set = dataset.loc[:, magnet_features + time]

    return {"audio": audio_set, "phonestate": ps_set, "location": loc_set, "acceleration": acc_set,
            "magnet": magnet_set, "target": dataset.loc[:, labels]}


if __name__ == "__main__":
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    dataPath = "../data/features_labels/"
    data = importData(dataPath)

    strat_train, strat_test = traintest(data, sampling="stratified", test_size=0.3)

    train, test = traintest(data, sampling=" ", test_size=0.3)

    ptrain = prepareData(train)
    ptest = prepareData(test)

