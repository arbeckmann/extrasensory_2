import numpy as np
import pandas as pd
import pickle

from scripts.importMaster import importData
from scripts.cleaningMaster import cleanData, scaleData
from scripts.preparationMaster import traintest, prepareData

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class DummyEstimator(BaseEstimator):
    def fit(self): pass
    def score(self): pass


if __name__ == "__main__":
    ## determine best model for each sensor for each label

    dataPath = "../data/features_labels/"
    data = importData(dataPath)
    # split
    train_set, test_set = traintest(data, sampling="stratified", test_size=0.5)
    # clean
    cleaned_train, imputer = cleanData(train_set, strategy="mean")
    # scale
    scaled_train, scaler = scaleData(cleaned_train, type="standard")
    # prepare
    prep_dict = prepareData(scaled_train)

    sensors = ["audio", "phonestate", "location", "acceleration", "magnet"]
    target_labels = prep_dict["target"].columns.values

    pipe = Pipeline([('clf', DummyEstimator())])

    search_space = [{'clf': [RandomForestClassifier()],  # Actual Estimator
                     'clf__n_estimators': [200],
                     'clf__max_depth': [25]},

                    {'clf': [XGBClassifier()],  # Actual Estimator
                     'clf__n_estimators': [200],
                     'clf__max_depth': [25]},

                    {'clf': [MLPClassifier()]},

                    {'clf': [LogisticRegression()]}
                    ]

    predictor_dict = {"audio": {},
                      "phonestate": {},
                      "location": {},
                      "acceleration": {},
                      "magnet": {}}

    for sensor in sensors:
        sensor_train = prep_dict[sensor].values
        print(sensor)

        for target in target_labels:
            labels = prep_dict["target"].loc[:, target].values
            gs = GridSearchCV(pipe, search_space, scoring="balanced_accuracy", n_jobs=-1, cv=3)
            gs.fit(sensor_train, labels)
            model = clone(gs.best_params_["clf"])

            predictor_dict[sensor][target] = model

    output = open('predictors.pkl', 'wb')
    pickle.dump(predictor_dict, output)
    output.close()





