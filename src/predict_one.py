## model training
import platform; print(platform.platform())
import sys; print("Python", sys.version)

# imports
import os
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

# setup
modelname = 'catboost'
seed=42
params_path = f'./training_files/{modelname}_best_params.joblib'

# load data
train = pd.read_csv('../data/final/train.csv')

TARGET = 'Transported'
FEATURES = [col for col in train.columns if col not in [TARGET]]

numerical = train[FEATURES].select_dtypes(include=np.number).columns.to_list()
categorical = train[FEATURES].select_dtypes(exclude=np.number).columns.to_list()

def get_features():
    out = {}
    print('\nSpace titanic model: enter features for prediction')
    for f in FEATURES:
        if f in categorical:
            vals = str(train[f].unique())
            while(1):
                raw_value = input(f'\t{f} (one of {vals}): ')
                try:
                    value = str(raw_value)
                except ValueError:
                    print('\tInsert a valid string')
                    continue
                if value in vals:
                    break
                else:
                    print(f'\tInsert one of {vals}')
        if f in numerical:
            valrange = [train[f].min(), train[f].max()]
            while(1):
                raw_value = input(f'\t{f} (train range {valrange}): ')
                try:
                    value = float(raw_value)
                except ValueError:
                    print('\tInsert a valid number')
                    continue
                break
        out[f] = [value]
    out = pd.DataFrame.from_dict(out)
    return out

if __name__ == '__main__':
    # load model
    model_path = f'./training_files/{modelname}_best_model'
    model = CatBoostClassifier()
    model.load_model(model_path)

    # get features
    xtest = get_features()
    pool_xtest = Pool(xtest, cat_features=categorical)

    # prediction
    preds = model.predict(pool_xtest)[0]
    print(f'\nThe prediction for {TARGET} is {bool(preds)}\n')
