## model training

import platform; print(platform.platform())
import sys; print("Python", sys.version)

# imports
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 

import joblib

# setup
modelname = 'xgb'
params_path = f'./src/training_files/{modelname}_best_params.joblib'

# load data
train = pd.read_csv('./data/final/train.csv')
train_ohe = pd.read_csv('./data/final_ohe/train.csv')

TARGET = 'Transported'
FEATURES = [col for col in train.columns if col not in [TARGET]]

numerical = train[FEATURES].select_dtypes(include=np.number).columns.to_list()
categorical = train[FEATURES].select_dtypes(exclude=np.number).columns.to_list()

# preprocessing
cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
preproc = ColumnTransformer(
    transformers=[('cat', cat_encoder, categorical)],
    verbose_feature_names_out=False,
    remainder='passthrough'
).set_output(transform='pandas')

# utility function
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
    
    # fit preprocessing pipeline
    preproc = preproc.fit(train[FEATURES]) 
    #print(preproc.get_feature_names_out())

    # load model
    model_path = f'./src/training_files/{modelname}_best_model.joblib'
    with open(model_path, 'rb') as file:
        model = joblib.load(file)

    # get features
    xtest = get_features()
    xtest = preproc.transform(xtest)
    xtest.columns = train_ohe.drop(columns=TARGET).columns

    # prediction
    preds = model.predict(xtest)
    probs = model.predict_proba(xtest)[0][1]
    print(f'\nThe prediction for {TARGET} is {bool(preds)} (predicted probability: {probs:.2%}).\n')
