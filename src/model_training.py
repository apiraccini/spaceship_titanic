## model training

import platform; print(platform.platform())
import sys; print("Python", sys.version)

# imports
import os
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import xgboost as xgb
import lightgbm as lgb
import catboost

# setup
modelname = 'lgb'
params_path = f'./src/training_files/studies/{modelname}_best_params.joblib'

# load data
train = pd.read_csv('./data/final_ohe/train.csv')
test = pd.read_csv('./data/final_ohe/test.csv')

TARGET = 'Transported'
FEATURES = [col for col in train.columns if col not in [TARGET]]

print(f'\nTarget: {TARGET}')
print(f'Features: {FEATURES}')
print(f'Shapes:\t{train.shape}\t{test.shape}')
print(f'Missing values:\t{train.isna().sum().sum()}, {test.isna().sum().sum()}\n')

# split data
x, x_val, y, y_val = train_test_split(
    train[FEATURES],
    train[TARGET],
    test_size = 0.2,
    random_state = 42
)

# get best hyperparameters
with open(params_path, 'rb') as path:
    best_params = joblib.load(path)
print(f"Model: {modelname}\nHyper parameters:")
for k, v in best_params.items():
    print(f"\t{k}: {v}")
print('Setting number of trees to 10000 and learning rate to 0.005')
best_params['n_estimators'] = 10000
best_params['learning_rate'] = 0.005

# define model
if modelname == 'xgb':
    model = xgb.XGBClassifier(**best_params)
elif modelname == 'lgb':
    model = lgb.LGBMClassifier(**best_params)
elif modelname == 'catboost':
    model = catboost.CatBoostClassifier(**best_params) 

# train model
print('\n')
print(120*'*')
print(f'Training {modelname}...\n')
if modelname == 'xgb':
    model.fit(
        x, y,
        eval_set= [(x_val, y_val)],
        verbose = 200
    )
elif modelname == 'lgb' :
    model.fit(
        x, y,
        eval_set= [(x_val, y_val)],
        callbacks=[lgb.log_evaluation(200)]
    )
print(120*'*', '\n')

# score on evaluation set
accuracy = accuracy_score(model.predict(x_val), y_val)
f1 = f1_score(model.predict(x_val), y_val)
auc = roc_auc_score(model.predict(x_val), y_val)
print(f'Metrics on evaluation set:\n\taccuracy: {accuracy:.4f}\tF1 score: {f1:.4f}\tAUC: {auc:.4f}')

# save best model
os.makedirs('./src/training_files', exist_ok=True)
model_path = f'./src/training_files/{modelname}_best_model.joblib'
with open(model_path, 'wb') as file:
    joblib.dump(model, file)

# predict test data
with open(model_path, 'rb') as file:
    model = joblib.load(file)
preds = model.predict(test)

sub = pd.read_csv('./data/raw/sample_submission.csv')
sub[TARGET] = preds.astype(bool)

os.makedirs('./submissions', exist_ok=True)
sub.to_csv(f'./submissions/{modelname}_tuned.csv', index=False)