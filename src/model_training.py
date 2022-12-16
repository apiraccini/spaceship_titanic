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
params_path = f'./src/training_files/{modelname}_best_params.joblib'

# load data
train = pd.read_csv('./data/final/train.csv')
test = pd.read_csv('./data/final/test.csv')

TARGET = 'Transported'
FEATURES = [col for col in train.columns if col not in [TARGET]]

numerical = train[FEATURES].select_dtypes(include=np.number).columns.to_list()
categorical = train[FEATURES].select_dtypes(exclude=np.number).columns.to_list()

train[categorical] = train[categorical].astype(str)
train[TARGET] = train[TARGET].astype(float)
test[categorical] = test[categorical].astype(str)

print(f'\nModel: {modelname}\n')
print(f'Target: {TARGET}')
print(f'Features:\n\tnumerical: {numerical}\n\tcategorical:{categorical}')
print(f'Shapes:\n\ttrain: {train.shape}\n\ttest: {test.shape}\n')

# split data
x, x_val, y, y_val = train_test_split(
    train[FEATURES],
    train[TARGET],
    test_size = 0.2,
    random_state = 42
)

# pool preparation
pool_train = Pool(x, y, cat_features=categorical)
pool_val = Pool(x_val, y_val, cat_features=categorical)
pool_test = Pool(test[FEATURES], cat_features=categorical)

# get best hyperparameters
with open(params_path, 'rb') as path:
    best_params = joblib.load(path)
best_params['learning_rate'] = 0.01
print("\nHyper parameters:")
for k, v in best_params.items():
    print(f"\t{k}: {v}")

# train model
print('\n')
print(120*'*')
print(f'Training {modelname}...\n')
model = CatBoostClassifier(**best_params)
model.fit(
    pool_train,
    eval_set = pool_val,
    early_stopping_rounds=15,
    verbose = 500
)
print(120*'*', '\n')

# score on eval.set
accuracy = model.best_score_['validation']['Accuracy']
f1 = model.best_score_['validation']['F1']
auc = model.best_score_['validation']['AUC']
print(f'Evaluation metrics:\n\taccuracy: {accuracy:.3f}\tF1 score: {f1:.3f}\tAUC: {auc:.3f}')

# save best model
os.makedirs('./src/training_files', exist_ok=True)
model_path = f'./src/training_files/{modelname}_best_model'
model.save_model(model_path)

# predict test data
model = CatBoostClassifier()
model.load_model(model_path)
preds = model.predict(pool_test)

sub = pd.read_csv('./data/raw/sample_submission.csv')
sub[TARGET] = preds.astype(bool)

os.makedirs('./submissions', exist_ok=True)
sub.to_csv('./submissions/catboost_tuned.csv', index=False)