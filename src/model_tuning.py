# imports
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

import optuna
from optuna.samplers import TPESampler
import joblib

# setup
modelname = 'catboost'
seed=42

# prepare data
train = pd.read_csv('./data/final/train.csv')
test = pd.read_csv('./data/final/test.csv')

TARGET = 'Transported'
FEATURES = [col for col in train.columns if col not in [TARGET]]

numerical = train[FEATURES].select_dtypes(include=np.number).columns.to_list()
categorical = train[FEATURES].select_dtypes(exclude=np.number).columns.to_list()

train[categorical] = train[categorical].astype(str)
train[TARGET] = train[TARGET].astype(float)
test[categorical] = test[categorical].astype(str)

print(f'\nModel: {modelname}')
print(f'Target: {TARGET}')
print(f'Fetaures:\n\tnumerical: {numerical}\n\tcategorical:{categorical}')
print(f'Shapes:\n\ttrain: {train.shape}\n\ttest: {test.shape}\n')

# data splitting
x, x_val, y, y_val = train_test_split(
    train[FEATURES],
    train[TARGET],
    test_size = 0.2,
    random_state = 42
)

# catboost fixed params
fixed_params = {
    'custom_metric': ['Accuracy', 'AUC', 'F1'],
    'num_trees': 1e4,
    'learning_rate': 0.1,
    'random_seed': 42,
    'bootstrap_type': 'Bayesian',
    'allow_writing_files': False
}

# objective function for optuna
def objective(trial, x, y):

    # prepare data
    x, x_val, y, y_val = train_test_split(
        x, y,
        test_size = 0.2,
        random_state = np.random.randint(0,1000)
    )

    pool_train = Pool(x, y, cat_features=categorical)
    pool_val = Pool(x_val, y_val, cat_features=categorical)

    # define parameter space
    tuning_params = {
        'loss_function': trial.suggest_categorical('loss_function', ['Logloss', 'CrossEntropy']),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 1, 25),
        'random_strength': trial.suggest_float('random_strength', 1, 5),
        'depth': trial.suggest_float('depth', 1, 12, step=1),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 0.8),
        'l2_leaf_reg':trial.suggest_float('l2_leaf_reg', 0.0001, 0.1, log=True),
    }

    params = {**fixed_params, **tuning_params}

    # build and train model with trial hyperparameters
    model = CatBoostClassifier(**params)
    model.fit(
        pool_train,
        eval_set = pool_val,
        early_stopping_rounds=15,
        verbose = 250
    )

    # return accuracy validation score for the model
    return model.best_score_['validation']['Accuracy']

# perform optimization
time_limit = 3600 * 1
np.random.seed(seed)
sampler = TPESampler(seed=seed)

study = optuna.create_study(
    sampler=sampler,
    study_name= f'{modelname}_optimization',
    direction='maximize')

print(f'Starting {modelname} optimization...\n')
study.optimize(
    lambda trial: objective(trial, x, y),
    n_trials = 10,
    timeout=time_limit,
)

# optimization results
print(f"\nNumber of finished trials: {len(study.trials)}")
print(f"Best score: {study.best_value}")
print("Best trial parameters:")
for k, v in study.best_trial.params.items():
    print(f"\t{k}: {v}")

# save final parameters
tuned_params = study.best_trial.params
best_params = {**fixed_params, **tuned_params}

params_path = f'./src/training_files/{modelname}_best_params'
with open(params_path, "wb") as file:
    joblib.dump(best_params, params_path)

# print results
best_params = joblib.load(params_path)
print("\nFinal parameters:")
for k, v in best_params.items():
    print(f"\t{k}: {v}")