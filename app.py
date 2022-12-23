## app

import platform; print(platform.platform())
import sys; print("Python", sys.version)

# imports

from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 

import joblib

# setup
modelname = 'lgb'
numerical = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Expenditure', 'CabinNum', 'GroupSize', 'FamilySize']
categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'NoSpending', 'CabinDeck', 'CabinSide', 'Solo']
FEATURES = numerical + categorical

# preprocessing
cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
preproc = ColumnTransformer(
    transformers=[('cat', cat_encoder, categorical)],
    verbose_feature_names_out=False,
    remainder='passthrough'
)

# load train data and fit preprocessing pipeline
train = pd.read_csv('./data/final/train.csv')
preproc = preproc.fit(train[FEATURES])

# declare a Flask app
app = Flask(__name__) # An instance of this class will be our WSGI application.

# main function
# we  use the route() decorator to tell Flask what URL should trigger our function
# the function returns the message we want to display in the user’s browser (here using {{output}}). The default content type is HTML
@app.route('/', methods=['GET', 'POST'])
def main():
    
    # if a form is submitted
    if request.method == "POST":
        
        # load model
        model_path = f'./src/training_files/{modelname}_best_model.joblib'
        with open(model_path, 'rb') as file:
            model = joblib.load(file)

        # get values through input bars
        newdata = [request.form.get(f) for f in FEATURES]
        
        # prepare prediction data
        xtest = pd.DataFrame([newdata], columns=FEATURES)
        hlp = ['float' if _ in numerical else 'str' for _ in FEATURES]
        typedict = {k:v for (k, v) in zip(FEATURES, hlp)}
        xtest = xtest.astype(typedict)
        xtest = preproc.transform(xtest)

        # prediction
        preds = model.predict(np.array(xtest))
        probs = model.predict_proba(xtest)[0][1]

        # output
        prediction = 'Transported' if preds==1 else 'Not transported'
        out = f'{prediction} ({probs:.2%})'

    else:
        out = "(waiting for input)"
        
    return render_template("website.html", output=out)

# run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0') # host=0.0.0.0