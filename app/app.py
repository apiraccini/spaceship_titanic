# imports

from flask import Flask, request, render_template

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, Pool


# declare a Flask app
app = Flask(__name__) # An instance of this class will be our WSGI application.

modelname = 'catboost'

# main function
# we  use the route() decorator to tell Flask what URL should trigger our function
# the function returns the message we want to display in the user’s browser (here using {{output}}). The default content type is HTML
@app.route('/', methods=['GET', 'POST'])
def main():
    
    # if a form is submitted
    if request.method == "POST":
        
        # load model
        model_path = f'./src/training_files/catboost_best_model'
        model = CatBoostClassifier()
        model.load_model(model_path)

        # get values through input bars
        newdata = []
        FEATURES = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupCount', 'CabinDeck', 'CabinSide']
        for f in FEATURES:
            newdata.append(request.form.get(f))
        
        # prepare prediction data
        newx = pd.DataFrame([newdata], columns=FEATURES)
        categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'CabinDeck', 'CabinSide']
        numerical = list(set(FEATURES)- set(categorical))
        hlp = ['float' if _ in numerical else 'str' for _ in FEATURES]
        typedict = {k:v for (k, v) in zip(FEATURES, hlp)}
        newx = newx.astype(typedict)
        pool_xtest = Pool(newx, cat_features=categorical)

        # get prediction
        pred = model.predict(pool_xtest)[0]
        prediction = 'Transported' if pred==1 else 'Not transported'
    else:
        prediction = ""
        
    return render_template("website.html", output=prediction)

# run the app
if __name__ == '__main__':
    app.run(debug = True) # host=0.0.0.0