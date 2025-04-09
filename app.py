# app.py
from flask import Flask, request, render_template
import numpy as np
import pickle
import os

# Load model and preprocessor from the models folder
MODEL_PATH = os.path.join("models", "dtr.pkl")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")

dtr = pickle.load(open(MODEL_PATH, 'rb'))
preprocessor = pickle.load(open(PREPROCESSOR_PATH, 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)

        raw_prediction = dtr.predict(transformed_features)[0]

        # Convert yield
        yield_hg_ha = round(raw_prediction, 2)
        yield_kg_ha = round(yield_hg_ha / 10, 2)
        yield_ton_ha = round(yield_kg_ha / 1000, 4)
        
        # Conversion to acre-based yield
        yield_kg_acre = round(yield_kg_ha / 2.47105, 2)
        yield_ton_acre = round(yield_kg_acre / 1000, 4)


        return render_template('index.html',
                       yield_hg_ha="{:,.2f}".format(yield_hg_ha),
                       yield_kg_ha="{:,.2f}".format(yield_kg_ha),
                       yield_ton_ha="{:,.4f}".format(yield_ton_ha),
                       yield_kg_acre="{:,.2f}".format(yield_kg_acre),
                       yield_ton_acre="{:,.4f}".format(yield_ton_acre))

    except Exception as e:
        return render_template('index.html', error="Something went wrong. Please check your inputs.")

if __name__ == '__main__':
    app.run(debug=True)
