from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models and encoders
clf1 = joblib.load("disease_type_model.pkl")
clf2 = joblib.load("disease_severity_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

input_features = ["Pollutant_Type", "Pollutant_Concentration_mg_kg", "Soil_pH", "Temperature_C"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = []
    for feature in input_features:
        val = request.form.get(feature)
        if feature in label_encoders:
            val = label_encoders[feature].transform([val])[0]
        else:
            val = float(val)
        data.append(val)

    input_array = np.array([data])

    # Predict
    disease_type = clf1.predict(input_array)[0]
    disease_severity = clf2.predict(input_array)[0]

    # Decode predictions if necessary
    if "Disease_Type" in label_encoders:
        disease_type = label_encoders["Disease_Type"].inverse_transform([disease_type])[0]
    if "Disease_Severity" in label_encoders:
        disease_severity = label_encoders["Disease_Severity"].inverse_transform([disease_severity])[0]

    return render_template('index.html', prediction_text=f"Disease Type: {disease_type}, Severity: {disease_severity}")

if __name__ == "__main__":
    app.run(debug=True)
