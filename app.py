from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='templates')

# Load the trained model
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the scaler object
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Define the home route
@app.route("/")
def home():
    return render_template("index.html")

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_features = [float(x) for x in request.form.values()]
        feature_names = ['Age',	'Sex',	'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',	'EKG results',	'Max HR',	'Exercise angina',	'ST depression',	'Slope of ST',	'Number of vessels fluro',	'Thallium']
        df = pd.DataFrame([input_features], columns=feature_names)
        scaled_features = scaler.transform(df)
        prediction = model.predict(scaled_features)
        return render_template("index.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)


    