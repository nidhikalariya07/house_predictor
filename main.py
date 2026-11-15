from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open('ridgemodel.pkl', 'rb'))


@app.route("/")
def index():
    locations = sorted(data['City'].unique())
    return render_template("index.html", locations=locations)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        location = request.form.get("city")
        bhk = int(request.form.get("bhk"))
        bath = int(request.form.get("bath"))
        sqft = float(request.form.get("total_sqft"))

        print(f"Received -> City: {location}, BHK: {bhk}, Bathrooms: {bath}, Super_Area_sqft: {sqft}")

        # FIX: Match the exact feature names and order expected by ridgemodel.pkl
        input_data = pd.DataFrame([[location, bhk, bath, sqft]],
                                  columns=['City', 'BHK', 'Bathrooms', 'Super_Area_sqft'])

        prediction = pipe.predict(input_data)[0]
        return str(np.round(prediction, 2))

    except Exception as e:
        print("CRITICAL ERROR:", str(e))
        return str(e), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)