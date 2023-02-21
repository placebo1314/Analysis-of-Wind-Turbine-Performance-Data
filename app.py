from flask import Flask, request
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('model.pkl')

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input parameters from the request
    windspeed = request.form['windspeed']
    direction = request.form['direction']
    month = request.form['month']
    hour = request.form['hour']

    # Make a prediction using the machine learning model
    prediction = model.predict([[windspeed, direction, month, hour]])

    # Return the prediction as a JSON response
    return {'prediction': prediction[0]}