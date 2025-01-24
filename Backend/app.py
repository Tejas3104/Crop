from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("crop_recommendation_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON input from frontend
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)
    return jsonify({"recommended_crop": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
