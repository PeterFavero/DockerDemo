from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("iris_model.pkl", "rb") as f:
    model = joblib.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

@app.route("/init", methods=["GET"])
def init():
    return "Iris Model Inference API is running."

# At the end of your Flask app file
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)