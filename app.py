from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Load model and imputers
model = joblib.load("xgb_wine_model.pkl")
imputer_mean = joblib.load("imputer_mean_density.pkl")
imputer_median = joblib.load("imputer_median_citric_pH.pkl")

@app.route("/")
def home():
    return "Wine Quality Predictor API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("📥 Received JSON:", data)

        input_data = np.array([
            data["fixed_acidity"],
            data["volatile_acidity"],
            data["citric_acid"],
            data["residual_sugar"],
            data["chlorides"],
            data["free_sulfur_dioxide"],
            data["total_sulfur_dioxide"],
            data["density"],
            data["pH"],
            data["sulphates"],
            data["alcohol"]
        ]).reshape(1, -1)

        print("🧪 Before imputation:", input_data)

        input_data[:, 2:3] = imputer_median.transform(input_data[:, 2:3])  # citric acid
        input_data[:, 7:8] = imputer_mean.transform(input_data[:, 7:8])    # density
        input_data[:, 8:9] = imputer_median.transform(input_data[:, 8:9])  # pH

        print("✅ After imputation:", input_data)

        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]

        print(f"🔮 Prediction: {prediction}, Confidence: {confidence}")

        return jsonify({
            "result": "Good" if prediction == 1 else "Not Good",
            "confidence": round(float(confidence) * 100, 2)
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

