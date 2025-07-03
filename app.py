from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)

# ✅ Allow CORS for ALL routes and your web frontend
CORS(app, resources={r"/predict": {"origins": "https://labexam1-c5b75.web.app"}})

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

        # Define expected feature names in correct order
        feature_names = [
            "fixed_acidity", "volatile_acidity", "citric_acid",
            "residual_sugar", "chlorides", "free_sulfur_dioxide",
            "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"
        ]

        # Create a DataFrame instead of a NumPy array
        import pandas as pd
        input_df = pd.DataFrame([[
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
        ]], columns=feature_names)

        # Apply imputers (now using DataFrame with column names)
        input_df[["citric_acid"]] = imputer_median.transform(input_df[["citric_acid"]])
        input_df[["density"]] = imputer_mean.transform(input_df[["density"]])
        input_df[["pH"]] = imputer_median.transform(input_df[["pH"]])

        # Predict using model
        prediction = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df)[0][prediction]

        return jsonify({
            "result": "Good" if prediction == 1 else "Not Good",
            "confidence": round(float(confidence) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
