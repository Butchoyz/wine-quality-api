from flask import Flask, request, jsonify, make_response
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model and imputers
model = joblib.load("xgb_wine_model.pkl")
imputer_mean = joblib.load("imputer_mean_density.pkl")
imputer_median = joblib.load("imputer_median_citric_pH.pkl")

# ✅ CORS middleware to allow Firebase frontend
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://labexam1-c5b75.web.app"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response

# ✅ Handle preflight OPTIONS requests
@app.route("/predict", methods=["OPTIONS"])
def handle_preflight():
    response = make_response()
    response.headers["Access-Control-Allow-Origin"] = "https://labexam1-c5b75.web.app"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response

@app.route("/")
def home():
    return "Wine Quality Predictor API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

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

        # ✅ Corrected imputation: pass citric_acid and pH together
        # Impute density (1 feature)
        input_data[:, 7:8] = imputer_mean.transform(input_data[:, 7:8])

        # Impute citric_acid and pH (2 features together)
        median_input = input_data[:, [2, 8]]
        median_output = imputer_median.transform(median_input)
        input_data[:, 2] = median_output[:, 0]  # citric_acid
        input_data[:, 8] = median_output[:, 1]  # pH

        # Predict
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]

        return jsonify({
            "result": "Good" if prediction == 1 else "Not Good",
            "confidence": round(float(confidence) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
