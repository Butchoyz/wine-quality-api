import React, { useState } from "react";

export default function WinePredictor() {
  const [formData, setFormData] = useState({
    fixed_acidity: "",
    volatile_acidity: "",
    citric_acid: "",
    residual_sugar: "",
    chlorides: "",
    free_sulfur_dioxide: "",
    total_sulfur_dioxide: "",
    density: "",
    pH: "",
    sulphates: "",
    alcohol: "",
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const apiUrl = "https://wine-quality-api-production.up.railway.app/predict";

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
  e.preventDefault();
  setLoading(true);
  setError(null);

  // Convert all form inputs to numbers
  const numericData = {};
  for (const key in formData) {
    const value = parseFloat(formData[key]);
    if (isNaN(value) || value < 0) {
      setError(`Invalid input for ${key.replaceAll("_", " ")}. Must be a non-negative number.`);
      setLoading(false);
      return;
    }
    numericData[key] = value;
  }

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(numericData),
    });

    const data = await response.json();
    if (response.ok) {
      setPrediction(data);
    } else {
      throw new Error(data.error || "API error.");
    }
  } catch (err) {
    setError("Something went wrong. Please try again.");
    console.error(err);
  } finally {
    setLoading(false);
  }
};
  return (
    <div className="min-h-screen bg-gradient-to-br from-red-100 to-yellow-50 flex items-center justify-center p-4">
      <div className="max-w-xl w-full bg-white rounded-2xl shadow-xl p-6">
        <h1 className="text-2xl font-bold mb-4 text-center text-red-700">🍷 Wine Quality Predictor</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          {Object.entries(formData).map(([key, value]) => (
            <div key={key}>
              <label className="block text-sm font-medium text-gray-700 capitalize">
                {key.replaceAll("_", " ")}
              </label>
              <input
                type="number"
                step="any"
                name={key}
                value={value}
                onChange={handleChange}
                required
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-red-500 focus:border-red-500"
              />
            </div>
          ))}
          <button
            type="submit"
            className="w-full py-2 px-4 bg-red-600 text-white rounded-xl shadow hover:bg-red-700 focus:outline-none"
            disabled={loading}
          >
            {loading ? "Predicting..." : "Predict Quality"}
          </button>
        </form>

        {prediction && (
          <div className="mt-6 text-center">
            <h2 className="text-xl font-semibold text-gray-800">
              Result: <span className={prediction.result === "Good" ? "text-green-600" : "text-red-600"}>{prediction.result}</span>
            </h2>
            <p className="text-sm text-gray-600 mt-1">Confidence: {prediction.confidence}%</p>
          </div>
        )}

        {error && <p className="text-red-500 mt-4 text-sm text-center">{error}</p>}
      </div>
    </div>
  );
}
