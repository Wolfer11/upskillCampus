from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os
from data_preprocessing import preprocess_data

# ✅ Paths
DATASET_PATH = r"E:\Smart city forecasting\Project9_smart-city-traffic-patterns\Project9_smart-city-traffic-patterns\smart-city-traffic-patterns\traffic_data.csv"
MODEL_PATH = r"E:\Smart city forecasting\traffic_forecast.h5"
SEQ_LENGTH = 24

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for Postman & other clients

# ✅ Load and preprocess data
try:
    df, scaler = preprocess_data(DATASET_PATH)
    print("✅ Data preprocessed successfully!")
except Exception as e:
    print(f"❌ Error in preprocessing data: {e}")
    scaler = None  # Avoid crash

# ✅ Load Model with Error Handling
try:
    @tf.keras.utils.register_keras_serializable()
    def mse(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"mse": mse})
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Avoid crashes

# ✅ Prediction API
@app.route('/predict', methods=['POST'])

def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': "Model or scaler not loaded. Check logs for details."}), 500

        data = request.get_json()
        
        # ✅ Validate input
        if not data or 'sequence' not in data:
            return jsonify({'error': "Missing 'sequence' in request data"}), 400
        
        sequence = data['sequence']

        # ✅ Ensure valid sequence
        if not isinstance(sequence, list) or len(sequence) != SEQ_LENGTH:
            return jsonify({'error': f"Expected list of {SEQ_LENGTH} values"}), 400

        input_sequence = np.array(sequence).reshape(1, SEQ_LENGTH, 1)

        # ✅ Make Prediction
        prediction = model.predict(input_sequence)
        forecasted_traffic = scaler.inverse_transform([[prediction[0][0]]])[0][0]

        return jsonify({'forecasted_traffic': forecasted_traffic})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(debug=True, port=5000)
