from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('./model/base_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Validate input data
        if 'features' not in data:
            return jsonify({'error': 'Invalid input data'}), 400
        # Convert the input into a NumPy array
        input_data = np.array([data['features']])
        # Perform prediction
        prediction = model.predict(input_data)
        # Send the result back to Node.js backend
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000)