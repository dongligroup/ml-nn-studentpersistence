from flask import Flask, request, jsonify
from flask_cors import CORS
from core import Model, Student

app = Flask(__name__)
CORS(app)

# Load the model (could be initialized once and reused)
model = Model()

@app.route('/', methods=['GET'])
def test():
    message = 'back end is on.'
    return message

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to accept a single data point and return the prediction.
    """
    try:
        data = request.get_json()  # Expecting JSON input
        student = Student.json_to_student(data)  # Convert JSON to Student object
        prediction = model.predict_one(student)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/train-one', methods=['POST'])
def train_one():
    """
    Endpoint to accept a single labeled data point and train the model.
    """
    try:
        data = request.get_json()  # Expecting JSON input
        student_labelled = Student.json_to_student(data)
        model.train_one(student_labelled)
        return jsonify({"message": "Model trained on one instance successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/train-batch', methods=['POST'])
def train_batch():
    """
    Endpoint to accept a CSV file containing labeled data and train the model.
    """
    try:
        file = request.files['file']  # Expecting the CSV file under 'file'
        file_path = f"./data/{file.filename}"
        file.save(file_path)
        acc, loss = model.train_batch(file_path)  # Train on the batch
        return jsonify({"accuracy": acc[-1], "loss": loss[-1]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
