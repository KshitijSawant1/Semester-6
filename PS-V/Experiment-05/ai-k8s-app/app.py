from flask import Flask, jsonify
import random
from datetime import datetime

app = Flask(__name__)

# Static student/project details
student_info = {
    "Name": "Kshitij Sawant",
    "Subject": "PS-V : AI DEVOPS",
    "Experiment": "Experiment 05 - Kubernetes Deployment",
    "Date": datetime.now().strftime("%Y-%m-%d"),
    "Roll Number": "66", 
    "Department": "Artificial Intelligence and Data Science"
}

@app.route('/')
def home():
    return jsonify({
        "Message": "Simple AI Application Running on Kubernetes",
        "Details": student_info
    })

@app.route('/predict')
def predict():
    prediction = random.choice(["Positive", "Negative"])
    
    return jsonify({
        "Prediction": prediction,
        "Details": student_info
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)