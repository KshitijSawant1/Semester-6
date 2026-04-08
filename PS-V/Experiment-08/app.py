from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "Name": "Kshitij Sawant",
        "Subject": "PS-V",
        "Experiment Title": "Implement a CI/CD pipeline for automated deployment of an AI application",
        "Message": "Experiment 08 CI/CD Running"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)