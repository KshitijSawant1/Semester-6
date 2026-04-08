from flask import Flask
from prometheus_client import start_http_server, Counter, Summary
import random
import time

app = Flask(__name__)

REQUEST_COUNT = Counter('app_requests_total', 'Total App Requests')
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

@app.route("/")
@REQUEST_TIME.time()
def home():
    REQUEST_COUNT.inc()
    time.sleep(random.random())
    return "Hello, Prometheus Monitoring!"

if __name__ == "__main__":
    start_http_server(8000)   # Metrics exposed here
    app.run(port=5000)