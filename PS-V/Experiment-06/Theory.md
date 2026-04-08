Below is a **step-by-step execution guide for Experiment-6** in a **clean, lab-ready format**, so you can perform it smoothly on your system.

---

# **Experiment-6: Python App → Prometheus → Grafana Monitoring**

---

# **1. Prerequisites**

Make sure these are installed:

```bash
python3 --version
pip3 --version
```

If not:

```bash
brew install python
```

---

# **2. Install Prometheus**

## Download Prometheus

```bash
brew install prometheus
```

## Run Prometheus

```bash
prometheus --config.file=/opt/homebrew/etc/prometheus.yml
```

### Open UI:

```
http://localhost:9090
```

---

# **3. Install Grafana**

```bash
brew install grafana
```

## Start Grafana

```bash
grafana-server
```

### Open:

```
http://localhost:3000
```

### Login:

* Username: admin
* Password: admin

---

# **4. Create Python Monitoring Application**

## Install libraries

```bash
pip3 install flask prometheus_client
```

---

## Create `app.py`

```python
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
```

---

## Run Application

```bash
python3 app.py
```

---

## Test Application

### Open:

```
http://localhost:5000
```

### Metrics endpoint:

```
http://localhost:8000
```

You should see metrics like:

```
app_requests_total
request_processing_seconds
```

---

# **5. Configure Prometheus**

## Locate config file

If installed via brew:

```bash
open /opt/homebrew/etc/prometheus.yml
```

---

## Add this inside `scrape_configs`

```yaml
scrape_configs:
  - job_name: 'python_app'
    static_configs:
      - targets: ['localhost:8000']
```

---

## Restart Prometheus

Stop and run again:

```bash
prometheus --config.file=/opt/homebrew/etc/prometheus.yml
```

---

# **6. Test in Prometheus**

Open:

```
http://localhost:9090
```

In query box type:

```
app_requests_total
```

Click **Execute**

### Expected:

Graph or increasing counter value

---

# **7. Connect Prometheus to Grafana**

## Steps

1. Open Grafana:

```
http://localhost:3000
```

2. Go to:

```
Settings → Data Sources
```

3. Click:

```
Add Data Source
```

4. Select:

```
Prometheus
```

5. Enter URL:

```
http://localhost:9090
```

6. Click:

```
Save & Test
```

---

# **8. Create Dashboard in Grafana**

## Steps

1. Click:

```
➕ → Dashboard
```

2. Click:

```
Add New Panel
```

3. In Query section enter:

```
app_requests_total
```

4. Visualization:

```
Time Series
```

5. Click:

```
Apply
```

---

# **9. Generate Data (Important Step)**

Keep refreshing:

```
http://localhost:5000
```

👉 This increases request count

---

# **10. Expected Output**

## Prometheus

* Shows increasing value of:

```
app_requests_total
```

## Grafana Dashboard

* Graph showing request count increasing over time

---

# **11. Full Flow Understanding**

```
Python App → exposes metrics (/metrics)
        ↓
Prometheus → scrapes metrics (localhost:8000)
        ↓
Grafana → visualizes data (dashboard)
```

---

# **12. Common Errors & Fixes**

## Error: No data in Prometheus

✔ Check:

```bash
http://localhost:8000
```

---

## Error: Grafana not connecting

✔ Check URL:

```
http://localhost:9090
```

---

## Error: Module not found

✔ Run:

```bash
pip3 install flask prometheus_client
```

---

## Error: Port already in use

✔ Change port in app.py:

```python
app.run(port=5001)
```

---

# **13. Result**

The Python application successfully exposed metrics using Prometheus client. Prometheus scraped these metrics, and Grafana visualized them in a dashboard, demonstrating real-time monitoring of application performance.

---

# **14. Conclusion**

This experiment demonstrates how monitoring is implemented in DevOps pipelines using Prometheus and Grafana. It shows how application metrics can be collected, analyzed, and visualized to ensure system performance, reliability, and observability.

---

If you want next, I can:

* Convert this into **lab record format (Aim, Theory, Procedure, Result)**
* Create **implementation table like previous experiment**
* Give **viva questions + answers**
* Help you make **better dashboard (CPU, latency, requests/sec)**
