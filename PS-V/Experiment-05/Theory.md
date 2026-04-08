Below is a **step-by-step guide to perform this experiment properly**, with corrected code, command sequence, and what to expect at each stage.

# Experiment: Deploy a Simple AI Application on Kubernetes

## Objective

To build a simple Flask-based AI application, containerize it using Docker, and deploy it on Kubernetes using Minikube.

---

# 1. Prerequisites

Before starting, make sure these are installed:

* Python
* Docker
* Minikube
* kubectl
* VS Code or any code editor

You should also ensure:

* Docker Desktop is running
* Minikube can access Docker
* kubectl is configured correctly

To verify installation, run:

```bash
docker --version
minikube version
kubectl version --client
python --version
```

---

# 2. Start Kubernetes Cluster

Run:

```bash
minikube start
```

This starts a local Kubernetes cluster.

### Expected result

You should see messages showing:

* Minikube is starting
* Kubernetes components are being set up
* Cluster is ready

To verify:

```bash
kubectl get nodes
```

### Expected output

You should see one node in `Ready` state.

---

# 3. Create Project Folder

Run:

```bash
mkdir ai-k8s-app
cd ai-k8s-app
```

This creates your project directory and moves into it.

---

# 4. Create the Flask AI Application

Create a file named:

```bash
app.py
```

Paste this corrected code:

```python
from flask import Flask, jsonify
import random

app = Flask(__name__)

@app.route('/')
def home():
    return "Simple AI Application Running on Kubernetes"

@app.route('/predict')
def predict():
    prediction = random.choice(["Positive", "Negative"])
    return jsonify({"Prediction": prediction})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```

### What this does

* `/` returns a simple message
* `/predict` returns a random prediction in JSON format

---

# 5. Create requirements.txt

Create a file named:

```bash
requirements.txt
```

Add:

```txt
flask
```

This tells Docker which Python package to install.

---

# 6. Create Dockerfile

Create a file named:

```bash
Dockerfile
```

Add:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
```

### What this does

* Uses Python 3.9 slim image
* Sets `/app` as working directory
* Copies project files
* Installs Flask
* Exposes port 5000
* Runs the Flask app

---

# 7. Build Docker Image Inside Minikube Environment

This step is important because Kubernetes inside Minikube should access the image directly.

## For macOS/Linux

Run:

```bash
eval $(minikube docker-env)
docker build -t ai-k8s-app .
```

## For Windows PowerShell

Run:

```powershell
minikube -p minikube docker-env --shell powershell | Invoke-Expression
docker build -t ai-k8s-app .
```

## For Windows CMD

Run:

```cmd
@FOR /f "tokens=*" %i IN ('minikube -p minikube docker-env --shell cmd') DO @%i
docker build -t ai-k8s-app .
```

### Expected result

Docker should build the image successfully with the tag:

```bash
ai-k8s-app
```

To verify:

```bash
docker images
```

You should see `ai-k8s-app` in the list.

---

# 8. Create Kubernetes Deployment File

Create a file named:

```bash
deployment.yaml
```

Add this corrected YAML:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-app
  template:
    metadata:
      labels:
        app: ai-app
    spec:
      containers:
        - name: ai-container
          image: ai-k8s-app
          imagePullPolicy: Never
          ports:
            - containerPort: 5000
```

### Why `imagePullPolicy: Never` is important

Since the image is built locally inside Minikube, Kubernetes should not try to pull it from Docker Hub.

---

# 9. Apply Deployment

Run:

```bash
kubectl apply -f deployment.yaml
```

### Expected output

```bash
deployment.apps/ai-deployment created
```

---

# 10. Verify Deployment and Pods

Run:

```bash
kubectl get deployments
kubectl get pods
```

### Expected result

You should see:

* deployment `ai-deployment`
* 2 pods running

Example:

```bash
NAME             READY   UP-TO-DATE   AVAILABLE   AGE
ai-deployment    2/2     2            2           20s
```

and

```bash
NAME                             READY   STATUS    RESTARTS   AGE
ai-deployment-xxxxxxxxxx-xxxxx   1/1     Running   0          20s
ai-deployment-xxxxxxxxxx-yyyyy   1/1     Running   0          20s
```

---

# 11. If Pod Has an Issue, Diagnose It

If pods are not running, use:

```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

Example:

```bash
kubectl logs ai-deployment-xxxxxxxxxx-xxxxx
```

This helps you check:

* image errors
* app startup issues
* dependency issues

---

# 12. Create Kubernetes Service File

Create a file named:

```bash
service.yaml
```

Add this corrected YAML:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-service
spec:
  type: NodePort
  selector:
    app: ai-app
  ports:
    - port: 80
      targetPort: 5000
      nodePort: 30007
```

### What this does

* Exposes the app outside the cluster
* Maps internal port 5000 to external NodePort 30007

---

# 13. Apply Service

Run:

```bash
kubectl apply -f service.yaml
```

### Expected output

```bash
service/ai-service created
```

---

# 14. Verify Service

Run:

```bash
kubectl get services
```

### Expected result

You should see `ai-service` with NodePort `30007`.

Example:

```bash
NAME         TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
ai-service   NodePort    10.x.x.x        <none>        80:30007/TCP   20s
```

---

# 15. Access the Application

## Option 1: Using Minikube Service

Run:

```bash
minikube service ai-service
```

This usually opens the application in your browser automatically.

## Option 2: Using Minikube IP

Run:

```bash
minikube ip
```

Suppose the IP returned is:

```bash
192.168.49.2
```

Then open these URLs in your browser:

### Home endpoint

```bash
http://192.168.49.2:30007
```

### Predict endpoint

```bash
http://192.168.49.2:30007/predict
```

---

# 16. Expected Output

## Home Page Output

When you open:

```bash
http://<minikube-ip>:30007
```

You should see:

```text
Simple AI Application Running on Kubernetes
```

## Predict Endpoint Output

When you open:

```bash
http://<minikube-ip>:30007/predict
```

You should see JSON like:

```json
{"Prediction":"Positive"}
```

or

```json
{"Prediction":"Negative"}
```

---

# 17. Manage Pods and Services

## View deployments

```bash
kubectl get deployments
```

## Scale pods

```bash
kubectl scale deployment ai-deployment --replicas=4
```

Then verify:

```bash
kubectl get pods
```

You should now see 4 running pods.

## Delete a pod

```bash
kubectl delete pod <pod-name>
```

Kubernetes will automatically create a new pod if deployment still exists.

## Delete service

```bash
kubectl delete service ai-service
```

## Delete deployment

```bash
kubectl delete deployment ai-deployment
```

---

# 18. Recommended File Structure

Your folder should look like this:

```bash
ai-k8s-app/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── deployment.yaml
└── service.yaml
```

---

# 19. Full Command Sequence

Here is the full command flow in order.

## Start cluster

```bash
minikube start
kubectl get nodes
```

## Create project

```bash
mkdir ai-k8s-app
cd ai-k8s-app
```

## Build image

### macOS/Linux

```bash
eval $(minikube docker-env)
docker build -t ai-k8s-app .
```

### Windows PowerShell

```powershell
minikube -p minikube docker-env --shell powershell | Invoke-Expression
docker build -t ai-k8s-app .
```

## Deploy app

```bash
kubectl apply -f deployment.yaml
kubectl get deployments
kubectl get pods
```

## Expose app

```bash
kubectl apply -f service.yaml
kubectl get services
```

## Access app

```bash
minikube service ai-service
```

or

```bash
minikube ip
```

---

# 20. Common Errors and Fixes

## Error: ImagePullBackOff

### Cause

Kubernetes is trying to pull image from Docker Hub.

### Fix

Ensure in `deployment.yaml`:

```yaml
imagePullPolicy: Never
```

Also rebuild image inside Minikube Docker environment.

---

## Error: Pods not starting

### Fix

Run:

```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

Check for:

* Python syntax errors
* missing Flask package
* wrong port

---

## Error: Service not accessible

### Fix

Check:

```bash
kubectl get services
minikube ip
```

Also ensure NodePort is correctly set to `30007`.

---

# 21. Result

The Flask-based AI application was successfully containerized using Docker and deployed on Kubernetes using Minikube. The deployment created multiple pods, and the service exposed the application externally through a NodePort. The `/predict` API returned random JSON predictions successfully.

---

# 22. Conclusion

This experiment demonstrates how a simple AI application can be deployed in a Kubernetes environment. It covers application development, Docker image creation, Kubernetes deployment, service exposure, scaling, and verification. It provides practical understanding of container orchestration and prepares the foundation for deploying larger AI and DevOps applications.

If you want, I can next turn this into a **proper lab record format with Aim, Theory, Procedure, Result, and Viva Questions**.
