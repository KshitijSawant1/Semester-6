
# Machine Learning Environment Setup (macOS)

## 1. Navigate to Project Directory

Open Terminal and move to the project folder:

```bash
cd ~/Desktop/ML\ BG
```


## 2. Create Virtual Environment

Create a virtual environment using Python:

```bash
python3 -m venv ml_env
```

---

## 3. Activate Virtual Environment

Activate the environment:

```bash
source ml_env/bin/activate
```

---

## 4. Upgrade pip

```bash
pip install --upgrade pip
```

---

## 5. Install Required Libraries

Install core machine learning libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Install Jupyter support:

```bash
pip install jupyter notebook ipykernel
```

(Optional) Install deep learning libraries:

```bash
pip install tensorflow
```

or

```bash
pip install torch torchvision
```

---

## 6. Configure Jupyter Kernel

```bash
python -m ipykernel install --user --name=ml_env --display-name "Python (ML BG)"
```

---

## 7. Open Project in VS Code

```bash
code .
```

Select the interpreter:

* Open Command Palette
* Choose "Python: Select Interpreter"
* Select `ml_env`

---

## 8. Save Dependencies

```bash
pip freeze > requirements.txt
```

---

## 9. Deactivate Environment

```bash
deactivate
```

---

## Conclusion

The machine learning environment is successfully configured and ready for use.



