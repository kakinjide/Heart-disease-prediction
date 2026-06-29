# Clinical Risk Stratification: Heart Disease Prediction Using Random Forest

An end-to-end machine learning pipeline built with Scikit-Learn to predict the presence of heart disease based on clinical patient variables. This project simulates an automated triage tool designed to optimize clinical workflows, reduce diagnostic bottlenecks, and improve patient outcomes.

## 📌 Project Overview
In clinical settings, manual patient risk stratification can cause critical diagnostic delays. This project frames heart disease detection as a binary classification challenge. By leveraging consumer health data and clinical metrics, the model automatically flags high-risk patients, enabling healthcare providers to prioritize care effectively.

## 🛠️ Tech Stack & Tools
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Model Serialization:** Pickle

## 📊 Dataset Features
The model processes 13 clinical features to predict the binary target (`0 = No Disease`, `1 = Disease Present`):
* **Demographics:** Age, Sex
* **Clinical Metrics:** Chest pain type (cp), Resting blood pressure (trestbps), Serum cholesterol (chol), Fasting blood sugar (fbs), Resting electrocardiographic results (restecg), Maximum heart rate achieved (thalach), Exercise-induced angina (exang), ST depression (oldpeak), Slope of peak exercise ST segment (slope), Number of major vessels (ca), Thalassemia type (thal).

## 🚀 Machine Learning Workflow

### 1. Data Processing & Exploratory Data Analysis (EDA)
* Automated data ingestion and verification using **Pandas**.
* Feature distribution tracking and clinical anomaly detection visualized using **Matplotlib**.

### 2. Model Selection & Reproducibility
* Implemented a **Random Forest Classifier** to naturally handle complex, non-linear clinical feature interactions.
* Enforced absolute pipeline reproducibility across data splits (`train_test_split`) and model initialization using deterministic random states (`random_state=42`).

### 3. Evaluation Strategy
In digital healthcare, mitigating false negatives is critical. Evaluation focuses on:
* **Accuracy Score:** Overall classification correctness.
* **Recall / Sensitivity:** Ensuring high-risk patients are not missed by the model.
* **Confusion Matrix:** Granular tracking of true vs. false classifications.

### 4. Production Serialization
* Serialized the final trained model into a production-ready `.pkl` file via **Pickle**, allowing for seamless integration into web applications or healthcare microservice APIs.

## 💻 Quick Start & Usage

### Installation
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Running the Pipeline
```python
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Ingest clinical data
heart_disease = pd.read_csv("heart-disease.csv")
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]

# 2. Reproducible train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model training
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. In-production simulation (Predicting a single patient pathway)
# Expects an array matching the 13 clinical features reshaped to 2D (1, 13)
sample_patient = np.array([63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]).reshape(1, -1)
prediction = clf.predict(sample_patient)
print(f"Patient Triage Category: {'High Risk (1)' if prediction[0] == 1 else 'Low Risk (0)'}")

# 5. Serialize for deployment
with open("random_forest_model_1.pkl", "wb") as f:
    pickle.dump(clf, f)
```

## 📈 Key Results & Takeaways
* **Clinical Accountability:** Random Forest feature importances were extracted to provide transparency into which clinical metrics (e.g., maximum heart rate, chest pain type) drove the risk scores.
* **Operational Impact:** Transitioning triage from manual review to an automated pre-assessment framework can scale operational efficiency and accelerate clinical service delivery.
