# ✈️ Aerospace Remaining Useful Life Prediction System

Predictive maintenance platform for turbofan engines using machine learning and deep learning models trained on NASA's C-MAPSS degradation dataset.

## 🚀 Project Overview

This project predicts the Remaining Useful Life (RUL) of aircraft turbofan engines by analyzing operational and sensor telemetry data.

The system simulates a real-world predictive maintenance workflow by:

* Forecasting engine degradation
* Estimating remaining operational cycles
* Generating maintenance recommendations
* Assigning maintenance priorities
* Delivering results through a Spring Boot API and Streamlit dashboard

The goal is to demonstrate how machine learning can support maintenance planning, reduce downtime, and improve fleet reliability.

## 🤖 Supported Prediction Models

| Model             | Type                 |
| ----------------- | -------------------- |
| LSTM              | Deep Learning        |
| GRU               | Deep Learning        |
| SimpleRNN         | Deep Learning        |
| Random Forest     | Ensemble Learning    |
| Linear Regression | Statistical Learning |

Users can switch between models directly from the dashboard and compare prediction outputs.

## 📊 Dataset

NASA C-MAPSS Turbofan Engine Degradation Dataset

Features include:

* Operational settings
* Sensor measurements
* Engine cycle history
* Simulated degradation patterns

The dataset is commonly used for predictive maintenance and Remaining Useful Life (RUL) forecasting research.

## 🏗️ System Architecture

Data Processing → Model Training → Prediction Engine → Spring Boot API → Streamlit Dashboard → Maintenance Alerts

## ⚙️ Technology Stack

### Machine Learning

* Python
* PyTorch
* Scikit-Learn
* NumPy
* Pandas

### Backend

* Java
* Spring Boot
* Maven

### Dashboard

* Streamlit

### Development

* Git
* GitHub
* VS Code

## 🔧 Key Features

* Multi-model prediction system
* Predictive maintenance recommendations
* Priority-based work order generation
* Interactive dashboard
* REST API integration
* Model comparison capability
* JSON maintenance alert generation
* Real-time prediction workflow

## 📈 Example Output

```json
{
  "asset_id": "ENGINE_1",
  "predicted_rul": 129.75,
  "maintenance_type": "Predictive",
  "priority": "LOW",
  "recommended_action": "Continue monitoring"
}
```
## Dashboard Screenshots

### Multi-Model Dashboard

Supports LSTM, GRU, SimpleRNN, Random Forest, and Linear Regression.

<img width="657" height="778" alt="Dashboard-Home" src="https://github.com/user-attachments/assets/3ab4212d-dadb-410a-b8a3-6f5ca0314276" />


### Prediction Results

Generated Remaining Useful Life (RUL) predictions and maintenance recommendations.

<img width="1431" height="783" alt="Prediction-Results" src="https://github.com/user-attachments/assets/e4bdd38d-26ec-4e38-849d-f7c4b6e98f3b" />


### Priority Filtering

Filter maintenance recommendations by severity.

<img width="654" height="570" alt="Priority-Filter " src="https://github.com/user-attachments/assets/a79ea901-ccf7-41e6-bfa0-f02058dc770f" />


### System Architecture

End-to-end predictive maintenance workflow.

<img width="491" height="639" alt="system-architecture" src="https://github.com/user-attachments/assets/6a70abed-480c-46f5-abbc-0e04f6b07208" />

## 🚀 Environment Setup

### Prerequisites

* Python 3.11
* Conda (Miniconda or Anaconda)

### Clone the repository

```bash
git clone https://github.com/loveLaceLogic/aerospace-rul-prediction.git
cd aerospace-rul-prediction
```

### Create the Conda environment

```bash
conda env create -f environment.yml
```

### Activate the environment

```bash
conda activate aerospace-rul
```

### Train the model

```bash
python -m scripts.train
```

### Launch the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```
