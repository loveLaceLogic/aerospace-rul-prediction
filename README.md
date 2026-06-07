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
