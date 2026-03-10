# Federated Learning for Electronic Health Records (EHR)

This module implements a privacy-preserving federated learning pipeline for predicting hospital readmission using Electronic Health Record (EHR) data.

## Key Components

The framework integrates several modern machine learning techniques:

• **Federated Representation Learning (FedRep)**
• **Differential Privacy (DP-SGD via Opacus)**
• **Federated Local Normalization (FedBN-style domain alignment)**
• **Class imbalance handling (Focal Loss + balanced batching)**
• **Hybrid ensemble with XGBoost**

The goal is to enable **collaborative learning across hospitals without sharing patient data**.

---

## Dataset

Diabetes hospital readmission dataset

Download from:

https://www.kaggle.com/datasets/brandao/diabetes

Place the file here:

```
data/diabetic_data.csv
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Federated Training

```
python train_ehr_federated.py
```

---

## Privacy Analysis

```
python privacy_analysis.py
```

This evaluates the **privacy–utility trade-off** by varying the differential privacy noise multiplier.

---

## Benchmarks

```
python benchmark_models.py
```

This compares the proposed model with classical baselines:

• Logistic Regression
• Random Forest
• XGBoost
• Federated baselines (FedAvg / FedProx)

---

## Research Contribution

The proposed hybrid federated framework demonstrates:

* strong privacy guarantees
* improved predictive performance
* robustness to heterogeneous hospital data

This implementation accompanies our research on **privacy-preserving federated learning for healthcare analytics**.
