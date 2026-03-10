# Privacy-Preserving Federated Learning for Multi-Modal Healthcare AI

This repository implements a **privacy-preserving federated learning framework for medical AI** across multiple healthcare data modalities.

The system enables **hospitals to collaboratively train machine learning models without sharing sensitive patient data**, combining federated learning, differential privacy, and modern deep learning techniques.

The project simulates real-world clinical collaboration where patient data cannot be centralized due to **HIPAA / GDPR privacy regulations**.

---

# Project Overview

Modern medical AI systems often require **large datasets from multiple hospitals**, but privacy regulations prevent sharing raw patient data.

This repository provides a **secure federated learning framework** where:

* Hospitals train models **locally**
* Only **privacy-protected model updates** are shared
* A **global medical AI model** is collaboratively learned

The framework supports **multiple healthcare modalities**.

---

# Supported Medical Modalities

## 1️⃣ Electronic Health Records (EHR)

Predicts **hospital readmission risk** using structured clinical features.

Key components:

* Federated Representation Learning (FedRep)
* Federated Local Normalization (FedBN-style)
* Differential Privacy using DP-SGD
* Hybrid neural network + XGBoost ensemble

Folder:

```
ehr federated learning/
```

---

## 2️⃣ Brain Tumor MRI Classification

Federated deep learning for **multi-class brain tumor detection** using MRI scans.

Key components:

* ResNet18 CNN architecture
* Federated training using Flower
* Non-IID hospital simulation via Dirichlet partitioning
* Adaptive differential privacy noise injection

Folder:

```
mri federated learning/
```

---

## 3️⃣ Chest X-Ray Pneumonia Detection

Privacy-preserving federated learning for **pneumonia diagnosis from chest X-rays**.

Key components:

* ResNet18 feature extractor
* FedProx optimization for heterogeneous hospitals
* Differential Privacy with Opacus
* Cost-sensitive loss to reduce missed diagnoses

Folder:

```
xray federated learning/
```

---

# Repository Structure

```
Federated-Healthcare-DP-Learning

README.md

ehr federated learning
    train_ehr_federated.py
    privacy_analysis.py
    benchmark_models.py

mri federated learning
    train_mri_federated.py
    privacy_modes_experiment.py

xray federated learning
    train_xray_federated.py
   
```

---

# Core Technologies

This framework combines several modern machine learning techniques:

• Federated Learning
• Differential Privacy (DP-SGD)
• Deep Neural Networks (CNNs)
• Medical Cost-Sensitive Learning
• Non-IID Data Simulation (Dirichlet Partitioning)

Key libraries used:

* PyTorch
* Flower
* Opacus
* Scikit-learn
* XGBoost

---

# Privacy Architecture

The system provides multiple layers of privacy protection.

```
Hospitals
    ↓
Local Model Training
    ↓
Differential Privacy (DP-SGD)
    ↓
Federated Aggregation
    ↓
Global Medical AI Model
```

Hospitals **never share raw medical data**.

Only **privacy-protected gradients or model parameters** are exchanged.

---

# Installation

Install the required dependencies:

```bash
pip install torch torchvision opacus flwr numpy scikit-learn matplotlib seaborn xgboost
```

---

# Running Experiments

Each module can be run independently.

### EHR Federated Model

```bash
python ehr federated learning/train_ehr_federated.py
```

### MRI Federated Model

```bash
python mri federated learning/train_mri_federated.py
```

### Chest X-ray Federated Model

```bash
python xray federated learning/train_xray_federated.py
```

---

# Datasets

The datasets used in this project are publicly available but **not included in the repository** due to size and licensing restrictions.

### EHR Dataset

Diabetes hospital readmission dataset
https://www.kaggle.com/datasets/brandao/diabetes

### Brain MRI Dataset

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Chest X-ray Dataset

https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets

---

# Research Contributions

This repository demonstrates how federated learning can be applied to **multi-modal healthcare data while maintaining strict privacy guarantees**.

Key contributions include:

* Privacy-preserving training across hospitals
* Federated learning under non-IID clinical data
* Differential privacy integration in medical AI
* Multi-modal healthcare learning (EHR + MRI + X-ray)

---

# License

This project is released under the **MIT License** for research and educational use.

---

# Citation

If you use this code for research purposes, please cite the associated paper (to be added).
