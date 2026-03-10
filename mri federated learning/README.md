# Federated Learning for Brain Tumor MRI Classification

This module implements a privacy-preserving federated learning framework for multi-class brain tumor classification using MRI scans.

## Key Components

The pipeline integrates:

• Federated Learning using **Flower**
• **ResNet18 CNN** for medical image classification
• **Non-IID Dirichlet data distribution** across hospitals
• **FedProx optimization** for heterogeneous clients
• **Adaptive Differential Privacy noise**

The goal is to simulate collaborative learning across hospitals while preserving patient privacy.

---

## Dataset

Brain Tumor MRI Dataset

Source:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Download the dataset and place it inside a `data/` directory.

---

## Installation

```bash
pip install flwr torch torchvision numpy scikit-learn kagglehub
```

---

## Running Federated Training

```bash
python train_mri_federated.py
```

---

## Output

The script performs federated training across multiple simulated hospitals and reports:

• classification accuracy
• privacy-aware federated training metrics

---

## Research Context

This implementation accompanies research on:

**Privacy-Preserving Federated Learning for Multi-Modal Healthcare Data**

combining:

* Electronic Health Records
* Chest X-ray imaging
* Brain MRI imaging
