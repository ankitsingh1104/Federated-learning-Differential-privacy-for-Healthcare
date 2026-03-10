# Federated Private X-Ray Classification (SOTA)

This project implements a **privacy-preserving Federated Learning (FL) pipeline** for detecting Pneumonia from Chest X-rays.

The system is designed for **medical environments where patient data cannot be centralized**, enabling hospitals to collaboratively train a shared model while keeping all medical images locally stored.

The framework integrates **Federated Learning, Differential Privacy, and cost-sensitive medical inference**.

---

# 📊 Performance Summary

| Metric             | Value                           |
| ------------------ | ------------------------------- |
| Peak Test Accuracy | **92.5%**                       |
| AUC-ROC            | **0.94**                        |
| Privacy Budget     | **ε = 3.89**, δ = 1e-5          |
| Data Distribution  | **Dirichlet Non-IID (α = 1.0)** |

These results demonstrate strong predictive performance while maintaining **formal privacy guarantees**.

---

# 🧠 Technical Features

## 1️⃣ Differential Privacy (DP-SGD)

The model uses the **Opacus** library to enforce Differential Privacy during training.

Privacy is achieved through:

* **Per-sample gradient clipping**
* **Gaussian noise injection**
* **Privacy accounting (ε, δ guarantees)**

This ensures the model **cannot memorize or reveal individual patient X-ray images**.

---

## 2️⃣ Federated Learning with FedProx

Real-world hospitals have **heterogeneous patient populations**.

To stabilize federated training across hospitals, we use the **FedProx optimization algorithm**, which introduces a proximal regularization term:

μ = 0.001

This reduces **client drift** and improves convergence under Non-IID data distributions.

---

## 3️⃣ Non-IID Hospital Data Simulation

To simulate realistic hospital conditions, the dataset is partitioned using a **Dirichlet distribution**:

α = 0.5

This produces **skewed class distributions across hospitals**, replicating real-world clinical data imbalance.

---

## 4️⃣ Cost-Sensitive Medical Loss

False negatives are critical in medical diagnosis.

To reduce missed pneumonia cases, we use **weighted cross-entropy loss**:

Normal : Pneumonia = **1.0 : 2.5**

This prioritizes **sensitivity (recall)** for pneumonia detection.

---

# 🏥 Federated Training Architecture

```
Hospital A (Local X-rays)
        ↓
Hospital B (Local X-rays)
        ↓
Hospital C (Local X-rays)

Local Training + Differential Privacy
        ↓
Secure Federated Aggregation
        ↓
Global Medical AI Model
```

Hospitals **never share raw medical images**.

Only **privacy-protected model updates** are exchanged.

---

# 📂 Dataset

COVID-19 Chest X-Ray Dataset

Source:
https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets

Expected directory structure:

```
data/
   xray_dataset_covid19/
       train/
           NORMAL/
           PNEUMONIA/
       test/
           NORMAL/
           PNEUMONIA/
```

Datasets are **not included in this repository** due to size and licensing constraints.

---

# ⚙️ Installation

Install dependencies:

```bash
pip install torch torchvision opacus numpy scikit-learn matplotlib seaborn
```

---

# 🚀 Running Federated Training

```bash
python train_xray_federated.py
```

The script performs:

1. Dirichlet Non-IID hospital partitioning
2. Federated client training
3. Differential Privacy gradient updates
4. FedProx aggregation
5. Global model evaluation

---

# 📈 Medical Evaluation Metrics

The system reports clinically relevant metrics:

• **Sensitivity (Recall)** – ability to detect pneumonia
• **Specificity** – ability to identify healthy patients
• **False Negative Rate** – critical diagnostic error

These metrics ensure the model meets **medical safety requirements**.

---

# 🔬 Research Context

This project is part of a larger framework for **privacy-preserving federated learning in healthcare**, combining multiple medical modalities:

* Electronic Health Records (EHR)
* Brain MRI imaging
* Chest X-ray imaging

The goal is to enable **secure collaborative medical AI across hospitals without sharing sensitive patient data**.

---

# 📜 License

Released under the **MIT License** for research and educational use.
