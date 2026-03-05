# Federated Private X-Ray Classification (SOTA)

This project implements a privacy-preserving Federated Learning (FL) pipeline for detecting Pneumonia from Chest X-rays. It is specifically designed for medical environments where data cannot be shared centrally due to HIPAA/GDPR constraints.



## 📊 Performance Summary
- **Peak Test Accuracy:** 92.5%
- **AUC-ROC Score:** 0.94
- **Privacy Budget:** ε = 3.89, δ = 1e-5
- **Data Distribution:** Dirichlet Non-IID (α = 1.0)

## 🧠 Technical Features

### 1. Differential Privacy (DP)
We integrate the **Opacus** engine to apply Rényi Differential Privacy. By clipping per-sample gradients and adding Gaussian noise, we provide a mathematical guarantee that the model does not memorize specific patient data.

### 2. Heterogeneous Data Handling (FedProx)
In a real-world federated setup, hospitals have different patient demographics (Non-IID data). We use the **FedProx** optimizer with a proximal term ($\mu = 0.001$) to stabilize training and prevent local model drift across the 3 decentralized clients.

### 3. Medical Inference Strategy
To reach clinical-grade accuracy, we implemented:
* **Test-Time Augmentation (TTA):** Predictions are averaged across multiple augmented views (rotations, flips, brightness shifts) to ensure robust classification.
* **Weighted Cross-Entropy:** Class weights were tuned (1.0 : 1.8) to prioritize sensitivity, reducing the rate of missed Pneumonia cases.



## 🛠️ Installation & Usage

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/your-username/medical-fl-dp.git](https://github.com/your-username/medical-fl-dp.git)
   cd medical-fl-dp/xray_classification
