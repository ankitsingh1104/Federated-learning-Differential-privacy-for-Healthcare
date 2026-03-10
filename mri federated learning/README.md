# Federated Learning for Brain Tumor MRI Classification

This module implements a **privacy-preserving federated learning framework** for multi-class brain tumor classification using MRI scans.

The system simulates **collaborative learning across multiple hospitals**, where each hospital trains locally on its own MRI data and shares only model updates instead of patient data.

The framework integrates **Federated Learning, Differential Privacy, and Deep Learning** to enable secure distributed medical AI.

---

# Key Features

The pipeline includes the following components:

• **Federated Learning using Flower (FLWR)**
• **ResNet18 Convolutional Neural Network** for MRI classification
• **Non-IID Dirichlet data distribution** across simulated hospitals
• **FedProx optimization strategy** for heterogeneous hospital data
• **Adaptive Differential Privacy noise injection**
• **Utility–privacy trade-off analysis**

This setup reflects **real-world hospital collaborations**, where patient data cannot be centralized due to privacy regulations.

---

# Dataset

Brain Tumor MRI Dataset

Source:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Dataset contains MRI scans belonging to four tumor classes:

* Glioma
* Meningioma
* Pituitary
* No Tumor

Due to licensing and size restrictions, the dataset is **not included in this repository**.

Download the dataset manually and place it in a local folder.

Example structure:

```text
data/
   brain-tumor-mri-dataset/
       Training/
           glioma/
           meningioma/
           pituitary/
           notumor/
```

---

# Installation

Install required dependencies:

```bash
pip install torch torchvision flwr numpy scikit-learn matplotlib seaborn kagglehub
```

---

# Running Federated MRI Training

To start federated training across simulated hospitals:

```bash
python train_mri_federated.py
```

This will:

1. Load the MRI dataset
2. Split data across hospitals using **Dirichlet non-IID sampling**
3. Train a **ResNet18 CNN model in a federated setup**
4. Apply **adaptive differential privacy noise**
5. Aggregate model updates using **FedProx**

---

# Privacy Experiment

To reproduce the privacy comparison experiment:

```bash
python privacy_modes_experiment.py
```

This experiment compares three training modes:

| Mode     | Description                       |
| -------- | --------------------------------- |
| None     | No privacy protection             |
| Standard | Fixed differential privacy noise  |
| Adaptive | Proposed adaptive noise mechanism |

This experiment evaluates the **privacy–utility trade-off in federated medical AI systems**.

---

# Output and Visualizations

The training scripts generate several research figures:

• Non-IID hospital data distribution
• Accuracy vs training rounds
• Privacy budget vs model utility
• Confusion matrix for final classification results

These figures are used in the **experimental evaluation section of the research paper**.

---

# Research Context

This module is part of a larger research framework exploring **privacy-preserving federated learning for multi-modal healthcare data**, including:

* Electronic Health Records (EHR)
* Brain MRI imaging
* Chest X-ray imaging

The goal is to develop **secure collaborative AI models that can learn across hospitals without sharing sensitive patient data**.

---

# License

This project is released under the **MIT License** for research and educational use.
