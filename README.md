# 🧠 Eyeblink Artifact Removal from Single-Channel EEG using 1D Convolutional Denoising Autoencoder

This project implements a **1D Convolutional Denoising Autoencoder (CDAE)** to automatically remove eyeblink artifacts from single-channel EEG signals, improving signal quality for Brain-Computer Interface (BCI) and EEG analysis applications.

---

## 🗂️ Project Overview

Eyeblink artifacts significantly distort EEG signals and must be removed to ensure accurate brain activity analysis. This project aims to eliminate such artifacts using deep learning, specifically a **1D CDAE** model trained on synthetic EEG data with eyeblink noise.

---

## 📊 Dataset

* **Source**: \[EEGdenoiseNet Dataset (publicly available)]
* **Training Samples**: 27,200
* **Validation Samples**: 3,400
* **Test Samples**: 3,400
* **Sequence Length**: 512
* **Channels**: Single-channel EEG

---

## ⚙️ Methodology

### 1. **EEG Data Collection**

* EEG signals recorded via scalp electrodes.
* Eyeblink artifacts synthetically injected into clean EEG signals.

### 2. **Preprocessing**

* Band-pass filtering applied to remove baseline drift and high-frequency noise.

### 3. **Model Architecture**

* **Encoder** compresses noisy EEG into a latent representation.
* **Decoder** reconstructs the clean signal from the compressed version.
* Built using **1D Convolutional Layers**.

---

## 📈 Model Evaluation

| Metric                                | Value  |
| ------------------------------------- | ------ |
| **Mean MSE**                          | 0.0621 |
| **Relative RMSE (RRMSE)**             | 0.4448 |
| **Mean Correlation Coefficient (CC)** | 0.8747 |

The model outperforms many existing state-of-the-art techniques in artifact removal from single-channel EEG.

---

## ✅ Key Features

* 📦 Lightweight model suitable for real-time applications
* 🔬 Works on **single-channel EEG**, unlike traditional multichannel methods
* 📈 High correlation with original clean signal
* ⚡ Faster inference using convolutional architecture

---

## 🧠 Applications

* Brain-Computer Interfaces (BCI)
* Cognitive neuroscience
* EEG-based diagnostics and neurofeedback systems

---

## 📁 Repository Structure

```
├── data/                # EEG dataset
├── models/              # Trained CDAE model
├── scripts/             # Preprocessing and training scripts
├── results/             # Evaluation metrics and visualizations
└── README.md            # Project documentation
```

---

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/eeg-artifact-removal.git
cd eeg-artifact-removal
pip install -r requirements.txt
python train_model.py
```
