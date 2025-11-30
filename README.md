# 🧠 Neuro-Fatigue Predictor: Deep Learning BCI

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)
![MNE](https://img.shields.io/badge/Neuro-MNE-red)
![Status](https://img.shields.io/badge/Status-Research_Complete-green)

### [Medium Article](https://medium.com/@visionbyangelic/decoding-the-unconscious-how-i-built-a-deep-learning-system-to-predict-burnout-99-recall-9d5757a92641)  

### 🚨 The Problem
Cognitive fatigue causes >90% of accidents in logistics and heavy industry. Subjective self-reporting is unreliable due to the "metacognitive gap"—the brain masks exhaustion until it's too late.

### 💡 The Solution
A **Deep Learning Safety System** that analyzes raw EEG data from consumer-grade hardware (Muse Headband) to predict fatigue **20 minutes before** subjective awareness.

---

## 🔬 Scientific Methodology
This project implements the **Trejo Protocol (2005)**, identifying **Frontal Theta Asymmetry (4-8Hz)** as the primary biomarker for "Unconscious Fatigue."

### Key Results
| Metric | Score | Impact |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **99.18%** | Caught 121 out of 122 fatigue events. |
| **False Negative Rate** | **0.82%** | Extremely low risk of "Fatal Errors." |
| **Training Accuracy** | **~99.5%** | High signal-to-noise ratio in Theta band. |

---

## 🛠️ Tech Stack
* **Data Engineering:** Sliding Window Segmentation (4s Epochs, 50% Overlap).
* **Signal Processing:** MNE-Python (Bandpass Filter 1-50Hz, Artifact Removal).
* **Model Architecture:** Time-Distributed **1D-CNN** (Convolutional Neural Network).

## 📊 Visualizations

### 1. Biological Proof (Signal Processing)
*The "Red" (Fatigued) state shows a massive energy spike in the 4-8Hz Theta range.*
![Spectral Density](https://miro.medium.com/v2/resize:fit:640/format:webp/1*KHeQTySqzRV5pcVt3gu2zQ.png)
![Spectral Density](https://miro.medium.com/v2/resize:fit:640/format:webp/1*r8m74LCRBBvXKHZCcB7Tqg.png)

### 2. Safety Analysis (Confusion Matrix)
*The model missed only 1 event out of 122 positive samples.*
![Confusion Matrix](https://miro.medium.com/v2/resize:fit:640/format:webp/1*UGlLKkPjAZ0s6Ltox8hCbQ.png)

---

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone https://github.com/visionbyangelic/Neuro-Fatigue-Prediction.git
    ```
2.  Install dependencies:
    ```bash
    pip install tensorflow mne numpy pandas matplotlib seaborn scikit-learn
    ```
3.  Run the Notebook:
    ```bash
    jupyter notebook notebooks/fatigue_prediction_CNN.ipynb
    ```
## ⚠️ Experimental Design & Limitations
**Why Intra-Subject?**
For this Proof-of-Concept, I deliberately utilized an **Intra-Subject Design** to control for inter-subject variability (skull thickness, impedance). This validates that the physiological biomarker (Theta Spike) is detectable by the 1D-CNN before introducing morphological confounders.

**Current Constraints:**
1. **Generalization:** Future work requires **LOSO (Leave-One-Subject-Out)** validation to extend the model to new users without calibration.
2. **Signal Artifacts:** While Bandpass filtering removes drift, EOG (eye blink) artifacts in the Delta band (1-4Hz) remain a potential confounder. Future iterations will implement **ICA (Independent Component Analysis)** for cleaner neural isolation.

---
## 📚 References
* *Trejo, L. J., et al. (2005).* Measures and models for predicting cognitive fatigue.
* *Ma, P., et al. (2025).* Monitoring nap deprivation-induced fatigue using fNIRS and deep learning.

---
**Author:** Angelic Charles | [LinkedIn](https://linkedin.com/in/angeliccharles)
