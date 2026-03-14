# 🧠 Neuro-Fatigue Predictor: Deep Learning BCI
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)
![MNE](https://img.shields.io/badge/Neuro-MNE-red)
![Status](https://img.shields.io/badge/Status-Research_Complete-green)
![EEG](https://img.shields.io/badge/Signal-EEG-purple)
![BCI](https://img.shields.io/badge/Domain-BCI%20%7C%20Neuroscience-blueviolet)

### 📖 [Read the Full Case Study on Medium](https://medium.com/@visionbyangelic/decoding-the-unconscious-how-i-built-a-deep-learning-system-to-predict-burnout-99-recall-9d5757a92641)
**Author:** Angelic Charles | [LinkedIn](https://linkedin.com/in/angeliccharles) | [GitHub](https://github.com/visionbyangelic) | [X / Twitter](https://x.com/visionbyangelic)

---

## 🚨 The Problem

Cognitive fatigue causes >90% of accidents in logistics and heavy industry. The core issue is a phenomenon called the **"Metacognitive Gap"** — the brain masks exhaustion from the conscious mind until it is already too late. Research shows that cognitive fatigue degrades reaction time **20 minutes before** a human subjectively feels tired. By the time the driver reaches for a coffee, they have already entered a micro-sleep state multiple times.

Existing solutions rely on expensive clinical-grade EEG equipment, making real-world, in-vehicle deployment impractical.

---

## 💡 The Solution

A **Deep Learning Safety System** that analyzes raw EEG data from consumer-grade hardware (the **$200 Muse Headband**) to detect fatigue **before** subjective awareness. The system applies rigorous signal processing and a 1D-CNN to learn the brain's specific "fatigue fingerprint" directly from the raw time-series — no manual feature extraction required.

**The Mission:** Democratize cognitive safety monitoring. Hospital-grade signal processing + deep learning, on hardware anyone can buy.

---

## 📁 Repository Structure

```
Neuro-Fatigue-Prediction/
│
├── notebooks/
│   ├── eeg-time-series-predicting-fatigue-with-1D-CNN.ipynb   ← Main pipeline
│   └── fork-of-predicting-fatigue-cross-check.ipynb           ← Forensic audit
│
├── README.md
└── requirements.txt
```

### 📓 Notebook Descriptions

| File | Description |
|:---|:---|
| [`eeg-time-series-predicting-fatigue-with-1D-CNN.ipynb`](notebooks/eeg-time-series-predicting-fatigue-with-1D-CNN.ipynb) | **Primary research notebook.** End-to-end pipeline: data ingestion from FatigueSet, MNE-Python signal processing, spectral analysis (PSD), sliding window epoching, 1D-CNN model architecture, training, and confusion matrix evaluation. Achieves **99.18% Recall** on Subject 01. |
| [`fork-of-predicting-fatigue-cross-check.ipynb`](notebooks/fork-of-predicting-fatigue-cross-check.ipynb) | **Forensic audit / cross-check notebook.** Addresses the risk of look-ahead bias in the original model. Implements a strict **Chronological Split** (training on first 80% of session, testing on last 20%) and a **Permutation Sanity Check** (scrambled labels collapse accuracy to ~50%). Confirms that the 99% Recall is driven by real physiological signal, not data leakage. |

---

## 🔬 Scientific Methodology

### The Trejo Protocol (2005)

This project implements the **Trejo Protocol**, a landmark neurophysiological framework for fatigue detection. The science is grounded in a specific, measurable biomarker:

- **Theta Waves (4–8 Hz):** Slow, high-amplitude oscillations generated in the frontal lobe during drowsiness and cognitive idling.
- **The "30% Rule":** Frontal Theta power spikes by >30% during the transition from an alert to a fatigued state — a detectable, consistent signature.
- **Target Channels:** AF7 and AF8 (Frontal Lobe sensors on the Muse Headband) are the primary measurement sites, matching clinical EEG placement at Fp1/Fp2.

### Hypothesis

> *If a 1D-CNN is trained on raw EEG time-series data, it can learn to detect the specific "Theta Wobble" in the Frontal Lobe to classify fatigue with >90% sensitivity — using only a $200 consumer device.*

---

## 📊 Dataset: FatigueSet

**Dataset Source:** [Kaggle — Mental Fatigue Level Detection (FatigueSet)](https://www.kaggle.com/datasets/tanjemahamed/mental-fatigue-level-detection-fatigueset-data)

**Original Publication:**
> Kalanadhabhatta, M., Min, C., Montanari, A., & Kawsar, F. (2021). *FatigueSet: A Multi-modal Dataset for Modeling Mental Fatigue and Fatigability.* International Conference on Pervasive Computing Technologies for Healthcare. [Semantic Scholar](https://www.semanticscholar.org/paper/FatigueSet:-A-Multi-modal-Dataset-for-Modeling-and-Kalanadhabhatta-Min/c55e81d7341e0b2ed97adc8ac1ac016b58f1ffa9)

**Official Dataset Homepage:** [esense.io/datasets/fatigueset](https://www.esense.io/datasets/fatigueset/)

### What is FatigueSet?

FatigueSet is a multi-modal physiological dataset collected under controlled experimental conditions. Participants performed physically and mentally demanding tasks across three sessions on three different days (up to 19 days apart), allowing researchers to study both acute and chronic fatigue.

**Data was collected from four wearable devices simultaneously:**

| Device | Signal |
|:---|:---|
| **Muse S EEG Headband** | Brainwaves (EEG) — 4 channels: TP9, AF7, AF8, TP10 |
| Empatica E4 Wristband | EDA, PPG, accelerometry |
| Nokia eSense Earables | IMU, microphone |
| Zephyr BioHarness | ECG, respiration |

**Additional annotations:** Subjective fatigue ratings and mental fatigability scores from two cognitive tasks — the **Choice Reaction Time task** and the **2-back task**.

**Study Design:**
- Sessions were conducted at roughly the same time of day to control for circadian effects.
- Participants completed a preliminary demographic questionnaire, a Big Five Inventory (BFI-10), and a Munich Chronotype Questionnaire (MCTQ).

### This Project's Data Slice

This work isolates the **EEG channel from Subject 01**, using `forehead_eeg_raw.csv` from:
- **Session 01** → "Alert" baseline state
- **Session 03** → "Fatigued" state (after prolonged cognitive task)

```
/fatigueset/01/01/forehead_eeg_raw.csv  ← Alert
/fatigueset/01/03/forehead_eeg_raw.csv  ← Fatigued
```

**Sampling rate:** 256 Hz | **Channels:** TP9, AF7, AF8, TP10 (4-channel EEG)

---

## 🧠 Why 1D-CNN?

This is not an arbitrary architecture choice. It is a deliberate match between model type and signal structure.

### The Core Argument

EEG is a **temporal** signal. Fatigue is not encoded in a single data point — it emerges as a *pattern across time*: a specific oscillation frequency (the Theta wobble at 4–8 Hz) that appears, sustains, and grows as the brain enters a fatigued state. The model must learn to **read time**, not just classify features.

| Model Type | Why it fails here |
|:---|:---|
| **Standard (2D) CNN** | Designed for spatial data (images). Treats EEG as a grid, not a waveform. Loses temporal ordering. |
| **LSTM / RNN** | Processes time steps sequentially, one at a time. Slow to train, prone to vanishing gradients on long sequences. Overkill for a localized frequency biomarker. |
| **FFT + Classical ML** | Manual feature extraction (Theta power, band ratios) is brittle and requires expert-defined thresholds. Fails on noise, subject variability, and edge cases. |
| **✅ 1D-CNN** | Slides a small filter across the time axis, learning *what shape* in the waveform signals fatigue — automatically. Computationally efficient. No manual feature engineering needed. |

### Technical Reasoning

A **1D Convolutional filter** scans along the time axis (1024 time points = 4 seconds at 256 Hz). At each position, it computes a dot product with a learned kernel. Over thousands of training examples, the kernel learns to activate on the specific temporal patterns that correspond to Theta oscillations in the frontal cortex.

```
Input Shape: (1024, 4)
 └─ 1024 time points × 4 EEG channels (TP9, AF7, AF8, TP10)

Architecture:
Conv1D(16 filters, kernel_size=64)  → Detects short Theta bursts
MaxPooling1D                         → Selects most prominent activations
Conv1D(32 filters, kernel_size=32)  → Detects sustained Theta patterns
MaxPooling1D
Flatten
Dense(64, activation='relu')
Dropout(0.5)                         → Prevents memorizing subject-specific artifacts
Dense(1, activation='sigmoid')       → Binary: Alert (0) or Fatigued (1)
```

**Why 4-second windows?** Trejo et al. (2005) identified 4 seconds as the minimum epoch length to reliably detect a Theta oscillation cycle. Shorter windows miss the pattern; longer windows blur the temporal boundary between alert and fatigued transitions.

**The result:** The model requires no manual feature engineering. It learns the Trejo Protocol's biomarker *directly from voltage*.

---

## 🛠️ Tech Stack

| Tool | Role |
|:---|:---|
| **MNE-Python** | EEG data loading, bandpass filtering (1–50 Hz), artifact removal, PSD via Welch's Method |
| **TensorFlow / Keras** | 1D-CNN model definition, training, evaluation |
| **NumPy** | Sliding window epoching, array operations |
| **Pandas** | CSV ingestion, data structuring |
| **Scikit-learn** | Train/test split, confusion matrix, classification report |
| **Matplotlib / Seaborn** | PSD plots, training curves, confusion matrix heatmap |

---

## ⚙️ Signal Processing Pipeline

1. **Raw Ingestion:** Load 4-channel Muse EEG from CSV (TP9, AF7, AF8, TP10).
2. **MNE Conversion:** Wrap into a `RawArray` object at 256 Hz.
3. **Bandpass Filter (1–50 Hz):**
   - High-pass (1 Hz): Removes DC offset and slow skin-impedance drift.
   - Low-pass (50 Hz): Removes electrical grid noise (50/60 Hz).
4. **Spectral Analysis:** Welch's Method Power Spectral Density — visual proof that the Theta spike is biologically present before any modeling.
5. **Sliding Window Epoching:**
   - Window: 4 seconds (1,024 samples)
   - Stride: 2 seconds (50% overlap)
   - Output: 1,353 labeled samples from two sessions (effectively 2× data augmentation)
6. **Label Assignment:** Session 01 → `0` (Alert), Session 03 → `1` (Fatigued)

---

## 📊 Key Results

| Metric | Score | Interpretation |
|:---|:---|:---|
| **Recall (Sensitivity)** | **99.18%** | Caught 121 out of 122 fatigue events |
| **False Negative Rate** | **0.82%** | Near-zero risk of "Fatal Errors" |
| **Training Accuracy** | **~99.5%** | High signal-to-noise in Theta band |
| **Chronological Test Accuracy** | **99.26%** | Confirmed on strictly future data |
| **Permutation Check Accuracy** | **~49.82%** | Collapsed to chance — confirms no data leakage |

### Visualizations

**Biological Proof: PSD shows the Theta spike is real**

![PSD Alert vs Fatigued](https://miro.medium.com/v2/resize:fit:640/format:webp/1*KHeQTySqzRV5pcVt3gu2zQ.png)
![PSD Detail](https://miro.medium.com/v2/resize:fit:640/format:webp/1*r8m74LCRBBvXKHZCcB7Tqg.png)

*The "Fatigued" state (Red) shows a massive energy spike in the 4–8 Hz Theta range compared to the "Alert" state (Blue). This confirms the biological signal exists before any model is applied.*

**Safety Analysis: Confusion Matrix**

![Confusion Matrix](https://miro.medium.com/v2/resize:fit:640/format:webp/1*UGlLKkPjAZ0s6Ltox8hCbQ.png)

*Out of 122 positive fatigue samples, the model missed exactly 1.*

---

## 🚀 How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/visionbyangelic/Neuro-Fatigue-Prediction.git
cd Neuro-Fatigue-Prediction
```

**2. Install dependencies:**
```bash
pip install tensorflow mne numpy pandas matplotlib seaborn scikit-learn
```

**3. Download the dataset from Kaggle:**
[Mental Fatigue Level Detection (FatigueSet)](https://www.kaggle.com/datasets/tanjemahamed/mental-fatigue-level-detection-fatigueset-data)

Place the dataset such that the paths resolve to:
```
/fatigueset/01/01/forehead_eeg_raw.csv
/fatigueset/01/03/forehead_eeg_raw.csv
```
*(Or update the paths in the notebook to match your local setup.)*

**4. Run the primary notebook:**
```bash
jupyter notebook notebooks/eeg-time-series-predicting-fatigue-with-1D-CNN.ipynb
```

**5. (Optional) Run the forensic audit:**
```bash
jupyter notebook notebooks/fork-of-predicting-fatigue-cross-check.ipynb
```

---

## ⚠️ Experimental Design & Limitations

### Why Intra-Subject?

For this Proof-of-Concept, I deliberately used an **Intra-Subject Design** (training and testing on the same individual). EEG signals suffer from extreme **Inter-Subject Variability** — differences in skull thickness, hair density, electrode impedance, and neural folding create unique biometric fingerprints for every human. Mixing subjects in early-stage training causes the model to learn *who the person is*, not *how tired they are*.

By isolating Subject 01, I controlled for morphological confounders and validated that the 1D-CNN is learning the **physiological biomarker** (Frontal Theta Asymmetry), not just the anatomy.

### Known Constraints

1. **Generalizability:** Future work requires **LOSO (Leave-One-Subject-Out)** cross-validation to extend the model to new users without calibration sessions.
2. **EOG Artifacts:** Bandpass filtering removes major noise, but eye-blink artifacts in the Delta band (1–4 Hz) remain a potential confounder. Future iterations will implement **ICA (Independent Component Analysis)** for cleaner neural isolation.
3. **Single Subject:** Results are validated on one participant. Multi-subject validation is the next milestone.
4. **Deployment Threshold:** A production safety system should shift the decision threshold from `p > 0.5` to `p > 0.3` to achieve 100% Recall at the cost of slightly more false alarms.

---

## 🔍 Forensic Audit

After initial publication, I proactively investigated whether the high performance was due to **look-ahead bias** — a common failure mode in time-series ML where overlapping windows allow information from the future to contaminate training.

**Test 1 — Chronological Split:**
Replaced random shuffling with a strict temporal split: first 80% of the recording for training, last 20% (future) for testing.
→ Result: **99.26% accuracy** on strictly unseen future data. ✅

**Test 2 — Permutation Sanity Check:**
Trained the identical architecture on randomly scrambled labels.
→ Result: Accuracy collapsed to **49.82%** (chance). ✅

**Conclusion:** The model is learning real biology, not memorizing artifacts.

---

## 📚 References

- Trejo, L. J., et al. (2005). *Measures and models for predicting cognitive fatigue.* Proceedings of SPIE.
- Kalanadhabhatta, M., Min, C., Montanari, A., & Kawsar, F. (2021). *FatigueSet: A Multi-modal Dataset for Modeling Mental Fatigue and Fatigability.* PervasiveHealth.
- Ma, P., et al. (2025). *Monitoring nap deprivation-induced fatigue using fNIRS and deep learning.*
- Krigolson, O. E., et al. *Choosing MUSE: Validation of a Low-Cost, Portable EEG System for ERP Research.* Frontiers in Neuroscience.

---

## 🔗 Links

| Resource | Link |
|:---|:---|
| 📖 Medium Article | [Decoding the Unconscious: 99% Recall](https://medium.com/@visionbyangelic/decoding-the-unconscious-how-i-built-a-deep-learning-system-to-predict-burnout-99-recall-9d5757a92641) |
| 📦 Dataset (Kaggle) | [Mental Fatigue Level Detection — FatigueSet](https://www.kaggle.com/datasets/tanjemahamed/mental-fatigue-level-detection-fatigueset-data) |
| 📄 FatigueSet Paper | [Semantic Scholar](https://www.semanticscholar.org/paper/FatigueSet:-A-Multi-modal-Dataset-for-Modeling-and-Kalanadhabhatta-Min/c55e81d7341e0b2ed97adc8ac1ac016b58f1ffa9) |
| 🌐 FatigueSet Homepage | [esense.io/datasets/fatigueset](https://www.esense.io/datasets/fatigueset/) |
| 💼 LinkedIn | [Angelic Charles](https://linkedin.com/in/angeliccharles) |
| 🐙 GitHub | [visionbyangelic](https://github.com/visionbyangelic) |
| 🐦 X / Twitter | [@visionbyangelic](https://x.com/visionbyangelic) |

---

**Author:** Angelic Charles — Computational Neuroscience | Deep Learning | BCI Research

