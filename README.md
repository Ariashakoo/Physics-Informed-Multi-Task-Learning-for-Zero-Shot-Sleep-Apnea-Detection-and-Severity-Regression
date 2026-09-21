# Physics-Informed Multi-Task Learning for Zero-Shot Sleep Apnea Detection and Severity Regression

This repository contains the official PyTorch implementation of the paper "Physics-Informed Multi-Task Learning for Zero-Shot Sleep Apnea Detection and Severity Regression".

## Project Overview

Sleep apnea is a severe and highly prevalent sleep disorder traditionally diagnosed using Polysomnography (PSG), which is costly, intrusive, and requires overnight clinical observation. While single-lead Electrocardiogram (ECG) tracking offers a promising and accessible alternative, deep learning models trained purely on ECG waveforms often fail to generalize across different clinical cohorts. This failure is typically due to models overfitting to dataset-specific noise or recording equipment artifacts rather than learning the underlying physiology.

This project introduces PIMTL-CPC (Physics-Informed Multi-Task Learning), a novel architecture that forces the neural network to ground its predictions in known physiological phenomena. By jointly learning to predict sleep apnea, regress the overall Apnea-Hypopnea Index (AHI), and reconstruct explicit physiological markers—specifically Cardiopulmonary Coupling (CPC) and ECG-Derived Respiration (EDR)—the model achieves state-of-the-art zero-shot transfer performance. This robustness is demonstrated by training entirely on the Apnea-ECG dataset and evaluating zero-shot on the diverse UCDDB dataset.

## Architecture & Methodology

### The PIMTL-CPC Architecture
Our proposed model is a multi-branch Convolutional Neural Network specifically designed for 1D time-series data:
* **Wide-Kernel Stem:** Uses a exceptionally large convolutional kernel (k=15) to capture the full QRS-T morphology in raw ECG segments without destroying phase relationships, which often occurs with standard small-kernel approaches.
* **Residual Squeeze-and-Excitation (SE):** The network utilizes three stages of residual blocks enhanced with 1D SE attention mechanisms. This allows the network to dynamically recalibrate feature channels based on temporal importance, emphasizing clinically relevant heartbeat morphologies.
* **Attentive Temporal Pooling:** Instead of standard average or max pooling, we deploy an attention mechanism to pool temporal features. This enables the network to focus on brief, transient hypopneic arousal bursts within a 60-second epoch rather than treating all seconds equally.
* **Decoupled Multi-Task Heads:** The pooled representation is routed to three separate prediction heads:
  1. **Classification Head:** Uses Cross-Entropy to predict Apnea vs. Normal for the given epoch.
  2. **Regression Head:** Uses Smooth L1 loss for global AHI severity estimation, scaling the regression constraint to balance gradient flow.
  3. **Physics-Informed Head:** Predicts the CPC spectral components (Low Frequency, High Frequency, and their ratio), acting as a physiological regularization constraint. This forces the latent space to encode respiratory-cardiac interactions.

### Baseline Models Implemented
For comprehensive benchmarking, this repository also implements several modern time-series architectures:
* **PASE-MST:** Pre-Activation SE Multi-Stream ResNet, which explicitly processes raw ECG, EDR, R-R intervals, and CPC in parallel streams before cross-fusion.
* **DREAM:** A 2D ResNet applied to Morlet Continuous Wavelet Transform (CWT) Scalograms, converting the 1D signal into a 2D time-frequency image space.
* **CNN-BiGRU:** A Spatio-Temporal network combining local feature extraction with Bidirectional Gated Recurrent Units for sequential modeling.
* **CNN-Transformer-LSTM:** A hybrid multi-scale network combining local convolutions with self-attention (Transformer Encoder) and recurrence (LSTM).

## Project Structure


sleep-cohort-analysis/
├── requirements.txt           # Python dependencies (mne, wfdb, neurokit2, torch, etc.)
├── utils/
│   ├── __init__.py
│   └── signal_processing.py   # Signal cleaning, CPC/EDR extraction, CWT Scalograms
├── data/
│   ├── __init__.py
│   ├── preprocessing.py       # WFDB/MNE parsing, artifact/NaN removal, dataset serialization
│   └── dataset.py             # PyTorch Dataset handling dynamic scaling and batched loading
├── models/
│   ├── __init__.py
│   └── architectures.py       # PIMTL-CPC and baseline neural network definitions
├── scripts/
│   ├── train.py               # Multi-task training loop, Cosine Annealing, checkpointing
│   └── evaluate_and_plot.py   # Inference engine, clinical metrics, and IEEE-style plotting
└── README.md
Installation
Clone the repository:

Bash
git clone [https://github.com/yourusername/Physics-Informed-Multi-Task-Learning-for-Zero-Shot-Sleep-Apnea-Detection-and-Severity-Regression.git](https://github.com/yourusername/Physics-Informed-Multi-Task-Learning-for-Zero-Shot-Sleep-Apnea-Detection-and-Severity-Regression.git)
cd Physics-Informed-Multi-Task-Learning-for-Zero-Shot-Sleep-Apnea-Detection-and-Severity-Regression
Create a virtual environment and install dependencies:

Bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
Usage
1. Data Preparation
The pipeline expects the Apnea-ECG and UCDDB datasets (accessible via PhysioNet) in your raw data directory.
To extract 60-second segments, drop corrupted (lead-off) epochs, compute physiological features, and build serialized .pkl files, run:

Bash
python -m data.preprocessing
2. Model Training
To train the PIMTL-CPC model (and comparative baselines) on the Apnea-ECG training partition, run:

Bash
python -m scripts.train
The training script automatically computes dynamic class weights to handle dataset imbalances, applies Cosine Annealing learning rate scheduling to avoid local minima, and saves the optimal model weights (best_pimtl.pth) based on validation Macro-F1 scores.

3. Evaluation & Plotting
To evaluate the trained models on the internal Apnea-ECG test set and perform zero-shot evaluation on the UCDDB dataset, run:

Bash
python -m scripts.evaluate_and_plot
This script generates publication-ready IEEE-style figures in the paper_plots/ directory:

Fig 1: Dual ROC Curves comparing internal test performance versus zero-shot generalization.

Fig 2: Confusion Matrices mapping True Physiological States against Model Predictions.

Fig 3: Clinical AHI Regression Analysis (Predicted vs. Reference Scatter plots and Bland-Altman agreement plots).

Fig 4: Interpretability overlays showcasing the Attentive Temporal Pooling activations directly overlaid on raw apneic ECG epochs.
