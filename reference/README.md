# 🧠 Brain Tumor Detection using Machine Learning

A professional ML pipeline for classifying brain tumors from MRI scans using deep learning and transfer learning.

**Test Accuracy: 94.8%** | **Model: EfficientNet-B0** | **Classes: 4**

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Comparison](#model-comparison)
- [Error Analysis](#error-analysis)
- [For the Back-End Team](#for-the-back-end-team)
- [Future Work](#future-work)
- [Team](#team)

---

## Overview

This project builds a deep learning model to classify brain MRI scans into four categories:

| Class | Description |
|---|---|
| **Glioma** | Tumor originating from glial cells in the brain |
| **Meningioma** | Tumor growing from the protective membranes (meninges) |
| **Pituitary** | Tumor on the pituitary gland at the base of the brain |
| **No Tumor** | Healthy brain with no tumor detected |

The model uses **transfer learning** with an EfficientNet-B0 backbone pretrained on ImageNet, fine-tuned on 7,200 brain MRI images. It achieves **94.8% accuracy** on the held-out test set.

---

## Dataset

- **Source:** [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
- **Total Images:** 7,200
- **Split:** 4,457 training / 1,114 validation / 1,600 testing
- **Classes:** Perfectly balanced (equal images per class)
- **Image Sizes:** Variable (150px to 1,375px), resized to 224×224
- **Data Cleaning:** 29 edge-detected/processed images removed from the training set

---

## Architecture

```
Input Image (any size, any format)
    ↓
Preprocessing: Convert RGB → Resize 224×224 → Normalize (ImageNet stats)
    ↓
EfficientNet-B0 Backbone (pretrained on ImageNet)
    ↓ extracts visual features (1,280-dimensional)
Custom Classification Head:
    Flatten → Dropout(0.5) → Linear(1280→256) → ReLU → Dropout(0.3) → Linear(256→4)
    ↓
Output: 4 class probabilities → predicted class + confidence score
```

**Model Details:**
- **Backbone:** EfficientNet-B0 (5.3M parameters)
- **Training:** 25 epochs, AdamW optimizer, StepLR scheduler
- **Augmentation:** Random crop, flip, rotation, affine, color jitter, Gaussian blur
- **Mixed Precision:** FP16 training for GPU memory efficiency

---

## Results

### Final Model Performance (EfficientNet-B0)

| Metric | Value |
|---|---|
| **Test Accuracy** | **94.8%** |
| **Total Misclassified** | 84 / 1,600 |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Glioma | 98.9% | 87.8% | 93.0% |
| Meningioma | 87.9% | 97.8% | 92.5% |
| No Tumor | 95.0% | 99.3% | 97.1% |
| Pituitary | 98.7% | 94.3% | 96.4% |

### Key Findings

- **Strongest performance:** No Tumor class (99.3% recall) — the model rarely misses a healthy brain
- **Main challenge:** Glioma ↔ Meningioma confusion — these tumor types appear visually similar in MRI scans
- **Overconfidence issue:** Some incorrect predictions have very high confidence scores

---

## Project Structure

```
brain-tumor-detection/
│
├── data/                        # Dataset (not in git)
│   ├── raw/                     # Original downloaded dataset
│   ├── processed/               # Cleaned data
│   └── splits/                  # Train/val/test splits
│
├── notebooks/                   # Jupyter notebooks
│   └── 01_eda.ipynb             # Exploratory Data Analysis
│
├── src/                         # Core source code
│   ├── data/                    # Data loading & transforms
│   │   ├── dataset.py           # PyTorch Dataset & DataLoader
│   │   └── transforms.py        # Image preprocessing & augmentation
│   ├── models/                  # Model architectures
│   │   └── classifier.py        # BrainTumorClassifier (multi-backbone)
│   ├── training/                # Training pipeline
│   │   └── trainer.py           # Training loop, validation, checkpointing
│   ├── evaluation/              # Evaluation & analysis
│   │   └── metrics.py           # Metrics, confusion matrix, error analysis
│   └── inference/               # Prediction interface
│       └── predictor.py         # BrainTumorPredictor class
│
├── scripts/                     # Runnable scripts
│   ├── prepare_data.py          # Data cleaning & split creation
│   ├── train.py                 # Model training entry point
│   ├── evaluate.py              # Model evaluation on test set
│   ├── export_model.py          # Export model for deployment
│   ├── test_data_pipeline.py    # Data pipeline verification
│   ├── test_model.py            # Model verification
│   └── test_predictor.py        # Predictor interface verification
│
├── configs/                     # Experiment configurations
│   ├── default.yaml             # ResNet-18 baseline config
│   ├── efficientnet.yaml        # EfficientNet-B0 config (best model)
│   └── densenet.yaml            # DenseNet-121 config
│
├── outputs/                     # Generated outputs (not in git)
│   ├── models/                  # Saved model checkpoints
│   ├── logs/                    # MLflow experiment tracking
│   ├── figures/                 # Plots and visualizations
│   └── deployment/              # Deployment package for back-end team
│
├── .gitignore
├── README.md
├── requirements.txt             # Pinned dependencies
└── setup.py                     # Package installation
```

---

## Setup & Installation

### Prerequisites

- Python 3.11 or 3.12 (3.13+ is NOT supported by PyTorch yet)
- NVIDIA GPU recommended (works on CPU too, just slower)

### Installation

```bash
# Clone the repository
git clone https://github.com/Adam-Yasser/brain-tumor-detection.git
cd brain-tumor-detection

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Step 1: Install PyTorch for YOUR machine
# Visit https://pytorch.org/get-started/locally/ and select your setup
# OR use one of these:

# GPU (NVIDIA with CUDA 12.4):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU only (no GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining dependencies
pip install -r requirements.txt

# Step 3: Install project package
pip install -e .
```

### Data Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
2. Extract into `data/raw/`
3. Run data preparation:

```bash
python scripts/prepare_data.py
```

---

## Usage

### Training

```bash
# Train with default config (ResNet-18)
python scripts/train.py

# Train with a specific config
python scripts/train.py --config configs/efficientnet.yaml
```

### Evaluation

```bash
python scripts/evaluate.py
```

### Prediction

```python
from src.inference.predictor import BrainTumorPredictor

predictor = BrainTumorPredictor("outputs/models/efficientnet_b0.pth")
result = predictor.predict("path/to/mri_image.jpg")

print(result)
# {
#     "predicted_class": "glioma",
#     "confidence": 0.9542,
#     "probabilities": {
#         "glioma": 0.9542,
#         "meningioma": 0.0301,
#         "notumor": 0.0098,
#         "pituitary": 0.0059
#     }
# }
```

---

## Model Comparison

Three models were trained and compared:

| Model | Parameters | Test Accuracy | Misclassified | Training Time |
|---|---|---|---|---|
| ResNet-18 (Baseline) | 11.3M | 92.1% | 126 / 1,600 | ~15 min |
| **EfficientNet-B0** | **5.3M** | **94.8%** | **84 / 1,600** | ~38 min |
| DenseNet-121 | 8.0M | 94.3% | 91 / 1,600 | ~71 min |

**EfficientNet-B0 was selected** as the final model for having the highest accuracy, fewest errors, and smallest model size.

---

## Error Analysis

### Main Findings

1. **Glioma ↔ Meningioma confusion** accounts for most errors. These tumor types appear visually similar in MRI scans — a known challenge in medical imaging.

2. **The model is overconfident** on some wrong predictions (100% confidence on incorrect classifications). This is a common issue with neural networks.

3. **No Tumor detection is excellent** (99.3% recall) — the model almost never tells a sick person they are healthy.

### Visualizations

All analysis plots are saved in `outputs/figures/`:
- `sample_images.png` — Sample MRI images per class
- `class_distribution.png` — Dataset balance visualization
- `size_distribution.png` — Image size analysis
- `training_curves.png` — Loss and accuracy over epochs
- `confusion_matrix.png` — Detailed error breakdown
- `per_class_accuracy.png` — Per-class performance
- `misclassified.png` — Most confidently wrong predictions
- `confidence_distribution.png` — Correct vs wrong confidence
- `glioma_meningioma_confusion.png` — Focused confusion analysis
- `model_comparison.png` — Three-model comparison charts

---

## For the Back-End Team

### Quick Start

The deployment package is in `outputs/deployment/`:

```
outputs/deployment/
├── brain_tumor_model.pth    (16.8 MB — trained model)
├── model_metadata.json      (model specifications)
└── usage_example.py         (working code example)
```

### API Integration

The model accepts an MRI image and returns a JSON-compatible response:

**Input:** Any MRI brain scan image (JPG, PNG — any size)

**Output:**
```json
{
    "predicted_class": "glioma",
    "confidence": 0.9542,
    "probabilities": {
        "glioma": 0.9542,
        "meningioma": 0.0301,
        "notumor": 0.0098,
        "pituitary": 0.0059
    }
}
```

### Requirements

```bash
pip install torch torchvision Pillow
```

### Notes

- First prediction takes ~2-3 seconds (model loading). Subsequent predictions are under 1 second.
- GPU is optional. The model works on CPU too.
- See `model_metadata.json` for preprocessing specifications.

---

## Future Work

- **More data:** Collecting additional MRI scans, especially for glioma and meningioma, could improve the main confusion point
- **Medical pretraining:** Using models pretrained on medical images (e.g., RadImageNet) instead of ImageNet
- **Ensemble methods:** Combining predictions from multiple models for higher accuracy
- **Attention mechanisms:** Helping the model focus specifically on tumor regions
- **Confidence calibration:** Addressing the overconfidence issue so prediction probabilities better reflect true accuracy
- **Grad-CAM visualization:** Showing which regions of the image the model focuses on for each prediction

---

## Team

- **Adam Yasser** and **Akram Gamal** — Machine Learning Pipeline
- **Osama Ahmed** and **Tarek Mostafa** — Back-End Development
- **Anas Osama** and **Abdelrahman Sayed** — Front-End Development

---

## Tools & Technologies

- **Python 3.12** — Programming language
- **PyTorch** — Deep learning framework
- **torchvision** — Pretrained models and image transforms
- **scikit-learn** — Evaluation metrics
- **MLflow** — Experiment tracking
- **Matplotlib / Seaborn** — Data visualization
- **Git / GitHub** — Version control