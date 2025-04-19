
# Pulsar Detection – Machine Learning Final Project

This project applies various machine learning models to classify pulsar stars from non-pulsar signals using the HTRU2 dataset. The goal was to analyze model behavior on an **imbalanced binary classification task**, comparing classical ML techniques through statistical evaluation and calibration methods.

## 📊 Dataset

The [HTRU2 dataset](https://archive.ics.uci.edu/ml/datasets/HTRU2) consists of 17,898 samples with 8 continuous features derived from radio astronomy signal profiles.

- **Class 1:** Real pulsars (1,639)
- **Class 0:** Noise / RFI (16,259)

## 🔧 Tools & Technologies

- Python (NumPy, SciPy, Matplotlib)
- PCA for dimensionality reduction
- Gaussian classifiers: Naive Bayes, Full Covariance, Tied models
- Logistic Regression (with L2 regularization)
- Support Vector Machines (Linear, RBF, Polynomial Kernels)
- Gaussian Mixture Models
- Score Calibration (minDCF, actDCF)

## 🔍 Methodology

1. **Data Preprocessing:**
   - Z-normalization
   - PCA (tested with m=7,6,5...)

2. **Model Evaluation:**
   - Used both **Single Split** and **k-Fold Cross-Validation**
   - Measured performance using **minDCF** and **ROC curves**

3. **Score Calibration:**
   - Linear calibration applied to improve decision thresholds

4. **Best Performing Models:**
   - Logistic Regression (lambda = 1e-5) with PCA (m=7)
   - Linear SVM (C = 0.01)
   - GMM Full Covariance (8 components)

## ✅ Key Findings

- **Linear classifiers** (LogReg, SVM) outperformed quadratic and non-linear models
- PCA improved performance when reduced to **7 principal components**
- Score calibration significantly improved decision-making in imbalanced settings

## 📁 Structure

```
├── data/                 # Raw and processed datasets
├── src/                  # Core scripts (training, PCA, evaluation)
├── plots/                # Evaluation plots (ROC, DCF curves, etc.)
└── report/               # Final PDF report
```

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yurnero14/MachineLearningPR-Project-2021-22
   cd MachineLearningPR-Project-2021-22
   ```

2. Make sure you have Python 3.x and the following libraries:
   - numpy
   - scipy
   - matplotlib

3. Run the analysis script:
   ```bash
   python src/train_and_evaluate.py
   ```

## 📬 Contact

For any questions, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/your-link) or email.
