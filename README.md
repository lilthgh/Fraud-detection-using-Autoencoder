# Fraud-detection-using-Autoencoder
Credit card fraud detection using an Autoencoder-based anomaly detection approach. Includes data preprocessing, feature scaling, train/test split, and reconstruction error thresholding to identify suspicious transactions. Evaluated using precision, recall, F1-score, ROC-AUC, and PR-AUC.

# Credit Card Fraud Detection with Autoencoder

This project implements a credit card fraud detection pipeline using an **Autoencoder** for anomaly detection.  
The goal is to detect fraudulent transactions by learning the normal transaction pattern and identifying unusual behavior through reconstruction error.

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem, where fraudulent transactions represent only a very small percentage of the total data.  
Instead of training a standard classifier directly, this project uses an Autoencoder trained only on **normal transactions**. After training, the model reconstructs input data, and transactions with high reconstruction error are flagged as potential fraud.

## Dataset

The project uses the well-known **Credit Card Fraud Detection** dataset from Kaggle.  
It contains anonymized transaction features, along with:

- `Time`
- `Amount`
- `Class`  
  - `0` = normal transaction  
  - `1` = fraudulent transaction  

## Workflow

### 1. Data Loading
The dataset is loaded using Pandas.

### 2. Data Cleaning
- Checked for missing values
- Checked for duplicate rows
- Removed duplicate records

### 3. Exploratory Checks
- Analyzed class distribution
- Confirmed severe class imbalance

### 4. Feature Scaling
The `Time` and `Amount` features were scaled using `RobustScaler`, since this method is less sensitive to outliers.

### 5. Train-Test Split
The data was split into training and testing sets using stratified sampling to preserve the fraud ratio in both sets.

### 6. SMOTE
SMOTE was applied to the training set for future classification experiments.  
However, for the Autoencoder model, only **normal transactions** were used for training.

### 7. Autoencoder Model
A simple dense Autoencoder was built with:
- Encoder: `30 -> 16 -> 8`
- Decoder: `8 -> 16 -> 30`

The model was trained to reconstruct normal transactions.

### 8. Fraud Detection
After training:
- reconstruction error was calculated
- a threshold was selected
- transactions with error above the threshold were classified as fraud

### 9. Evaluation
The model was evaluated using:
- Confusion Matrix
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- TensorFlow / Keras

## Key Idea

The Autoencoder learns how normal transactions look.  
If a transaction is reconstructed poorly, it may indicate fraudulent or anomalous behavior.

## Future Improvements

- Optimize reconstruction error threshold
- Compare percentile-based and F1-optimized threshold selection
- Implement **Autoencoder + Attention**
- Implement **GRU**
- Implement **GRU + Attention**
- Compare interpretability and detection performance across models

## How to Run

1. Upload or download the dataset
2. Install dependencies
3. Run the notebook or Python script step by step

### Example dependencies
```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn tensorflow
Results

The Autoencoder achieved strong fraud recall, showing that it can detect a large portion of fraudulent transactions.
Further threshold tuning is needed to reduce false positives and improve precision.

Author

Lilit Gharibi
