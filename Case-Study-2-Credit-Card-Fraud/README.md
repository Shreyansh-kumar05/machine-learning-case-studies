# Credit Card Fraud Detection

## Case Study Overview

This project focuses on detecting fraudulent credit card transactions using machine learning.

The problem is treated as a *binary classification* task:

- 0 → Genuine transaction
- 1 → Fraudulent transaction

## Objective

The objective is to identify fraudulent transactions from a highly imbalanced transaction dataset.

## Machine Learning Model

### XGBoost

XGBoost is used as the main machine learning model. It is a tree-based ensemble algorithm that builds multiple decision trees sequentially to improve predictions.

## Data Preprocessing

The project includes:

1. Loading the dataset.
2. Checking the dataset structure and missing values.
3. Separating features and target.
4. Splitting the data into training and testing sets.
5. Handling class imbalance using SMOTE.

## SMOTE

The fraud class is usually much smaller than the genuine transaction class.

*SMOTE (Synthetic Minority Over-sampling Technique)* creates synthetic samples of the minority class to help the model learn from fraudulent transactions.

SMOTE is applied only to the training data so that the test data remains representative of the original dataset.

## Threshold Tuning

The model produces a probability of fraud.

Different decision thresholds can be used to convert this probability into a final class prediction.

Threshold selection affects:

- Precision
- Recall
- False Positives
- False Negatives

## Evaluation

The model is evaluated using:

- Precision
- Recall
- F1-score

These metrics are useful for fraud detection because the dataset is highly imbalanced and accuracy alone may be misleading.

## Feature Importance

XGBoost feature importance is used to identify which features contribute more to the model's predictions.

Feature importance shows model influence and does not necessarily mean that a feature causes fraud.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Google Colab

## Project Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
SMOTE on Training Data
   ↓
XGBoost
   ↓
Probability Prediction
   ↓
Threshold Tuning
   ↓
Precision / Recall / F1-score
   ↓
Feature Importance
```

## Conclusion

This project demonstrates how XGBoost and SMOTE can be used to detect fraudulent credit card transactions while considering the challenges of highly imbalanced data.
