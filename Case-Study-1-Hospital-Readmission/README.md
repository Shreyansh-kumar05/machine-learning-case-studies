# Hospital Readmission Prediction (Case Study 1)

Predicting 30-day hospital readmission risk in diabetic patients using *Logistic Regression with L2 regularization*.

## 📌 Problem Statement

Use patient records (diagnosis codes, vitals, prior visit history) to predict whether a patient will be *readmitted within 30 days* of discharge, and evaluate the clinical trade-off between false negatives and false positives.

## 📊 Dataset

- *Source:* UCI Diabetes 130-US Hospitals Dataset
- *File:* diabetic_data.csv
- *Rows:* 101,766 encounters
- *Target column:* readmitted → converted to binary readmitted_30
  - 1 → readmitted within <30 days (11,357 cases)
  - 0 → not readmitted within 30 days (90,409 cases)

### Features Used
| Category | Columns |
|---|---|
| Demographics | age, race, gender |
| Visit history | time_in_hospital, number_outpatient, number_emergency, number_inpatient |
| Diagnosis codes | diag_1, diag_2, diag_3, number_diagnoses |
| Labs / vitals | num_lab_procedures, num_procedures, num_medications, A1Cresult, max_glu_serum, insulin |
| Treatment | change, diabetesMed |

## ⚙️ Methodology

1. Load dataset and create binary target (readmitted_30)
2. Select relevant features (diagnosis codes, vitals, prior visits)
3. Handle missing values (? → Unknown)
4. Label-encode categorical columns
5. Train-test split (80/20, stratified)
6. Standardize features (StandardScaler)
7. Train *Logistic Regression* with *L2 regularization* (penalty='l2')
8. Evaluate using ROC-AUC, confusion matrix, precision/recall

## 📈 Results

| Metric | Value |
|---|---|
| ROC-AUC | *0.64* |
| Accuracy | 0.89 |
| Recall (readmitted class) | 0.01 |
| Precision (readmitted class) | 0.49 |

*Confusion Matrix*

|              | Predicted: No | Predicted: Yes |
|--------------|---------------|-----------------|
| *Actual: No*  | 18,049 | 34 |
| *Actual: Yes* | 2,238 | 33 |

The dataset is *imbalanced* (~11% positive class), which causes the model to favor the majority class and miss most true readmissions (very low recall).

## 🏥 Clinical Discussion: False Negatives vs False Positives

- *False Negative* (model predicts no readmission, but patient IS readmitted): The costlier error clinically — the patient misses extra follow-up care or monitoring, risking complications, worse outcomes, or even penalty costs for the hospital.
- *False Positive* (model predicts readmission risk, but patient is NOT readmitted): Lower-risk error — leads to some unnecessary precautionary care/follow-up, costing staff time and resources, but no harm to the patient.

*Conclusion:* In this clinical setting, false negatives are more costly than false positives. The model should be tuned (e.g., adjusting the classification threshold or using class_weight='balanced') to prioritize recall/sensitivity, even at the cost of more false positives.

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- scikit-learn (Logistic Regression, StandardScaler, train_test_split)
- Matplotlib (ROC curve)

## ▶️ How to Run

\\\`bash
pip install pandas numpy scikit-learn matplotlib
python readmission_prediction.py
\\\`

Or open readmission_prediction.ipynb in Google Colab and run all cells (upload diabetic_data.csv when prompted).

## 📁 Repo Structure

\\\`
├── diabetic_data.csv
├── readmission_prediction.ipynb
├── README.md
\\\`

## 🔮 Future Improvements

- Handle class imbalance (class_weight='balanced', SMOTE)
- Try tree-based models (Random Forest, XGBoost) for comparison
- Hyperparameter tuning on regularization strength (C)
- Feature importance analysis on diagnosis codes
