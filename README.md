# Loan Default Prediction — End-to-End Classification Pipeline
**ProSensia AI/ML Bootcamp | Week 2 Weekend Project**

## Project Overview
End-to-end classification pipeline to predict loan defaults. Three models trained and compared using ROC-AUC curves, F1-Score, Precision, and Recall. Final model selected based on performance on imbalanced data.

## Project Structure
```
├── loan_default_pipeline.ipynb    # Main Jupyter Notebook
├── loan_default_dataset.csv       # Dataset (5000 samples, 14 features)
├── loan_default_rf_model.pkl      # Saved winning model
├── requirements.txt               # Python dependencies
├── AI_Utilization_Report.md       # LLM usage documentation
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook loan_default_pipeline.ipynb
```

## Dataset
- 5,000 loan applications | 14 features | Target: `Loan_Default` (binary)
- Default rate: ~20% — class imbalance handled with SMOTE

## Pipeline Steps
1. Data loading & inspection
2. EDA & correlation heatmap
3. Feature engineering (Loan-to-Income ratio, Risk Score, etc.)
4. 80/20 Train-Test Split (`random_state=42`, `stratify=y`)
5. SMOTE on `X_train` only (no leakage)
6. Train 3 models: Logistic Regression, Decision Tree, Random Forest
7. ROC-AUC curve comparison
8. Confusion matrices & classification reports
9. Feature importance analysis
10. Best model saved as `.pkl`

## Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | ~75% | — | — | — | — |
| Decision Tree | ~79% | — | — | — | — |
| **Random Forest** | **~84%** | **best** | **best** | **best** | **best** |

## Winner: Random Forest
Best ROC-AUC and F1-Score. Handles non-linear feature interactions. Ensemble of 100 trees reduces overfitting via bagging.

## Key Business Insight
Credit Score, Debt-To-Income Ratio, and Num_Late_Payments are the strongest predictors of loan default. High-risk applicants: Credit Score < 580 AND DTI > 0.45.

## No Data Leakage — Verified
- SMOTE applied ONLY after train-test split
- StandardScaler fitted ONLY on X_train (inside Pipeline)
- `random_state=42` set throughout for reproducibility
