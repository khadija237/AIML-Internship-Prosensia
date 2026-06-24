# SMOTE & Feature Interpretability — Day 9
**ProSensia AI/ML Bootcamp | Week 2, Day 4**

## Project Overview
Identified severe class imbalance in the target variable, applied SMOTE to balance the training data, retrained the optimized Random Forest, and generated a Feature Importance chart for business interpretability.

## Project Structure
```
├── smote_interpretability_day9.ipynb   # Main Jupyter Notebook
├── production_rf_model.pkl             # Final production model
├── ecommerce_cleaned.csv               # Cleaned dataset
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook smote_interpretability_day9.ipynb
```

## What is SMOTE?
**Synthetic Minority Over-sampling Technique** — uses K-Nearest Neighbors to generate synthetic data points for the minority class in the training set only.

## Critical Rules Followed
- SMOTE applied **ONLY** to `X_train` / `y_train`
- `X_test` / `y_test` kept completely untouched (real data only)
- Model evaluated on real test data — no synthetic data in evaluation

## Why Testing on Synthetic Data is Dangerous
Synthetic data is generated from patterns in training data. Testing on it would make the model appear to perform perfectly — but it would fail on real-world data. This is a catastrophic statistical error.

## Business Interpretation of Feature Importance
The model primarily relies on financial features (Revenue, Cost, Profit, Discount) to predict loss-making orders. **Actionable insight:** Orders with Discount > 20% AND high Shipping_Cost should trigger a pricing review.

## Tools & Libraries
- Python 3.x | Pandas | NumPy | Scikit-Learn | imbalanced-learn | Matplotlib | Seaborn | Joblib
