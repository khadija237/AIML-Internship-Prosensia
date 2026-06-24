# Baseline ML Model — Logistic Regression
**ProSensia AI/ML Bootcamp | Week 2, Day 1**

## Project Overview
First baseline Machine Learning model built on the cleaned e-commerce dataset from Week 1. Predicts whether an order will result in a financial loss using Logistic Regression.

## Project Structure
```
├── baseline_model.ipynb      # Main Jupyter Notebook
├── ecommerce_cleaned.csv     # Cleaned dataset from Week 1
├── requirements.txt          # Python dependencies
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook baseline_model.ipynb
```

## Target Variable
`Is_Loss` — Binary classification (1 = loss order, 0 = profitable order)

## Key Steps
1. Load cleaned dataset from Week 1
2. Separate features (X) and target (y)
3. 80/20 Train-Test Split with `random_state=42` and `stratify=y`
4. Train Logistic Regression inside a Pipeline (with StandardScaler)
5. Predict on unseen test data
6. Evaluate with accuracy score, classification report, confusion matrix

## Important: Data Leakage Prevention
The StandardScaler is fit **only on X_train** inside a Pipeline. This ensures no information from the test set leaks into the training process — keeping the evaluation statistically valid.

## Results
- Model: Logistic Regression (C=1.0, max_iter=1000, solver=lbfgs)
- Split: 80/20, random_state=42, stratified
- Evaluation: Accuracy, Precision, Recall, F1-Score

## Tools & Libraries
- Python 3.x | Pandas | NumPy | Scikit-Learn | Matplotlib | Seaborn
