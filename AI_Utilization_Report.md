# AI Utilization Report
**ProSensia AI/ML Bootcamp | Week 2, Day 5 — Weekend Project**
**Student:** Dija | **Date:** Week 2 Friday

---

## LLM Tools Used

| Tool | Purpose |
|------|---------|
| Claude (Anthropic) | Scaffolding Matplotlib/Seaborn boilerplate for ROC-AUC curves, confusion matrix plots, and feature importance charts |
| Claude (Anthropic) | Generating pipeline structure (SMOTE + StandardScaler + train-test split ordering) |

---

## What Was AI-Generated
- Matplotlib figure layout for multi-panel plots
- Seaborn heatmap boilerplate with mask for upper triangle
- ROC curve plotting loop across 3 models
- Classification report formatting

## What Was Manually Written / Understood
- All model instantiation logic (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)
- SMOTE application order (strictly after train-test split)
- Feature engineering (Loan_To_Income_Ratio, Risk_Score, etc.)
- Model selection justification based on F1 and ROC-AUC
- Business interpretation of feature importances

## Mathematical Concepts Articulated Without AI
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- Why ROC-AUC > Accuracy for imbalanced datasets
- Why SMOTE must never touch the test set (data leakage)
- Bias-Variance Tradeoff in Random Forest vs Decision Tree

---

*AI was used for boilerplate scaffolding only. All ML logic, evaluation, and business insights were independently understood and articulated.*
