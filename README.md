# Loan Default Prediction API - Day 17
ProSensia AI/ML Bootcamp | Week 4 Day 2

## what i updated
added pydantic schema for input validation and full prediction logic in POST /predict.

## files
- main.py
- production_rf_model.pkl
- requirements.txt
- README.md

## how to run
pip install -r requirements.txt
uvicorn main:app --reload

open: http://127.0.0.1:8000/docs

## endpoints
- GET  /health-check  -> {"status": "API is live"}
- POST /predict       -> returns {"loan_default_prediction": 0 or 1}

## sample input for /predict
{
  "Age": 35,
  "Annual_Income": 55000,
  "Loan_Amount": 20000,
  "Loan_Term_Months": 36,
  "Credit_Score": 620,
  "Employment_Years": 5,
  "Num_Credit_Lines": 4,
  "Debt_To_Income_Ratio": 0.35,
  "Num_Late_Payments": 2,
  "Has_Mortgage": 0,
  "Education": 1,
  "Loan_Purpose": 0,
  "Marital_Status": 1,
  "Loan_To_Income": 0.36,
  "Risk_Score": 7.8
}

## why pydantic validation
if wrong data type is sent, pydantic returns 422 error automatically.
model never receives bad data - prevents garbage in garbage out problem.
