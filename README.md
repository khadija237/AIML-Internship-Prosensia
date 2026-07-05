# Loan Default Prediction API - Response Models
ProSensia AI/ML Bootcamp | Week 4 Day 4

## what i did today
added PredictionResponse pydantic model to structure the api output.
now /predict returns prediction and confidence_score in a strict schema.
swagger docs auto-generate the response structure at /docs.

## files
- main.py
- production_rf_model.pkl
- requirements.txt
- README.md

## how to run
pip install -r requirements.txt
D:\python\python.exe -m uvicorn main:app --reload

open: http://127.0.0.1:8000/docs

## endpoints
- GET  /health-check  -> {"status": "API is live"}
- POST /predict       -> PredictionResponse

## input example
{
  "Age": 35, "Annual_Income": 55000, "Loan_Amount": 20000,
  "Loan_Term_Months": 36, "Credit_Score": 620,
  "Employment_Years": 5, "Num_Credit_Lines": 4,
  "Debt_To_Income_Ratio": 0.35, "Num_Late_Payments": 2,
  "Has_Mortgage": 0, "Education": 1, "Loan_Purpose": 0,
  "Marital_Status": 1, "Loan_To_Income": 0.36, "Risk_Score": 7.8
}

## output
{
  "prediction": 0,
  "confidence_score": 0.87
}

## why response models
- strict output schema, no random fields returned
- swagger auto-generates documentation
- frontend/backend integration becomes easy
- no raw dict returned = cleaner API contract
