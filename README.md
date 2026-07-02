# Loan Default Prediction API - FastAPI
ProSensia AI/ML Bootcamp | Week 4 Day 1

## what i built
deployed the week 2 random forest model as a REST API using FastAPI.
model is loaded into memory on startup and serves predictions via POST /predict.

## files
- main.py
- production_rf_model.pkl
- requirements.txt
- README.md

## how to run
pip install -r requirements.txt
uvicorn main:app --reload

then open: http://127.0.0.1:8000/docs

## endpoints
- GET  /              -> api running message
- GET  /health-check  -> {"status": "API is live"}
- POST /predict       -> takes loan features, returns prediction

## example POST /predict input
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

## what is pickling
converting a python object (trained model) into a byte stream saved as .pkl file.
joblib.dump() saves it, joblib.load() loads it back.
security risk: never load .pkl files from unknown sources — they can execute arbitrary code on load.

## why fastapi not flask
fastapi is async, faster, auto-generates swagger docs, built-in data validation with pydantic.
flask is older and slower for ML inference workloads.
