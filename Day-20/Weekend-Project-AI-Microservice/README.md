# Loan Approval Prediction - AI Microservice
ProSensia AI/ML Bootcamp ( Week 4 Weekend Project)

## overview
production-ready AI microservice that predicts whether a loan application
will be approved or rejected based on applicant financial profile.

## files
- main.py
- loan_approval_model.pkl
- loan_approval_dataset.csv
- train_model.ipynb
- requirements.txt
- README.md

## how to run
pip install -r requirements.txt
uvicorn main:app --reload

swagger UI: http://127.0.0.1:8000/docs

## endpoints
- GET  /health-check  -> {"status": "API is live"}
- POST /predict       -> LoanPredictionResponse

## input example
{
  "Age": 35,
  "Annual_Income": 75000,
  "Loan_Amount": 25000,
  "Loan_Term_Months": 36,
  "Credit_Score": 720,
  "Employment_Years": 8,
  "Num_Dependents": 2,
  "Existing_Debt": 15000,
  "Education_Level": 1,
  "Property_Ownership": 1,
  "Loan_Purpose": 0,
  "Num_Prev_Loans": 2,
  "Num_Late_Payments": 0,
  "Debt_To_Income": 0.20,
  "Loan_To_Income": 0.33
}

## output
{
  "prediction": 1,
  "result": "Approved",
  "confidence_score": 0.82
}

## pipeline
- 3000 samples generated
- train/test split 80/20 stratified
- SMOTE for class imbalance
- random forest (100 trees, max_depth=10)
- pydantic input validation
- OOD guardrails (400 error)
- structured response model
