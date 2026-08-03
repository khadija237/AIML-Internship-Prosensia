from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Approval Prediction API")

model = joblib.load("loan_approval_model.pkl")
print("model loaded!")


# training data boundaries
BOUNDS = {
    "Age":               (21, 65),
    "Annual_Income":     (25000, 200000),
    "Loan_Amount":       (5000, 150000),
    "Loan_Term_Months":  (12, 60),
    "Credit_Score":      (300, 850),
    "Employment_Years":  (0, 35),
    "Num_Dependents":    (0, 6),
    "Existing_Debt":     (0, 80000),
    "Education_Level":   (0, 3),
    "Property_Ownership":(0, 1),
    "Loan_Purpose":      (0, 3),
    "Num_Prev_Loans":    (0, 8),
    "Num_Late_Payments": (0, 10),
}


# input schema
class LoanApplicationInput(BaseModel):
    Age: int
    Annual_Income: float
    Loan_Amount: float
    Loan_Term_Months: int
    Credit_Score: int
    Employment_Years: int
    Num_Dependents: int
    Existing_Debt: float
    Education_Level: int
    Property_Ownership: int
    Loan_Purpose: int
    Num_Prev_Loans: int
    Num_Late_Payments: int
    Debt_To_Income: float
    Loan_To_Income: float


# output schema
class LoanPredictionResponse(BaseModel):
    prediction: int
    result: str
    confidence_score: float


@app.get("/health-check")
def health_check():
    return {"status": "API is live"}


@app.post("/predict", response_model=LoanPredictionResponse)
def predict(data: LoanApplicationInput):
    input_dict = data.model_dump()

    # OOD guardrails
    for field, (min_val, max_val) in BOUNDS.items():
        value = input_dict[field]
        if value < min_val or value > max_val:
            raise HTTPException(
                status_code=400,
                detail=f"Data Out of Bounds: {field} value {value} is outside training range ({min_val} - {max_val})"
            )

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    confidence = round(float(probability[int(prediction)]), 4)
    result = "Approved" if prediction == 1 else "Rejected"

    return LoanPredictionResponse(
        prediction=int(prediction),
        result=result,
        confidence_score=confidence
    )
