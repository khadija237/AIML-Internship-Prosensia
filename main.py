from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Default Prediction API")

model = joblib.load("loan_default_rf_model.pkl")
print("model loaded!")


# training data boundaries (from week 2 loan dataset)
BOUNDS = {
    "Age":                  (22, 65),
    "Annual_Income":        (20000, 150000),
    "Loan_Amount":          (5000, 100000),
    "Loan_Term_Months":     (12, 60),
    "Credit_Score":         (300, 850),
    "Employment_Years":     (0, 30),
    "Num_Credit_Lines":     (1, 15),
    "Debt_To_Income_Ratio": (0.05, 0.65),
    "Num_Late_Payments":    (0, 10),
    "Has_Mortgage":         (0, 1),
}


class LoanApplication(BaseModel):
    Age: int
    Annual_Income: float
    Loan_Amount: float
    Loan_Term_Months: int
    Credit_Score: int
    Employment_Years: int
    Num_Credit_Lines: int
    Debt_To_Income_Ratio: float
    Num_Late_Payments: int
    Has_Mortgage: int
    Education_encoded: int
    Loan_Purpose_encoded: int
    Marital_Status_encoded: int
    Loan_To_Income_Ratio: float
    Income_Per_Credit_Line: float
    Monthly_Payment_Est: float
    Risk_Score: float


@app.get("/health-check")
def health_check():
    return {"status": "API is live"}


@app.post("/predict")
def predict(data: LoanApplication):
    input_dict = data.model_dump()

    # OOD guardrails - check values against training bounds
    for field, (min_val, max_val) in BOUNDS.items():
        value = input_dict[field]
        if value < min_val or value > max_val:
            raise HTTPException(
                status_code=400,
                detail=f"Data Out of Bounds: {field} value {value} is outside training range ({min_val} - {max_val})"
            )

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]

    return {"loan_default_prediction": int(prediction)}
