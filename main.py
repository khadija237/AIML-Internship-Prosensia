from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Default Prediction API")

model = joblib.load("loan_default_rf_model.pkl")
print("model loaded!")


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
    print("received:", input_dict)

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]

    return {"loan_default_prediction": int(prediction)}
