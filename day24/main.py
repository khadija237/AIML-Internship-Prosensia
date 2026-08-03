# ============================================================
# Loan Approval Prediction API - Day 26
# model loads ONCE on startup for performance
# no for loops in inference path - vectorized operations only
# random seed set for reproducibility
# ============================================================

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import joblib
import time

np.random.seed(42)

app = FastAPI(title="Loan Approval Prediction API - prosensia-ml-service:v1")

# model loaded ONCE on startup - not inside route
# this avoids reloading model on every request (latency optimization)
model = joblib.load("loan_approval_model.pkl")
print("model loaded on startup!")


# input schema
class LoanApplicationInput(BaseModel):
    Age: int = Field(..., ge=21, le=65)
    Annual_Income: float = Field(..., ge=25000, le=200000)
    Loan_Amount: float = Field(..., ge=5000, le=150000)
    Loan_Term_Months: int = Field(..., ge=12, le=60)
    Credit_Score: int = Field(..., ge=300, le=850)
    Employment_Years: int = Field(..., ge=0, le=35)
    Num_Dependents: int = Field(..., ge=0, le=6)
    Existing_Debt: float = Field(..., ge=0, le=80000)
    Education_Level: int = Field(..., ge=0, le=3)
    Property_Ownership: int = Field(..., ge=0, le=1)
    Loan_Purpose: int = Field(..., ge=0, le=3)
    Num_Prev_Loans: int = Field(..., ge=0, le=8)
    Num_Late_Payments: int = Field(..., ge=0, le=10)
    Debt_To_Income: float = Field(..., ge=0.0, le=1.0)
    Loan_To_Income: float = Field(..., ge=0.0, le=10.0)


# output schema
class LoanPredictionResponse(BaseModel):
    prediction: int
    result: str
    confidence_score: float
    response_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# OOD bounds check
BOUNDS = {
    "Age": (21, 65), "Annual_Income": (25000, 200000),
    "Loan_Amount": (5000, 150000), "Credit_Score": (300, 850),
    "Num_Late_Payments": (0, 10),
}


def run_prediction(input_dict: dict):
    # vectorized - no for loops
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    confidence = round(float(probability[int(prediction)]), 4)
    result = "Approved" if prediction == 1 else "Rejected"
    return int(prediction), result, confidence


@app.get("/health-check", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="API is live", model_loaded=True)


@app.post("/predict", response_model=LoanPredictionResponse)
async def predict(data: LoanApplicationInput):
    start = time.time()
    input_dict = data.model_dump()

    # OOD check
    for field, (min_val, max_val) in BOUNDS.items():
        val = input_dict[field]
        if val < min_val or val > max_val:
            raise HTTPException(
                status_code=400,
                detail=f"Out of Bounds: {field}={val} range ({min_val}-{max_val})"
            )

    prediction, result, confidence = await run_in_threadpool(
        run_prediction, input_dict
    )

    return LoanPredictionResponse(
        prediction=prediction,
        result=result,
        confidence_score=confidence,
        response_time_ms=round((time.time() - start) * 1000, 2)
    )
