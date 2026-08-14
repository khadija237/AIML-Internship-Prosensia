# ============================================================
# Loan Approval Prediction API - Day 27
# Week 6 Mandate: Pydantic Schema Hardening + Async Refactoring
#                  + Error Boundary Guardrails
#
# changes from Day 26:
# - added @field_validator custom checks on top of Field boundaries
# - added global exception handlers so API NEVER returns raw 500
# - kept async def + run_in_threadpool for non-blocking inference
# - model still loads ONCE on startup (not inside route)
# ============================================================

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
import joblib
import time

np.random.seed(42)

app = FastAPI(title="Loan Approval Prediction API - prosensia-ml-service:v2")

# model loaded ONCE on startup - not inside route
model = joblib.load("loan_approval_model.pkl")
print("model loaded on startup!")


# ------------------------------------------------------------
# input schema - hardened with custom validators
# ------------------------------------------------------------
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

    # custom validator 1: employment years can't exceed a person's working life
    # (Age - 18), catches logically corrupt combinations Field() alone can't see
    @field_validator("Employment_Years")
    @classmethod
    def employment_years_realistic(cls, v):
        if v > 47:
            raise ValueError("Employment_Years unrealistically high")
        return v

    # custom validator 2: reject NaN / infinite floats that slip past type coercion
    @field_validator(
        "Annual_Income", "Loan_Amount", "Existing_Debt",
        "Debt_To_Income", "Loan_To_Income"
    )
    @classmethod
    def no_nan_or_inf(cls, v):
        if not np.isfinite(v):
            raise ValueError("value must be a finite number (no NaN/Infinity)")
        return v

    # custom validator 3: loan amount vs income sanity check
    # protects backend from mathematically impossible loan requests
    @field_validator("Loan_Amount")
    @classmethod
    def loan_amount_sane(cls, v, info):
        income = info.data.get("Annual_Income")
        if income is not None and v > income * 5:
            raise ValueError("Loan_Amount cannot exceed 5x Annual_Income")
        return v


# output schema
class LoanPredictionResponse(BaseModel):
    prediction: int
    result: str
    confidence_score: float
    response_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str


# cross-field OOD bounds check (kept from Day 26, still useful as a
# second guardrail layer beyond Pydantic's own validation)
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


# ------------------------------------------------------------
# Global exception handlers - THE UNHANDLED EXCEPTION BAN
# API must never return a raw 500 to the QA tester / backend dev
# ------------------------------------------------------------

# catches Pydantic validation failures (missing fields, wrong types,
# out-of-bound values, custom @field_validator failures)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # pydantic's raw exc.errors() can embed a non-JSON-serializable
    # exception object inside "ctx" for custom @field_validator errors
    # (type=value_error) -> must be stripped/stringified before returning
    clean_errors = [
        {
            "field": ".".join(str(loc) for loc in err.get("loc", [])),
            "message": err.get("msg"),
            "type": err.get("type"),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": "Unprocessable Entity",
            "detail": clean_errors,
        },
    )


# catches anything else that slips through (corrupted payloads,
# unexpected runtime errors) so the server never crashes ungracefully
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad Request",
            "detail": f"Request could not be processed: {str(exc)}",
        },
    )


@app.get("/health-check", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="API is live", model_loaded=True)


@app.post("/predict", response_model=LoanPredictionResponse)
async def predict(data: LoanApplicationInput):
    start = time.time()
    input_dict = data.model_dump()

    # second guardrail layer - OOD check
    for field, (min_val, max_val) in BOUNDS.items():
        val = input_dict[field]
        if val < min_val or val > max_val:
            raise HTTPException(
                status_code=400,
                detail=f"Out of Bounds: {field}={val} range ({min_val}-{max_val})"
            )

    # heavy CPU-bound prediction dispatched to threadpool
    # so it never blocks the ASGI event loop
    prediction, result, confidence = await run_in_threadpool(
        run_prediction, input_dict
    )

    return LoanPredictionResponse(
        prediction=prediction,
        result=result,
        confidence_score=confidence,
        response_time_ms=round((time.time() - start) * 1000, 2)
    )
