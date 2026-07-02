from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Loan Default Prediction API")

model = joblib.load("production_rf_model.pkl")
print("model loaded!")


@app.get("/health-check")
def health_check():
    return {"status": "API is live"}


@app.post("/predict")
def predict(data: dict):
    # placeholder - just printing incoming json to terminal
    print("received payload:", data)
    return {"message": "payload received", "data": data}
