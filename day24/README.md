# Loan Approval Prediction - Containerized ML Microservice
prosensia-ml-service:v1 | Day 26

## project structure
```
day26/
├── main.py
├── loan_approval_model.pkl
├── Dockerfile
├── requirements.txt
└── README.md
```

## how to build
```bash
docker build -t prosensia-ml-service:v1 .
```

## how to run
```bash
docker run -d -p 8000:8000 prosensia-ml-service:v1
```

## verify container running
```bash
docker ps
```

## how to test
swagger UI: http://127.0.0.1:8000/docs

curl:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35, "Annual_Income": 75000,
    "Loan_Amount": 25000, "Loan_Term_Months": 36,
    "Credit_Score": 720, "Employment_Years": 8,
    "Num_Dependents": 2, "Existing_Debt": 15000,
    "Education_Level": 1, "Property_Ownership": 1,
    "Loan_Purpose": 0, "Num_Prev_Loans": 2,
    "Num_Late_Payments": 0, "Debt_To_Income": 0.20,
    "Loan_To_Income": 0.33
  }'
```

## expected response
```json
{
  "prediction": 1,
  "result": "Approved",
  "confidence_score": 0.82,
  "response_time_ms": 15.6
}
```

## api contract
| endpoint | method | input | output |
|----------|--------|-------|--------|
| /health-check | GET | none | status, model_loaded |
| /predict | POST | LoanApplicationInput | LoanPredictionResponse |

## performance
- model loaded ONCE on startup (not per request)
- async endpoints with run_in_threadpool
- vectorized pandas operations (no for loops)
- p95 latency < 500ms
