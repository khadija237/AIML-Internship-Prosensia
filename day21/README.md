# Loan Approval Prediction - Dockerized AI Microservice
ProSensia AI/ML Bootcamp | Month 2 Week 5 Day 21

## project structure
```
day21/
├── main.py
├── loan_approval_model.pkl
├── Dockerfile
├── requirements.txt
└── README.md
```

## how to build docker image
```bash
docker build -t loan-approval-api .
```

## how to run container
```bash
docker run -p 8000:8000 loan-approval-api
```

## how to test
open browser: http://127.0.0.1:8000/docs

or use curl:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## expected response
```json
{
  "prediction": 1,
  "result": "Approved",
  "confidence_score": 0.82
}
```

## docker vs virtual machines
- VM: har app ka apna Guest OS hota hai, hypervisor overhead bohot zyada
- Docker: sab apps ek hi OS kernel share karte hain, sirf app isolate hoti hai
- Docker containers start in seconds, VMs take minutes
- Docker images are MBs, VM images are GBs

## why docker for ML
- model + dependencies ek saath package ho jaate hain
- "works on my machine" problem khatam
- easy deployment on any server
- consistent environment everywhere
