# adversarial security test script
# run: python security_tests.py

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

valid_payload = {
    "Age": 35, "Annual_Income": 75000, "Loan_Amount": 25000,
    "Loan_Term_Months": 36, "Credit_Score": 720,
    "Employment_Years": 8, "Num_Dependents": 2,
    "Existing_Debt": 15000, "Education_Level": 1,
    "Property_Ownership": 1, "Loan_Purpose": 0,
    "Num_Prev_Loans": 2, "Num_Late_Payments": 0,
    "Debt_To_Income": 0.20, "Loan_To_Income": 0.33
}

tests = [
    {
        "name": "valid input",
        "payload": valid_payload,
        "expected": 200
    },
    {
        "name": "negative age (OOD bounds)",
        "payload": {**valid_payload, "Age": -5},
        "expected": 422
    },
    {
        "name": "extremely high income (OOD bounds)",
        "payload": {**valid_payload, "Annual_Income": 999999999},
        "expected": 422
    },
    {
        "name": "sql injection in string field",
        "payload": {**valid_payload, "Age": "SELECT * FROM users"},
        "expected": 422
    },
    {
        "name": "script injection attempt",
        "payload": {**valid_payload, "Age": "<script>alert(1)</script>"},
        "expected": 422
    },
    {
        "name": "missing required field",
        "payload": {"Age": 35, "Annual_Income": 75000},
        "expected": 422
    },
    {
        "name": "wrong data type",
        "payload": {**valid_payload, "Credit_Score": "abc"},
        "expected": 422
    },
    {
        "name": "credit score out of range",
        "payload": {**valid_payload, "Credit_Score": 9999},
        "expected": 422
    },
]

print("SECURITY PENETRATION TEST RESULTS")
print("=" * 50)
passed = 0
for test in tests:
    try:
        r = requests.post(f"{BASE_URL}/predict", json=test["payload"], timeout=5)
        status = "PASS" if r.status_code == test["expected"] else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"[{status}] {test['name']} -> got {r.status_code} expected {test['expected']}")
    except Exception as e:
        print(f"[ERROR] {test['name']} -> {e}")

print(f"\n{passed}/{len(tests)} tests passed!")
