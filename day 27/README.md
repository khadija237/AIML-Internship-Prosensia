# Loan Approval Prediction - Hardened ML Microservice
prosensia-ml-service:v2 | Day 27

## what changed since Day 26
Day 26 got the model containerized and serving predictions. Day 27 focuses on
making that same service **survive bad input** without crashing, and making
sure heavy prediction work doesn't block other requests.

## project structure
```
day27/
├── main.py
├── loan_approval_model.pkl
├── Dockerfile
├── requirements.txt
└── README.md
```

## 1. validation strategy
Every request goes through two layers before it reaches the model:

**Layer 1 - Pydantic `Field()` constraints**
Every input field has a `ge`/`le` boundary (e.g. `Credit_Score` must be
between 300-850). Anything outside that range, missing, or the wrong type
(string instead of int, etc.) is rejected automatically by Pydantic before
it ever reaches our code.

**Layer 2 - custom `@field_validator` checks**
`Field()` boundaries only check one value at a time. Some bad data only
makes sense when you compare fields against each other, so custom
validators handle:
- `Employment_Years` can't be unrealistically high (>47 years)
- `Annual_Income`, `Loan_Amount`, `Existing_Debt`, `Debt_To_Income`,
  `Loan_To_Income` must be finite numbers (rejects NaN / Infinity payloads,
  which `Field()` alone does not catch)
- `Loan_Amount` cannot be more than 5x `Annual_Income` (cross-field sanity
  check — protects the model from mathematically implausible requests)

**Layer 3 - manual OOD bounds check inside `/predict`**
A final dictionary-based bounds check runs right before inference as a
belt-and-suspenders guard against any value that could still confuse the
model.

If a request fails layer 1 or 2 -> **422 Unprocessable Entity**.
If a request fails layer 3 -> **400 Bad Request**.
The API **never** returns a raw, unhandled 500 — a global exception handler
catches anything unexpected and converts it into a clean 400 response
instead of crashing the server.

## 2. why asynchronous endpoints improve performance
`/predict` and `/health-check` are both declared with `async def`. FastAPI
runs on an ASGI event loop that handles many requests concurrently on a
single thread — but only if none of those requests block that thread.

Model inference (`model.predict`) is **CPU-bound**, not I/O-bound, so
awaiting it directly would freeze the event loop and stall every other
request while one prediction runs. That's why inference is dispatched with
`run_in_threadpool(...)`: it hands the CPU-heavy work to a separate worker
thread, so the event loop stays free to accept and route new incoming
requests (like health checks or other predictions) while that one
prediction finishes in the background.

I/O-bound operations (calling a database, another API, reading a file)
behave differently — those can be awaited directly with `async def`
because they spend most of their time *waiting*, not *computing*, so the
event loop can do other work during that wait automatically.

## 3. purpose of the custom validators
Custom validators exist to catch corrupted or adversarial payloads that
pass basic type-checking but are still nonsensical — e.g. a syntactically
valid float that's actually `NaN`, or an employment history longer than a
person's working life. Without them, bad-but-technically-valid data could
reach the model and produce garbage predictions instead of a clean error.

## 4. how invalid requests are handled
- **422 Unprocessable Entity** — Pydantic-level failures: missing fields,
  wrong types, out-of-range values, or a custom validator rejecting the
  data. Response includes a clean `field` / `message` / `type` breakdown
  per error (raw pydantic error objects are sanitized before being
  returned, since they aren't JSON-serializable as-is).
- **400 Bad Request** — request passed schema validation but failed the
  manual OOD bounds check, or triggered an unexpected runtime error.
- The server process itself never crashes or returns a bare 500 — every
  failure path returns a structured JSON error object.

## how to build
```bash
docker build -t prosensia-ml-service:v2 .
```

## how to run
```bash
docker run -d -p 8000:8000 prosensia-ml-service:v2
```

## verify container running
```bash
docker ps
```

## how to test
swagger UI: http://127.0.0.1:8000/docs

**valid request:**
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

**chaos test - missing field (expect 422):**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 35, "Annual_Income": 75000}'
```

**chaos test - corrupted type (expect 422):**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": "not_a_number", "Annual_Income": 75000, "Loan_Amount": 25000,
       "Loan_Term_Months": 36, "Credit_Score": 720, "Employment_Years": 8,
       "Num_Dependents": 2, "Existing_Debt": 15000, "Education_Level": 1,
       "Property_Ownership": 1, "Loan_Purpose": 0, "Num_Prev_Loans": 2,
       "Num_Late_Payments": 0, "Debt_To_Income": 0.20, "Loan_To_Income": 0.33}'
```

## expected response (valid request)
```json
{
  "prediction": 1,
  "result": "Approved",
  "confidence_score": 0.82,
  "response_time_ms": 15.6
}
```

## expected response (validation failure - 422)
```json
{
  "error": "Unprocessable Entity",
  "detail": [
    {"field": "body.Credit_Score", "message": "Field required", "type": "missing"}
  ]
}
```

## api contract
| endpoint | method | input | output |
|----------|--------|-------|--------|
| /health-check | GET | none | status, model_loaded |
| /predict | POST | LoanApplicationInput | LoanPredictionResponse or ErrorResponse |

## performance
- model loaded ONCE on startup (not per request)
- async endpoints with `run_in_threadpool` for CPU-bound inference
- vectorized pandas operations (no for loops)
- p95 latency < 500ms
- global exception handlers guarantee zero unhandled 500 errors
