# Loan Approval Prediction - OOD-Guarded ML Microservice
prosensia-ml-service:v3 | Day 28

## what changed since Day 27
Day 27 hardened the *shape* of incoming data (types, individual field
ranges, cross-field sanity checks). Day 28 hardens the *statistical
plausibility* of incoming data — a payload can pass every single Field()
boundary and still look nothing like anything the model was trained on.
That's what today's OOD (Out-of-Distribution) Boundary Interceptor catches.

## project structure
```
day28/
├── main.py                 # FastAPI app + OODInterceptorMiddleware
├── ood_checker.py           # OODBoundaryChecker class
├── generate_baseline.py     # builds baseline dataset + stats (run once)
├── baseline_stats.json      # per-feature mean/std/IQR/bounds
├── loan_approval_model.pkl
├── Dockerfile
├── requirements.txt
└── README.md
```

## 1. why OOD guardrails matter for enterprise ML microservices
A trained model has only ever seen data inside a certain statistical
envelope. Feed it something wildly outside that envelope — an income of
199,999 combined with 79,999 in existing debt, for example — and it will
still confidently return a prediction. The model has no built-in concept
of "I've never seen anything like this, I don't actually know." That
silent failure mode is worse than an error: it looks like a normal,
trustworthy response to every downstream system that consumes it.

The **Garbage Prediction Ban** exists because of this: it's better to
reject a statistically implausible request with a clear error than to let
the model guess and hand back a confident-looking number nobody should
trust.

## 2. how the statistical baseline was built
`generate_baseline.py` builds a synthetic dataset (3,000 rows, seed=42)
shaped like realistic loan applicants — each continuous feature drawn from
a normal distribution centered on a plausible typical value, then clipped
to the same hard domain limits already enforced by `Field(ge=, le=)` in
`main.py`. From that baseline, per feature, it computes:
- mean, standard deviation
- Q1 (25th percentile), Q3 (75th percentile), IQR = Q3 − Q1
- **Tukey's fences**: `lower = Q1 − 1.5×IQR`, `upper = Q3 + 1.5×IQR`,
  clipped back to the field's hard domain limits

Low-cardinality categorical fields (`Education_Level`, `Property_Ownership`,
`Loan_Purpose`) are excluded — every valid category is equally "in
distribution" for a label with 2–4 possible values, so there's no
statistical outlier concept to apply there. They're still fully guarded by
`Field(ge=, le=)` from Day 26/27.

## 3. mathematical justification — IQR vs. Z-score vs. Isolation Forest
**Chosen: IQR / Tukey's fences (1.5×IQR beyond Q1/Q3)**

- **Why not Z-score (mean ± 3σ):** Z-score assumes the underlying feature
  is approximately normally distributed. Several features here
  (`Loan_To_Income`, `Existing_Debt`, `Num_Late_Payments`) are right-skewed
  — most applicants cluster low, with a long tail of high-debt/high-ratio
  cases. On skewed data, a symmetric Z-score cutoff either lets through
  extreme values on the long-tail side or wrongly flags normal values on
  the short side. IQR bounds are computed from percentiles, not the mean,
  so they don't assume symmetry and adapt to the actual shape of each
  feature.
- **Why not Isolation Forest:** Isolation Forest is a trained multivariate
  detector — good for catching anomalies in the *combination* of features,
  but it's a black box (harder to explain a specific rejection reason to a
  QA tester or backend dev) and adds real training/maintenance overhead
  for a service this size. Since the requirement here is a fast, explainable
  per-field boundary check, a lightweight statistical checker is the
  better fit. It's kept as a documented alternative, not implemented, to
  avoid over-engineering the interceptor.
- **Avoiding excessive false positives:** the 1.5× multiplier (Tukey's
  standard convention) is deliberately looser than the stricter 3×IQR
  "extreme outlier" convention — 1.5× flags genuine statistical outliers
  without rejecting the normal spread of real applicants. Bounds are also
  clipped to each field's hard domain limits so the fence can never be
  tighter than what Day 26/27 already allows as a legitimate value.

## 4. how the interceptor works
`OODInterceptorMiddleware` (in `main.py`) is registered as global FastAPI
middleware. For every `POST /predict`:
1. It reads the raw request body **before** Pydantic ever sees it.
2. It replays that body downstream (via a patched `receive()` callable) so
   the route handler still gets a normal, unconsumed request.
3. It checks each numeric field against `baseline_stats.json`'s IQR fences
   using `OODBoundaryChecker.check()`.
4. If any field is outside its fence → returns **400 Bad Request: Data
   Out of Bounds** immediately, with a `violations` list naming exactly
   which fields failed and why. **The model is never called.**
5. If everything is in-distribution → the request proceeds to Pydantic
   validation (Day 27's `@field_validator` checks) and then the model.

This means invalid data is intercepted in one central place before it can
reach either the validation layer or the model — same principle as
`/health-check` and any future routes being automatically exempt, since
the middleware only inspects `POST /predict`.

## how to build
```bash
docker build -t prosensia-ml-service:v3 .
```

## how to run
```bash
docker run -d -p 8000:8000 prosensia-ml-service:v3
```

## verify container running
```bash
docker ps
```

## how to test
swagger UI: http://127.0.0.1:8000/docs

**normal request (in-distribution, expect 200):**
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

**extreme OOD outlier (technically Field-valid, statistically implausible,
expect 400 "Data Out of Bounds"):**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35, "Annual_Income": 199999,
    "Loan_Amount": 148000, "Loan_Term_Months": 36,
    "Credit_Score": 720, "Employment_Years": 8,
    "Num_Dependents": 2, "Existing_Debt": 79999,
    "Education_Level": 1, "Property_Ownership": 1,
    "Loan_Purpose": 0, "Num_Prev_Loans": 2,
    "Num_Late_Payments": 0, "Debt_To_Income": 0.20,
    "Loan_To_Income": 0.33
  }'
```

## expected response (OOD rejection)
```json
{
  "error": "Data Out of Bounds",
  "detail": "Request rejected before reaching the model: one or more values fall outside the statistical training distribution.",
  "violations": [
    {
      "field": "Annual_Income",
      "value": 199999,
      "expected_range": [25000, 146634.1019],
      "reason": "Annual_Income=199999 falls outside the statistical baseline range [25000, 146634.1019] (IQR fences)"
    }
  ]
}
```

## api contract
| endpoint | method | input | output |
|----------|--------|-------|--------|
| /health-check | GET | none | status, model_loaded |
| /predict | POST | LoanApplicationInput (screened by OOD middleware first) | LoanPredictionResponse or ErrorResponse |

## performance
- model AND OOD checker both loaded ONCE on startup (not per request)
- async endpoints with `run_in_threadpool` for CPU-bound inference
- OOD check is O(1) per feature (dictionary lookup + comparison) — negligible
  latency added before routing
- global exception handlers + OOD middleware guarantee zero unhandled 500s
  and zero garbage predictions on out-of-distribution input
