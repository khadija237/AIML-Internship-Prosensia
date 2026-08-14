# ============================================================
# generate_baseline.py - Day 28
# Generates a synthetic baseline dataset consistent with the
# Field() constraints already enforced in main.py (Day 26/27),
# then computes IQR-based statistical bounds per feature.
#
# This baseline stands in for "the training set" and is what
# the OOD interceptor uses to judge whether an incoming request
# looks like the data the model was actually trained on.
# ============================================================

import numpy as np
import pandas as pd
import json

np.random.seed(42)
N = 3000

# each feature generated as a clipped normal distribution centered
# near the middle of its realistic operating range, NOT the full
# Field() min/max (which are hard outer limits, not the typical
# distribution of real applicants)
data = {
    "Age": np.clip(np.random.normal(38, 10, N), 21, 65).round().astype(int),
    "Annual_Income": np.clip(np.random.normal(72000, 28000, N), 25000, 200000),
    "Loan_Amount": np.clip(np.random.normal(28000, 15000, N), 5000, 150000),
    "Loan_Term_Months": np.clip(np.random.normal(36, 12, N), 12, 60).round().astype(int),
    "Credit_Score": np.clip(np.random.normal(660, 90, N), 300, 850).round().astype(int),
    "Employment_Years": np.clip(np.random.normal(9, 6, N), 0, 35).round().astype(int),
    "Num_Dependents": np.clip(np.random.normal(1.8, 1.3, N), 0, 6).round().astype(int),
    "Existing_Debt": np.clip(np.random.normal(14000, 10000, N), 0, 80000),
    "Debt_To_Income": np.clip(np.random.normal(0.28, 0.15, N), 0.0, 1.0),
    "Loan_To_Income": np.clip(np.random.normal(0.55, 0.6, N), 0.0, 10.0),
    "Num_Prev_Loans": np.clip(np.random.normal(2.2, 1.6, N), 0, 8).round().astype(int),
    "Num_Late_Payments": np.clip(np.random.normal(1.1, 1.5, N), 0, 10).round().astype(int),
}

df = pd.DataFrame(data)

# only continuous / count features get statistical OOD bounds.
# low-cardinality categorical fields (Education_Level, Property_Ownership,
# Loan_Purpose) are already fully constrained by Field(ge=, le=) in
# main.py -- every valid category is equally "in distribution" there's
# no statistical outlier concept for a 4-category label, so they're
# excluded from the baseline stats file.
CONTINUOUS_FEATURES = list(data.keys())

# hard domain limits (same as the Field(ge=, le=) bounds already
# enforced in main.py). IQR fences can mathematically extend past
# these -- e.g. a lower bound of -2 dependents -- which is physically
# meaningless, so the OOD bound is always clipped back to the domain.
DOMAIN_LIMITS = {
    "Age": (21, 65), "Annual_Income": (25000, 200000),
    "Loan_Amount": (5000, 150000), "Loan_Term_Months": (12, 60),
    "Credit_Score": (300, 850), "Employment_Years": (0, 35),
    "Num_Dependents": (0, 6), "Existing_Debt": (0, 80000),
    "Debt_To_Income": (0.0, 1.0), "Loan_To_Income": (0.0, 10.0),
    "Num_Prev_Loans": (0, 8), "Num_Late_Payments": (0, 10),
}

baseline_stats = {}
for col in CONTINUOUS_FEATURES:
    series = df[col]
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    domain_min, domain_max = DOMAIN_LIMITS[col]

    # Tukey's fences: 1.5x IQR beyond Q1/Q3.
    # chosen over a fixed Z-score cutoff because several features
    # (Loan_To_Income, Existing_Debt, Num_Late_Payments) are right-skewed,
    # not normally distributed -- IQR bounds don't assume normality,
    # while a 3-sigma Z-score rule would either flag too many legitimate
    # high earners or miss genuine low-end outliers on skewed data.
    lower_bound = max(q1 - 1.5 * iqr, domain_min)
    upper_bound = min(q3 + 1.5 * iqr, domain_max)

    baseline_stats[col] = {
        "mean": round(float(series.mean()), 4),
        "std": round(float(series.std()), 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
    }

with open("baseline_stats.json", "w") as f:
    json.dump(baseline_stats, f, indent=2)

print("baseline_stats.json written.")
print(json.dumps(baseline_stats, indent=2))
