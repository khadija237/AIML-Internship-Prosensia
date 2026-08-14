# ============================================================
# ood_checker.py - Day 28
# Lightweight statistical OOD (Out-of-Distribution) checker.
#
# Loads the baseline stats computed by generate_baseline.py and
# checks incoming feature values against Tukey's IQR fences
# (lower_bound / upper_bound per feature). Anything outside those
# fences is flagged as OOD -- statistically implausible given the
# distribution the model was trained on, even if it technically
# passes the hard Field(ge=, le=) limits in main.py.
# ============================================================

import json
from pathlib import Path

BASELINE_PATH = Path(__file__).parent / "baseline_stats.json"


class OODBoundaryChecker:
    def __init__(self, baseline_path: Path = BASELINE_PATH):
        with open(baseline_path, "r") as f:
            self.baseline_stats = json.load(f)

    def check(self, payload: dict) -> list[dict]:
        """
        Checks every feature present in `payload` that has a baseline
        entry. Returns a list of violation dicts (empty list = in
        distribution, safe to send to the model).
        """
        violations = []
        for field, bounds in self.baseline_stats.items():
            if field not in payload:
                continue
            value = payload[field]
            if not isinstance(value, (int, float)):
                continue  # type errors are Pydantic's job, not OOD's

            lower, upper = bounds["lower_bound"], bounds["upper_bound"]
            if value < lower or value > upper:
                violations.append({
                    "field": field,
                    "value": value,
                    "expected_range": [lower, upper],
                    "reason": (
                        f"{field}={value} falls outside the statistical "
                        f"baseline range [{lower}, {upper}] (IQR fences)"
                    ),
                })
        return violations
