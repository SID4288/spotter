"""Smoke tests for the market_index imputation fix.

Run with: python -m pytest tests/ -q  (pytest not required; plain asserts work too)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

import train as T


def test_missing_market_uses_batch_date_median():
    # Training lookup covers Jan only; target batch is Feb with one missing row.
    train = pd.DataFrame({
        "weight": [30000.0, 30000.0],
        "equipment": ["Dry Van", "Dry Van"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "market_index": [1.5, 1.5],
        "is_corrupt": [False, False],
    })
    prep = T.fit_preprocessors(train)
    target = pd.DataFrame({
        "weight": [30000.0, 30000.0, 30000.0],
        "equipment": ["Dry Van"] * 3,
        "date": pd.to_datetime(["2025-02-01"] * 3),
        "market_index": [0.9, 0.94, float("nan")],
        "pickup_lat": [0.0] * 3, "pickup_lon": [0.0] * 3,
        "delivery_lat": [1.0] * 3, "delivery_lon": [1.0] * 3,
        "distance": [360.0] * 3,
        "equipment_code": [0] * 3,
    })
    out = T.transform(target, prep)
    # Same-date median of the batch is 0.92, not the global training median 1.5.
    assert abs(out.loc[2, "market_index"] - 0.92) < 1e-9, out.loc[2, "market_index"]
    assert out["market_index"].notna().all()


def test_december_lane_exists():
    train = pd.read_csv(T.DATA / "train_test.csv", parse_dates=["date"])
    assert ((train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")).any()


if __name__ == "__main__":
    test_missing_market_uses_batch_date_median()
    test_december_lane_exists()
    print("tests passed")
