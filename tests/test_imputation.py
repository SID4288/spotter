"""Smoke tests for the market_index imputation fix.

Run with: python -m pytest tests/ -q  (pytest not required; plain asserts work too)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
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


def test_flag_fitted_on_train_only():
    rng = __import__("numpy").random.RandomState(0)
    d = rng.uniform(300, 1500, 30)
    r = np.exp(0.5 * np.log(d) + 6.7 + rng.normal(0, 0.03, 30))
    df = pd.DataFrame({"posted_rate": r, "distance": d})
    df.loc[30] = [8000.0, 360.0]
    tr, va = df.iloc[:30], df.iloc[30:]
    coef, med, mad = T._flag_params(tr)
    assert bool(T._apply_flag(va, coef, med, mad).iloc[0])
    assert T._apply_flag(tr, coef, med, mad).mean() < 0.1


def test_unknown_equipment_and_distance_guards():
    train = pd.DataFrame({
        "weight": [30000.0, 30000.0],
        "equipment": ["Dry Van", "Dry Van"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "market_index": [1.0, 1.0],
        "pickup": ["A", "A"], "delivery": ["B", "B"],
        "pickup_lat": [0.0, 0.0], "pickup_lon": [0.0, 0.0],
        "delivery_lat": [1.0, 1.0], "delivery_lon": [1.0, 1.0],
        "distance": [360.0, 370.0],
    })
    prep = T.fit_preprocessors(train)
    target = pd.DataFrame({
        "weight": [float("nan")],
        "equipment": ["MysteryRig"],
        "date": pd.to_datetime(["2025-02-01"]),
        "market_index": [float("nan")],
        "pickup": ["A"], "delivery": ["B"],
        "pickup_lat": [0.0], "pickup_lon": [0.0],
        "delivery_lat": [1.0], "delivery_lon": [1.0],
        "distance": [float("nan")],
    })
    out = T.transform(target, prep)
    assert out["equipment_code"].iloc[0] == -1
    assert out[T.FEATURES].notna().all().all()


def test_pipeline_selects_only_features():
    pipe = T.make_pipeline()
    names = [n for n, _ in pipe.steps]
    assert names == ["weight_distance", "market", "features", "select", "model"]
    assert "geo_distance" not in T.FEATURES  # dropped: corr 1.00 with distance


def test_pipeline_artifact_roundtrip():
    import pathlib
    if not pathlib.Path(T.PIPE_PATH).exists():
        return
    pipe = T.load_pipeline()
    assert [n for n, _ in pipe.steps] == ["weight_distance", "market", "features", "select", "model"]
    assert pipe.named_steps.model.n_jobs == -1


if __name__ == "__main__":
    test_missing_market_uses_batch_date_median()
    test_december_lane_exists()
    test_flag_fitted_on_train_only()
    test_unknown_equipment_and_distance_guards()
    test_pipeline_selects_only_features()
    test_pipeline_artifact_roundtrip()
    print("tests passed")
