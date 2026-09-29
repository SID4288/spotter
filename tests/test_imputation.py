"""Production guards: inductive imputation, strict schema, compact HGB pipeline.

Run with: python -m pytest tests/ -q  (plain asserts work too)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

import train as T


def test_market_inductive_no_batch_peek():
    """Missing market must NOT use batch same-date median (transductive leak)."""
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
        "pickup": ["A"] * 3, "delivery": ["B"] * 3,
        "pickup_lat": [0.0] * 3, "pickup_lon": [0.0] * 3,
        "delivery_lat": [1.0] * 3, "delivery_lon": [1.0] * 3,
        "distance": [360.0] * 3,
    })
    out = T.transform(target, prep)
    # Feb unseen: month miss (train Jan only), dow miss (Wed vs Sat) -> global 1.5.
    # Old transductive code returned batch median 0.92; inductive must not.
    assert abs(out.loc[2, "market_index"] - 1.5) < 1e-9, out.loc[2, "market_index"]
    assert out["market_index"].notna().all()


def test_market_month_fallback_inductive():
    train = pd.DataFrame({
        "weight": [30000.0] * 4,
        "equipment": ["Dry Van"] * 4,
        "date": pd.to_datetime(["2025-01-05", "2025-01-06", "2025-02-03", "2025-02-04"]),
        "market_index": [1.0, 1.0, 2.0, 2.0],
    })
    prep = T.fit_preprocessors(train)
    target = pd.DataFrame({
        "weight": [30000.0],
        "equipment": ["Dry Van"],
        "date": pd.to_datetime(["2025-02-10"]),  # unseen exact date, seen month
        "market_index": [float("nan")],
        "pickup": ["A"], "delivery": ["B"],
        "pickup_lat": [0.0], "pickup_lon": [0.0],
        "delivery_lat": [1.0], "delivery_lon": [1.0],
        "distance": [360.0],
    })
    out = T.transform(target, prep)
    assert abs(out.loc[0, "market_index"] - 2.0) < 1e-9


def test_single_row_parity():
    """Batch row transform must equal single-row transform (inductive proof)."""
    rng = np.random.RandomState(0)
    train = pd.DataFrame({
        "weight": [30000.0, 31000.0, 29000.0],
        "equipment": ["Dry Van", "Reefer", "Dry Van"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-02-01"]),
        "market_index": [1.0, 1.1, 1.2],
        "pickup": ["A", "B", "A"], "delivery": ["B", "C", "B"],
        "pickup_lat": [0.0, 1.0, 0.0], "pickup_lon": [0.0, 1.0, 0.0],
        "delivery_lat": [1.0, 2.0, 1.0], "delivery_lon": [1.0, 2.0, 1.0],
        "distance": [360.0, 500.0, 370.0],
    })
    prep = T.fit_preprocessors(train)
    batch = pd.DataFrame({
        "weight": [30000.0, float("nan")],
        "equipment": ["Dry Van", "Dry Van"],
        "date": pd.to_datetime(["2025-03-01", "2025-03-01"]),
        "market_index": [float("nan"), float("nan")],
        "pickup": ["A", "ZZZ"], "delivery": ["B", "B"],
        "pickup_lat": [0.0, 9.0], "pickup_lon": [0.0, 9.0],
        "delivery_lat": [1.0, 1.0], "delivery_lon": [1.0, 1.0],
        "distance": [360.0, 360.0],
    })
    full = T.transform(batch, prep)
    single = T.transform(batch.head(1), prep)
    assert np.isclose(full["market_index"].iloc[0], single["market_index"].iloc[0])
    assert full["lane_freq"].iloc[1] == 0  # unseen pickup -> lane 0


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
    assert pd.isna(out["equipment_cat"].iloc[0])  # unseen -> NaN bin for HGB


def test_validate_frame_strict():
    base = pd.DataFrame({
        "pickup": ["A"], "delivery": ["B"],
        "pickup_lat": [0.0], "pickup_lon": [0.0],
        "delivery_lat": [1.0], "delivery_lon": [1.0],
        "distance": [360.0], "equipment": ["Dry Van"],
        "weight": [30000.0], "date": pd.to_datetime(["2025-01-01"]),
        "market_index": [1.0],
    })
    T.validate_frame(base)  # clean passes
    bad = base.copy(); bad["distance"] = 0
    try:
        T.validate_frame(bad); assert False, "distance<=0 must raise"
    except ValueError:
        pass
    bad = base.copy(); bad["weight"] = -5.0
    try:
        T.validate_frame(bad); assert False, "negative weight must raise"
    except ValueError:
        pass
    bad = base.copy(); bad["quote_signal"] = 2.0
    try:
        T.validate_frame(bad); assert False, "quote_signal must raise"
    except ValueError:
        pass


def test_pipeline_selects_only_features():
    pipe = T.make_pipeline()
    names = [n for n, _ in pipe.steps]
    assert names == ["weight_distance", "market", "features", "select", "model"]
    assert "geo_distance" not in T.FEATURES  # dropped: corr 1.00 with distance
    from modeling import DollarHGBRegressor
    assert isinstance(pipe.named_steps["model"], DollarHGBRegressor)
    assert "pickup_cat" in pipe.named_steps["select"].columns


def test_pipeline_artifact_roundtrip():
    import pathlib
    import joblib
    if not pathlib.Path(T.PIPE_PATH).exists():
        return
    # New artifacts must load with plain joblib.load (no __main__ hack).
    pipe = joblib.load(T.PIPE_PATH)
    assert [n for n, _ in pipe.steps] == ["weight_distance", "market", "features", "select", "model"]
    assert hasattr(pipe.named_steps["model"], "smearing_")


if __name__ == "__main__":
    test_market_inductive_no_batch_peek()
    test_market_month_fallback_inductive()
    test_single_row_parity()
    test_december_lane_exists()
    test_flag_fitted_on_train_only()
    test_unknown_equipment_and_distance_guards()
    test_validate_frame_strict()
    test_pipeline_selects_only_features()
    test_pipeline_artifact_roundtrip()
    print("tests passed")
