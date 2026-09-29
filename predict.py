"""Standalone single-row inference for the freight-rate pipeline.

Usage:
    python predict.py --input '{"pickup":"Lexington","delivery":"Fort Wayne",...}'
    python predict.py --file payload.json

Payload fields (all required):
    pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon,
    distance (>0), equipment (Dry Van/Reefer/Flatbed; unknown allowed -> -1/NaN),
    weight (>=0 or null), date (YYYY-MM-DD), market_index (float or null).

Loads output/pipeline.joblib with plain joblib.load (no namespace hacks;
requires preprocessing.py + modeling.py on PYTHONPATH). Validates schema via
train.validate_frame, runs pipe.predict (log-RPM + smearing inside), returns
rate = exp(pred) * distance. Prints latency; warns if > 50ms.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PIPE_PATH = ROOT / "output" / "pipeline.joblib"

# Ensure local modules resolve when invoked from any cwd.
sys.path.insert(0, str(ROOT))
import preprocessing  # noqa: F401  (needed for unpickling)
import modeling  # noqa: F401
from train import validate_frame


def load_artifact(path: Path | str = PIPE_PATH):
    """Standard deserialization (no __main__ aliasing)."""
    return joblib.load(path)


def validate_payload(payload: dict) -> pd.DataFrame:
    required = ["pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat",
                "delivery_lon", "distance", "equipment", "weight", "date", "market_index"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"missing payload fields: {missing}")
    df = pd.DataFrame([payload])
    # Coerce numerics; keep NaN for imputable fields.
    for col in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
                "weight", "market_index"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not np.isfinite(df["distance"].iloc[0]) or float(df["distance"].iloc[0]) <= 0:
        raise ValueError(f"distance must be a positive number, got {payload.get('distance')!r}")
    w = df["weight"].iloc[0]
    if pd.notna(w) and float(w) < 0:
        raise ValueError(f"weight must be >= 0 or null, got {payload.get('weight')!r}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError(f"date must be parseable YYYY-MM-DD, got {payload.get('date')!r}")
    validate_frame(df, require_label=False)
    return df


def predict_one(payload: dict, pipe=None) -> tuple[float, float]:
    """Returns (predicted_rate, latency_ms)."""
    pipe = pipe or load_artifact()
    frame = validate_payload(payload)
    t0 = time.perf_counter()
    log_rpm = pipe.predict(frame)
    rate = float(np.clip(np.exp(log_rpm) * frame["distance"].to_numpy(), 1.0, None)[0])
    ms = (time.perf_counter() - t0) * 1000.0
    return rate, ms


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-row freight-rate inference.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="JSON object payload")
    g.add_argument("--file", help="Path to JSON file with payload object")
    ap.add_argument("--artifact", default=str(PIPE_PATH))
    ap.add_argument("--latency-budget-ms", type=float, default=50.0)
    args = ap.parse_args()
    payload = json.loads(args.input) if args.input else json.loads(Path(args.file).read_text())
    pipe = load_artifact(args.artifact)
    # Warmup (HGB lazy init) then timed run.
    _, _ = predict_one(payload, pipe)
    rate, ms = predict_one(payload, pipe)
    print(json.dumps({"predicted_rate": round(rate, 2), "latency_ms": round(ms, 2)}))
    if ms > args.latency_budget_ms:
        print(f"WARNING: latency {ms:.1f}ms exceeds budget {args.latency_budget_ms:.0f}ms",
              file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
