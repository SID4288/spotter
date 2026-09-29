"""Official format validator + December chart + slice/drift guards.

Base contract (unchanged): validates 12k predictions + 31 December rows,
writes scorer_results/candidate_december.png. Extra guards are opt-in via
--validation-inputs / --train-data / --labels / --metrics-json so existing
CI (`score.py --predictions ... --december-predictions ...`) keeps passing.

Guards:
  1. Slice report: equipment, distance deciles, seen vs unseen cities/lanes,
     holiday windows (Dec 24-26, 31). With --labels prints MAE per slice,
     else prediction means/counts (distribution check).
  2. Drift: missing-value fallback proxy > 2% alert; unseen-lane rate
     (lane_freq==0 proxy) > 15% alert; output bounds; metrics.json CI.
  Use --fail-on-drift to exit(2) on alerts for CI gating.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_ROWS = 12_000
EXPECTED_IDS = {f"TE-{index:06d}" for index in range(1, EXPECTED_ROWS + 1)}
DECEMBER_DATES = pd.date_range("2025-12-01", "2025-12-31", freq="D")
FIXED_PICKUP = "Lexington"
FIXED_DELIVERY = "Fort Wayne"
FIXED_DISTANCE = 360.0
FIXED_EQUIPMENT = "Dry Van"
FIXED_WEIGHT = 32_000.0

FALLBACK_ALERT = 0.02
UNSEEN_LANE_ALERT = 0.15
VAL_MEAN_BOUNDS = (1500.0, 3000.0)
DEC_MEAN_BOUNDS = (600.0, 1000.0)
HOLIDAYS = ("2025-12-24", "2025-12-25", "2025-12-26", "2025-12-31")


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        fail(f"{label} file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        fail(f"could not read {label}: {exc}")


def numeric_series(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        fail(f"{label} contains invalid {column} values")
    return values.astype(float)


def validate_predictions(predictions: pd.DataFrame) -> None:
    if list(predictions.columns) != ["load_id", "predicted_rate"]:
        fail("predictions must contain exactly two columns in this order: load_id,predicted_rate")
    if len(predictions) != EXPECTED_ROWS:
        fail(f"predictions must contain exactly {EXPECTED_ROWS:,} rows")
    if predictions["load_id"].isna().any() or predictions["load_id"].duplicated().any():
        fail("predictions contains missing or duplicate load_id values")

    submitted_ids = set(predictions["load_id"].astype(str))
    missing = EXPECTED_IDS - submitted_ids
    extra = submitted_ids - EXPECTED_IDS
    if missing or extra:
        fail(
            "prediction IDs do not match the validation set "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

    predicted_rate = numeric_series(predictions, "predicted_rate", "predictions")
    if (predicted_rate <= 0).any():
        fail("predictions contains non-positive predicted_rate values")


def validate_december(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    if list(frame.columns) != columns:
        fail("December predictions must keep the original seven columns and column order")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        fail("December predictions contains invalid dates")
    result["distance"] = numeric_series(result, "distance", "December predictions")
    result["weight"] = numeric_series(result, "weight", "December predictions")
    result["predicted_rate"] = numeric_series(result, "predicted_rate", "December predictions")

    if result["date"].duplicated().any():
        fail("December predictions contains duplicate dates")
    if len(result) != 31 or set(result["date"]) != set(DECEMBER_DATES):
        fail("December predictions must contain one row for every day from 2025-12-01 to 2025-12-31")
    if not result["pickup"].eq(FIXED_PICKUP).all():
        fail(f"December pickup must be {FIXED_PICKUP} for all rows")
    if not result["delivery"].eq(FIXED_DELIVERY).all():
        fail(f"December delivery must be {FIXED_DELIVERY} for all rows")
    if not np.isclose(result["distance"], FIXED_DISTANCE).all():
        fail(f"December distance must be {FIXED_DISTANCE:g} for all rows")
    if not result["equipment"].eq(FIXED_EQUIPMENT).all():
        fail(f"December equipment must be {FIXED_EQUIPMENT} for all rows")
    if not np.isclose(result["weight"], FIXED_WEIGHT).all():
        fail(f"December weight must be {FIXED_WEIGHT:g} for all rows")
    if (result["predicted_rate"] <= 0).any():
        fail("December predicted_rate values must be positive")
    return result.sort_values("date")


def save_december_chart(december: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    color = "#064A56"
    axis.plot(
        december["date"],
        december["predicted_rate"],
        color=color,
        linewidth=2.6,
        marker="o",
        markersize=3.2,
    )
    floor = float(december["predicted_rate"].min())
    axis.fill_between(
        december["date"],
        december["predicted_rate"],
        floor - max(10.0, floor * 0.02),
        color=color,
        alpha=0.08,
    )
    axis.set_title("Candidate: December 2025 Predicted Load Rate", loc="left", fontsize=15, fontweight="bold", pad=12)
    axis.set_ylabel("Predicted rate ($)")
    axis.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9DAFB3")
    axis.tick_params(axis="x", rotation=35)
    axis.text(
        0,
        -0.40,
        "Fixed inputs: Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb | only date changes",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#455A60",
    )
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _mae(a: pd.Series, p: pd.Series) -> float:
    return float(np.mean(np.abs(a.to_numpy() - p.to_numpy())))


def slice_report(
    inputs: pd.DataFrame,
    predictions: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    train_inputs: pd.DataFrame | None = None,
) -> dict:
    """Print + return slice diagnostics. Labels optional (MAE vs means)."""
    merged = inputs.merge(predictions, on="load_id", how="inner")
    has_labels = labels is not None and "posted_rate" in (labels.columns if labels is not None else [])
    if has_labels:
        merged = merged.merge(labels[["load_id", "posted_rate"]], on="load_id", how="left")
    out: dict = {}

    print("\n== Slice: equipment ==")
    for eq, g in merged.groupby("equipment"):
        line = f"  {eq:10s} n={len(g):5d} mean_pred=${g['predicted_rate'].mean():.2f}"
        if has_labels:
            line += f" MAE=${_mae(g['posted_rate'], g['predicted_rate']):.2f}"
        print(line)
        out[f"equipment:{eq}"] = {"n": int(len(g)), "mean_pred": float(g["predicted_rate"].mean())}

    print("== Slice: distance deciles ==")
    try:
        merged["dist_bin"] = pd.qcut(merged["distance"], 10, duplicates="drop")
        for b, g in merged.groupby("dist_bin", observed=True):
            line = f"  {b} n={len(g):5d} mean_pred=${g['predicted_rate'].mean():.2f}"
            if has_labels:
                line += f" MAE=${_mae(g['posted_rate'], g['predicted_rate']):.2f}"
            print(line)
    except Exception as exc:
        warn(f"distance deciles skipped: {exc}")

    if train_inputs is not None:
        train_cities = set(train_inputs["pickup"]) | set(train_inputs["delivery"])
        touches = merged["pickup"].isin(train_cities) & merged["delivery"].isin(train_cities)
        print("== Slice: seen vs unseen cities ==")
        for name, mask in (("seen", touches), ("unseen_city", ~touches)):
            g = merged[mask]
            line = f"  {name:12s} n={len(g):5d} ({len(g)/max(len(merged),1):.1%}) mean_pred=${g['predicted_rate'].mean() if len(g) else float('nan'):.2f}"
            if has_labels and len(g):
                line += f" MAE=${_mae(g['posted_rate'], g['predicted_rate']):.2f}"
            print(line)
            out[name] = {"n": int(len(g)), "rate": float(len(g) / max(len(merged), 1))}
        train_lanes = set(train_inputs["pickup"].astype(str) + "__" + train_inputs["delivery"].astype(str))
        va_lanes = merged["pickup"].astype(str) + "__" + merged["delivery"].astype(str)
        unseen_lane_rate = float((~va_lanes.isin(train_lanes)).mean())
        print(f"  unseen_lane_rate={unseen_lane_rate:.3f} (alert if > {UNSEEN_LANE_ALERT})")
        out["unseen_lane_rate"] = unseen_lane_rate

    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    hol = merged[merged["date"].dt.strftime("%Y-%m-%d").isin(HOLIDAYS)]
    if len(hol):
        print(f"== Slice: holiday peak (Dec 24-26, 31) n={len(hol)} mean_pred=${hol['predicted_rate'].mean():.2f} ==")
        if has_labels:
            print(f"   holiday MAE=${_mae(hol['posted_rate'], hol['predicted_rate']):.2f}")
        out["holiday"] = {"n": int(len(hol)), "mean_pred": float(hol["predicted_rate"].mean())}
    return out


def drift_checks(
    predictions: pd.DataFrame,
    december: pd.DataFrame,
    inputs: pd.DataFrame | None,
    train_inputs: pd.DataFrame | None,
    metrics_path: Path | None,
) -> list[str]:
    """Return alert strings (empty = clean). Pure checks, no exceptions."""
    alerts: list[str] = []
    vmean = float(predictions["predicted_rate"].mean())
    if not (VAL_MEAN_BOUNDS[0] <= vmean <= VAL_MEAN_BOUNDS[1]):
        alerts.append(f"validation mean ${vmean:.2f} outside bounds {VAL_MEAN_BOUNDS}")
    dmean = float(december["predicted_rate"].mean())
    if not (DEC_MEAN_BOUNDS[0] <= dmean <= DEC_MEAN_BOUNDS[1]):
        alerts.append(f"december mean ${dmean:.2f} outside bounds {DEC_MEAN_BOUNDS}")
    if inputs is not None:
        # True fallback rate via frozen pipeline when available (preferred):
        # global fallback = rows falling through date->month->dow to global.
        true_global = None
        try:
            art = Path(__file__).resolve().parent / "output" / "pipeline.joblib"
            if art.is_file():
                import joblib as _jl
                _pipe = _jl.load(str(art))
                _tmp = inputs.copy()
                _tmp["date"] = pd.to_datetime(_tmp["date"], errors="coerce")
                _tmp = _pipe.named_steps["weight_distance"].transform(_tmp)
                _tmp = _pipe.named_steps["market"].transform(_tmp)
                true_global = float(
                    _pipe.named_steps["market"].last_stats_.get("global_fallback_rate", 0.0)
                )
                print(f"  market global_fallback_rate={true_global:.4f} (alert if > {FALLBACK_ALERT})")
        except Exception as exc:
            warn(f"pipeline fallback probe skipped: {exc}")
        if true_global is not None:
            if true_global > FALLBACK_ALERT:
                alerts.append(f"imputation global fallback {true_global:.2%} > {FALLBACK_ALERT:.0%}")
        else:
            # Proxy when artifact absent: raw missing rates.
            for col in ("market_index", "weight"):
                if col in inputs.columns:
                    miss = float(inputs[col].isna().mean())
                    if miss > FALLBACK_ALERT:
                        alerts.append(f"imputation fallback proxy: {col} missing {miss:.2%} > {FALLBACK_ALERT:.0%}")
        if train_inputs is not None and "pickup" in inputs.columns:
            train_lanes = set(train_inputs["pickup"].astype(str) + "__" + train_inputs["delivery"].astype(str))
            va_lanes = inputs["pickup"].astype(str) + "__" + inputs["delivery"].astype(str)
            rate = float((~va_lanes.isin(train_lanes)).mean())
            if rate > UNSEEN_LANE_ALERT:
                alerts.append(f"unmapped lanes (lane_freq==0) {rate:.1%} > {UNSEEN_LANE_ALERT:.0%}")
    if metrics_path is not None and metrics_path.is_file():
        try:
            m = json.loads(metrics_path.read_text())
            if "december_mean_rate" in m:
                ref = float(m["december_mean_rate"])
                if abs(dmean - ref) > 25.0:
                    alerts.append(
                        f"december mean ${dmean:.2f} drifted from metrics.json ${ref:.2f} (> $25). "
                        "Regenerate metrics.json via train.py or fix regression."
                    )
        except Exception as exc:
            alerts.append(f"metrics.json check failed: {exc}")
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate candidate output files and generate the fixed December chart."
    )
    parser.add_argument("--predictions", required=True, help="CSV with load_id,predicted_rate")
    parser.add_argument(
        "--december-predictions",
        required=True,
        help="Completed data/december_chart_inputs.csv",
    )
    parser.add_argument("--output-dir", default="scorer_results")
    parser.add_argument("--validation-inputs", default=None, help="data/validation.csv for slices/drift")
    parser.add_argument("--train-data", default=None, help="data/train_test.csv for unseen rates")
    parser.add_argument("--labels", default=None, help="optional CSV with load_id,posted_rate for MAE slices")
    parser.add_argument("--metrics-json", default=None, help="notebooks/results/modelling/metrics.json for CI")
    parser.add_argument("--fail-on-drift", action="store_true", help="exit 2 on drift alerts")
    # Tolerate PowerShell users typing `\` instead of backtick for line continuation,
    # e.g. `...\csv \--december-predictions ...` arrives as `\--december-predictions`.
    argv = []
    for token in sys.argv[1:]:
        if token in {"\\", "`"}:
            continue
        while len(token) > 2 and token.startswith("\\") and token[1] == "-":
            token = token[1:]
        argv.append(token)
    args = parser.parse_args(argv)

    preds = read_csv(Path(args.predictions), "predictions")
    validate_predictions(preds)
    december = validate_december(read_csv(Path(args.december_predictions), "December predictions"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    chart = output / "candidate_december.png"
    save_december_chart(december, chart)

    print(f"Validated {EXPECTED_ROWS:,} final predictions.")
    print("Validated 31 fixed December predictions.")
    print(f"Created chart: {chart}")
    print("Final validation metrics are calculated by Spotter after submission.")

    # Opt-in guards (never break base contract unless --fail-on-drift).
    inputs = read_csv(Path(args.validation_inputs), "validation inputs") if args.validation_inputs else None
    train_inputs = read_csv(Path(args.train_data), "train data") if args.train_data else None
    labels = read_csv(Path(args.labels), "labels") if args.labels else None
    if inputs is not None:
        slice_report(inputs, preds, labels, train_inputs)
    alerts = drift_checks(
        preds, december, inputs, train_inputs,
        Path(args.metrics_json) if args.metrics_json else None,
    )
    for a in alerts:
        warn(a)
    if alerts and args.fail_on_drift:
        raise SystemExit(f"ERROR: drift guards failed ({len(alerts)} alerts)")


if __name__ == "__main__":
    main()
