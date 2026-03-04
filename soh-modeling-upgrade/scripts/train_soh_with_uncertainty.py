#!/usr/bin/env python3
"""Train SOH regression models with uncertainty proxies and target-domain drift checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SOH model with uncertainty and drift checks")
    parser.add_argument("--input", required=True, help="SOH feature CSV")
    parser.add_argument("--soh-label-col", required=True, help="SOH target column")
    parser.add_argument("--group-col", default="", help="Optional group column for strict train/test split")
    parser.add_argument("--id-col", default="", help="Optional sample id column")
    parser.add_argument("--output-dir", default="./soh-output", help="Output directory")
    parser.add_argument("--model", choices=["rf", "gb"], default="rf", help="Model type")
    parser.add_argument("--n-estimators", type=int, default=300, help="RF trees")
    parser.add_argument("--max-depth", type=int, default=10, help="RF max_depth")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--n-bootstrap", type=int, default=20, help="Bootstrap reps for uncertainty")
    parser.add_argument("--target-csv", default="", help="Optional target-domain feature CSV")
    parser.add_argument("--target-label-col", default="", help="SOH label in target domain (if available)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="RandomForest n_jobs")
    return parser.parse_args()


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def _safe_split(df: pd.DataFrame, y: pd.Series, group_col: str, test_size: float, random_state: int):
    if group_col and group_col in df.columns and df[group_col].nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        groups = df[group_col]
        trn_idx, tst_idx = next(splitter.split(df, y, groups=groups))
        return trn_idx, tst_idx
    return train_test_split(
        df.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )


def _build_model(model_name: str, n_estimators: int, max_depth: int, random_state: int, n_jobs: int):
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            n_jobs=n_jobs,
        )
    # Fallback: still return RandomForest (more stable for bootstrap uncertainty output)
    return RandomForestRegressor(
        n_estimators=max(50, n_estimators // 2),
        random_state=random_state,
        max_depth=max_depth,
        n_jobs=n_jobs,
    )


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mpe = float(np.mean((y_pred - y_true) / np.where(y_true == 0, np.nan, y_true))) * 100 if np.any(y_true != 0) else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mpe_percent": mpe,
    }


def _bootstrap_mean_std(base_model, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    preds = []
    for k in range(n_boot):
        idx = rng.integers(0, len(X_train), size=len(X_train))
        m = clone(base_model)
        m.set_params(random_state=seed + k)
        m.fit(X_train.iloc[idx], y_train.iloc[idx])
        preds.append(m.predict(X_eval))
    pred_stack = np.column_stack(preds)
    return pred_stack.mean(axis=1), pred_stack.std(axis=1)


def _drift_metrics(train: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    common_cols = [c for c in train.columns if c in target.columns]
    rows: list[dict[str, float | str]] = []
    max_shift = 0.0
    for col in common_cols:
        s = train[col].to_numpy(dtype=float)
        t = target[col].to_numpy(dtype=float)
        finite_s = s[np.isfinite(s)]
        finite_t = t[np.isfinite(t)]
        if len(finite_s) < 5 or len(finite_t) < 5:
            mean_shift = float("nan")
            std_shift = float("nan")
            quant_shift = float("nan")
        else:
            mean_shift = float(np.abs(finite_t.mean() - finite_s.mean()))
            std_shift = float(finite_t.std(ddof=1) / (finite_s.std(ddof=1) + 1e-9))
            quantiles = np.arange(0.1, 1.0, 0.1)
            shift = np.abs(np.quantile(finite_t, quantiles) - np.quantile(finite_s, quantiles))
            quant_shift = float(np.max(shift))
            max_shift = max(max_shift, float(quant_shift))
        rows.append(
            {
                "feature": col,
                "mean_shift": mean_shift,
                "std_shift_ratio": std_shift,
                "quantile_shift": quant_shift,
            }
        )
    drift = pd.DataFrame(rows)
    summary = {
        "num_features": int(len(drift)),
        "max_quantile_shift": max_shift,
        "high_quantile_features": int((drift["quantile_shift"] > 0.05).sum()) if "quantile_shift" in drift else 0,
    }
    return drift, summary


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    if args.soh_label_col not in df.columns:
        raise ValueError(f"Missing SOH label column: {args.soh_label_col}")

    feature_df = df.drop(columns=[c for c in [args.soh_label_col] if c == args.soh_label_col])
    sample_id = df[args.id_col] if args.id_col and args.id_col in df.columns else pd.Series(np.arange(len(df)))
    y = pd.to_numeric(df[args.soh_label_col], errors="coerce").astype(float)

    if args.id_col and args.id_col in feature_df.columns:
        feature_df = feature_df.drop(columns=[args.id_col])
    if args.group_col and args.group_col in feature_df.columns:
        feature_df = feature_df.drop(columns=[args.group_col], errors="ignore")

    feature_df = _ensure_numeric(feature_df).fillna(0.0)
    y = y.fillna(y.median())
    valid = np.isfinite(y.to_numpy())
    feature_df = feature_df.loc[valid]
    y = y.loc[valid]
    sample_id = sample_id.loc[valid]

    train_idx, test_idx = _safe_split(feature_df, y, args.group_col, args.test_size, args.random_state)
    X_train = feature_df.loc[train_idx]
    X_test = feature_df.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    id_test = sample_id.loc[test_idx].tolist()

    base_model = _build_model(args.model, args.n_estimators, args.max_depth, args.random_state, args.n_jobs)
    base_model.fit(X_train, y_train)

    pred_test = base_model.predict(X_test)
    test_metrics = _regression_metrics(y_test, pred_test)
    tree_std = np.zeros_like(pred_test, dtype=float)
    if hasattr(base_model, "estimators_"):
        per_tree = np.column_stack([t.predict(X_test) for t in base_model.estimators_])
        tree_std = per_tree.std(axis=1)

    bootstrap_mean, bootstrap_std = _bootstrap_mean_std(
        base_model, X_train, y_train, X_test, max(1, args.n_bootstrap), args.random_state + 100
    )

    z = 1.96
    uncertainty = np.maximum(tree_std, bootstrap_std)
    lower = bootstrap_mean - z * uncertainty
    upper = bootstrap_mean + z * uncertainty
    width = upper - lower
    coverage = float(np.mean((y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper)))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame(
        {
            "sample_id": id_test,
            "y_true": y_test.to_numpy(),
            "y_pred": pred_test,
            "bootstrap_mean": bootstrap_mean,
            "bootstrap_std": bootstrap_std,
            "tree_std": tree_std,
            "uncertainty": uncertainty,
            "y_pi_lower_95": lower,
            "y_pi_upper_95": upper,
            "pi_width": width,
            "residual": y_test.to_numpy() - pred_test,
        }
    )
    pred_df.to_csv(out / "soh_predictions.csv", index=False)

    pred_unc = pd.DataFrame(
        {
            "sample_id": id_test,
            "uncertainty_95": uncertainty,
            "interval_width_95": width,
            "coverage_95": coverage,
        }
    )
    pred_unc.to_csv(out / "soh_uncertainty.csv", index=False)

    feature_names = X_train.columns.tolist()
    if hasattr(base_model, "feature_importances_"):
        imp = pd.DataFrame({"feature": feature_names, "importance": base_model.feature_importances_}).sort_values(
            "importance", ascending=False
        )
        imp.to_csv(out / "soh_feature_importance.csv", index=False)

    metrics: dict[str, Any] = {
        "n_rows": int(len(feature_df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(feature_names)),
        "test": test_metrics,
        "test_pi_coverage_95": coverage,
        "test_mean_interval_width_95": float(np.mean(width)) if len(width) else float("nan"),
        "bootstrap_reps": int(args.n_bootstrap),
    }

    if args.target_csv:
        target_df = pd.read_csv(args.target_csv)
        target_features = target_df.drop(columns=[c for c in [args.target_label_col] if c == args.target_label_col], errors="ignore")
        target_features = target_features.reindex(columns=feature_df.columns, fill_value=np.nan)
        target_features = _ensure_numeric(target_features).fillna(0.0)
        target_pred = base_model.predict(target_features)
        drift_df, drift_summary = _drift_metrics(X_train, target_features)
        drift_df.to_csv(out / "soh_drift_features.csv", index=False)

        drift_metrics = {
            **metrics,
            **drift_summary,
            "target_rows": int(len(target_df)),
        }
        if args.target_label_col and args.target_label_col in target_df.columns:
            y_t = pd.to_numeric(target_df[args.target_label_col], errors="coerce").astype(float)
            valid_t = np.isfinite(y_t.to_numpy())
            if np.any(valid_t):
                tgt_pred = target_pred[valid_t]
                tgt_true = y_t.loc[valid_t]
                drift_metrics["target"] = _regression_metrics(tgt_true, tgt_pred)

        (out / "soh_drift_summary.json").write_text(
            json.dumps(drift_metrics, indent=2),
            encoding="utf-8",
        )

        target_pred_df = pd.DataFrame(
            {
                "sample_id": target_df.index.to_numpy(),
                "y_pred": target_pred,
                "y_true": y_t.to_numpy() if "y_t" in locals() else [np.nan] * len(target_pred),
            }
        )
        target_pred_df.to_csv(out / "soh_target_predictions.csv", index=False)
    else:
        (out / "soh_drift_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    joblib.dump(base_model, out / "soh_model.joblib")
    (out / "soh_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote artifacts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
