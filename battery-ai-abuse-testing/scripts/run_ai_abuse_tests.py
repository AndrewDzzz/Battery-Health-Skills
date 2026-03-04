#!/usr/bin/env python3
"""Run controlled abuse robustness checks for battery ML models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple AI abuse tests.")
    parser.add_argument("--input", required=True, help="Feature CSV with label column")
    parser.add_argument("--label-col", required=True, help="Ground-truth label column")
    parser.add_argument("--sample-id-col", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--output-dir", default="./ai-abuse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--flip-samples", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def _load_model(path: str, n_features: int):
    if path:
        model = joblib.load(path)
        return model
    return GradientBoostingClassifier(random_state=0)


def _build_features(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(df[label_col], errors="coerce")
    if y.isna().all():
        raise ValueError("Label column must be numeric or coercible.")
    X = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, y


def _predict(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(X)
    pred = model.predict(X)
    conf = proba.max(axis=1)
    return pred, conf


def _apply_noise(df: pd.DataFrame, noise_std: float, rng: np.random.Generator) -> pd.DataFrame:
    arr = df.to_numpy(dtype=float)
    scale = np.nanstd(arr, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale), scale, 0.0)
    return pd.DataFrame(arr + rng.normal(0, noise_std * (scale + 1e-9), size=arr.shape), columns=df.columns)


def _inject_sign_flip(df: pd.DataFrame, cols: list[str], rng: np.random.Generator, frac: float) -> pd.DataFrame:
    out = df.copy()
    if not cols:
        return out
    n = max(1, int(len(out) * frac))
    idx = rng.choice(len(out), size=n, replace=False)
    cidx = rng.choice(len(cols), size=n, replace=True)
    for row_i, col_i in zip(idx, cidx):
        col = cols[int(col_i)]
        out.iloc[int(row_i), int(col)] *= -1
    return out


def _permute_rows(df: pd.DataFrame, rng: np.random.Generator, frac: float) -> pd.DataFrame:
    out = df.copy()
    n = int(len(out) * frac)
    if n <= 1:
        return out
    idx = rng.choice(len(out), size=n, replace=False)
    perm = rng.permutation(idx)
    out.iloc[idx] = out.iloc[perm].to_numpy()
    return out


def _run_case(name: str, X_test: pd.DataFrame, y_test: pd.Series, pred_clean, conf_clean, model, X_adv) -> dict:
    pred_adv, conf_adv = _predict(model, X_adv)
    flip = float(np.mean(pred_adv != pred_clean))
    conf_delta = float(np.mean(conf_clean - conf_adv))
    return {
        "case": name,
        "flip_rate": flip,
        "mean_confidence_delta": conf_delta,
        "acc_on_adv": float(accuracy_score(y_test, pred_adv)),
    }


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    X, y = _build_features(df, args.label_col)
    ids = df[args.sample_id_col].tolist() if args.sample_id_col and args.sample_id_col in df.columns else list(range(len(df)))

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=args.test_size, random_state=args.seed, stratify=y if y.nunique() > 1 else None
    )

    model = _load_model(args.model_path, X.shape[1])
    if not args.model_path:
        model.fit(X_train, y_train)
        scaler = None
    else:
        scaler = None

    pred_clean, conf_clean = _predict(model, X_test)
    acc_clean = float(accuracy_score(y_test, pred_clean))

    rng = np.random.default_rng(args.seed)
    base_features = X_test.reset_index(drop=True)
    y_test = pd.Series(y_test.reset_index(drop=True).to_numpy())
    pred_clean_arr = pred_clean
    conf_clean_arr = conf_clean

    cases = [
        ("noise_perturb", _apply_noise(base_features, args.noise_std, rng)),
        ("sign_injection", _inject_sign_flip(base_features, [c for c in ["current", "voltage", "temperature", "cell_voltage"] if c in base_features.columns], rng, args.flip_samples)),
        ("row_replay", _permute_rows(base_features, rng, args.flip_samples)),
    ]

    detail_rows = []
    for name, X_adv in cases:
        row = _run_case(name, base_features, y_test, pred_clean_arr, conf_clean_arr, model, X_adv.reset_index(drop=True))
        row["status"] = "hold" if row["flip_rate"] > 0.25 or row["mean_confidence_delta"] > 0.2 else "pass"
        row["top_k"] = args.top_k
        detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)
    high_risk = detail_df[(detail_df["flip_rate"] > 0.25) | (detail_df["mean_confidence_delta"] > 0.2)]
    unstable_idx = []
    for row in high_risk.itertuples(index=False):
        unstable_idx.append({"case": row.case, "evidence": f"flip_rate={row.flip_rate:.3f}, conf_delta={row.mean_confidence_delta:.3f}"})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(out_dir / "abuse_details.csv", index=False)

    abuse_summary = {
        "baseline_accuracy": acc_clean,
        "samples_test": int(len(y_test)),
        "flip_threshold": 0.25,
        "confidence_delta_threshold": 0.2,
        "cases": detail_rows,
        "high_risk_count": int(len(high_risk)),
    }
    (out_dir / "abuse_summary.json").write_text(json.dumps(abuse_summary, indent=2), encoding="utf-8")

    (out_dir / "high-risk-cases.csv").write_text("case,evidence\n" + "\n".join([f'{x["case"]},{x["evidence"]}' for x in unstable_idx]), encoding="utf-8")

    print(f"Wrote AI abuse outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

