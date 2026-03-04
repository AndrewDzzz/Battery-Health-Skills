#!/usr/bin/env python3
"""Train a GBDT model and compute SHAP attributions for battery fault classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train interpretable battery diagnosis model")
    parser.add_argument("--input", required=True, help="Input feature CSV")
    parser.add_argument("--label-col", required=True, help="Column containing target fault label")
    parser.add_argument("--sample-id-col", default="", help="Optional ID column to track rows")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--top-k", type=int, default=8, help="Top contributors per sample for attribution export")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--hard-boundary-margin", type=float, default=0.12, help="Margin threshold for weak-confidence cases")
    parser.add_argument("--model-kwargs", default="", help="Optional JSON for GradientBoostingClassifier kwargs")
    return parser.parse_args()


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    try:
        if y_proba.ndim == 1:
            metrics["auroc"] = float(roc_auc_score(y_true, y_proba))
        elif y_proba.shape[1] == 2:
            y_true_bin = (y_true == classes[1]).astype(int)
            metrics["auroc"] = float(roc_auc_score(y_true_bin, y_proba[:, 1]))
        else:
            metrics["auroc"] = float(roc_auc_score(pd.get_dummies(y_true), y_proba, average="macro", multi_class="ovr"))
    except Exception:
        metrics["auroc"] = float("nan")
    return metrics


def _select_top_attributions(sv: np.ndarray, feature_names: list[str], sample_ids: list[str | int], pred: np.ndarray, true_y: pd.Series, top_k: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(len(sample_ids)):
        row_values = sv[i]
        top_idx = np.argsort(np.abs(row_values))[::-1][: top_k]
        for rank, fid in enumerate(top_idx, start=1):
            rows.append(
                {
                    "sample_id": sample_ids[i],
                    "predicted_label": str(pred[i]),
                    "true_label": str(true_y.iloc[i]),
                    "rank": int(rank),
                    "feature": feature_names[int(fid)],
                    "shap_value": float(row_values[int(fid)]),
                    "abs_shap": float(abs(row_values[int(fid)])),
                }
            )
    return rows


def _to_sample_shap(pred: np.ndarray, y_proba: np.ndarray, shap_values, feature_count: int) -> np.ndarray:
    n = len(pred)
    if isinstance(shap_values, list):
        arr = np.zeros((n, feature_count), dtype=float)
        if len(shap_values) == 2:
            # Binary case, shap may be [class0, class1]
            class_idx = np.argmax(y_proba, axis=1)
            for i in range(n):
                arr[i] = np.asarray(shap_values[int(class_idx[i])])[i]
            return arr
        for cidx in range(len(shap_values)):
            # Fallback to highest-support class for each sample
            cls_mask = pred == np.unique(pred)[cidx]
            arr[cls_mask] = np.asarray(shap_values[cidx])[cls_mask]
        return arr

    if isinstance(shap_values, np.ndarray):
        arr = np.asarray(shap_values)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            # shape may be (samples, features, classes)
            cls_index = np.argmax(y_proba, axis=1)
            out = np.zeros((n, feature_count), dtype=float)
            for i in range(n):
                out[i] = arr[i, :, int(cls_index[i])]
            return out
    return np.zeros((n, feature_count), dtype=float)


def _hard_boundary_mask(y_proba: np.ndarray, margin: float) -> np.ndarray:
    if y_proba.ndim == 1:
        return np.abs(y_proba - 0.5) < margin
    if y_proba.shape[1] == 2:
        return np.abs(y_proba[:, 1] - 0.5) < margin
    sorted_proba = np.sort(y_proba, axis=1)
    gap = sorted_proba[:, -1] - sorted_proba[:, -2]
    return gap < margin


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    feature_df = df.drop(columns=[args.label_col])
    if args.sample_id_col and args.sample_id_col in feature_df.columns:
        sample_ids = feature_df[args.sample_id_col].tolist()
        feature_df = feature_df.drop(columns=[args.sample_id_col])
    else:
        sample_ids = list(range(len(df)))

    X = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[args.label_col]
    feature_names = X.columns.tolist()

    params = json.loads(args.model_kwargs) if args.model_kwargs else {}
    model = GradientBoostingClassifier(**params)

    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, sample_ids, test_size=args.test_size, random_state=args.random_state, stratify=y if y.nunique() > 1 else None
    )

    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)

    classes = model.classes_
    metrics = _compute_metrics(y_test, pred, proba, classes)
    metrics.update(
        {
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "n_features": int(len(feature_names)),
            "classes": [str(c) for c in classes],
            "test_size": float(args.test_size),
        }
    )

    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model_path = out_dir / "model.joblib"
    dump(model, model_path)

    hard_boundary_mask = _hard_boundary_mask(proba, args.hard_boundary_margin)
    hard_cases = pd.DataFrame(
        {
            "sample_id": np.array(id_test, dtype=object)[hard_boundary_mask],
            "predicted_label": pred[hard_boundary_mask],
            "true_label": y_test.reset_index(drop=True)[hard_boundary_mask],
            "max_prob": np.max(proba[hard_boundary_mask], axis=1),
        }
    )
    hard_cases.to_csv(out_dir / "hard_boundary_cases.csv", index=False)

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
        sample_shap = _to_sample_shap(pred, proba, shap_values, len(feature_names))

        global_abs = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.mean(np.abs(sample_shap), axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        global_abs.to_csv(out_dir / "global_importance.csv", index=False)

        rows = _select_top_attributions(
            sample_shap,
            feature_names,
            list(np.array(id_test, dtype=object)),
            pred,
            y_test.reset_index(drop=True),
            args.top_k,
        )
        pd.DataFrame(rows).to_csv(out_dir / "top_attributions.csv", index=False)

        print(f"Wrote SHAP outputs to {out_dir}")
    except Exception as exc:  # pragma: no cover
        warning = {
            "shap_error": str(exc),
            "note": "Run `pip install shap` to enable SHAP outputs.",
        }
        (out_dir / "eval_metrics.json").write_text(
            json.dumps({**metrics, **warning}, indent=2),
            encoding="utf-8",
        )

    print(f"Wrote model to {model_path}")
    print(f"Wrote metrics to {out_dir / 'eval_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
