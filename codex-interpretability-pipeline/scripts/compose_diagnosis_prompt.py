#!/usr/bin/env python3
"""Turn SHAP attribution outputs into compact, model-to-LLM diagnosis prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose concise BatteryAgent-style prompts from attributions.")
    parser.add_argument("--attributions", required=True, help="CSV of top_attributions.csv")
    parser.add_argument("--mapping", default="", help="Optional JSON mapping feature->mechanism")
    parser.add_argument("--max-features", type=int, default=5, help="Top-k contributors included per sample")
    parser.add_argument("--sample-limit", type=int, default=100, help="Maximum number of samples to build prompts for")
    parser.add_argument("--output", required=True, help="Output JSONL file path for sample prompts")
    parser.add_argument("--write-markdown", default="", help="Optional markdown summary path")
    parser.add_argument("--format", default="markdown", choices=["markdown", "json"], help="Prompt body format")
    return parser.parse_args()


def load_mapping(mapping_path: str) -> dict[str, str]:
    if not mapping_path:
        return {}
    raw = Path(mapping_path).read_text(encoding="utf-8")
    return json.loads(raw)


def build_prompt(
    sample_id: str | int,
    pred_label: str,
    true_label: str,
    features: list[tuple[str, float]],
    mapping: dict[str, str],
    format_mode: str,
) -> tuple[str, dict]:
    support_rows = []
    support_features: list[dict] = []
    for rank, (name, val) in enumerate(features, start=1):
        direction = "supports" if val > 0 else "opposes"
        mechanism = mapping.get(name, "unknown mechanism")
        row = {
            "rank": rank,
            "feature": name,
            "shap_value": val,
            "mechanism": mechanism,
            "direction": direction,
        }
        support_rows.append(row)
        support_features.append(row)

    summary = {
        "sample_id": str(sample_id),
        "predicted_fault": pred_label,
        "true_fault": true_label,
        "top_contributors": support_features,
        "requested_task": "Produce diagnosis explanation, likely root-cause path, confidence, and verification checks.",
    }

    if format_mode == "json":
        body = json.dumps(summary, ensure_ascii=False, indent=2)
    else:
        bullet_lines = [
            f"- [{f['rank']}] {f['feature']} ({f['direction']}): {f['mechanism']} (SHAP {f['shap_value']:.3f})"
            for f in support_features
        ]
        body = "\n".join(
            [
                f"# Diagnosis Prompt for sample_id={sample_id}",
                f"- Predicted: {pred_label}",
                f"- True (if known): {true_label}",
                "- Context:",
                "  - Use the top contributing features below to infer the likely fault mechanism path.",
                "- Top feature evidence:",
                *bullet_lines,
                "- Output fields:",
                "  - likely_faults (ranked)",
                "  - confidence (0-1)",
                "  - immediate_actions",
                "  - verification_steps",
            ]
        )
    return body, summary


def main() -> int:
    args = parse_args()
    mapping = load_mapping(args.mapping)

    at_df = pd.read_csv(args.attributions)
    for col in ["sample_id", "predicted_label", "true_label", "rank", "feature", "shap_value"]:
        if col not in at_df.columns:
            raise ValueError(f"Missing required attribution column: {col}")

    grouped = (
        at_df.sort_values(["sample_id", "rank"])
        .groupby("sample_id", sort=True)
        .head(args.max_features)
    )

    output_lines: list[str] = []
    summary_rows: list[dict] = []
    for sample_id, grp in grouped.groupby("sample_id", sort=True):
        pred_label = str(grp["predicted_label"].iloc[0])
        true_label = str(grp["true_label"].iloc[0]) if "true_label" in grp.columns else ""
        feature_pairs = [(str(r["feature"]), float(r["shap_value"])) for _, r in grp.iterrows()]
        prompt, summary = build_prompt(sample_id, pred_label, true_label, feature_pairs, mapping, args.format)
        payload = {
            "sample_id": str(sample_id),
            "format": args.format,
            "prompt": prompt,
            "summary": summary,
        }
        output_lines.append(json.dumps(payload, ensure_ascii=False))
        summary_rows.append(summary)
        if len(summary_rows) >= args.sample_limit:
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(output_lines), encoding="utf-8")

    if args.write_markdown:
        md_path = Path(args.write_markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with md_path.open("w", encoding="utf-8") as f:
            f.write("# LLM Prompt Pack\n\n")
            for payload in output_lines:
                body = json.loads(payload)["prompt"]
                f.write(body + "\n\n---\n\n")

    print(f"Wrote {len(summary_rows)} sample prompts to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

