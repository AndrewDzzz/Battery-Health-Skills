# Battery-Health-Skills

Operational AI skill pack for battery telemetry, SOH modeling, and deployment-grade security checks.

## Skills in this repository

- `feature-engineering`
  - Compact mechanism-aware features for interpretable fault classification.
- `interpretability-pipeline`
  - Deterministic model training, SHAP attribution, and LLM-ready prompts.
- `battery-telemetry-integrity`
  - Structural validation for timestamp gaps, duplicates, replay patterns, and outlier jumps.
- `battery-security-audit`
  - Lightweight security baseline with threat ranking, owners, and gate planning.
- `soh-modeling-upgrade`
  - Degradation-oriented SOH feature extraction, uncertainty-aware regression, and drift checks.
- `soh-field-demo`
  - End-to-end real-world workflow: simulate or ingest telemetry, validate integrity, estimate SOH, and publish a rollout report.

## Quick start

From repo root:

```bash
python feature-engineering/scripts/extract_features.py --input telemetry.csv --output features.csv --timestamp-col timestamp --cycle-col cycle_id --current-col current --voltage-col pack_voltage --temp-col temp --soc-col soc

python interpretability-pipeline/scripts/train_gbdt_shap.py --input features.csv --label-col fault_label --output-dir runs/interpretability

python battery-telemetry-integrity/scripts/check_telemetry_integrity.py --input telemetry.csv --time-col timestamp --id-col cell_id --voltage-col voltage --current-col current --temp-col temperature --output-dir runs/telemetry

python battery-security-audit/scripts/generate_security_audit.py --assets "CellTelemetry,ChargingService,ModelRegistry,InferenceAPI" --output-dir runs/security
```

## SOH-specific workflow example

```bash
python soh-modeling-upgrade/scripts/extract_soh_features.py \
  --input telemetry.csv \
  --output soh_features.csv \
  --timestamp-col timestamp \
  --cycle-col cycle_id \
  --current-col current \
  --voltage-col voltage \
  --temp-col temperature \
  --soc-col soc \
  --soh-proxy-col capacity

python soh-modeling-upgrade/scripts/train_soh_with_uncertainty.py \
  --input soh_features.csv \
  --soh-label-col soh_proxy \
  --output-dir runs/soh \
  --target-csv soh_features_shifted.csv \
  --target-label-col soh_proxy
```

## Real-world demo (battery fleet readiness check)

```bash
python soh-field-demo/scripts/run_battery_field_demo.py \
  --generate-data \
  --n-cells 6 \
  --cycles-per-cell 14 \
  --points-per-cycle 120 \
  --out-dir ./runs/ev-fleet-demo
```

This creates:

- a synthetic EV-like telemetry dataset,
- integrity diagnostics,
- SOH feature/uncertainty outputs,
- a security register stub,
- and a consolidated `demo_summary.json`.

## Reference material

- SOH modeling papers and notes are collected under:
  - `soh-modeling-upgrade/references/soh-paper-notes.md`
  - `soh-field-demo/references/field-case-notes.md`
- Security and telemetry details are in each skill's `references/` folder.

## Notes

- Scripts are designed to be composable. Use the same telemetry file across multiple skills.
- Use `soh-field-demo` before running to validate a production-like path with clear failure gates.

## Website

- Open the included static project site: [docs/index.html](./docs/index.html)
- To publish on GitHub Pages:
  - enable Pages in repository settings and set source to `main` branch and `/docs` folder.
