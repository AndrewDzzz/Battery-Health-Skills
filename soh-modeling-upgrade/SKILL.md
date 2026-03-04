---
name: soh-modeling-upgrade
description: Improve lithium-ion State-of-Health workflows with mechanism-aware feature extraction, uncertainty-aware SOH regression, and domain-shift checks for transfer between temperatures, SOC ranges, and cycling protocols.
---

# SOH Modeling Upgrade

Use this skill when you need a more realistic SOH modeling layer than fixed fault-style features:
- degradation-oriented feature extraction from raw telemetry cycles,
- uncertainty-aware SOH prediction with regression diagnostics,
- and shift checks before deploying a model to new cells or operating domains.

## Recommended workflow

1. Convert raw telemetry into cycle-level SOH features.
2. Train a regression model and request uncertainty diagnostics.
3. Run target-domain drift checks before rollout.

### 1) Build SOH features

```bash
python scripts/extract_soh_features.py \
  --input /path/to/telemetry.csv \
  --output /path/to/soh_features.csv \
  --timestamp-col timestamp \
  --cycle-col cycle_id \
  --current-col current \
  --voltage-col voltage \
  --temp-col temperature \
  --soc-col soc \
  --soh-proxy-col capacity \
  --stats-output /path/to/soh_feature_stats.json
```

If no explicit cycle column exists, pass `--samples-per-cycle`.

### 2) Train SOH model with uncertainty and drift diagnostics

```bash
python scripts/train_soh_with_uncertainty.py \
  --input /path/to/soh_features.csv \
  --soh-label-col soh_proxy \
  --group-col cycle_id \
  --output-dir /path/to/soh-model \
  --n-estimators 300 \
  --n-bootstrap 20
```

Optional target-domain audit:

```bash
python scripts/train_soh_with_uncertainty.py \
  --input /path/to/source_features.csv \
  --soh-label-col soh_proxy \
  --target-csv /path/to/target_features.csv \
  --target-label-col soh_proxy \
  --output-dir /path/to/domain-audit
```

## Outputs to expect

- `soh_features.csv`: per-cycle degradation features.
- `soh_feature_stats.json`: NaN rates and timing checks.
- `soh_model.joblib`: trained regressor.
- `soh_metrics.json`: MAE/RMSE/R2 for test (and target if provided).
- `soh_predictions.csv`: test predictions and residuals.
- `soh_uncertainty.csv`: per-test prediction mean/std and interval width.
- `soh_drift_features.csv` and `soh_drift_summary.json` when target data is supplied.
- `soh_feature_importance.csv`: feature ranking from tree-based model.

## Reference material

- `references/soh-paper-notes.md`: practical mapping from recent SOH papers to this skill.
- `scripts/extract_soh_features.py`: physics-informed feature extractor.
- `scripts/train_soh_with_uncertainty.py`: uncertainty + drift model trainer.

## Real-world example

Run a practical end-to-end field drill using the new integration skill:

```bash
python ../soh-field-demo/scripts/run_battery_field_demo.py \
  --generate-data \
  --n-cells 8 \
  --cycles-per-cell 18 \
  --points-per-cycle 140 \
  --out-dir ./field-demo
```
