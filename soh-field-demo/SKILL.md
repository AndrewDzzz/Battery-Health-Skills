---
name: soh-field-demo
description: Run a real-world battery-field workflow that combines telemetry integrity checks, SOH feature extraction, uncertainty-aware SOH modeling, and a security preflight for deployment readiness.
---

# SOH Field Deployment Demo

Use this skill when you need an operator-style end-to-end workflow for battery fleets (EV depots, ESS cabinets, robotics battery banks):
- generate or ingest telemetry,
- validate stream integrity,
- estimate SOH with uncertainty,
- and create a combined audit+readiness summary.

## Inputs

- Raw telemetry CSV with at least:
  - `timestamp`
  - `cell_id`
  - `cycle_id`
  - `current`
  - `voltage`
  - `temperature`
  - `soc`
  - optional `capacity` (SOH proxy)

## Real-world flow

```bash
python soh-field-demo/scripts/run_battery_field_demo.py \
  --generate-data \
  --n-cells 8 \
  --cycles-per-cell 18 \
  --points-per-cycle 140 \
  --out-dir ./runs/field-demo
```

## Using your own telemetry

```bash
python soh-field-demo/scripts/run_battery_field_demo.py \
  --telemetry-csv /path/to/telemetry.csv \
  --out-dir ./runs/field-demo \
  --security-assets "CellTelemetry,ChargingService,ModelRegistry,InferenceAPI"
```

## Pipeline outputs

- `telemetry/demo_telemetry.csv` (ingested or generated data)
- `telemetry/telemetry_integrity_report.json` (integrity status)
- `soh/soh_features.csv` (degradation features)
- `soh/soh_predictions.csv` (predictions + uncertainty interval)
- `soh/soh_metrics.json` (MAE/RMSE/R2 and interval statistics)
- `soh/soh_drift_summary.json` (target-domain checks when provided)
- `security/risk_summary.json` (baseline security scorecard)
- `demo_summary.json` (single merged artifact with all status flags)

## References

- `references/field-case-notes.md`: deployment playbook, gates, and example interpretation.
- `scripts/generate_ev_telemetry_demo.py`: synthetic EV-like telemetry generator.
- `scripts/run_battery_field_demo.py`: orchestrator that executes the full pipeline.
