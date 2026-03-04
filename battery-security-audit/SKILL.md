---
name: battery-security-audit
description: Build a structured battery security baseline for data ingestion, BMS operations, cloud transport, and model-serving surfaces with explicit owners and risk priorities.
---

# Battery Security Baseline Audit

Use this skill when you need a quick but repeatable security baseline for battery systems.

## Scope

- Onboarding and threat scoping of:
  - Sensor telemetry ingestion
  - BMS/edge firmware update paths
  - Transport/storage for battery health data
  - Model training and inference endpoints

## Quick Start

```bash
python scripts/generate_security_audit.py \
  --assets "CellTelemetry,InverterGateway,ChargingService,ModelRegistry,InferenceAPI" \
  --output-dir /path/to/audit-out \
  --threat-profile /path/to/threats.yaml
```

## What it produces

- `risk_register.md`: ranked asset-to-threat register
- `risk_register.csv`: machine-readable risk table
- `controls_plan.md`: owner/task/gate view for follow-up
- `risk_summary.json`: severity counts and top recommendations

## Inputs

- `--assets`: comma-separated assets and optional owners (for quick mode)
- `--assets-csv`: optional CSV with columns `asset,owner,criticality`
- `--threat-profile`: optional YAML/JSON override to match your environment

## Reference

- `references/security-baseline-notes.md`
- `scripts/generate_security_audit.py`

