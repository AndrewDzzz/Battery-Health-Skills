---
name: battery-telemetry-integrity
description: Detect integrity and plausibility anomalies in battery telemetry streams (timestamp gaps, duplicates, replay-like patterns, and impossible signal jumps).
---

# Battery Telemetry Integrity

Use this skill when you need to gate battery telemetry before feature extraction or model serving.

## Quick Start

```bash
python scripts/check_telemetry_integrity.py \
  --input /path/to/telemetry.csv \
  --time-col timestamp \
  --id-col cell_id \
  --voltage-col cell_voltage \
  --current-col current \
  --temp-col temperature \
  --output-dir /path/to/telemetry-check
```

## Checks included

- Required-column validation
- Timestamp monotonicity per asset
- Duplicate packet detection
- Outlier jump flags for voltage/current/temperature
- Flatline and prolonged gaps
- Exact-duplicate replay pattern detection

## Outputs

- `telemetry_integrity_report.json`: summary + severity counts
- `integrity_flags.csv`: row-level flags
- `summary.md`: concise audit-ready narrative

## Reference

- `references/integrity-check-notes.md`
- `scripts/check_telemetry_integrity.py`

