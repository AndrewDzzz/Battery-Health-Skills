---
name: codex-feature-engineering
description: Extract mechanism-aware lithium-ion battery features for interpretable fault diagnosis, following the BatteryAgent physical perception layer design (usage-history, voltage, and thermodynamic features).
---

# BatteryAgent Feature Engineering

Use this skill when you need physically grounded, compact features from raw battery telemetry before interpretable ML diagnosis.

## Scope

This skill implements the physical perception layer ideas from:
- 10 mechanism-based features derived from electrochemical principles
- Three feature groups (usage history, voltage characteristics, thermodynamics)
- A low-dimension representation suitable for interpretable classifiers and attribution workflows

## Quick Start

1. Prepare a per-sample cycle dataset (CSV) with telemetry columns for time and key sensors.
2. Run `scripts/extract_features.py` to generate one row per cycle and a 10-column feature vector.
3. Review basic diagnostics from the log output and feature NaN rates.

```bash
python scripts/extract_features.py \
  --input /path/to/telemetry.csv \
  --output /path/to/features.csv \
  --timestamp-col timestamp \
  --cycle-col cycle \
  --current-col current \
  --voltage-col pack_voltage \
  --cell-voltage-col cell_voltage \
  --temp-col temperature \
  --soc-col soc
```

## Feature Set

- Usage history domain
  - `cycle_number`: cycle identifier captured from the dataset
  - `cc_phase_ratio`: proportion of points where voltage slope is flat enough to represent constant-current-like behavior
  - `soc_max`: maximum SOC in the cycle
- Voltage domain
  - `pack_cell_voltage_ratio`: mean pack-to-cell voltage ratio
  - `voltage_correlation`: Pearson correlation between pack and cell voltage trajectories
  - `initial_min_voltage`: minimum voltage in the first 20% of the cycle window
  - `voltage_gradient`: mean absolute voltage gradient
- Thermodynamics domain
  - `max_temp_diff`: cycle max minus min terminal temperature
  - `max_temp_rate`: maximum absolute temperature change rate
  - `terminal_temp`: final cycle terminal temperature

These features balance model input compactness and physical interpretability.

## Output Contract

Expected outputs:
- One-row-per-cycle CSV with these feature columns and `cycle_id`
- Optional `cycle_feature_stats.csv` (if `--write-stats` is enabled) with summary checks
- A concise processing summary printed to stdout

## Reference Usage

- `references/paper-extraction-notes.md`: notes from the paper on feature intent and module boundaries
- `scripts/extract_features.py`: end-to-end extractor and validation helper
