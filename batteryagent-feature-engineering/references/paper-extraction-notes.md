# BatteryAgent Feature Extraction Notes

These notes support the extractor and map each feature to a mechanism channel.

## Paper-grounded assumptions

- The framework in [BatteryAgent (arXiv)](https://arxiv.org/abs/2512.24686) emphasizes interpretable fault detection with:
  - a knowledge-driven module,
  - a dynamic feature generator,
  - and an LLM-based fault reasoning layer.
- The feature generator described there is represented here as 10 compact, mechanism-aware telemetry features.

## Mapping to failure mechanism intent

1. Usage history mechanisms
   - `cycle_number`, `cc_phase_ratio`, `soc_max`
2. Voltage instability mechanisms
   - `pack_cell_voltage_ratio`, `voltage_correlation`, `initial_min_voltage`, `voltage_gradient`
3. Thermal imbalance mechanisms
   - `max_temp_diff`, `max_temp_rate`, `terminal_temp`

## Data quality checks to run before modeling

- Confirm cycle grouping is consistent (no partial cycles accidentally merged).
- Check for sensor dropouts in voltage and temperature columns before correlation/gradient features.
- Keep sensor units aligned (voltage in volts, current in amperes, temperature in °C, SOC in [0,1] or [0,100]).
- Require minimum points per cycle for derivative calculations (`>= 5` recommended).

