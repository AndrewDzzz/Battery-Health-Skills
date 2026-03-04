# Battery Telemetry Integrity Checks

## Why these checks

Telemetry used for battery health models is vulnerable to:
- sensor malfunction drift,
- replay or duplicated packet attacks,
- timing manipulation from compromised gateways,
- and injection noise that looks statistically normal unless contextualized.

## Rule defaults

- Timestamp should be non-decreasing per device.
- Voltage jump limit default: `3.0` volts per sample interval.
- Current jump limit default: `25.0` amps per sample interval.
- Temperature jump limit default: `8.0` °C per sample interval.
- Flatline window: 20 consecutive samples with unchanged voltage and current.
- Gap interval: more than 3× median inter-sample spacing is suspicious.

## Flag mapping

- `duplicate_row`: duplicate same `time_col + id_col`
- `timestamp_reorder`: non-monotonic timestamp
- `voltage_jump`: jump magnitude above threshold
- `current_jump`: current magnitude above threshold
- `temp_jump`: temperature magnitude above threshold
- `gap`: large sampling gap
- `replay_pattern`: repeated payload sequence with same values repeated > 2 times

