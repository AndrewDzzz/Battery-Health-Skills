# Field deployment casebook: SOH and security in production

This playbook is written for a real EV/ESS battery field program with fragmented telemetry and multiple operating domains.

## Real-world scenario used in this repo

- Context: 8-cell pilot fleet at an EV depot, mixed ambient temperature.
- Goal: detect SOH degradation early while blocking bad telemetry before model impact.
- Minimum acceptable checks before model release:
  - No high-severity telemetry integrity violations.
  - SOH prediction interval width below rollout threshold.
  - At least one security risk owner per high-priority row.

## Rollout gate checklist

1. **Telemetry integrity**
   - Timestamp monotonicity for each cell.
   - Duplicate packet ratio below drift threshold.
   - Voltage/current/temperature jumps bounded.
2. **SOH modeling**
   - R2 and MAE within expected pilot baseline.
   - Interval coverage stays stable after deployment window.
   - Significant feature drift (`soh_drift_summary.json`) gets reviewed before rollout.
3. **Security baseline**
   - `security/risk_summary.json` records open/high-risk items and owners.
   - Add model and dataset owner for every `Asset:Threat` pair with high score.

## Additional papers to read

- [CyFormer: Accurate State-of-Health Prediction of Lithium-Ion Batteries via Cyclic Attention (2023)](https://arxiv.org/abs/2304.08502)
  - Useful for cyclic-feature extraction and transfer learning in cross-domain settings.
- [Knowledge-Aware Modeling with Frequency Adaptive Learning for Battery Health Prognostics (2025)](https://arxiv.org/abs/2510.02839)
  - Useful for uncertainty-aware degradation decomposition and domain-aware feature decomposition.
- [End-to-End Framework for Predicting Remaining Useful Life of Lithium-Ion Batteries (2025)](https://arxiv.org/abs/2505.16664)
  - Useful for hybrid CNN/attention/ODE pipelines and transfer-learning strategy.
- [Enhanced Gaussian Process Dynamical Models with Knowledge Transfer for Long-term Battery Degradation Forecasting (2022)](https://arxiv.org/abs/2212.01609)
  - Useful for knowledge transfer and early-lifetime projection.
- [A knowledge distillation based cross-modal learning framework for the lithium-ion battery state of health estimation (2024)](https://link.springer.com/article/10.1007/s40747-024-01458-4)
  - Useful for lightweight deployment and physically-consistent outputs.
- [State of Health Prediction for Lithium-Ion Batteries Using Transformer–LSTM Fusion Model (2025)](https://www.mdpi.com/2076-3417/15/7/3747)
  - Useful for multi-domain feature construction (time, frequency, temperature).
