# SOH papers used for this skill

The references below were selected to strengthen the new SOH workflow:

1. **arXiv:2502.19586** - sequence/transformer style multi-task SOH estimation
   - Why it matters: supports richer temporal context and protocol-flexible feature blocks.

2. **arXiv:2503.23858** - PINN-style SOH models with physical priors
   - Why it matters: motivates adding physically interpretable proxies (voltage/current/temperature dynamics) before black-box models.

3. **arXiv:2503.04664** - continuous-time / Neural-ODE style battery state modeling
   - Why it matters: supports derivative-based degradation features and interpolation between sparse cycles.

4. **arXiv:2503.06353** - hybrid physics-ML battery prognostics
   - Why it matters: reinforces combining proxy physics features with data-driven regressors.

5. **arXiv:2402.00068** - test-time/domain-adaptive battery prognostics
   - Why it matters: motivates explicit target-domain drift checks before deployment.

6. **2512.24686** (initial paper you referenced)
   - Why it matters: aligns with a mechanism-aware decomposition approach; this new skill builds on that direction with uncertainty and transfer checks.

## Further reading (2024-2025)

- [A novel Neural-ODE model for the state of health estimation of lithium-ion battery using charging curve (arXiv:2505.05803)](https://arxiv.org/abs/2505.05803)
  - Useful for charging-curve representation and generalization on sparse/shifted domain datasets.
- [AI-Driven Prognostics for State of Health Prediction in Li-ion Batteries: A Comprehensive Analysis (arXiv:2504.05728)](https://arxiv.org/abs/2504.05728)
  - Useful benchmark reference for model-family selection and evaluation patterns.
- [End-to-End Framework for Predicting the Remaining Useful Life of Lithium-Ion Batteries (arXiv:2505.16664)](https://arxiv.org/abs/2505.16664)
  - Useful for signal preprocessing + deep stack architecture comparisons in deployment pipelines.
- [MambaLithium: Selective state space model for RUL, SOH, and SOC estimation (arXiv:2403.05430)](https://arxiv.org/abs/2403.05430)
  - Useful for sequence-model alternatives and compute-efficient deployment considerations.
