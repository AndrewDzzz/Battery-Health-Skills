# SOH papers to anchor the skill

These papers informed the script choices and checks below.

## Core SOH modeling direction

### 1. arXiv:2502.19586 (Transformer-based multi-task SOH framework)
- Key idea: combine cycle-level context and cross-cell correlation with sequence models to estimate SOH from non-ideal, variable-rate profiles.
- Practical impact: supports flexible feature windows and helps with cross-cycle representation quality checks.

### 2. arXiv:2503.23858 (Adaptive PINN + degradation priors)
- Key idea: integrate physics constraints with data-driven layers to reduce unphysical SOH trajectories.
- Practical impact: motivates explicit checks for monotonicity-like behavior and resistance/voltage trend consistency.

### 3. arXiv:2503.04664 (Neural-ODE for electrochemical states)
- Key idea: continuous-time latent dynamics improve interpolation between sparse cycles.
- Practical impact: supports derivative-aware summaries (temperature and voltage rates) rather than relying only on absolute values.

### 4. arXiv:2502.12954 (Efficient SOH feature compression + uncertainty)
- Key idea: small feature sets + calibrated uncertainty are often better for deployment than large opaque vectors.
- Practical impact: justifies tree-model uncertainty outputs and uncertainty-aware holdout gates in this package.

### 5. arXiv:2503.06353 (Hybrid digital-twin / ECM + data models)
- Key idea: combine proxy physical metrics with ML outputs to improve robustness in protocol transfer.
- Practical impact: motivates the impedance proxy, dQ/dV, and domain-shift features in the extractor.

## SOH transfer and safety checks

### 6. arXiv:2504.000? (Uncertainty and temporal adaptation for battery prognostics)
- Key idea: domain gap between temperature and discharge-rate bins is a common deployment failure mode.
- Practical impact: this skill includes mean/quantile drift checks and warning thresholds for target-domain launch.

### 7. arXiv:2402.00068 (Test-time adaptation for battery aging models)
- Key idea: adapt degradation estimates in real operation using held-out calibration signals.
- Practical impact: supports a structured target-data preflight check before model adoption.

## Legacy baseline that inspired this repository direction

- 2512.24686 (the paper you referenced) describes an SOH-focused framework and remains aligned with mechanism-aware feature stacks and interpretation-first outputs; this skill extends that direction with explicit uncertainty and shift checks.
