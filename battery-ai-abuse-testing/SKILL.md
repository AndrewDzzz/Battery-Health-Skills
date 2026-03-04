---
name: battery-ai-abuse-testing
description: Run controlled abuse checks on battery ML models by applying noise/injection/feature-perturbation tests and reporting prediction stability and risk.
---

# Battery AI Abuse Testing

Use this skill to evaluate whether battery health models are robust to common abuse cases.

## Quick Start

```bash
python scripts/run_ai_abuse_tests.py \
  --input /path/to/features.csv \
  --label-col fault_label \
  --model-path /path/to/model.joblib \
  --output-dir /path/to/ai-abuse \
  --seed 42
```

If no model is supplied, a default `GradientBoostingClassifier` baseline is trained for reproducibility.

## Included test families

- Feature noise: gaussian perturbation on numeric features
- Sensor inversion: sign flips on current and voltage to mimic tamper conditions
- Replay substitution: row permutation and mixed-context perturbation

## Outputs

- `abuse_summary.json`: pass/fail summary and drift metrics
- `abuse_details.csv`: per-test flip rates and confidence shifts
- `high-risk-cases.csv`: sample IDs where prediction stability is low

## References

- `references/abuse-test-notes.md`
- `scripts/run_ai_abuse_tests.py`

