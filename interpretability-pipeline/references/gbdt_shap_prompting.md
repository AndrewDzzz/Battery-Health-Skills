# Interpretable Battery Diagnosis: GBDT + SHAP + LLM Bridge

This reference file contains a practical contract for model explainability and report generation.

## Training contract

- Use a gradient-boosting tree model as a stable, interpretable baseline.
- Keep classes defined and documented (normal vs fault categories).
- Persist both:
  - the fitted model artifact
  - metric snapshot (`eval_metrics.json`) to make releases auditable.

## Attribution contract

- Use SHAP as the default explanation layer for tree models.
- Persist:
  - `global_importance.csv`: average absolute SHAP per feature
  - `top_attributions.csv`: per-cycle sorted positive/negative contributors
- Include class/prediction context in attribution rows.

## LLM bridge contract

- LLM input should be structured as:
  - context fields (`sample_id`, predicted label, true label, top support features)
  - mapped mechanism signals (`cycle_*`, voltage, thermodynamics)
  - explicit request for causal narrative + confidence + verification actions.
- Keep prompt payload short (3-8 bullets for top features).

## Suggested quality gates

1. `hard_boundary_cases` proportion:
   - binary: `|p_true - 0.5| < 0.12` and in production mode review every event.
   - multiclass: `top1 - top2 < 0.12` review every flagged event.
2. Feature group coverage:
   - at least one top contributor from each domain in 10+ case reports over each quarter.
3. Artifact completeness:
   - model + metrics + attributions + prompt pack in each training run.

