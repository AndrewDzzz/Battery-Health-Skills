---
name: interpretability-pipeline
description: Train interpretable battery fault classifiers with GBDT, compute SHAP attributions, and turn top feature contributions into structured LLM-ready diagnostics.
---

# BatteryAgent Interpretability Pipeline

Use this skill when you need to build the model+reasoning half of the BatteryAgent stack:
- deterministic training artifact generation
- attribution output for model transparency
- machine-consumable prompt packs for LLM fault reasoning

## Workflow

1. Validate data: feature table + label column with one label per cycle.
2. Train a GBDT classifier (`GradientBoostingClassifier`) and save the trained model.
3. Produce standard quality metrics and hard-boundary samples.
4. Compute SHAP attributions and ranked feature contributions.
5. Convert ranked attributions into LLM-ready prompts with optional mechanism mapping.

```bash
python scripts/train_gbdt_shap.py \
  --input /path/to/features.csv \
  --label-col fault_label \
  --sample-id-col cycle_id \
  --output-dir /path/to/out \
  --test-size 0.25 \
  --random-state 42

python scripts/compose_diagnosis_prompt.py \
  --attributions /path/to/out/top_attributions.csv \
  --mapping /path/to/out/feature_to_mechanism.json \
  --output /path/to/out/llm_prompts.jsonl
```

## Default Outputs

- `model.joblib`: trained model artifact
- `eval_metrics.json`: AUROC/accuracy/precision/recall/F1 and sample counts
- `global_importance.csv`: mean absolute SHAP per feature
- `top_attributions.csv`: per-sample top contributing features
- `hard_boundary_cases.csv`: samples with weak class confidence

## References

- `references/gbdt_shap_prompting.md`: expected modeling + prompting structure
- `scripts/train_gbdt_shap.py`: training and explainability pipeline
- `scripts/compose_diagnosis_prompt.py`: SHAP-to-prompt compiler for LLMs
