# Battery AI Abuse Testing Notes

## Test purpose

Abuse tests should answer one question before deployment:
`Will minor telemetry distortion or manipulative input patterns cause unsafe diagnostic outcomes?`

## Default tolerances

- Acceptable flip-rate delta (abuse vs clean) should be reviewed by safety.
- Default target: <= 5% absolute increase for binary case.
- High-risk if top-1 confidence drops by more than `0.25` on normal samples.

## Suggested evidence bundle

- Raw sample count and class balance
- Baseline metrics
- For each abuse case: flip rate, confidence delta, and top unstable features

## Review rules

- If any test has `status = hold`, block rollout and require remediation.
- For recurrent false alarms, tighten thresholds or tune feature clipping rules before reopening.

