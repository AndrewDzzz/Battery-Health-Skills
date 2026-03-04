# Battery Security Baseline Notes

## Core security domains

- Edge/Device: tamper resistance, OTA integrity, key lifecycle, boot trust
- Telemetry: schema enforcement, replay protection, transport encryption, anomaly logging
- Storage: immutable audit trail, access control, retention, secret handling
- AI/ML stack: data provenance, training isolation, model artifact integrity, inference policy gates

## Default threat list

1. Sensor data spoofing
2. Replay injection
3. Timestamp manipulation
4. OTA downgrade/fake image
5. Secrets leakage
6. Poisoned training files
7. Membership inference / model theft
8. Prompt or input injection on explanation stack
9. Denial-of-service on inference endpoints
10. Insider misuse and over-privilege

## Priority mapping

- Criticality scale: 1–5 (impact), Likelihood scale: 1–5
- Priority score = `impact * likelihood`
- High-risk: score >= 20
- Medium-risk: 9–19
- Low-risk: <= 8

## Remediation templates

- Add immutable logging for `cycle_id` and ingestion hash.
- Gate model rollout on security and data checks passing.
- Require signed OTA payloads with rollback protection.
- Keep a separate security owner and on-call escalation for each high-risk row.

