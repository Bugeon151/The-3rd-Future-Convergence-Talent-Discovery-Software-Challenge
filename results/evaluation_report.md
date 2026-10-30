# Evaluation Report

Measured on the deterministic synthetic scenario suite (see nearzero/sim/scenarios.py).
Every number below is produced by running the code in this repository (`scripts/run_eval.ps1`).

## 1. Runtime benchmark

- Detector device: cuda:0
- Detector inference: 48.7 ms
- Core pipeline (perception->verify, no vision): 181.6 ms

## 2. Policy comparison (per-scenario)

| policy | scenario | recall | mean lead (s) | false alarms |
|---|---|---|---|---|
| A: distance_threshold | * | 1.00 | 0.50 | 1 |
| A: distance_threshold | head_on | 1.00 | 0.40 | 0 |
| A: distance_threshold | crossing | 1.00 | 0.50 | 0 |
| A: distance_threshold | parallel | NULL | NULL | 0 |
| A: distance_threshold | safe_pass | NULL | NULL | 1 |
| A: distance_threshold | sudden_stop | 1.00 | 0.40 | 0 |
| A: distance_threshold | sudden_turn | NULL | NULL | 0 |
| A: distance_threshold | zone_intrusion | 1.00 | 0.90 | 0 |
| A: distance_threshold | demo_avoided | 1.00 | 0.30 | 0 |
| B: ttc_threshold | * | 1.00 | 2.62 | 2 |
| B: ttc_threshold | head_on | 1.00 | 2.60 | 0 |
| B: ttc_threshold | crossing | 1.00 | 2.10 | 0 |
| B: ttc_threshold | parallel | NULL | NULL | 0 |
| B: ttc_threshold | safe_pass | NULL | NULL | 1 |
| B: ttc_threshold | sudden_stop | 1.00 | 2.60 | 0 |
| B: ttc_threshold | sudden_turn | NULL | NULL | 1 |
| B: ttc_threshold | zone_intrusion | 1.00 | 3.20 | 0 |
| B: ttc_threshold | demo_avoided | 1.00 | 2.60 | 0 |
| C: trajectory_prediction | * | 1.00 | 2.70 | 2 |
| C: trajectory_prediction | head_on | 1.00 | 2.60 | 0 |
| C: trajectory_prediction | crossing | 1.00 | 2.10 | 0 |
| C: trajectory_prediction | parallel | NULL | NULL | 0 |
| C: trajectory_prediction | safe_pass | NULL | NULL | 1 |
| C: trajectory_prediction | sudden_stop | 1.00 | 2.60 | 0 |
| C: trajectory_prediction | sudden_turn | NULL | NULL | 1 |
| C: trajectory_prediction | zone_intrusion | 1.00 | 3.60 | 0 |
| C: trajectory_prediction | demo_avoided | 1.00 | 2.60 | 0 |
| D: nearzero_x | * | 1.00 | 1.52 | 1 |
| D: nearzero_x | head_on | 1.00 | 1.50 | 0 |
| D: nearzero_x | crossing | 1.00 | 1.90 | 0 |
| D: nearzero_x | parallel | NULL | NULL | 0 |
| D: nearzero_x | safe_pass | NULL | NULL | 0 |
| D: nearzero_x | sudden_stop | 1.00 | 1.50 | 0 |
| D: nearzero_x | sudden_turn | NULL | NULL | 1 |
| D: nearzero_x | zone_intrusion | 1.00 | 1.40 | 0 |
| D: nearzero_x | demo_avoided | 1.00 | 1.30 | 0 |

## 3. Ablation (NEAR-ZERO X components)

| ablation | recall | mean_lead (s) | false_alarms | intervention_success | mean_risk_reduction |
|---|---|---|---|---|---|
| full | 1.00 | 1.43 | 0 | 1.00 | 44.82 |
| no_uncertainty | 1.00 | 2.40 | 0 | 1.00 | 54.07 |
| no_counterfactual | 1.00 | 1.43 | 0 | NULL | NULL |
| no_verification | 1.00 | 1.43 | 0 | 1.00 | 44.82 |

## 4. Honest limitations

- Synthetic evaluation isolates algorithmic safety quality from camera noise.
- Lead time and recall are computed on ground-truth contact intervals.
- Values labeled NULL are scenarios where the metric does not apply.
