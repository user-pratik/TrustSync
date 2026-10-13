## Headline Metrics

| Metric | Count | Percentage | 95% Wilson CI |
| :--- | :---: | :---: | :---: |
| Decision Accuracy | 13/13 | 100.0% | [77.2%, 100.0%] |
| Spoof FAR | 0/13 | 0.0% | [0.0%, 22.8%] |
| All-Reject FAR | 0/13 | 0.0% | [0.0%, 22.8%] |
| False Rejection Rate (FRR) | 0/13 | 0.0% | [0.0%, 22.8%] |
| Attribution Accuracy | 13/13 | 100.0% | [77.2%, 100.0%] |
| Addressee Accuracy | 13/13 | 100.0% | [77.2%, 100.0%] |
| Liveness F1 | -- | 100.0% | -- |

## Gating Ablation Study (B0 to A5)

| Variant | Accuracy | FAR (All Reject) | FRR | Liveness F1 |
| :--- | :---: | :---: | :---: | :---: |
| B0 | 53.8% | 100.0% | 0.0% | 100.0% |
| A1 | 76.9% | 50.0% | 0.0% | 100.0% |
| A2 | 76.9% | 50.0% | 0.0% | 100.0% |
| A3 | 100.0% | 0.0% | 0.0% | 100.0% |
| A4 | 100.0% | 0.0% | 0.0% | 100.0% |
| A5 | 100.0% | 0.0% | 0.0% | 100.0% |

## Per-Scenario Evaluation

| Scenario | N | Accuracy | Accept Rate |
| :--- | :---: | :---: | :---: |
| live_single | 3 | 100.0% | 100.0% |
| multi_one_speaks | 3 | 100.0% | 100.0% |
| not_addressed | 3 | 100.0% | 0.0% |
| replay_spoof | 3 | 100.0% | 0.0% |
| silence | 1 | 100.0% | 0.0% |

## Confusion Matrix

| Expected | ACCEPT | REJECT_SPOOF | REJECT_AMBIGUOUS | REJECT_NOT_ADDRESSED | REJECT_LOW_QUALITY | REJECT_NO_FACE | NONE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ACCEPT | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| NONE | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| REJECT_NOT_ADDRESSED | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| REJECT_SPOOF | 0 | 3 | 0 | 0 | 0 | 0 | 0 |

## Leave-One-Subject-Out Cross-Validation

| Subject | Clips | Accuracy | FAR |
| :--- | :---: | :---: | :---: |
| sub_a | 5 | 100.0% | 0.0% |
| sub_b | 4 | 100.0% | 0.0% |
| sub_c | 4 | 100.0% | 0.0% |

## Dataset Distribution

| Category | Count |
| :--- | :---: |
| Total Clips | 13 |
| Scenario: live_single | 3 |
| Scenario: multi_one_speaks | 3 |
| Scenario: not_addressed | 3 |
| Scenario: replay_spoof | 3 |
| Scenario: silence | 1 |
