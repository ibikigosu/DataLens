# Deterministic baseline results

## Result

The deterministic baseline detects every injected issue in both fiscal years.
It preserves 100% recall for high and critical controlled defects.
Its main weakness is review ordering and duplicate attribution rather than missed structural defects.

| Metric | FY2024 development | FY2025 temporal holdout |
| --- | ---: | ---: |
| Evaluated records | 23,929 | 21,403 |
| Controlled defects | 360 | 360 |
| Findings | 441 | 441 |
| Issue-level precision | 81.63% | 81.63% |
| Issue-level recall | 100.00% | 100.00% |
| Issue-level F1 | 89.89% | 89.89% |
| Record-level precision | 81.82% | 81.63% |
| Record-level recall | 100.00% | 100.00% |
| Macro issue-level F1 | 92.46% | 92.46% |
| High and critical issue recall | 100.00% | 100.00% |
| Top-50 precision | 48.00% | 50.00% |
| False alarms per 1,000 records | 3.39 | 3.78 |

The metrics were generated with seed `20240622` and 40 defects of each of the nine controlled issue types per fiscal year.

## Interpretation

The baseline has perfect controlled-defect recall because its rules directly represent structural constraints and logically impossible values.
That result does not imply complete procurement data quality coverage.
The controlled catalog intentionally excludes ambiguous natural anomalies.

Duplicate vendor and transaction identifiers each have 50% issue-level precision.
When a duplicate pair exists, the rule must flag both records because it has no evidence identifying which member is the erroneous copy.
This creates 80 expected false alarms per fiscal year.

The baseline also flags one naturally occurring performance date ordering anomaly in each period.
Those findings are not counted as controlled true positives because naturally unusual source records are not ground truth.

Top-50 precision is lower than overall precision because critical duplicate findings sort to the front of the review queue.
The deterministic score treats both members of a duplicate pair equally.
This establishes a useful target for later anomaly and false-alarm ranking work.

## What the baseline proves

- Controlled defects can be reproduced over real-shaped source data.
- Issue-level labels remain separate from source anomalies and future review feedback.
- Vendor and transaction rules can be evaluated through one consistent finding contract.
- Severe deterministic issues can be protected with 100% recall.
- FY2025 produces materially consistent behavior without changing the rule set.

## What remains unresolved

The baseline cannot rank the likely incorrect member of a duplicate pair.
It cannot detect near-duplicate vendors, unusual category combinations, amount anomalies, or inconsistent attributes that require context.
It does not calibrate model confidence.
It does not use reviewer feedback.
These limitations motivate feature engineering, separate vendor and transaction anomaly models, and a later false-alarm reranker.

