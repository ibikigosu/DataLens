# Statistical model comparison results

## Decision

The development comparison selected Isolation Forest for vendor records and Local Outlier Factor for transaction records.
The selected statistical workflow was not promoted over the deterministic baseline.
It remains supplemental anomaly evidence because it missed most controlled defects and generated more false alarms.
The deterministic baseline remains the primary issue detector.

FY2024 alone controlled model selection.
The table winners were locked before FY2025 was scored.
FY2025 was used only for temporal evaluation.

## Model comparison

| Metric | FY2024 deterministic baseline | FY2024 selected statistical workflow | FY2025 deterministic baseline | FY2025 selected statistical workflow |
| --- | ---: | ---: | ---: | ---: |
| Precision | 81.82% | 8.48% | 81.63% | 4.50% |
| Recall | 100.00% | 3.89% | 100.00% | 2.78% |
| Macro F1 across tables | 89.90% | 5.41% | 89.80% | 4.52% |
| Top-50 precision | 96.00% | 4.00% | 92.00% | 0.00% |
| False alarms per 1,000 records | 3.34 | 6.31 | 3.78 | 9.91 |
| High and critical record recall | 100.00% | 3.13% | 100.00% | 2.50% |

These values use a record-level comparison so statistical anomaly rankings and deterministic issue findings are evaluated at a shared grain.
The earlier deterministic baseline report retains the separate issue-level metrics.

## Why the table winners were selected

Vendor Isolation Forest achieved 12% FY2024 top-50 precision and 5.77% F1.
Vendor Local Outlier Factor achieved 8% FY2024 top-50 precision and 4.74% F1.
Isolation Forest therefore won the vendor comparison.

Transaction Local Outlier Factor achieved 6% FY2024 top-50 precision and 5.05% F1.
Transaction Isolation Forest achieved 0% FY2024 top-50 precision and 0% F1.
Local Outlier Factor therefore won the transaction comparison.

The selection order was top-50 precision, F1, precision, and then fewer false positives.
No FY2025 metric was available to the selection function.

## Promotion result

Promotion requires the statistical workflow to be non-inferior to the deterministic baseline for top-50 precision, macro F1, and false alarms.
It also requires selection from a development period and 100% guarded high and critical recall.

The selected statistical workflow failed all three performance non-inferiority gates.
It passed the development-only selection gate.
The guarded queue passed the deterministic critical-finding preservation gate.

The result is a deliberate non-promotion.
The model artifacts and experiment runs remain useful for iteration, but they do not replace deterministic detection or ranking.

## Deterministic critical guardrail

The guarded review queue is an outer union of deterministic records and statistical-only anomaly candidates.
Deterministic critical records receive the highest priority tier.
Deterministic non-critical records receive the next tier.
Statistical-only candidates receive the lowest tier.

The FY2024 and FY2025 guarded queues both retained 100% high and critical controlled-defect recall.
The union increased false alarms, which is another reason the statistical layer is not promoted as a primary ranking source.

## Bounded anomaly evidence

Each statistical record contains at most three feature deviations.
The serialized evidence is limited to 1,000 characters.
The anomaly score is bounded from 0 to 100 and represents a relative rank against FY2024 training scores.
The evidence explicitly states that anomaly rank is not business severity and does not identify a specific issue.

The generated evidence is stored in `artifacts/modeling/fy2025-bounded-anomaly-evidence.json`.
The full selected scores are stored in `artifacts/modeling/fy2025-selected-scores.parquet`.
The guarded review queue is stored in `artifacts/modeling/fy2025-guarded-review-queue.parquet`.

## MLflow tracking

MLflow uses the local SQLite tracking store at `artifacts/mlflow.db`.
Each model family run records parameters, FY2024 and FY2025 metrics, dataset identity, schema version, feature-pipeline metadata, table-specific estimators, and serializable model bundles.
The selection run records the table winners, promotion decision, comparison summary, selected scores, guarded queue, and bounded evidence.

## Interpretation

The poor statistical results are consistent with the current feature boundary.
Several controlled defects concern names, UEI format, date relationships, duplicate attribution, and cross-table references that are not represented strongly in the first statistical feature matrices.
The deterministic rules directly encode those structural constraints and should remain authoritative.

The anomaly models still provide a useful experimental baseline for unusual amount and category combinations.
Future iteration should improve context-rich features and reviewer-feedback reranking without weakening deterministic critical coverage.

Anomaly percentiles are not calibrated defect probabilities.
The rank calibration tables show that controlled-defect prevalence does not increase consistently enough across score deciles to support probability language.
