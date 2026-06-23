# DataLens model card

## Model details

DataLens combines an active deterministic data quality baseline with table-specific statistical anomaly candidates.
The active model version is `deterministic-baseline-v1`.
The evaluated anomaly policy version is `anomaly-v1`.

The selected vendor candidate uses Isolation Forest.
The selected transaction candidate uses Local Outlier Factor.
Both candidates remain rejected because they did not pass the configured promotion gates.

## Purpose and intended use

DataLens helps procurement data stewards rank potential data quality issues for human review.
It is intended for exploratory review, workflow demonstration, and engineering evaluation.

DataLens does not detect fraud.
An unusual record is not proof of an error, misconduct, or business risk.
The statistical anomaly score is supplemental review evidence and is not business severity.

## Data

Development and model selection use GSA Public Buildings Service contract data from USAspending for fiscal year 2024.
Fiscal year 2025 is retained as a temporal holdout and is not used to select model families or preprocessing.

Evaluation labels come from controlled defects injected reproducibly into real-shaped public procurement records.
Controlled defects are not naturally observed production labels.
They provide a known test target for missing values, invalid identifiers, invalid relationships, duplicates, date errors, and invalid numeric values.

Reviewer feedback in the proof of concept may be simulated to demonstrate the human-in-the-loop workflow.
Simulated feedback must remain distinguishable from verified reviewer decisions.

## Features

Vendor features include transaction counts, address variation, country, state, and business-size codes.
Transaction features include obligation values, offer counts, award type, pricing type, action type, product or service code, NAICS, competition, solicitation procedure, and set-aside codes.

Numeric values use configured transformations and robust scaling.
Categorical values use learned frequency encoding.
Preprocessing is fitted on development data and reused unchanged for temporal scoring.

## Evaluation

On fiscal year 2024 controlled-defect evaluation:

- The deterministic baseline achieved 0.96 top-50 precision, 1.00 recall, 0.899 macro F1, and 3.34 false alarms per 1,000 records.
- The selected anomaly workflow achieved 0.04 top-50 precision, 0.039 recall, 0.054 macro F1, and 6.31 false alarms per 1,000 records.
- The guarded combined queue preserved 1.00 high and critical recall but increased false alarms to 9.53 per 1,000 records.

On the fiscal year 2025 temporal holdout:

- The deterministic baseline achieved 0.92 top-50 precision, 1.00 recall, 0.898 macro F1, and 3.78 false alarms per 1,000 records.
- The selected anomaly workflow achieved 0.00 top-50 precision, 0.028 recall, 0.045 macro F1, and 9.91 false alarms per 1,000 records.
- The guarded combined queue preserved 1.00 high and critical recall but increased false alarms to 13.55 per 1,000 records.

## Promotion decision

The anomaly candidates were rejected.
They failed top-50 precision non-inferiority, macro F1 non-inferiority, and false-alarm-rate non-inferiority gates.
The deterministic critical-finding preservation gate passed.

The deterministic baseline therefore remains active.
The rejected candidate versions remain registered in MLflow with the `candidate` alias and rejection tags for auditability.

## Feedback reranker demonstration

The final proof of concept trains a supervised reranker from decisive persisted feedback.
The mentor demo seeds 200 simulated correct flags and 200 simulated false alarms before adding one interactive reviewer decision.
The candidate is evaluated on held-out feedback with a top-50 review budget.
In the controlled demo, candidate top-50 precision improves from approximately 50% to 100% while overall recall remains 100% because no deterministic finding is removed.

This result is optimistic because simulated labels have a strong relationship with issue type.
It demonstrates the feedback, evaluation, promotion, artifact, and activation workflow.
It is not production evidence.

## Limitations and risks

Controlled defects cover only known issue families and may be easier to detect than real production errors.
Naturally unusual records in USAspending are not labeled as correct or incorrect.
Public procurement data does not represent every enterprise procurement system, policy, geography, or data-entry process.
The temporal holdout covers one later fiscal year and is not equivalent to long-term monitoring.
The proof of concept does not assess fairness impacts across vendor groups.
The proof of concept does not establish calibrated probabilities.
The proof of concept does not validate reviewer consistency or feedback quality.

## Future validation

Production use would require adjudicated real-world labels, multiple temporal and organizational holdouts, subgroup analysis, drift monitoring, feedback quality controls, security review, privacy review, and operational load testing.
Any future supervised reranker must be evaluated separately from the controlled-defect anomaly comparison.
