# DataLens Model Card

## Intended Use

DataLens ranks procurement data quality findings so reviewers can prioritize records that need human attention.
DataLens does not detect fraud.
The system is intended for audit preparation, data stewardship, and workflow triage over controlled defects in procurement-style records.

## Data And Evaluation

The demonstration data is simulated and contains controlled defects that exercise known data quality scenarios.
Current evaluation uses top-50 precision to measure whether the highest-priority findings contain relevant issues for review.
Validation uses a temporal holdout so later records are used to check whether model behavior generalizes beyond the training slice.

## Candidate Models

The anomaly candidates were rejected.
The deterministic baseline remains the safe default when a feedback-trained reranker is not active or does not improve validation performance.

## Limitations

Model scores should be treated as prioritization aids, not final determinations.
Performance depends on the quality and representativeness of reviewer feedback.
Future validation should include larger labeled datasets, additional agencies and vendors, and monitoring for drift across fiscal years.
