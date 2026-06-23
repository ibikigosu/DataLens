# DataLens problem framing and deterministic baseline

## Product problem

Data stewards and procurement analysts often receive vendor and transaction datasets that are too large for complete manual review.
DataLens must identify specific data quality issues, explain the evidence, and rank the resulting findings for human review.
It must not classify fraud, infer misconduct, or automatically modify source records.

The primary operational output is a ranked list of findings.
Each finding identifies one issue type on one vendor record or transaction record.
A record may have several independent findings.
The risk score ranks review priority from 0 to 100 and is not a probability.

## Target workflow

1. A data steward supplies a validated pair of vendor and transaction tables.
2. DataLens runs deterministic rules and later statistical models.
3. Findings are ranked by business severity and supporting evidence.
4. The reviewer investigates the highest-priority findings first.
5. Review feedback is recorded at issue level.
6. Source correction remains outside DataLens and under human control.

## Evaluation design

Naturally unusual USAspending records are not reliable ground truth.
The baseline therefore uses controlled defects injected into real-shaped FY2024 and FY2025 data.
Every controlled defect retains its table, immutable record identifier, issue type, severity, field, original value, injected value, and random seed.

FY2024 is the development period.
FY2025 is a temporal holdout and uses the same defect catalog, seed policy, rules, severity mapping, and evaluation code.
No FY2025 distribution is used to change the baseline.

Issue-level evaluation treats each `(table, record, issue type)` tuple as one label.
Record-level evaluation treats a record as positive when it contains at least one controlled defect.
This distinction matters because detecting one issue on a multi-issue record does not mean every issue was found.

## Controlled defect catalog

| Issue type | Table | Severity | Rationale |
| --- | --- | --- | --- |
| `missing_vendor_name` | Vendor | High | A reviewable vendor needs an identifying name. |
| `invalid_vendor_uei` | Vendor | Critical | UEI is the primary federal recipient identifier and must have 12 alphanumeric characters. |
| `invalid_domestic_state` | Vendor | High | A domestic USA address must use a recognized state or territory code. |
| `duplicate_vendor_id` | Vendor | Critical | Vendor identifiers must be unique within a procurement dataset. |
| `orphan_vendor_reference` | Transaction | Critical | Every transaction must identify one vendor record. |
| `duplicate_transaction_key` | Transaction | Critical | Transaction identifiers must be unique. |
| `invalid_performance_date_order` | Transaction | High | A current performance end date cannot precede its start date. |
| `negative_offer_count` | Transaction | Medium | A count of offers cannot be negative. |
| `action_date_outside_fiscal_year` | Transaction | High | A period extract must contain action dates inside its declared fiscal year. |

Forty defects of each type are injected per fiscal year.
The injector supports immutable record identities so duplicate business-key defects can be evaluated without losing the affected record.
Duplicate-key rules intentionally flag every member of a duplicate group.
This protects recall but creates false alarms on the unmodified partner record, which is an important baseline limitation.

## Ranking and metrics

Critical, high, medium, and low findings begin at risk scores of 100, 75, 50, and 25.
Additional independent issues add a bounded contribution without allowing a score above 100.
The deterministic baseline reports:

- Issue-level precision, recall, and F1
- Record-level precision, recall, and F1
- Macro issue-level F1
- High and critical issue recall
- Top-50 precision
- False alarms per 1,000 evaluated records

High and critical recall protects severe data integrity checks.
Top-50 precision measures whether a reviewer sees useful findings in a realistic first work queue.
False alarms per 1,000 records makes review burden comparable across datasets of different sizes.

## Baseline limitations

The controlled catalog covers structural and logically impossible defects rather than every useful procurement quality issue.
It does not establish that naturally rare values, negative obligations, zero-dollar actions, missing DUNS, or absent parent awards are defects.
Exact duplicate rules cannot know which member of a duplicate pair is the incorrect record.
The baseline does not detect near-duplicate names, context-dependent missingness, unusual amounts, or temporal vendor changes.
The rule set is a meaningful reference because it is transparent, reproducible, and strong on deterministic constraints, but it is not sufficient as the final ranking system.

