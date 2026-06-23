# USAspending Public Buildings Service EDA findings

## Scope

DataLens uses contract actions awarded by the General Services Administration's Public Buildings Service.
FY2024 covers October 1, 2023 through September 30, 2024 and is the development period.
FY2025 covers October 1, 2024 through September 30, 2025 and is reserved as a temporal holdout.
The EDA profiles FY2024 only.
FY2025 values are not inspected or visualized before temporal evaluation.
The extracts contain contract award type codes A, B, C, and D.

This scope was selected because it provides a coherent procurement domain and naturally falls near the planned 20,000 to 50,000 transaction size.
The complete two-year scope contains 41,735 transaction records without arbitrary row truncation.

## Grain and relationships

A transaction record represents one USAspending contract action.
Its primary key is `contract_transaction_unique_key`.
A vendor record is derived from the latest observed representation of one recipient UEI.
The derived `vendor_id` uses UEI and retains DUNS only as a legacy fallback.

FY2024 contains 22,064 transaction records and 1,785 vendor records.
The FY2025 preparation manifest records 19,671 transaction records and 1,652 vendor records without exposing holdout distributions to the EDA.
FY2024 has unique transaction keys and unique vendor keys.
Every FY2024 transaction joins to a vendor.

## Data quality observations

DUNS is completely absent in FY2024.
This is expected after the federal transition to UEI and is not a defect.

Recipient doing-business-as name and address line 2 are usually absent.
These are optional fields and their absence is not independently actionable.

Parent award fields are absent for standalone awards.
Set-aside fields, offer count, solicitation date, and action type have context-dependent missingness.
Rules for these fields must account for award and action context rather than treating all missing values as defects.

Negative federal action obligations are common and represent deobligations or corrections.
Zero-dollar actions are also common and frequently represent administrative changes.
In FY2024, 7.40% of actions are negative, 30.90% are zero, and 61.70% are positive.
Amount sign alone must not be interpreted as a quality issue.

One FY2024 transaction has a current performance end date before its performance start date.
These records merit review, but naturally unusual source records remain unlabeled unless evidence establishes a defect.

Hundreds of vendors have more than one address representation across their transactions.
This affects 239 FY2024 vendors, or 13.39% of the derived vendor table.
This could indicate correction, relocation, formatting variation, or contradictory source values.
Future consistency features should preserve time and source context.

## Distribution observations

Delivery orders and BPA calls account for most FY2024 records.
Together they represent 83.28% of transactions.
Facilities support services and commercial or institutional building construction are the dominant NAICS categories.
Transaction volume is concentrated among a relatively small number of vendors.
The FY2024 median vendor has 3 transactions, while the largest vendor has 582.
The ten most active vendors account for 17.63% of transactions, and the top 100 account for 49.95%.
This concentration means record-level metrics can be dominated by high-volume vendors.

Monthly transaction volume rises sharply near the fiscal-year end.
September has 2,714 actions compared with 1,188 in October.
Later feature work should consider fiscal position without allowing the model to treat year-end volume alone as a defect.

Federal action obligations are strongly right-skewed and include large positive and negative values.
The FY2024 median is $6,033.74, while the maximum is $523.65 million and the minimum is negative $57.43 million.
Robust statistics and transformed amount features will be more informative than unscaled raw values.

## Leakage and evaluation risks

FY2025 must remain isolated from EDA-driven rule tuning, defect design calibration, feature scaling, and model fitting.
Fields derived from later updates, such as `last_modified_date`, may encode information that was not available at action time.
Such fields may support provenance and recency checks but require careful treatment as model features.

Controlled defects must retain their original values, issue type, severity, and random seed.
Naturally unusual records must not be converted into positive labels merely because a rule or model flags them.
Record-level metrics must be complemented by issue-level metrics because one record may contain several independent defects.

## Sampling and temporal limitations

The dataset covers one GSA sub-agency and is not representative of every federal agency.
It also does not reproduce all fields found in a private enterprise vendor master or purchase system.
The derived vendor table reflects recipient attributes repeated on contract actions rather than an authoritative standalone vendor registry.

USAspending can correct historical records.
The versioned manifests record the acquisition time, source update date, filters, row counts, and file hashes so this analysis remains auditable.

## Recommended checks for the baseline

- Missing required identifiers and names
- Invalid identifier formats
- Duplicate transaction identifiers
- Broken transaction-to-vendor relationships
- Impossible performance date ordering
- Negative offer counts
- Inconsistent vendor attributes within a procurement dataset

These checks are deterministic, explainable, and suitable for controlled-defect evaluation.
They intentionally exclude amount sign, rare categories, and unusual natural values unless a separate source contract proves them invalid.
