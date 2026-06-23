# Milestone 8 thermo-nuclear code quality review

## Scope

This review covers configuration and reproducibility changes only.

## Findings and resolutions

### Dynamic model table access obscured the contract

The first implementation used `getattr` to select vendor or transaction model settings.
That was unnecessary magic around a two-value domain.

Resolution: `FamilyModelConfiguration.for_table` now owns the explicit table selection.

### Relationship validation leaked a hard-coded join column

The first implementation validated transaction references through a literal `vendor_id`.
That contradicted the goal of making the versioned schema authoritative.

Resolution: paired validation now reads the transaction-to-vendor relationship from the schema.
The schema requires exactly one such relationship.

### Schema scoring weights were optional at the canonical rules boundary

The first implementation silently fell back to severity-derived scores when callers omitted configured weights.
That left two sources of truth for scoring behavior.

Resolution: every deterministic scoring call must now provide the schema scoring weights.
The obsolete severity-to-score fallback was removed.

### Configured paths did not fully reach execution

The first implementation loaded path settings but some training and baseline functions still used module constants.
That made path configuration partly decorative.

Resolution: baseline and modeling entry points now pass resolved configured data, manifest, artifact, dataset, baseline, and MLflow locations into execution.

## Approval

No file approaches the 1,000-line threshold because of this milestone.
Configuration models, loading, and dataframe validation remain separate cohesive modules.
The review found no remaining blocker for the milestone after the resolutions above.
