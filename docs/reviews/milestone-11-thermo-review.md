# Milestone 11 thermo-nuclear code quality review

## Scope

This review covers MLflow registration, activation evidence, and the model card.

## Findings and resolutions

### Experiment tracking made the modeling workflow structurally mixed

The first registry implementation added MLflow registration inside an already large experiment orchestration module.
That combined data loading, training, evaluation, artifact persistence, tracking, and registry behavior in one file.

Resolution: MLflow runs and selected-candidate registration moved to `modeling/tracking.py`.
The workflow now orchestrates the experiment while tracking owns MLflow-specific behavior.

### Candidate registration could be mistaken for activation

Registering the selected anomaly models without explicit status would make “best candidate” easy to confuse with “active model.”

Resolution: selected table models receive the `candidate` alias and explicit promoted or rejected tags.
The application registry manifest separately records the deterministic baseline as active.
The `active` MLflow alias is assigned only when all gates pass.

### Rejection evidence needed to survive outside logs

Console output and experiment parameters were not sufficient audit evidence for future reviewers.

Resolution: the registry manifest records active identity, candidate versions, aliases, status, reasons, and the complete promotion decision.
The same manifest is retained as a local artifact and an MLflow artifact.

## Approval

Registry behavior is isolated from experiment orchestration.
Activation cannot occur through candidate registration alone.
The model card states intended use, metrics, limitations, controlled-defect assumptions, simulated feedback, and future validation needs.
