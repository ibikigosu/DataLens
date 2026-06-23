# Milestone 12 thermo-nuclear code quality review

## Scope

This review covers the feedback reranker, demo data, final Streamlit workflow, demo script, and retrospective.

## Findings and resolutions

### Retraining did not originally use reviewer feedback

The earlier manual retraining endpoint repeated the unsupervised comparison and ignored persisted feedback.
That contradicted the human-in-the-loop domain model.

Resolution: retraining now builds a supervised feedback reranker from the latest decisive issue-level reviews and evaluates it on held-out feedback.

### Duplicate defects were blocked before scoring

The schema validator originally required business keys to be unique.
That made duplicate-vendor and duplicate-transaction rules unreachable.

Resolution: keys must be present and nonblank, while duplicates remain valid scoring inputs and become findings.

### Reranker artifacts needed a safe, portable activation boundary

The first artifact manifest stored an absolute model path and loaded the model on every scoring request.

Resolution: active metadata stores a version-relative path, safe skops types are validated, and the active pipeline is cached by version.

### The mentor flow required too many manual review actions

Training from one newly submitted review is neither statistically valid nor practical in a live demonstration.

Resolution: a public-API demo script seeds 400 clearly labeled simulated historical reviews in one transactional batch.
The interface still requires and demonstrates a real reviewer submission.

### Visual polish had to preserve application ownership

Loading demo data or computing model metrics inside Streamlit would have duplicated FastAPI behavior.

Resolution: the interface sends demo files through the same public scoring API and presents API-owned feedback and retraining results.

## Approval

The reranker, persistence, API, and UI responsibilities remain separated.
The interface is visually reviewed in initial, scored, feedback, and promoted states.
The final documentation states both successful readings and validation weaknesses.
