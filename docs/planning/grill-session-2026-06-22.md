# DataLens Design Decisions

This document preserves the decisions reached during the DataLens planning and grilling session.
It is the detailed internal design record behind the shorter mentor-facing GitHub project issue.

## Product direction

- DataLens targets enterprise data stewards and procurement analysts.
- It analyzes vendor records together with purchase transactions.
- It addresses data quality only and must not describe findings as fraud.
- It scores vendors and transactions separately, then produces a combined vendor summary.
- It is described as human-in-the-loop rather than self-learning.
- The central demonstration claim is improved top-50 precision without reducing high-severity recall.

## Data strategy

- USAspending GSA contract data is the real data foundation.
- FY2024 is used for development and FY2025 is reserved for temporal evaluation.
- The working dataset should contain approximately 20,000 to 50,000 transactions.
- Public recipient records are mapped to vendors and contract actions are mapped to transactions.
- Missing enterprise-only fields may be added through clearly documented enrichment.
- Real-shaped data is combined with controlled defect injection to avoid training only on wholly synthetic distributions.
- Controlled defects retain their original value, defect type, severity, ground-truth label, and random seed.
- A record may contain up to three controlled defects.
- Naturally occurring anomalies are not treated as known defects without evidence.
- A small processed sample is included later so the demonstration works without internet access.

## Input and schema contract

- Batch input consists of exactly two CSV files: vendors and transactions.
- An approved schema configuration is required before scoring.
- The schema defines fields, types, identifiers, required values, relationships, rules, and scoring weights.
- Scoring weights are configurable per schema with application defaults.
- Missing required columns, incompatible configured types, invalid encodings, unreadable files, and broken required join keys fail validation before scoring.
- Paired-file validation is atomic.
- Extra columns are accepted but ignored unless configured.
- DataLens never mutates uploaded source data or automatically generates a cleaned replacement.
- Uploaded files are temporary and are deleted after scoring.
- Dataset hashes, run metadata, and derived findings are retained.

## Detection and scoring

- Separate rule engines evaluate vendor and transaction records.
- Separate Isolation Forest models represent vendor and transaction anomaly behavior.
- Issue types remain deterministic and explainable through rules.
- A single supervised model learns false-alarm likelihood from feedback and reranks findings.
- Separate supervised classifiers are not trained for every issue type because feedback volume is insufficient.
- Each record receives a risk score from 0 to 100.
- Each finding includes an issue type, evidence, severity, and learned confidence when relevant.
- Severity is determined by configured business rules.
- The record score begins with the highest issue severity and adds bounded contributions from independent issues.
- Learned models may adjust ranking but may not lower or suppress a deterministic critical finding.
- Explanations use deterministic templates and bounded feature-level evidence.
- Generative language models are not used for explanations.

## Feedback and model lifecycle

- Feedback is attached to a specific finding.
- Supported outcomes are correct flag, false alarm, wrong issue type, missed issue, and unsure.
- Feedback may include an optional reviewer name, comment, and corrected value.
- Authentication, authorization, and simulated roles are not included.
- Submitted feedback remains separate from injected ground-truth labels.
- Retraining is manually triggered.
- The demonstration may seed clearly marked simulated feedback from held-out injected labels.
- MLflow tracks experiments, parameters, metrics, artifacts, and candidate models.
- Promotion requires every deterministic critical finding to remain preserved.
- Promotion must not reduce high or critical recall.
- Promotion requires at least a two-percentage-point improvement in macro issue-level F1.
- Promotion must improve top-50 precision or reduce false alarms.
- Promotion must pass FY2025 temporal evaluation.
- Failed promotion records the rejection reason and leaves the active model unchanged.

## Application architecture

- DataLens is a production-shaped modular monolith.
- FastAPI owns validation, scoring, persistence, feedback, retraining, and model lifecycle behavior.
- Streamlit is a thin review client that communicates through FastAPI.
- Streamlit does not import machine learning modules or access the database directly.
- PostgreSQL stores scoring runs, findings, feedback, schema versions, and model lifecycle metadata.
- MLflow uses PostgreSQL metadata and mounted local artifact storage.
- Object storage is excluded from the proof of concept.
- Docker Compose runs FastAPI, Streamlit, PostgreSQL, and MLflow.
- Batch scoring is synchronous.
- Each uploaded file is limited to 10 MB.
- A batch contains at most 50,000 combined rows.
- A completed batch returns a stable run identifier and compact summary.
- Full findings are retrieved separately as JSON or CSV.
- Single vendor and single transaction JSON scoring use the same scoring services as batch scoring.
- The API is versioned under `/api/v1`.
- The API exposes batch scoring, single-record scoring, run retrieval, finding retrieval, feedback, retraining, model listing, health, and readiness behavior.

## Quality and delivery

- The mentor-provided internship guide is the delivery contract for the project.
- Progress is tracked in `docs/planning/internship-roadmap.md`.
- The roadmap follows the guide's twelve milestones from environment setup through final retrospective.
- A milestone is marked complete only after verification and merge, or explicit user confirmation.
- Weekly mentor check-ins record progress, evidence, decisions, blockers, and the next step.
- Structured logs include request ID, run ID, schema version, model version, row counts, duration, and failure stage.
- The system provides health and readiness endpoints.
- Tests cover validators, defect injection, features, rules, aggregation, explanations, feedback, promotion, persistence, MLflow, API behavior, Streamlit integration, and Docker startup.
- One end-to-end flow covers upload, scoring, findings, feedback, retraining, and model selection.
- UI review includes visual inspection at common desktop dimensions.
- Lint failures, test failures, and flaky tests are release blockers.
- The repository is a mentor-facing demonstration of professional Git practice.
- Work proceeds chronologically through separate issues, branches, commits, and pull requests.
- The intended order is foundation, EDA, data preparation, validation, baseline rules, anomaly models, feedback learning, API completion, Streamlit completion, deployment, and final documentation.
- No commit, push, or pull request action occurs without the user's explicit instruction.

## Explicit exclusions

- Fraud detection
- Automatic data correction
- Authentication and role management
- Immediate online learning
- Generative explanations
- Asynchronous queues and workers
- Object storage
- Kubernetes and cloud deployment
- Permanent uploaded-file retention
- Direct enterprise database connectors
- Excel and arbitrary file formats
- A general-purpose schema editor
- Multi-tenant behavior
