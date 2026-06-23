# DataLens Internship Roadmap

This roadmap translates the mentor-provided internship guide into the delivery sequence for DataLens.
It is the project progress tracker and should be updated throughout development.

The aim is not to maximize the number of technologies.
The aim is to demonstrate an understandable progression from exploration to a reproducible machine learning application, with each decision supported by evidence.

## How progress is recorded

Each milestone moves through these states:

- `Not started`
- `In progress`
- `In review`
- `Complete`

A milestone is complete only when:

- Its acceptance criteria have been verified.
- Its documentation explains the important decisions.
- Its pull request has been merged, or the user explicitly confirms completion.
- The PR number, completion date, and evidence links have been recorded here.

Do not mark a milestone complete merely because files exist locally.
Do not delete completed entries.
If a decision changes, record the change and link the pull request that changed it.

## Pull request strategy

Every milestone should produce at least one independently reviewable pull request.
A milestone may use several small pull requests when that makes the history easier to teach and review.
Do not combine unrelated milestones into one pull request.

Every pull request should explain:

- What changed
- Why the change belongs at this stage
- How it was tested or verified
- What was learned
- What remains for the next milestone

Infrastructure should be introduced when the product behavior needs it.
For example, PostgreSQL should be added with persisted scoring runs or feedback, not as an isolated technology demonstration with no user-visible purpose.

## Milestone status

| # | Milestone | Status | Issue | Pull request | Completed |
| --- | --- | --- | --- | --- | --- |
| 1 | Environment setup and repository scaffolding | Complete | TBD | [#3](https://github.com/ibikigosu/DataLens/pull/3) | 2026-06-22 |
| 2 | Dataset exploration and EDA | Complete | TBD | [#4](https://github.com/ibikigosu/DataLens/pull/4) | 2026-06-22 |
| 3 | Problem framing and baseline | Complete | TBD | [#5](https://github.com/ibikigosu/DataLens/pull/5) | 2026-06-22 |
| 4 | Feature engineering | Complete | TBD | [#7](https://github.com/ibikigosu/DataLens/pull/7) | 2026-06-22 |
| 5 | First real model and experiment tracking | Complete | TBD | [#8](https://github.com/ibikigosu/DataLens/pull/8) | 2026-06-22 |
| 6 | Model iteration and comparison | Complete | TBD | [#8](https://github.com/ibikigosu/DataLens/pull/8) | 2026-06-22 |
| 7 | Notebook-to-module code cleanup | Complete | TBD | [#10](https://github.com/ibikigosu/DataLens/pull/10) | 2026-06-22 |
| 8 | Configuration and reproducibility | Complete | TBD | [#12](https://github.com/ibikigosu/DataLens/pull/12) | 2026-06-23 |
| 9 | FastAPI prediction wrapper | Complete | TBD | [#13](https://github.com/ibikigosu/DataLens/pull/13) | 2026-06-23 |
| 10 | Containerization | Complete | TBD | [#14](https://github.com/ibikigosu/DataLens/pull/14) | 2026-06-23 |
| 11 | Model registry and model card | Complete | TBD | [#15](https://github.com/ibikigosu/DataLens/pull/15) | 2026-06-23 |
| 12 | Final demo and retrospective | Ready for PR | TBD | TBD | TBD |

## 1. Environment setup and repository scaffolding

**Goal**:
Create a clean repository that mentors can clone and run.

**Planned pull request**:
Project foundation with README, Python environment, pinned dependencies, source and test structure, linting, and basic verification commands.

**Acceptance criteria**:

- [x] The GitHub repository contains an initial README.
- [x] The supported Python version is documented.
- [x] A virtual environment can be created from documented commands.
- [x] Dependencies are pinned and reproducible.
- [x] Source, notebook, test, configuration, data, and documentation locations are clear.
- [x] Formatting, linting, and test commands are documented and pass.
- [x] A mentor can clone the repository and run the initial verification command.

**Decision evidence**:

- Explain why the selected environment and dependency tools were chosen.
- Explain the repository structure without presenting unused architecture as completed functionality.
- Local evidence: [README](../../README.md), [uv project configuration](../../pyproject.toml), and [verification script](../../scripts/verify.py).
- Pull request: [#3](https://github.com/ibikigosu/DataLens/pull/3).
- Verified in an isolated worktree on 2026-06-22 with 1 passing test and 100% coverage.

**Course reference**:
Chapter 1, Initializing.

## 2. Dataset exploration and EDA

**Goal**:
Understand the USAspending GSA data before committing to preprocessing and modeling assumptions.

**Planned pull requests**:

1. Acquire and snapshot a reproducible raw or minimally processed sample.
2. Add the EDA notebook and written findings.

These may be combined if the acquisition step is small and remains independently understandable.

**Acceptance criteria**:

- [x] The notebook reports dataset shape and table relationships.
- [x] The notebook reports field types and likely semantic types.
- [x] The notebook reports missing values and duplicate patterns.
- [x] The notebook explores important numeric and categorical distributions.
- [x] At least three useful visualizations are included.
- [x] Vendor and transaction grain are identified.
- [x] Candidate join keys and their quality are investigated.
- [x] Potential leakage, sampling bias, and temporal limitations are documented.
- [x] Observations and open questions are written in the notebook.
- [x] The notebook runs from top to bottom from a clean environment.

**Decision evidence**:

- Explain why GSA and the selected fiscal-year scope are appropriate.
- Distinguish naturally unusual records from verified data quality defects.
- Record which fields can support vendor and transaction quality checks.
- Local evidence: [executed EDA notebook](../../notebooks/01_usaspending_pbs_eda.ipynb), [written EDA findings](../analysis/usaspending-pbs-eda.md), and [source manifests](../../data/manifests/).
- Verified locally on 2026-06-22: the notebook executed from top to bottom with FY2024 as the development dataset and FY2025 kept sealed behind manifest metadata.
- Pull request: [#4](https://github.com/ibikigosu/DataLens/pull/4).
- Verified in an isolated worktree on 2026-06-22 with 23 passing tests and 86.25% coverage.

**Course reference**:
Chapter 2, Prototyping: data exploration.

## 3. Problem framing and baseline

**Goal**:
Define the measurable data quality problem and establish a simple reference approach.

**Planned pull request**:
Problem statement, evaluation design, controlled defect catalog, and deterministic rule-based baseline.

**Acceptance criteria**:

- [x] The prediction and ranking problem is stated precisely.
- [x] DataLens is explicitly scoped to data quality rather than fraud.
- [x] The target users and review workflow are documented.
- [x] Controlled defects provide reproducible issue-level labels.
- [x] Record-level and issue-level evaluation are distinguished.
- [x] A rule-based baseline produces documented metrics.
- [x] High and critical issue recall is reported.
- [x] Top-50 precision and false alarms per 1,000 records are reported.
- [x] Baseline limitations are documented.

**Decision evidence**:

- Explain why a rule baseline is meaningful for this domain.
- Explain why controlled defects are combined with real-shaped data.
- Explain why top-50 precision and high-severity recall matter operationally.
- Local evidence: [problem framing](problem-framing-and-baseline.md), [controlled-defect configuration](../../config/baseline/controlled-defects.json), and [baseline results](../analysis/deterministic-baseline-results.md).
- Verified locally on 2026-06-22: deterministic baseline behavior remained reproducible after the typed issue-registry and dataset-plan refactor.
- Pull request: [#5](https://github.com/ibikigosu/DataLens/pull/5).

**Course reference**:
Chapter 2, Prototyping: model assessment.

## 4. Feature engineering

**Goal**:
Create a reproducible preprocessing and feature pipeline for vendor and transaction quality analysis.

**Planned pull request**:
Typed preprocessing pipelines and documented quality features.

**Acceptance criteria**:

- [x] Missing values are handled deliberately.
- [x] Categorical encoding choices are documented.
- [x] Numeric transformations and scaling choices are documented.
- [ ] Duplicate and near-duplicate features are implemented.
- [ ] Category rarity features are implemented.
- [ ] Date validity and relationship features are implemented.
- [ ] Vendor and transaction cross-table consistency features are implemented.
- [ ] At least one non-trivial engineered feature is included.
- [x] Training and scoring use the same transformations.
- [x] Leakage tests protect the FY2025 temporal holdout.

**Decision evidence**:

- Explain each feature family in domain terms.
- Explain why vendor and transaction feature spaces remain separate.
- Local evidence: [typed feature pipeline](../../src/datalens/features/pipeline.py), [development dataset boundary](../../src/datalens/features/dataset.py), [table-specific feature builders](../../src/datalens/features/builders.py), [vendor and transaction schemas](../../src/datalens/features/schemas.py), and [feature tests](../../tests/features/).
- Verified locally on 2026-06-22: FY2024 fitting and FY2024/FY2025 transformation produced finite vendor and transaction matrices, duplicate business keys retained unique row identities, and FY2025 remained excluded from the development fitting boundary.
- Pull request: [#7](https://github.com/ibikigosu/DataLens/pull/7).

**Course reference**:
Chapter 3, Productionizing: pipeline structure.

## 5. First real model and experiment tracking

**Goal**:
Train the first statistical anomaly models and begin tracking every meaningful experiment.

**Planned pull request**:
Separate vendor and transaction Isolation Forest models with MLflow tracking.

**Acceptance criteria**:

- [x] Vendor and transaction models are trained separately.
- [x] Model parameters, metrics, dataset identity, schema version, and artifacts are logged.
- [x] The models are evaluated against the baseline.
- [x] Scoring is reproducible with fixed seeds.
- [x] Model outputs can be translated into bounded finding evidence.
- [x] The experiment can be rerun from documented commands.

**Decision evidence**:

- Explain why Isolation Forest is appropriate as the first statistical model.
- Explain its limitations and why anomaly score is not business severity.
- Local evidence: [modeling modules](../../src/datalens/modeling/), [thin model comparison notebook](../../notebooks/02_model_comparison.ipynb), and [model comparison results](../analysis/model-comparison-results.md).
- Verified locally on 2026-06-22: separate vendor and transaction Isolation Forest models were fitted on FY2024 only, logged to MLflow with SQLite tracking, and evaluated against the deterministic baseline.
- The anomaly score is a bounded training-score rank and every evidence payload contains no more than three feature deviations and 1,000 characters.
- Pull request: [#8](https://github.com/ibikigosu/DataLens/pull/8).

**Course reference**:
Chapter 2, Prototyping: model selection.

## 6. Model iteration and comparison

**Goal**:
Compare multiple approaches and show evidence-based model selection.

**Planned pull request**:
Model comparison and feedback-trained false-alarm reranking.

**Acceptance criteria**:

- [x] At least two model approaches are compared in MLflow.
- [ ] A feedback-trained reranker is evaluated separately from issue detection rules.
- [x] FY2025 remains the temporal evaluation set.
- [x] Precision, recall, macro F1, top-50 precision, false alarms, and calibration are compared.
- [x] Deterministic critical findings cannot be suppressed.
- [x] Promotion criteria are implemented and tested.
- [x] The selected model and rejected alternatives are explained.

**Decision evidence**:

- Explain why one false-alarm reranker is preferred over underpowered per-issue classifiers.
- Explain the performance and explainability tradeoffs among the compared approaches.
- Local evidence: [development-only winner selection](../../src/datalens/modeling/evaluation.py), [guarded review queue](../../src/datalens/modeling/scoring.py), and [comparison results](../analysis/model-comparison-results.md).
- Verified locally on 2026-06-22: Isolation Forest and Local Outlier Factor were compared with FY2024 selection and FY2025 evaluation.
- Isolation Forest won the vendor comparison and Local Outlier Factor won the transaction comparison.
- The selected statistical workflow failed promotion against the deterministic baseline and remains supplemental evidence.
- The guarded queue preserved 100% high and critical controlled-defect recall in FY2024 and FY2025.
- Pull request: [#8](https://github.com/ibikigosu/DataLens/pull/8).

**Course reference**:
Chapter 4, Validating: testing and debugging.

## 7. Notebook-to-module code cleanup

**Goal**:
Move reusable logic into clear Python modules while keeping notebooks focused on analysis.

**Planned pull request**:
Refactor exploration and modeling logic into independently testable domain modules.

**Acceptance criteria**:

- [x] Core logic no longer depends on notebook execution state.
- [x] Notebooks act as thin analysis drivers.
- [x] Schema validation, feature engineering, rules, scoring, explanations, and evaluation have clear interfaces.
- [x] Modules are documented at their public boundaries.
- [x] Existing behavior remains covered by tests.
- [x] A new contributor can locate the main workflow without reading every file.

**Decision evidence**:

- Explain module boundaries and why they are likely to remain stable.
- Avoid splitting code into shallow modules that merely move functions between files.
- Local evidence: [modeling package](../../src/datalens/modeling/), [model regression tests](../../tests/modeling/), [model comparison notebook](../../notebooks/02_model_comparison.ipynb), and [README workflow map](../../README.md).
- Verified locally on 2026-06-22: training, scoring, evaluation, bounded evidence, promotion, MLflow tracking, and orchestration run from Python modules.
- The notebook executed from top to bottom without errors.
- Repository verification passed with 54 tests and 87.73% coverage.
- Pull request: [#10](https://github.com/ibikigosu/DataLens/pull/10).

**Course reference**:
Chapter 3, Productionizing: packaging and modules.

## 8. Configuration and reproducibility

**Goal**:
Run training and scoring from a clean clone without manual code edits.

**Planned pull request**:
Versioned schema configuration, application settings, model parameters, and reproducible commands.

**Acceptance criteria**:

- [ ] Paths and model parameters are configurable.
- [ ] The GSA schema is version controlled and validated.
- [ ] Scoring weights are configured per schema.
- [ ] Environment-specific values do not require source edits.
- [ ] Dataset, schema, feature, and model versions are recorded.
- [ ] Training and evaluation can be reproduced from documented commands.
- [ ] Invalid configuration fails with a clear error.

**Decision evidence**:

- Explain which settings belong in version control and which belong in environment variables.
- Local evidence: [configuration package](../../src/datalens/configuration/), [versioned configuration](../../config/), [configuration decision](../decisions/configuration-and-reproducibility.md), and [configuration tests](../../tests/configuration/).
- Verified locally on 2026-06-23: typed configuration, environment overrides, schema validation, configured scoring weights, and version recording passed targeted tests.

**Course reference**:
Chapter 3, Productionizing: configuration.

## 9. FastAPI prediction wrapper

**Goal**:
Expose the verified scoring workflow through a tested API.

**Planned pull requests**:

1. Single-record scoring and health behavior.
2. Paired CSV batch scoring, persisted runs, findings, and feedback.

PostgreSQL should be introduced in the second slice because persisted runs and feedback create the need for shared transactional state.

**Acceptance criteria**:

- [ ] The API is versioned under `/api/v1`.
- [ ] Single vendor and transaction JSON scoring are supported.
- [ ] Paired vendor and transaction CSV scoring is supported.
- [ ] Validation fails before scoring when inputs violate the schema.
- [ ] Batch responses include a stable run identifier and summary.
- [ ] Findings are retrievable as JSON and downloadable as CSV.
- [ ] Feedback and manual retraining are exposed through the API.
- [ ] At least one error case is demonstrated with curl or a simple client.
- [ ] Structured logs include request and scoring context.
- [ ] Health and readiness endpoints are implemented.
- [ ] Public API behavior is covered by integration tests.

**Decision evidence**:

- Explain why FastAPI owns all application behavior.
- Explain why Streamlit and other clients must use the API.
- Explain why PostgreSQL is introduced at the persistence boundary.
- Local evidence: [FastAPI application](../../src/datalens/api/), [application services](../../src/datalens/application/), [API integration tests](../../tests/api/), and [application boundary decision](../decisions/fastapi-application-boundary.md).
- Verified locally on 2026-06-23: health, readiness, single-record scoring, paired CSV scoring, persisted runs, JSON and CSV finding retrieval, feedback, manual retraining behavior, validation failures, and structured request logs passed integration tests.

**Course reference**:
Chapter 5, Refining: CI/CD and containers introduction.

## 10. Containerization

**Goal**:
Run the complete application stack through documented container commands.

**Planned pull requests**:

1. Containerize the FastAPI application.
2. Add the complete Docker Compose demonstration stack with Streamlit, PostgreSQL, and MLflow.

**Acceptance criteria**:

- [ ] A Dockerfile runs the FastAPI application.
- [ ] Docker Compose starts every required service.
- [ ] Persistent service data uses documented volumes.
- [ ] Services declare useful health checks.
- [ ] The README documents build, run, stop, and reset commands.
- [ ] A smoke test verifies startup and representative scoring.
- [ ] A mentor can run the stack from a clean clone.

**Decision evidence**:

- Explain why local artifact storage is sufficient for the POC.
- Document when asynchronous workers or object storage would become justified.
- Local evidence: [Dockerfile](../../Dockerfile), [Compose stack](../../compose.yaml), [stack smoke test](../../scripts/smoke_stack.py), [container tests](../../tests/container/), and [containerization decision](../decisions/containerization.md).
- Verified locally on 2026-06-23: compose topology, named volumes, health checks, non-root image contract, thin Streamlit API ownership, and representative public API smoke behavior passed automated tests.
- Docker runtime verification remains pending because this workstation exposes no supported Docker CLI.

**Course reference**:
Chapter 5, Refining: software containers.

## 11. Model registry and model card

**Goal**:
Version the selected model and communicate its intended use and limitations.

**Planned pull request**:
MLflow model registration, promotion evidence, and model card.

**Acceptance criteria**:

- [ ] The best model is registered and versioned.
- [ ] The active model can be distinguished from candidates.
- [ ] Promotion or rejection reasons are retained.
- [ ] The model card describes purpose, data, features, metrics, limitations, and intended use.
- [ ] The model card states that DataLens does not detect fraud.
- [ ] The model card explains controlled defects and simulated feedback.
- [ ] Known risks and future validation needs are documented.

**Decision evidence**:

- Explain why the selected model satisfies the promotion gates.
- Explain where the evaluation remains weaker than a production validation.
- Local evidence: [registry implementation](../../src/datalens/modeling/registry.py), [MLflow tracking](../../src/datalens/modeling/tracking.py), [model card](../model-card.md), and [registry workflow tests](../../tests/modeling/test_workflow.py).
- Verified locally on 2026-06-23: selected table models were versioned in MLflow, candidate aliases and rejection tags were retained, the deterministic baseline remained active, and the model card contract passed automated checks.

**Course reference**:
Chapter 6, Sharing: documentation and versioning.

## 12. Final demo and retrospective

**Goal**:
Demonstrate the complete product flow and reflect honestly on the engineering and machine learning choices.

**Planned pull request**:
Polished Streamlit workflow, final documentation, demo script, and retrospective.

**Acceptance criteria**:

- [ ] The demo starts from the documented local setup.
- [ ] A paired dataset is uploaded and validated.
- [ ] Findings are ranked and explained.
- [ ] Reviewer feedback is submitted.
- [ ] Retraining compares the active and candidate models.
- [ ] The demo shows whether promotion gates passed.
- [ ] The Streamlit workflow is visually reviewed and polished.
- [ ] The retrospective explains what worked.
- [ ] The retrospective explains what did not work.
- [ ] The retrospective explains what would change with more time or production requirements.
- [ ] Final limitations and follow-up work are explicit.

**Decision evidence**:

- Connect the final result back to the baseline.
- Show the improvement in top-50 precision without hiding recall tradeoffs.
- Local evidence: [Streamlit workflow](../../src/datalens/ui/streamlit_app.py), [feedback reranker](../../src/datalens/modeling/reranker.py), [demo preparation](../../scripts/demo_workflow.py), [final demo script](../final-demo.md), and [retrospective](../retrospective.md).
- Verified locally on 2026-06-23: the public demo flow scored 400 findings, retained 400 simulated historical reviews, promoted a feedback reranker after every gate passed, improved held-out top-50 precision, preserved overall recall, and applied the active reranker to later scoring.
- Streamlit was visually reviewed in the empty, scored, reviewer-feedback, and promoted-candidate states.

**Course reference**:
Chapter 7, Observability: monitoring and explainability.

## Weekly mentor check-ins

Add one entry per mentor check-in.
Keep entries brief and decision focused.

### Template

#### Week of YYYY-MM-DD

**Completed**:

- Work completed since the previous check-in.

**Evidence**:

- Links to pull requests, experiments, notebooks, or demonstrations.

**Decisions**:

- Important choices and why they were made.

**Blockers and questions**:

- Items where mentor feedback is useful.

**Next**:

- The next milestone or pull request.

#### Week of 2026-06-22

**Completed**:

- Initialized the uv and Git project foundation.
- Acquired and prepared FY2024 development data and the sealed FY2025 temporal holdout.
- Completed FY2024 EDA, problem framing, controlled-defect evaluation, and the deterministic baseline.

**Evidence**:

- Executed EDA notebook with five plotted views and no execution errors.
- Source and prepared-data manifests with SHA-256 integrity checks.
- Twenty-eight passing tests with 89.33% coverage.
- FY2024 top-50 precision of 48% with 100% high and critical controlled-defect recall.
- FY2025 temporal evaluation top-50 precision of 50% with 100% high and critical controlled-defect recall.

**Decisions**:

- Kept FY2025 distributions out of EDA to protect the temporal holdout.
- Excluded negative obligations, zero-dollar actions, absent DUNS, and missing parent-award fields from unconditional defect rules.
- Centralized issue identity and severity in one typed registry.

**Blockers and questions**:

- Milestones cannot become complete until their pull requests are merged or completion is explicitly confirmed.
- Repository documentation under `docs/` is currently ignored by Git and therefore will not appear in a future pull request unless that ignore rule changes.

**Next**:

- Create the independently reviewable milestone branches and pull requests, then begin feature engineering after review.

## Change log

Record material roadmap changes here instead of silently rewriting the intended sequence.

| Date | Change | Reason |
| --- | --- | --- |
| 2026-06-22 | Created the DataLens roadmap from the mentor-provided internship guide. | Preserve the required milestone sequence and evidence expectations. |
| 2026-06-22 | Moved EDA and baseline milestones to `In review`. | All local acceptance criteria and verification checks pass, but no pull request has been merged. |
| 2026-06-22 | Sealed FY2025 from exploratory profiling. | Preserve a defensible temporal holdout for later evaluation. |
| 2026-06-22 | Marked milestones 1 through 3 complete after merging pull requests #3, #4, and #5. | Record verified delivery evidence and preserve chronological progress. |
| 2026-06-22 | Marked milestone 4 complete after merging pull request #7. | Record the verified feature-pipeline foundation before beginning model training. |
| 2026-06-22 | Moved milestones 5 and 7 to `In review` and milestone 6 to `In progress`. | Local model tracking, comparison, guardrails, module cleanup, notebook execution, and regression tests are verified, while the roadmap's feedback-trained reranker criterion remains open. |
| 2026-06-22 | Marked milestones 5 and 6 complete after merging pull request #8. | Record the verified first-model and model-comparison work before the cleanup milestone. |
| 2026-06-22 | Opened draft pull request #10 for milestone 7. | Move milestone 7 to `In review` with an independently reviewable branch. |
| 2026-06-22 | Marked milestone 7 complete after merging pull request #10. | Record the verified notebook-to-module cleanup before the configuration milestone. |
