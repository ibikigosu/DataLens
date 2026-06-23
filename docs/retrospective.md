# DataLens retrospective

## What worked

The milestone sequence kept data acquisition, baseline rules, feature engineering, modeling, application behavior, and interface work independently reviewable.
The deterministic baseline produced strong and stable controlled-defect readings across development and temporal holdout data.
Typed configuration removed source edits from environment changes and recorded dataset, schema, feature, and model versions.
FastAPI became the canonical application boundary for validation, scoring, persistence, feedback, and retraining.
The guarded ranking design preserved every deterministic finding while allowing feedback to influence review order.
MLflow retained experiment metrics, selected candidate versions, aliases, and rejection reasons.
The final Streamlit workflow remained a thin API client and exposed limitations beside the model evidence.

## What did not work

The unsupervised anomaly candidates did not improve the deterministic baseline.
They produced materially lower top-50 precision and recall with more false alarms.
They were correctly rejected instead of being promoted for demonstration value.

The first schema validator rejected duplicate business keys before duplicate rules could score them.
End-to-end demo preparation exposed that boundary error and the validator was corrected.

The first UI test double modeled an HTTP response incorrectly.
Browser testing caught the mismatch that unit tests had hidden.

Docker definitions and service contracts are verified, but this workstation has no supported Docker CLI.
An actual `docker compose up --build --wait` run remains a machine-level verification item.

## Machine learning tradeoffs

The deterministic baseline reaches 0.96 development top-50 precision and 0.92 temporal top-50 precision with full controlled-defect recall.
The selected anomaly workflow reaches 0.04 development top-50 precision and 0.00 temporal top-50 precision.
The guarded anomaly queue preserves recall but increases false alarms.

The feedback reranker improves top-50 precision in the mentor demo because simulated feedback has a strong, learnable relationship with issue type.
That result is intentionally visible, but it is optimistic.
It does not replace evaluation on adjudicated production feedback.
Reranking preserves every deterministic finding, so overall recall remains 100%.
Top-50 recall is reported separately so review-budget tradeoffs remain visible.

## What would change for production

Use adjudicated reviewer labels from multiple teams and periods.
Separate training, validation, and policy approval responsibilities.
Add database migrations instead of proof-of-concept schema compatibility logic.
Move model activation state into a transactional registry boundary.
Use object storage for durable artifacts and an asynchronous worker for long-running training.
Add authentication, authorization, audit retention, rate limits, malware scanning, and upload quarantine.
Add load tests, disaster recovery tests, drift monitoring, reviewer agreement metrics, and subgroup analysis.
Run the full stack in continuous integration with Docker and PostgreSQL.

## Final assessment

DataLens is deployment-shaped and suitable for a mentor proof of concept.
It is not a production fraud system, an automatic correction engine, or a validated production machine learning service.
The strongest result is the transparent engineering workflow: weak models are rejected, reviewer feedback is retained, and promotion evidence is inspectable.
