# Milestone 9 thermo-nuclear code quality review

## Scope

This review covers the FastAPI prediction wrapper and persisted application behavior.

## Findings and resolutions

### HTTP routes owned application orchestration

The first implementation made routes call scoring and persistence separately.
That duplicated active model metadata and allowed future routes to persist partial state differently.

Resolution: `ScoringCoordinator` now owns validation, scoring, version attachment, atomic run persistence, and finding retrieval.
Routes only parse transport inputs and present application results.

### Active model identity was duplicated as a string

The first implementation repeated `deterministic-baseline-v1` in scoring and retraining paths.
That made activation state vulnerable to drift.

Resolution: the versioned model configuration now declares the active model version once.
Scoring and retraining consume that value.

### Manual retraining escaped the injected runtime

The first implementation called the modeling command without passing the API runtime configuration.
Tests or deployments with overridden paths could therefore retrain against default locations.

Resolution: `run_modeling_experiment` accepts an injected runtime configuration and manual retraining passes the API runtime through.

### Failed requests were absent from structured logs

The first middleware logged only after a response existed.
Unhandled failures could bypass the request context log.

Resolution: request logging now records status 500 through a `finally` path while preserving successful response headers and status codes.

## Approval

The API is decomposed into contracts, routes, presenters, middleware, application orchestration, scoring, retraining, and persistence.
No changed file approaches 1,000 lines.
The review found no remaining structural blocker after these changes.
