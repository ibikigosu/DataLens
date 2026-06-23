# Use PostgreSQL and MLflow for shared state and model governance

DataLens will use PostgreSQL for scoring runs, findings, feedback, schema versions, and model lifecycle metadata, with MLflow tracking experiments and candidate models.
Although the proof-of-concept dataset is small, the API, interface, and model lifecycle require shared transactional state and reproducible promotion evidence.
Artifacts remain on a mounted local volume because object storage would not strengthen the demonstration.
