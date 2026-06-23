# Combine deterministic rules, anomaly models, and a false-alarm reranker

DataLens will use separate rule engines and Isolation Forest models for vendor and transaction records, then use one feedback-trained model to rerank likely false alarms.
Issue identity and severity remain deterministic, while learned models improve review ordering.
The reranker may never suppress a deterministic critical finding, which protects recall and keeps explanations auditable.
