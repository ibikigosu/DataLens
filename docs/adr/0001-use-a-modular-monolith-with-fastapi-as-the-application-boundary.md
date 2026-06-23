# Use a modular monolith with FastAPI as the application boundary

DataLens will be a production-shaped modular monolith in which FastAPI owns validation, scoring, persistence, feedback, retraining, and model lifecycle behavior.
Streamlit will remain a thin client of the public API and will not import machine learning code or access PostgreSQL directly.
This preserves one source of application behavior while demonstrating clear boundaries that can later be extracted if scale justifies it.
