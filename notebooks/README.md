# Notebooks

Notebooks are analysis drivers rather than the permanent home of reusable business logic.
They must run from top to bottom from a clean uv environment.
Reusable acquisition, preparation, rule, and evaluation behavior belongs under `src/datalens/`.

`01_usaspending_pbs_eda.ipynb` is the FY2024 exploratory analysis driver.
`02_model_comparison.ipynb` is the thin driver for model training, MLflow tracking, development selection, and FY2025 evaluation.
