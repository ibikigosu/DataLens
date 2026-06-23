"""Polished Streamlit interface that delegates product behavior to FastAPI."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from datalens.configuration.loader import load_runtime_config
from datalens.ui.api_client import ApiError, DataLensApiClient, Upload

st.set_page_config(
    page_title="DataLens review workspace",
    page_icon=":material/data_check:",
    layout="wide",
)

st.session_state.setdefault("scoring_run", None)
st.session_state.setdefault("retraining", None)
st.session_state.setdefault("feedback_count", 0)
st.session_state.setdefault("feedback_saved", False)

config = load_runtime_config()
client = DataLensApiClient(config.settings.api_base_url)


def _status_badge(label: str, passed: bool) -> None:
    st.badge(
        label,
        icon=":material/check:" if passed else ":material/close:",
        color="green" if passed else "red",
    )


def _render_header() -> None:
    heading, status = st.columns([4, 1], vertical_alignment="center")
    with heading:
        st.title("DataLens")
        st.caption(
            "Validate paired procurement data, review evidence, and test whether "
            "feedback improves ranking."
        )
    with status:
        try:
            client.health()
            st.badge("API ready", icon=":material/check_circle:", color="green")
        except ApiError as error:
            st.badge("API unavailable", icon=":material/error:", color="red")
            st.error(str(error))
            st.stop()


def _render_upload() -> None:
    with st.container(border=True):
        title, context = st.columns([3, 2], vertical_alignment="bottom")
        with title:
            st.subheader("1. Validate and score")
            st.caption("Upload vendor and transaction CSV files as one procurement dataset.")
        with context:
            fiscal_year = st.number_input(
                "Fiscal year",
                min_value=2000,
                max_value=2100,
                value=2024,
                step=1,
            )

        vendor_column, transaction_column = st.columns(2)
        with vendor_column:
            vendor_file = st.file_uploader(
                "Vendor records",
                type="csv",
                help="Use demo/vendors.csv for the mentor walkthrough.",
            )
        with transaction_column:
            transaction_file = st.file_uploader(
                "Transaction records",
                type="csv",
                help="Use demo/transactions.csv for the mentor walkthrough.",
            )

        action, demo_action, note = st.columns([1, 1, 2], vertical_alignment="center")
        with action:
            score = st.button(
                "Validate and score",
                type="primary",
                icon=":material/play_arrow:",
                disabled=vendor_file is None or transaction_file is None,
                width="stretch",
            )
        with demo_action:
            demo = st.button(
                "Run demo dataset",
                icon=":material/science:",
                width="stretch",
            )
        with note:
            st.caption(
                "Validation runs before scoring. Cross-table relationship rules require both files."
            )

        if score or demo:
            if demo:
                vendor_path = Path("demo/vendors.csv")
                transaction_path = Path("demo/transactions.csv")
                vendor_upload = Upload(
                    vendor_path.name,
                    io.BytesIO(vendor_path.read_bytes()),
                )
                transaction_upload = Upload(
                    transaction_path.name,
                    io.BytesIO(transaction_path.read_bytes()),
                )
            else:
                assert vendor_file is not None
                assert transaction_file is not None
                vendor_upload = Upload(vendor_file.name, vendor_file)
                transaction_upload = Upload(transaction_file.name, transaction_file)
            with st.status("Validating the schema and ranking findings...", expanded=False):
                try:
                    st.session_state.scoring_run = client.score_batch(
                        fiscal_year=int(fiscal_year),
                        vendors=vendor_upload,
                        transactions=transaction_upload,
                    )
                    st.session_state.retraining = None
                    st.write("Validation passed.")
                    st.write("Findings were persisted with a stable scoring run identifier.")
                except ApiError as error:
                    st.error(str(error))


def _filtered_findings(findings: pd.DataFrame) -> pd.DataFrame:
    severities = sorted(findings["severity"].dropna().unique().tolist())
    selected = st.pills(
        "Severity filter",
        severities,
        default=severities,
        selection_mode="multi",
    )
    if not selected:
        return findings.iloc[0:0]
    return findings.loc[findings["severity"].isin(selected)].reset_index(drop=True)


def _render_findings(run: dict[str, object]) -> None:
    summary = run["summary"]
    assert isinstance(summary, dict)
    with st.container(horizontal=True):
        st.metric("Vendor records", f"{summary['vendor_records']:,}", border=True)
        st.metric(
            "Transaction records",
            f"{summary['transaction_records']:,}",
            border=True,
        )
        st.metric("Ranked findings", f"{summary['finding_count']:,}", border=True)
        st.metric(
            "Feedback submitted",
            st.session_state.feedback_count,
            border=True,
        )

    st.subheader("2. Review ranked findings")
    st.caption(
        f"Run {run['run_id']} used schema {run['schema_version']} and model {run['model_version']}."
    )
    findings = pd.DataFrame(run["findings"])
    if findings.empty:
        st.success("No configured quality issues were found.", icon=":material/check_circle:")
        return

    filtered = _filtered_findings(findings)
    table_column, review_column = st.columns([3, 2], gap="large")
    with table_column:
        event = st.dataframe(
            filtered,
            column_order=[
                "severity",
                "review_priority",
                "risk_score",
                "model_confidence",
                "target_table",
                "record_id",
                "issue_type",
            ],
            column_config={
                "severity": st.column_config.TextColumn("Severity", pinned=True),
                "review_priority": st.column_config.NumberColumn(
                    "Review priority",
                    format="%.3f",
                ),
                "risk_score": st.column_config.NumberColumn(
                    "Risk score",
                    format="%.0f",
                ),
                "model_confidence": st.column_config.ProgressColumn(
                    "Model confidence",
                    min_value=0,
                    max_value=1,
                    format="percent",
                ),
                "target_table": st.column_config.TextColumn("Table"),
                "record_id": st.column_config.TextColumn("Record"),
                "issue_type": st.column_config.TextColumn("Issue type"),
            },
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=430,
        )

    selected_rows = event.selection.rows
    selected = filtered.iloc[selected_rows[0]] if selected_rows else filtered.iloc[0]
    with review_column, st.container(border=True):
        st.badge(selected["severity"].title(), icon=":material/flag:")
        st.subheader(str(selected["issue_type"]).replace("_", " "))
        st.caption(f"{selected['target_table']} · {selected['record_id']}")
        st.write(selected["evidence"])
        st.caption(
            "Risk score reflects configured business severity. "
            "Model confidence affects review ordering only."
        )

        with st.form("finding-feedback", clear_on_submit=True):
            verdict = st.segmented_control(
                "Review result",
                ["correct_flag", "false_alarm", "wrong_issue_type", "unsure"],
                default="correct_flag",
                format_func=lambda value: value.replace("_", " ").title(),
            )
            notes = st.text_area(
                "Reviewer notes",
                max_chars=2_000,
                placeholder="What did you verify in the source system?",
            )
            submitted = st.form_submit_button(
                "Save review",
                type="primary",
                icon=":material/rate_review:",
            )
        if submitted and verdict:
            try:
                client.submit_feedback(
                    str(selected["finding_id"]),
                    verdict=verdict,
                    notes=notes,
                )
                st.session_state.feedback_count += 1
                st.session_state.feedback_saved = True
                st.rerun()
            except ApiError as error:
                st.error(str(error))

    st.download_button(
        "Download visible findings",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"datalens-{run['run_id']}-findings.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def _render_model_comparison() -> None:
    st.subheader("3. Compare active and candidate ranking")
    st.caption(
        "Manual retraining uses decisive persisted feedback. "
        "Simulated demo feedback remains labeled as simulated."
    )
    if st.button(
        "Train feedback reranker",
        icon=":material/model_training:",
        type="secondary",
    ):
        with st.status("Training and evaluating a feedback reranker...", expanded=False):
            try:
                st.session_state.retraining = client.retrain()
                st.write("The candidate was evaluated on held-out feedback.")
                st.write("Every deterministic finding remains in the review queue.")
            except ApiError as error:
                st.error(str(error))

    result = st.session_state.retraining
    if not result:
        st.caption(
            "For the mentor demo, seed simulated historical feedback with "
            "`uv run python scripts/demo_workflow.py --seed-feedback`."
        )
        return

    promotion = result["promotion"]
    promoted = bool(result["promoted"])
    if promoted:
        st.success(
            f"{result['candidate_model_version']} passed every gate and became active.",
            icon=":material/check_circle:",
        )
    else:
        st.warning(
            promotion["reason"],
            icon=":material/warning:",
        )

    active = result["active"]
    candidate = result["candidate"]
    if active and candidate:
        comparison = pd.DataFrame(
            [
                {
                    "Ranking": "Active",
                    "Top-k precision": active["top_k_precision"],
                    "Top-k recall": active["top_k_recall"],
                    "False alarms": active["false_alarms_in_top_k"],
                    "Overall recall": active["overall_recall"],
                },
                {
                    "Ranking": "Candidate",
                    "Top-k precision": candidate["top_k_precision"],
                    "Top-k recall": candidate["top_k_recall"],
                    "False alarms": candidate["false_alarms_in_top_k"],
                    "Overall recall": candidate["overall_recall"],
                },
            ]
        )
        st.dataframe(
            comparison,
            column_config={
                "Top-k precision": st.column_config.NumberColumn(format="percent"),
                "Top-k recall": st.column_config.NumberColumn(format="percent"),
                "Overall recall": st.column_config.NumberColumn(format="percent"),
            },
            hide_index=True,
        )
        st.caption(
            f"Evaluation used top {result['evaluation_top_k']} on "
            f"{result['validation_examples']} held-out feedback examples. "
            "Overall recall remains 100% because reranking does not suppress findings."
        )

    with st.container(horizontal=True):
        for gate, passed in promotion["gates"].items():
            _status_badge(gate.replace("_", " ").title(), bool(passed))


def _render_empty_state() -> None:
    st.subheader("A review flow with an audit trail")
    st.markdown(
        """
        1. Upload the paired vendor and transaction files.
        2. Inspect ranked, evidence-backed quality findings.
        3. Submit reviewer feedback and compare ranking performance.
        """
    )
    st.caption(
        "DataLens identifies potential data quality issues. "
        "It does not detect fraud or automatically correct source data."
    )


_render_header()
if st.session_state.feedback_saved:
    st.toast("Review feedback saved.", icon=":material/check_circle:")
    st.session_state.feedback_saved = False
_render_upload()

run = st.session_state.scoring_run
if run:
    _render_findings(run)
    _render_model_comparison()
else:
    _render_empty_state()

retraining = st.session_state.retraining
active_model = (
    retraining["candidate_model_version"]
    if retraining and retraining["promoted"]
    else run["model_version"]
    if run
    else config.model.active_model_version
)
with st.sidebar:
    st.caption("DataLens proof of concept")
    st.markdown(f"**Schema**  \n{config.schema.schema_version}")
    st.markdown(f"**Current active model**  \n{active_model}")
    st.markdown(f"**Environment**  \n{config.settings.environment}")
    st.caption("FastAPI owns validation, scoring, persistence, feedback, and retraining.")
