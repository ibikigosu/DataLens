"""Polished Streamlit interface that delegates product behavior to FastAPI."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from datalens.configuration.loader import load_runtime_config
from datalens.demo import simulated_feedback_verdict
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
st.session_state.setdefault("simulated_feedback_seeded", False)
st.session_state.setdefault("source_record_keys", {})

config = load_runtime_config()
client = DataLensApiClient(config.settings.api_base_url)
DEMO_VENDOR_PATH = Path("demo/vendors.csv")
DEMO_TRANSACTION_PATH = Path("demo/transactions.csv")
ANOMALY_SUMMARY_PATH = Path("artifacts/milestone-11-modeling/comparison-summary.json")


def _status_badge(label: str, passed: bool) -> None:
    st.badge(
        label,
        icon=":material/check:" if passed else ":material/close:",
        color="green" if passed else "red",
    )


@st.cache_data
def _load_anomaly_summary() -> dict[str, object] | None:
    if not ANOMALY_SUMMARY_PATH.exists():
        return None
    return json.loads(ANOMALY_SUMMARY_PATH.read_text(encoding="utf-8"))


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


def _source_record_keys(vendors: pd.DataFrame, transactions: pd.DataFrame) -> dict[str, str]:
    keys: dict[str, str] = {}
    for position, vendor_id in enumerate(vendors["vendor_id"].astype("string")):
        keys[f"vendor:{position:08d}"] = str(vendor_id)
    for position, transaction_key in enumerate(
        transactions["contract_transaction_unique_key"].astype("string")
    ):
        keys[f"transaction:{position:08d}"] = str(transaction_key)
    return keys


def _uploads_from_csv_bytes(
    vendor_bytes: bytes,
    transaction_bytes: bytes,
    *,
    vendor_name: str,
    transaction_name: str,
) -> tuple[Upload, Upload, dict[str, str]]:
    vendors = pd.read_csv(io.BytesIO(vendor_bytes))
    transactions = pd.read_csv(io.BytesIO(transaction_bytes))
    return (
        Upload(vendor_name, io.BytesIO(vendor_bytes)),
        Upload(transaction_name, io.BytesIO(transaction_bytes)),
        _source_record_keys(vendors, transactions),
    )


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
                help="Use demo/vendors.csv for the user walkthrough.",
            )
        with transaction_column:
            transaction_file = st.file_uploader(
                "Transaction records",
                type="csv",
                help="Use demo/transactions.csv for the user walkthrough.",
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
                vendor_upload, transaction_upload, source_record_keys = _uploads_from_csv_bytes(
                    DEMO_VENDOR_PATH.read_bytes(),
                    DEMO_TRANSACTION_PATH.read_bytes(),
                    vendor_name=DEMO_VENDOR_PATH.name,
                    transaction_name=DEMO_TRANSACTION_PATH.name,
                )
            else:
                assert vendor_file is not None
                assert transaction_file is not None
                vendor_upload, transaction_upload, source_record_keys = _uploads_from_csv_bytes(
                    vendor_file.getvalue(),
                    transaction_file.getvalue(),
                    vendor_name=vendor_file.name,
                    transaction_name=transaction_file.name,
                )
            with st.status("Validating the schema and ranking findings...", expanded=False):
                try:
                    st.session_state.scoring_run = client.score_batch(
                        fiscal_year=int(fiscal_year),
                        vendors=vendor_upload,
                        transactions=transaction_upload,
                    )
                    st.session_state.retraining = None
                    st.session_state.simulated_feedback_seeded = False
                    st.session_state.source_record_keys = source_record_keys
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


def _mixed_queue(findings: pd.DataFrame) -> pd.DataFrame:
    mixed = findings.copy()
    mixed["_mix_key"] = pd.util.hash_pandas_object(
        mixed["finding_id"].astype("string"),
        index=False,
    )
    return mixed.sort_values("_mix_key", ignore_index=True).drop(columns=["_mix_key"])


def _confidence_queue(findings: pd.DataFrame) -> pd.DataFrame:
    return findings.sort_values(
        ["model_confidence", "risk_score", "target_table", "record_id", "issue_type"],
        ascending=[False, False, True, True, True],
        ignore_index=True,
    )


def _review_table(
    findings: pd.DataFrame,
    limit: int,
    source_record_keys: dict[str, str],
) -> pd.DataFrame:
    visible = findings.head(limit).copy()
    visible.insert(0, "queue_position", range(1, len(visible) + 1))
    visible["issue"] = visible["issue_type"].str.replace("_", " ").str.title()
    visible["record"] = (
        visible["target_table"].str.title() + " · " + visible["record_id"].astype(str)
    )
    columns = [
        "queue_position",
        "severity",
        "issue",
        "record",
    ]
    if source_record_keys:
        visible["source_key"] = visible["record_id"].map(source_record_keys).fillna("")
        columns.append("source_key")
    if visible["model_confidence"].notna().any():
        columns.append("model_confidence")
    if visible["risk_score"].nunique(dropna=True) > 1:
        columns.append("risk_score")
    return visible[columns]


def _simulated_feedback(findings: pd.DataFrame) -> list[dict[str, str | None]]:
    return [
        {
            "finding_id": str(finding["finding_id"]),
            "verdict": simulated_feedback_verdict(str(finding["issue_type"])),
            "notes": (
                "Simulated historical feedback for the user demonstration. "
                "This is not verified production ground truth."
            ),
        }
        for finding in findings.to_dict("records")
    ]


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
    if filtered.empty:
        st.info("No findings match the active filters.", icon=":material/filter_alt:")
        return
    default_limit = min(25, len(filtered)) if len(filtered) else 0
    review_limit = st.slider(
        "Visible review queue",
        min_value=1,
        max_value=max(1, len(filtered)),
        value=max(1, default_limit),
        help="Keep the user walkthrough focused while preserving the full run for export.",
        disabled=filtered.empty,
    )
    queue_mode = "Initial mixed queue"
    has_model_confidence = filtered["model_confidence"].notna().any()
    if has_model_confidence:
        queue_mode = st.segmented_control(
            "Queue view",
            ["Initial mixed queue", "Reranked by feedback"],
            default="Reranked by feedback",
        )

    ordered_findings = (
        _confidence_queue(filtered)
        if queue_mode == "Reranked by feedback"
        else _mixed_queue(filtered)
    )
    table_column, review_column = st.columns([3, 2], gap="large")
    with table_column:
        visible_findings = ordered_findings.head(review_limit).reset_index(drop=True)
        event = _render_queue_table(
            visible_findings,
            review_limit,
            st.session_state.source_record_keys,
            key=f"{queue_mode}-{run['run_id']}",
        )
    selected_rows = event.selection.rows
    selected = (
        visible_findings.iloc[selected_rows[0]] if selected_rows else visible_findings.iloc[0]
    )
    with review_column, st.container(border=True):
        st.badge(selected["severity"].title(), icon=":material/flag:")
        st.subheader(str(selected["issue_type"]).replace("_", " "))
        st.caption(f"{selected['target_table']} · {selected['record_id']}")
        source_key = st.session_state.source_record_keys.get(str(selected["record_id"]))
        if source_key:
            st.caption(f"Source key: {source_key}")
        st.write(selected["evidence"])
        st.caption(
            "Risk score reflects configured business severity. "
            "Feedback priority is learned from reviewer labels and only affects ordering."
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
        "Download filtered findings",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"datalens-{run['run_id']}-findings.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def _render_queue_table(
    findings: pd.DataFrame,
    limit: int,
    source_record_keys: dict[str, str],
    *,
    key: str,
):
    review_table = _review_table(findings, limit, source_record_keys)
    column_order = [
        column
        for column in [
            "queue_position",
            "severity",
            "issue",
            "record",
            "source_key",
            "model_confidence",
            "risk_score",
        ]
        if column in review_table.columns
    ]
    return st.dataframe(
        review_table,
        column_order=column_order,
        column_config={
            "queue_position": st.column_config.NumberColumn(
                "#",
                format="%d",
                width="small",
            ),
            "severity": st.column_config.TextColumn("Severity", pinned=True),
            "issue": st.column_config.TextColumn("Issue"),
            "record": st.column_config.TextColumn("Record"),
            "source_key": st.column_config.TextColumn("Source key"),
            "risk_score": st.column_config.NumberColumn(
                "Risk score",
                format="%.0f",
                help="Shown only when risk scores vary in the visible queue.",
            ),
            "model_confidence": st.column_config.ProgressColumn(
                "Feedback priority",
                min_value=0,
                max_value=1,
                format="percent",
                help=(
                    "Learned from reviewer feedback. "
                    "It does not override deterministic rule evidence."
                ),
            ),
        },
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=430,
        key=key,
    )


def _render_model_comparison() -> None:
    st.subheader("3. Compare active and candidate ranking")
    st.caption(
        "Manual retraining uses decisive persisted feedback. "
        "Simulated demo feedback remains labeled as simulated."
    )
    run = st.session_state.scoring_run
    findings = pd.DataFrame(run["findings"]) if run else pd.DataFrame()
    seed_disabled = findings.empty or st.session_state.simulated_feedback_seeded
    train_disabled = findings.empty

    with st.container(horizontal=True):
        seed_clicked = st.button(
            "Seed demo feedback",
            icon=":material/rate_review:",
            disabled=seed_disabled,
            help=(
                "Labels critical UEI and duplicate-key findings as correct, with "
                "non-critical demo issues as contrast examples."
            ),
        )
        train_clicked = st.button(
            "Train feedback reranker",
            icon=":material/model_training:",
            type="secondary",
            disabled=train_disabled,
        )
        reset_clicked = st.button(
            "Reset demo model",
            icon=":material/restart_alt:",
            help="Return future scoring runs to the deterministic baseline.",
        )

    if reset_clicked:
        with st.status("Resetting the active feedback reranker...", expanded=False):
            try:
                response = client.deactivate_active_reranker()
                st.session_state.scoring_run = None
                st.session_state.retraining = None
                st.session_state.feedback_count = 0
                st.session_state.simulated_feedback_seeded = False
                if response["deactivated"]:
                    st.write("Future runs will use the deterministic baseline.")
                else:
                    st.write("The deterministic baseline was already active.")
                st.rerun()
            except ApiError as error:
                st.error(str(error))

    if seed_clicked and run:
        with st.status("Saving simulated historical feedback...", expanded=False):
            try:
                response = client.submit_feedback_batch(
                    str(run["run_id"]),
                    _simulated_feedback(findings),
                )
                st.session_state.feedback_count += int(response["saved_feedback"])
                st.session_state.simulated_feedback_seeded = True
                st.write(f"Saved {response['saved_feedback']} simulated feedback labels.")
                st.write("The reranker can now be trained from the Streamlit demo.")
                st.rerun()
            except ApiError as error:
                st.error(str(error))

    if train_clicked:
        with st.status("Training and evaluating a feedback reranker...", expanded=False):
            try:
                st.session_state.retraining = client.retrain()
                st.write("The candidate was evaluated on held-out feedback.")
                st.write("Every deterministic finding remains in the review queue.")
            except ApiError as error:
                st.error(str(error))

    result = st.session_state.retraining
    if not result:
        st.caption("Seed demo feedback, then train the feedback reranker from this page.")
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


def _render_anomaly_models() -> None:
    st.subheader("4. Candidate anomaly models")
    summary = _load_anomaly_summary()
    if not summary:
        st.caption("No anomaly model comparison artifact is available.")
        return

    holdout = summary["temporal_holdout"]
    assert isinstance(holdout, dict)
    models = holdout["models"]
    assert isinstance(models, dict)
    baseline = holdout["baseline"]
    assert isinstance(baseline, dict)
    registry = summary.get("registry", {})
    candidates = registry.get("candidates", []) if isinstance(registry, dict) else []

    rows = [
        {
            "Model": "Deterministic rules",
            "Role": "Active detector",
            "Top-50 precision": baseline["top_50_precision"],
            "Recall": baseline["recall"],
            "False alarms per 1,000": baseline["false_alarms_per_1000_records"],
            "Status": "Active",
        }
    ]
    for model_name, metrics in models.items():
        candidate = next(
            (item for item in candidates if item.get("model_family") == model_name),
            {},
        )
        rows.append(
            {
                "Model": str(model_name).replace("_", " ").title(),
                "Role": "Supplemental anomaly evidence",
                "Top-50 precision": metrics["top_50_precision"],
                "Recall": metrics["recall"],
                "False alarms per 1,000": metrics["false_alarms_per_1000_records"],
                "Status": str(candidate.get("status", "candidate")).title(),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        column_config={
            "Top-50 precision": st.column_config.NumberColumn(format="percent"),
            "Recall": st.column_config.NumberColumn(format="percent"),
            "False alarms per 1,000": st.column_config.NumberColumn(format="%.2f"),
        },
        hide_index=True,
    )
    st.caption(
        "The current demo uses deterministic rules as the active detector because "
        "they performed better on the temporal holdout. The next planned modeling "
        "step is to use Isolation Forest and Local Outlier Factor as supplemental "
        "anomaly evidence for unusual vendors and transactions without mixing them "
        "into the feedback reranker."
    )


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
    _render_anomaly_models()
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
