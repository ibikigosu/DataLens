from datalens.modeling.evaluation import (
    ClassificationMetrics,
    RankingEvaluation,
    TableEvaluation,
)
from datalens.modeling.promotion import assess_promotion


def _metrics(
    *,
    top_k: float,
    macro_f1: float,
    false_alarms: float,
    high_critical_recall: float,
) -> RankingEvaluation:
    classification = ClassificationMetrics(1, 1, 1, 0.5, 0.5, 0.5)
    table = TableEvaluation(classification, top_k, false_alarms)
    return RankingEvaluation(
        evaluated_records=10,
        controlled_defect_records=2,
        predicted_records=2,
        classification=classification,
        macro_f1=macro_f1,
        high_critical_recall=high_critical_recall,
        top_k_precision=top_k,
        false_alarms_per_1000_records=false_alarms,
        per_table={"vendor": table, "transaction": table},
    )


def test_promotion_requires_non_inferiority_and_critical_preservation() -> None:
    decision = assess_promotion(
        _metrics(
            top_k=0.2,
            macro_f1=0.2,
            false_alarms=5.0,
            high_critical_recall=0.2,
        ),
        _metrics(
            top_k=0.8,
            macro_f1=0.8,
            false_alarms=3.0,
            high_critical_recall=1.0,
        ),
        _metrics(
            top_k=0.8,
            macro_f1=0.8,
            false_alarms=5.0,
            high_critical_recall=1.0,
        ),
        selection_period_role="development",
    )

    assert not decision.promoted
    assert decision.development_only_selection
    assert decision.deterministic_critical_findings_preserved
    assert not decision.top_k_precision_non_inferior
