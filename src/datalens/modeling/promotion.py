"""Typed model promotion criteria and decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from datalens.modeling.evaluation import RankingEvaluation


@dataclass(frozen=True)
class PromotionCriteria:
    minimum_top_k_ratio: float = 1.0
    minimum_macro_f1_ratio: float = 1.0
    maximum_false_alarm_increase_per_1000: float = 0.0
    required_guarded_high_critical_recall: float = 1.0


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    development_only_selection: bool
    top_k_precision_non_inferior: bool
    macro_f1_non_inferior: bool
    false_alarm_rate_non_inferior: bool
    deterministic_critical_findings_preserved: bool
    criteria: PromotionCriteria

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_promotion(
    candidate: RankingEvaluation,
    baseline: RankingEvaluation,
    guarded: RankingEvaluation,
    *,
    selection_period_role: str,
    criteria: PromotionCriteria | None = None,
) -> PromotionDecision:
    policy = criteria or PromotionCriteria()
    gates = {
        "development_only_selection": selection_period_role == "development",
        "top_k_precision_non_inferior": candidate.top_k_precision
        >= baseline.top_k_precision * policy.minimum_top_k_ratio,
        "macro_f1_non_inferior": candidate.macro_f1
        >= baseline.macro_f1 * policy.minimum_macro_f1_ratio,
        "false_alarm_rate_non_inferior": candidate.false_alarms_per_1000_records
        <= (baseline.false_alarms_per_1000_records + policy.maximum_false_alarm_increase_per_1000),
        "deterministic_critical_findings_preserved": guarded.high_critical_recall
        >= policy.required_guarded_high_critical_recall,
    }
    return PromotionDecision(
        promoted=all(gates.values()),
        criteria=policy,
        **gates,
    )
