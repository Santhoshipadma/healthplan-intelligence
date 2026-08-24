"""Translate a model result into language a person can quickly understand."""

from __future__ import annotations

from typing import Any


def result_summary(probability: float) -> tuple[str, str]:
    """Return a friendly label and tone for the interface."""

    if probability >= 0.70:
        return "Higher cost exposure", "high"
    if probability >= 0.40:
        return "Moderate cost exposure", "medium"
    return "Lower cost exposure", "low"


def explain_prediction(
    plan: dict[str, Any], probability: float, benchmarks: dict[str, float]
) -> list[str]:
    """Describe the strongest understandable factors behind a prediction.

    This is a transparent explanation layer, not a claim that every sentence is
    an exact causal interpretation of the random-forest model.
    """

    reasons: list[str] = []

    if plan["deductible"] > benchmarks["deductible"] * 1.20:
        reasons.append(
            "The deductible is well above the typical plan in this demo, so you may pay more before coverage contributes."
        )
    elif plan["deductible"] < benchmarks["deductible"] * 0.75:
        reasons.append(
            "The deductible is below the typical demo plan, which lowers your early cost exposure."
        )

    if plan["coinsurance_percent"] > benchmarks["coinsurance_percent"] + 6:
        reasons.append(
            "The coinsurance percentage is relatively high, meaning your share can remain significant after the deductible."
        )
    elif plan["coinsurance_percent"] < benchmarks["coinsurance_percent"] - 6:
        reasons.append(
            "The coinsurance percentage is relatively low, which may reduce your share of covered costs."
        )

    if plan["primary_care_copay"] > benchmarks["primary_care_copay"] + 15:
        reasons.append(
            "The primary-care copay is higher than the demo benchmark, increasing routine visit costs."
        )

    if plan["plan_type"] == "Bronze":
        reasons.append(
            "Bronze plans in the demo usually trade lower premiums for more cost exposure when care is used."
        )
    elif plan["plan_type"] in {"Gold", "Platinum"}:
        reasons.append(
            f"{plan['plan_type']} plans in the demo generally provide stronger cost protection when care is used."
        )

    if plan["hsa_eligible"]:
        reasons.append(
            "HSA eligibility can provide a tax-advantaged way to prepare for eligible expenses, although it does not lower the plan's limits."
        )

    if not reasons:
        reasons.append(
            "The plan is close to the demo's typical values, so no single input strongly dominates the result."
        )

    confidence_note = (
        "The model sees several signals pointing in the same direction."
        if probability >= 0.75 or probability <= 0.25
        else "The result is close enough that comparing the full plan documents is especially important."
    )
    return reasons[:3] + [confidence_note]
