from healthplan_intelligence import (
    explain_prediction,
    generate_demo_data,
    predict_high_cost_risk,
    train_model,
)


SAMPLE_PLAN = {
    "plan_type": "Silver",
    "deductible": 4_500,
    "primary_care_copay": 40,
    "coinsurance_percent": 25,
    "annual_premium": 5_400,
    "hsa_eligible": False,
}


def test_demo_data_has_both_outcomes():
    data = generate_demo_data(rows=300)

    assert len(data) == 300
    assert set(data["high_out_of_pocket"].unique()) == {0, 1}


def test_prediction_is_a_probability():
    data = generate_demo_data(rows=300)
    model, _, _ = train_model(data)

    probability = predict_high_cost_risk(model, SAMPLE_PLAN)

    assert 0.0 <= probability <= 1.0


def test_explanation_uses_plain_language():
    benchmarks = {
        "deductible": 3_500,
        "primary_care_copay": 35,
        "coinsurance_percent": 20,
        "annual_premium": 5_400,
    }

    reasons = explain_prediction(SAMPLE_PLAN, 0.68, benchmarks)

    assert reasons
    assert all(isinstance(reason, str) for reason in reasons)
