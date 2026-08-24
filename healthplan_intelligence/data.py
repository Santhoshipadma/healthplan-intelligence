"""Create safe demo data for the HealthPlan Intelligence project.

The project intentionally starts with synthetic data. That keeps the repository
easy to run and avoids publishing anyone's private insurance information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "plan_type",
    "deductible",
    "primary_care_copay",
    "coinsurance_percent",
    "annual_premium",
    "hsa_eligible",
]


def generate_demo_data(rows: int = 700, seed: int = 42) -> pd.DataFrame:
    """Return reproducible, fictional health-plan examples.

    The relationships are deliberately understandable: larger deductibles,
    copays, and coinsurance tend to increase out-of-pocket exposure, while plan
    type and premium add realistic variation. Randomness prevents the model
    from behaving like a hard-coded calculator.
    """

    if rows < 50:
        raise ValueError("Please generate at least 50 rows for a useful demo model.")

    rng = np.random.default_rng(seed)
    plan_types = rng.choice(
        ["Bronze", "Silver", "Gold", "Platinum"],
        size=rows,
        p=[0.28, 0.42, 0.23, 0.07],
    )

    deductible_centers = {
        "Bronze": 6_200,
        "Silver": 4_200,
        "Gold": 2_000,
        "Platinum": 700,
    }
    premium_centers = {
        "Bronze": 330,
        "Silver": 460,
        "Gold": 610,
        "Platinum": 790,
    }

    deductibles = np.array(
        [rng.normal(deductible_centers[name], 950) for name in plan_types]
    ).clip(0, 9_500)
    monthly_premiums = np.array(
        [rng.normal(premium_centers[name], 85) for name in plan_types]
    ).clip(180, 1_100)

    copays = rng.normal(42, 17, rows).clip(0, 100)
    coinsurance = rng.normal(24, 9, rows).clip(0, 50)
    hsa_eligible = rng.binomial(1, 0.38, rows)

    plan_adjustment = pd.Series(plan_types).map(
        {"Bronze": 0.65, "Silver": 0.10, "Gold": -0.45, "Platinum": -0.75}
    ).to_numpy()

    risk_signal = (
        -3.15
        + deductibles / 2_200
        + copays / 85
        + coinsurance / 18
        - monthly_premiums / 1_100
        - hsa_eligible * 0.20
        + plan_adjustment
        + rng.normal(0, 0.48, rows)
    )
    probability = 1 / (1 + np.exp(-risk_signal))
    high_out_of_pocket = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "plan_type": plan_types,
            "deductible": deductibles.round().astype(int),
            "primary_care_copay": copays.round().astype(int),
            "coinsurance_percent": coinsurance.round().astype(int),
            "annual_premium": (monthly_premiums * 12).round().astype(int),
            "hsa_eligible": hsa_eligible.astype(bool),
            "high_out_of_pocket": high_out_of_pocket,
        }
    )
