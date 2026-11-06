"""Fast unit tests: metrics, the CANN zero-initialisation property, chain-ladder vs package."""

import numpy as np
import pandas as pd
import pytest

from deep_actuarial_science.metrics import balance_ratio, gini_index, poisson_deviance
from deep_actuarial_science.models import frequency as fm
from deep_actuarial_science.models import reserving as rm


def test_poisson_deviance_zero_for_perfect_fit():
    y = np.array([0.0, 1, 2, 0])
    v = np.array([1.0, 0.5, 1, 0.2])
    assert poisson_deviance(y, np.maximum(y, 1e-12), v) == pytest.approx(0, abs=1e-6)


def test_balance_and_gini():
    y, mu, v = np.array([0, 1, 3.0]), np.array([0.5, 1.0, 2.5]), np.ones(3)
    assert balance_ratio(y, mu) == pytest.approx(1.0)
    assert gini_index(y, mu, v) > 0


def _synthetic_policies(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Exposure": rng.uniform(0.1, 1, n),
            "Area": pd.Categorical(rng.choice(list("ABCDEF"), n)),
            "VehPower": rng.integers(4, 15, n),
            "VehAge": rng.integers(0, 20, n),
            "DrivAge": rng.integers(18, 90, n),
            "BonusMalus": rng.integers(50, 150, n),
            "VehBrand": pd.Categorical(rng.choice(["B1", "B2", "B3"], n)),
            "VehGas": pd.Categorical(rng.choice(["Diesel", "Regular"], n)),
            "Density": rng.integers(1, 10000, n),
            "Region": pd.Categorical(rng.choice(["R11", "R24", "R82"], n)),
        }
    )
    lam = 0.08 * np.exp(0.01 * (df.BonusMalus - 80))
    df["ClaimNb"] = rng.poisson(lam * df.Exposure).astype(float)
    return df


def test_cann_starts_exactly_at_glm():
    df = _synthetic_policies()
    glm = fm.GLM().fit(df, {"formula": "ClaimNb ~ BonusMalusGLM + VehGas + DensityGLM"})
    cann = fm.CANN().fit(df, {"epochs": 0, "hidden": [5], "embedding_dims": [1, 1]}, base=glm)
    np.testing.assert_allclose(cann.predict(df), glm.predict(df), rtol=1e-5)


def _triangles(n_comp=3, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    pattern = np.array([0.4, 0.25, 0.12, 0.08, 0.05, 0.04, 0.03, 0.015, 0.01, 0.005])
    for c in range(n_comp):
        for ay in range(1988, 1998):
            prem = 1000 * (1 + c)
            incr = prem * 0.7 * pattern * rng.lognormal(0, 0.1, 10)
            for j, cum in enumerate(np.cumsum(incr), start=1):
                rows.append(
                    {
                        "lob": "test",
                        "company": c,
                        "GRNAME": f"c{c}",
                        "ay": ay,
                        "dev": j,
                        "cum_paid": cum,
                        "premium": prem,
                    }
                )
    d = pd.DataFrame(rows)
    d["incr_paid"] = d.groupby(["company", "ay"]).cum_paid.diff().fillna(d.cum_paid)
    d["calendar"] = d.ay + d.dev - 1
    d["upper"] = d.calendar <= rm.LAST_YEAR
    return d


def test_chain_ladder_matches_package():
    from deep_actuarial_science.pipelines.reserving import package_chain_ladder

    tri = _triangles()
    pred, _ = rm.chain_ladder(tri)
    ours = rm.reserves(pred).set_index(rm.KEY).predicted_reserve
    pkg = package_chain_ladder(tri).reindex(ours.index)
    np.testing.assert_allclose(ours, pkg, rtol=1e-9)


def test_odp_fitted_values_preserve_row_sums():
    tri = _triangles(1)
    up = tri[tri.upper]
    mu = rm.chain_ladder_fitted_upper(tri)
    np.testing.assert_allclose(mu.groupby(up.ay).sum(), up.groupby("ay").incr_paid.sum(), rtol=1e-9)
