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
    df["RandN"] = rng.normal(0, 1, n)
    df["RandU"] = rng.uniform(-np.sqrt(3), np.sqrt(3), n)
    lam = 0.08 * np.exp(0.01 * (df.BonusMalus - 80))
    df["ClaimNb"] = rng.poisson(lam * df.Exposure).astype(float)
    return df


def test_cann_starts_exactly_at_glm():
    df = _synthetic_policies()
    glm = fm.GLM().fit(df, {"formula": "ClaimNb ~ BonusMalusGLM + VehGas + DensityGLM"})
    cann = fm.CANN().fit(df, {"epochs": 0, "hidden": [5], "embedding_dims": [1, 1]}, base=glm)
    np.testing.assert_allclose(cann.predict(df), glm.predict(df), rtol=1e-5)


def test_localglmnet_starts_exactly_at_its_glm():
    """Zero attention weights and GLM-coefficient biases: the network equals its start GLM."""
    df = _synthetic_policies()
    m = fm.LocalGLMnet().fit(df, {"epochs": 0, "hidden": [5]})
    X = m.design(df)
    eta = m.glm_baseline["const"] + X[m.glm_baseline.index[1:]].to_numpy() @ m.glm_baseline.iloc[1:]
    np.testing.assert_allclose(m.predict(df), np.exp(eta) * df.Exposure, rtol=1e-4)
    assert len(m.features) == 3 + 3 + 7 + 2  # one-hot brands and regions, 7 scaled, 2 controls


def test_multi_seed_networks_report_each_run():
    df = _synthetic_policies(n=2000)
    m = fm.FNN().fit(df, {"epochs": 1, "hidden": [5], "embedding_dims": [1, 1], "n_runs": 3})
    runs = m.predict_runs(df)
    assert len(runs) == 3
    np.testing.assert_allclose(m.predict(df), np.mean(runs, axis=0))


def test_mlflow_ui_in_colab_links_the_proxy_url(monkeypatch):
    """In Colab the new-tab link is Colab's proxied URL, not a localhost address."""
    import sys
    import types

    from deep_actuarial_science import notebook

    calls, shown = [], []
    output = types.SimpleNamespace(
        eval_js=lambda js: calls.append(js) or "https://5000-abc.prod.colab.dev/",
        serve_kernel_port_as_iframe=lambda port, height: calls.append(("iframe", port)),
    )
    colab = types.ModuleType("google.colab")
    colab.output = output
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(notebook, "_port_open", lambda port: True)  # pretend MLflow runs
    monkeypatch.setattr("os.path.exists", lambda p: False)
    import IPython.display

    monkeypatch.setattr(IPython.display, "display", lambda obj: shown.append(obj.data))
    notebook.mlflow_ui(port=5000)
    assert calls[0] == "google.colab.kernel.proxyPort(5000)"
    assert 'href="https://5000-abc.prod.colab.dev/"' in shown[0]
    assert "localhost" not in shown[0]
    assert ("iframe", 5000) in calls


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


def test_wiki_has_colab_link_for_every_lesson(tmp_path):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("build_wiki", root / "scripts" / "build_wiki.py")
    bw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bw)
    assert bw.main(str(tmp_path)) == 0

    sidebar = (tmp_path / "_Sidebar.md").read_text()
    notebooks = (tmp_path / "Notebooks.md").read_text()
    for lesson in sorted((root / "lessons").glob("*.md")):
        url = bw.colab_url(lesson.read_text())
        assert url, f"{lesson.name} has no Colab badge"
        assert (root / url.split("/blob/main/")[1]).exists(), f"{url} points at a missing notebook"
        page = (tmp_path / lesson.name).read_text()
        assert page.count(f"[▶ Open in Colab]({url})") == 2  # nav bar top and bottom
        assert url in sidebar and url in notebooks
        assert "_class:" not in page
    # a [[Text|Page]] wiki link inside a table row splits the cell at its "|"
    import re

    for f in tmp_path.glob("*.md"):
        for line in f.read_text().splitlines():
            if line.startswith("|"):
                assert not re.search(r"\[\[[^\]]*\|[^\]]*\]\]", line), f"wiki link in {f.name}"
