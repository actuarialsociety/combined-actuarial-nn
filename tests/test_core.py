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


LOCAL_NOTEBOOKS = {"run_all_mps.ipynb"}  # Apple silicon only: no cloud launchers


def test_run_all_mps_notebook_trains_on_mps_and_runs_every_pipeline():
    import json
    from pathlib import Path

    nb = json.loads(
        (Path(__file__).resolve().parent.parent / "notebooks" / "run_all_mps.ipynb").read_text()
    )
    assert "colab" not in nb["metadata"] and "kaggle" not in nb["metadata"]
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'os.environ["DAS_DEVICE"] = "mps"' in code
    assert "torch.backends.mps.is_available()" in code  # falls back off a Mac (CI)
    assert '("frequency", "mortality", "reserving")' in code
    assert "reconciliation.csv" in code


def test_colab_notebooks_force_a_gpu_runtime():
    """Every notebook requests a T4 GPU in its Colab metadata and stops on a CPU runtime."""
    import json
    from pathlib import Path

    notebooks = sorted((Path(__file__).resolve().parent.parent / "notebooks").glob("*.ipynb"))
    assert notebooks
    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text())
        if nb_path.name in LOCAL_NOTEBOOKS:
            continue
        assert nb["metadata"]["accelerator"] == "GPU", nb_path.name
        assert nb["metadata"]["colab"]["gpuType"] == "T4", nb_path.name
        setup = "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])
        guard, clone = setup.find("torch.cuda.is_available()"), setup.find("git clone")
        assert 0 <= guard < clone, f"{nb_path.name}: GPU check must run before cloning"


def test_device_honours_das_device(monkeypatch):
    import torch

    from deep_actuarial_science.models.torch_utils import device

    monkeypatch.setenv("DAS_DEVICE", "cpu")
    assert device().type == "cpu"
    monkeypatch.setenv("DAS_DEVICE", "mps")
    if torch.backends.mps.is_available():
        assert device().type == "mps"
    else:
        with pytest.raises(RuntimeError, match="no MPS"):
            device()
    monkeypatch.delenv("DAS_DEVICE")
    assert device().type == ("cuda" if torch.cuda.is_available() else "cpu")  # MPS is opt-in


def test_notebooks_launch_anywhere():
    """Every notebook links Colab, Kaggle and SageMaker Studio, asks Kaggle for a GPU and
    internet, and its setup cell clones the repo only when it is not inside a checkout."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for nb_path in sorted((root / "notebooks").glob("*.ipynb")):
        if nb_path.name in LOCAL_NOTEBOOKS:
            continue
        nb = json.loads(nb_path.read_text())
        assert nb["metadata"]["kaggle"]["isGpuEnabled"], nb_path.name
        assert nb["metadata"]["kaggle"]["isInternetEnabled"], nb_path.name
        header = "".join(nb["cells"][0]["source"])
        gh = f"github/actuarialsociety/deep-actuarial-science/blob/main/notebooks/{nb_path.name}"
        assert f"https://colab.research.google.com/{gh}" in header
        assert f"https://kaggle.com/kernels/welcome?src=https://github.com/{gh[7:]}" in header
        assert "https://console.aws.amazon.com/sagemaker/home#/studio-landing" in header
        assert "studiolab" not in header  # Studio Lab is closed to new customers
        setup = "".join(nb["cells"][1]["source"])
        assert "if REPO_DIR is None:" in setup and setup.index("REPO_DIR is None") < setup.index(
            "git clone"
        )


def test_platform_detection(monkeypatch):
    import sys

    from deep_actuarial_science import notebook

    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert notebook.platform() == "kaggle"
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE")
    monkeypatch.setenv("SAGEMAKER_SPACE_NAME", "my-space")
    assert notebook.platform() == "sagemaker"


def test_mlflow_ui_on_kaggle_explains_instead_of_linking(monkeypatch):
    import IPython.display

    from deep_actuarial_science import notebook

    shown = []
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setattr(IPython.display, "display", lambda obj: shown.append(obj.data))
    monkeypatch.setattr(notebook, "_port_open", lambda port: pytest.fail("must not start MLflow"))
    notebook.mlflow_ui()
    assert "search_runs" in shown[0]


def test_wandb_hook_logs_metrics_figures_and_tables(monkeypatch):
    """With WANDB_PROJECT set, a pipeline run is one W&B run with config, metrics, figures and
    benchmark tables; without it the hook does nothing."""
    import sys
    import types

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from deep_actuarial_science.wandb_hook import WandbHook

    calls = {"log": [], "summary": {}}
    run = types.SimpleNamespace(
        log=calls["log"].append,
        summary=types.SimpleNamespace(update=calls["summary"].update),
        finish=lambda exit_code=0: calls.setdefault("finish", exit_code),
    )
    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: calls.setdefault("init", kw) and run
    fake.Image = lambda fig: ("image", fig)
    fake.Table = lambda dataframe: ("table", len(dataframe))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    catalog = types.SimpleNamespace(
        load=lambda name: {"reserving": {"data": {"lines": ["ppauto"]}}}
    )
    run_params = {"pipeline_name": "reserving", "env": "smoke"}

    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    hook = WandbHook()
    hook.before_pipeline_run(run_params, None, catalog)
    assert hook.run is None and "init" not in calls

    monkeypatch.setenv("WANDB_PROJECT", "test")
    monkeypatch.setenv("DAS_DEVICE", "cpu")
    hook.before_pipeline_run(run_params, None, catalog)
    assert calls["init"]["config"]["data.lines"] == ["ppauto"]
    assert "device:cpu" in calls["init"]["tags"] and calls["init"]["job_type"] == "reserving"
    hook.after_dataset_saved("neural_odp.res_metrics", {"mape_ultimate": 0.04, "ok": True}, None)
    hook.after_dataset_saved("neural_odp.res_figures", {"fit": plt.figure()}, None)
    hook.after_dataset_saved("neural_odp.res_reconciliation", pd.DataFrame({"a": [1, 2]}), None)
    hook.after_pipeline_run()
    assert calls["summary"] == {"neural_odp/mape_ultimate": 0.04}
    assert calls["log"][1]["neural_odp.res_figures/fit"][0] == "image"
    assert calls["log"][2] == {"neural_odp.res_reconciliation": ("table", 2)}
    assert calls["finish"] == 0 and hook.run is None


def test_reserving_reconciliation_checks_the_models_own_metrics():
    """Chain-ladder is reconciled on its MAPE / RMSPE (Kuo 2019, Table 1), not only on the
    package cross-check: before the fix both metrics always came out MISSING."""
    from deep_actuarial_science.pipelines.reserving import reconcile

    tri = _triangles()
    pred, _ = rm.chain_ladder(tri)
    bench = [
        {"source": "paper", "tolerance": 0.01, "expected": {"mape_ultimate": 0.0}},
        {"source": "package", "tolerance": 1e-6, "expected": {"max_rel_diff_vs_package": 0.0}},
    ]
    rec = reconcile({"mape_ultimate": 0.005}, pred, tri, bench, {"kind": "chain_ladder"})
    assert dict(zip(rec.metric, rec.status)) == {
        "mape_ultimate": "PASS",
        "max_rel_diff_vs_package": "PASS",
    }


def test_networks_use_keras_initialisation():
    """The benchmark papers train in Keras: Glorot kernels, zero biases, uniform(+-0.05)
    embeddings, an orthogonal LSTM recurrent kernel and forget-gate bias 1."""
    import torch

    from deep_actuarial_science.models import mortality as mm
    from deep_actuarial_science.models.torch_utils import keras_init

    net = fm.EmbeddingFNN(7, [11, 22], [2, 2], [20, 15, 10], bias0=-2.0)
    assert all(e.weight.abs().max() <= 0.05 for e in net.emb)
    first = net.hidden[0]
    assert first.weight.abs().max() <= (6 / (first.in_features + first.out_features)) ** 0.5
    assert torch.all(first.bias == 0) and torch.all(net.out.weight == 0)  # zero-output start
    assert net.out.bias.item() == -2.0

    lstm = keras_init(mm.LSTM(10, 90, q=20, units=20)).lstm
    bias = lstm.gates.bias.view(4, 20)
    assert torch.all(bias[1] == 1) and torch.all(bias[[0, 2, 3]] == 0)
    rec = lstm.gates.weight[:, 20:]  # (4 * units) x units: orthonormal columns
    torch.testing.assert_close(rec.T @ rec, torch.eye(20), atol=1e-5, rtol=0)


def test_single_published_run_is_checked_against_the_seed_distribution():
    from deep_actuarial_science.benchmarks import reconcile

    bench = {"source": "Ch. 8", "tolerance": "2sd", "expected": {"rmse_out": 0.3531}}
    ok = reconcile({"rmse_out": 0.40, "rmse_out_sd": 0.03}, bench, True, "cnn")
    far = reconcile({"rmse_out": 0.50, "rmse_out_sd": 0.03}, bench, True, "cnn")
    single = reconcile({"rmse_out": 0.36}, bench, True, "cnn")  # no sd: one seed only
    assert ok.status[0] == "PASS" and ok.tolerance[0] == pytest.approx(0.06)
    assert far.status[0] == "FAIL" and single.status[0] == "MISSING"
