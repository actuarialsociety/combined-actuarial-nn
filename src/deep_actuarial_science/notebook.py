"""Run and inspect the Kedro pipelines from a notebook (local, container or Colab).

    from deep_actuarial_science.notebook import run, load, show, reconciliation, findings
    run("frequency", models=["glm", "cann"], params={"frequency.data.scale": 0.1})
    reconciliation("frequency", ["glm", "cann"])   # paper | reference run | this run
    findings("frequency", ["glm", "cann"])         # which gaps are real, vs the reference run
    show("frequency", "cann")

"Reference run" is the frozen full run committed in `reference/` (catalog: `reference_*`).

Each call opens a fresh KedroSession (a session can only run once), so notebooks can
re-run with different parameters. Every run is tracked in MLflow like `kedro run`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

ROOT = Path(__file__).resolve().parents[2]
PREP_OUTPUTS = {
    "frequency": ["freq_learn", "freq_test"],
    "mortality": ["mort_data"],
    "reserving": ["res_triangles"],
}
METRICS = {"frequency": "freq_metrics", "mortality": "mort_metrics", "reserving": "res_metrics"}
PREFIX = {"frequency": "freq", "mortality": "mort", "reserving": "res"}  # dataset names
OUTPUT = {"frequency": "predictions", "mortality": "results", "reserving": "predictions"}


def _env(env: str | None) -> str:
    # DAS_ENV=smoke makes every notebook run small and fast (used by CI).
    return env or os.environ.get("DAS_ENV", "local")


def _session(env, params):
    bootstrap_project(ROOT)
    return KedroSession.create(project_path=ROOT, env=_env(env), runtime_params=params or {})


def run(
    pipeline: str,
    models: list[str] | None = None,
    params: dict | None = None,
    env: str | None = None,
) -> None:
    """Run a pipeline. With `models`, prepare the data, then run only those namespaces."""
    os.chdir(ROOT)
    if models:
        with _session(env, params) as s:
            s.run(pipeline_name=pipeline, to_outputs=PREP_OUTPUTS[pipeline])
        with _session(env, params) as s:
            s.run(pipeline_name=pipeline, namespaces=models)
    else:
        with _session(env, params) as s:
            s.run(pipeline_name=pipeline)


def catalog(env: str | None = None, params: dict | None = None):
    os.chdir(ROOT)
    with _session(env, params) as s:
        return s.load_context().catalog


def load(name: str, env: str | None = None):
    return catalog(env).load(name)


def parameters(env: str | None = None, params: dict | None = None) -> dict:
    os.chdir(ROOT)
    with _session(env, params) as s:
        return s.load_context().params


def leaderboard(pipeline: str, models: list[str], env: str | None = None) -> pd.DataFrame:
    """Metrics of the given models side by side, with the benchmark status."""
    cat = catalog(env)
    rows = []
    for m in models:
        row = {"model": m, **cat.load(f"{m}.{METRICS[pipeline]}")}
        rec_name = f"{m}.{METRICS[pipeline].replace('metrics', 'reconciliation')}"
        try:
            rec = cat.load(rec_name)
            row["benchmark"] = ", ".join(sorted(set(rec.status))) if len(rec) else "none"
        except Exception:  # noqa: BLE001 - reconciliation is optional
            row["benchmark"] = "n/a"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def _models(pipeline: str, models: list[str] | None) -> list[str]:
    if models:
        return models
    from importlib import import_module

    return list(import_module(f"deep_actuarial_science.pipelines.{pipeline}").MODELS)


def reconciliation(
    pipeline: str, models: list[str] | None = None, env: str | None = None
) -> pd.DataFrame:
    """Each benchmarked metric three ways: the paper's table, the frozen reference run
    (`reference/reconciliation.csv`) and this run. `status` is this run against the paper;
    away from the reference setup (smaller data, fewer epochs) it is SKIPPED."""
    cat = catalog(env)
    rec = pd.concat(
        [cat.load(f"{m}.{PREFIX[pipeline]}_reconciliation") for m in _models(pipeline, models)]
    )
    ref = cat.load("reference_reconciliation").set_index(["model", "metric"]).actual
    out = rec.join(ref.rename("reference_run"), on=["model", "metric"])
    out = out.rename(columns={"expected": "paper", "actual": "this_run"})
    cols = ["model", "metric", "paper", "reference_run", "this_run", "tolerance", "status"]
    return out[cols].reset_index(drop=True)


def findings(
    pipeline: str, models: list[str] | None = None, env: str | None = None
) -> pd.DataFrame:
    """This run's findings for `models` (default: all), with the frozen reference run's in
    the `*_reference` columns (see `deep_actuarial_science.findings`).

    - frequency, reserving: each step down the ranking, `worse` to `better`, with the gap in
      loss, its 95% interval over test policies (companies) and the verdict; then what
      averaging a network's seeds adds (`<model>_xN`).
    - mortality: where the published single run falls among our seeds, and how many seeds
      beat Lee-Carter.
    """
    from deep_actuarial_science import findings as f

    cat, ctx = catalog(env), parameters(env)
    models = _models(pipeline, models)
    data = {m: cat.load(f"{m}.{PREFIX[pipeline]}_{OUTPUT[pipeline]}") for m in models}
    ref = cat.load(f"reference_{pipeline}_findings")
    if pipeline == "mortality":
        from deep_actuarial_science.pipelines.mortality import published_rmse

        published = published_rmse(
            ctx["benchmarks"]["mortality"], ctx["mortality"]["data"], ctx["reference_setup"]
        )
        table = f.mortality(data, published)[0].set_index("model")
        cols = ["seed_mean", "seeds_at_or_below_published", "seeds_beating_lee_carter"]
        return table.join(ref.set_index("model")[cols].add_suffix("_reference"))
    table = f.summary(f.frequency(data) if pipeline == "frequency" else f.reserving(data))
    ref = ref.set_index(["worse", "better"])[["gap", "verdict"]].add_suffix("_reference")
    cols = ["worse", "better", "gap", "ci_low", "ci_high", "verdict"]
    return table[cols].join(ref, on=["worse", "better"])


def figure_dir(pipeline: str, model: str | None = None) -> Path:
    base = ROOT / "data" / "08_reporting" / pipeline
    return base / "comparison" if model is None else base / model / "figures"


def platform() -> str:
    """Where this notebook runs: colab, kaggle, sagemaker, azureml, docker or local."""
    import sys

    if "google.colab" in sys.modules:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    home = os.path.expanduser("~")
    if (
        any(
            os.path.isdir(d)
            for d in ("/home/studio-lab-user", "/home/sagemaker-user", "/home/ec2-user/SageMaker")
        )
        or "SAGEMAKER_SPACE_NAME" in os.environ
    ):
        return "sagemaker"
    if os.path.isdir("/mnt/batch/tasks/shared/LS_root") or os.path.isdir(f"{home}/cloudfiles"):
        return "azureml"  # Azure ML compute instance (cloudfiles = the workspace file share)
    if os.path.exists("/.dockerenv"):
        return "docker"
    return "local"


def device_summary() -> None:
    """Print the platform and the device the networks will train on."""
    import torch

    from deep_actuarial_science.models.torch_utils import device

    dev = device()
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else None
    print(f"Platform: {platform()} · training device: {dev}" + (f" ({gpu})" if gpu else ""))
    if dev.type == "cpu" and torch.backends.mps.is_available():
        print("Apple GPU available: set DAS_DEVICE=mps to train on it (opt-in, see README).")
    elif dev.type == "cpu":
        print("No GPU: the full frequency networks are slow on a CPU; FAST/smoke runs are fine.")


# Hosted Jupyter (SageMaker, Azure ML) reaches a kernel-side port through
# jupyter-server-proxy at <jupyter base url>/proxy/<port>/. The link is built in the browser,
# because only the page knows the public host and base URL.
PROXY_LINK_JS = """
const cfg = document.getElementById('jupyter-config-data');
let base = cfg ? JSON.parse(cfg.textContent).baseUrl : location.pathname.split('/lab')[0];
if (!base.endsWith('/')) base += '/';
const url = location.origin + base + 'proxy/PORT/';
element.innerHTML = '<a href="' + url + '" target="_blank" rel="noopener">Open MLflow in a ' +
  'new tab</a> <small>(' + url + '; needs jupyter-server-proxy, preinstalled on SageMaker)</small>';
"""


# In Colab, MLflow listens on 127.0.0.1 of a disposable VM that only Colab's authenticated
# proxy can reach, and the proxy's domain is Colab's to choose. MLflow 3 rejects unknown Host
# headers, cross-origin writes and framing, so behind a hosted proxy (Colab, SageMaker,
# Azure ML) only those checks are opened up. Local and container runs keep MLflow's defaults.
COLAB_MLFLOW_FLAGS = [
    "--allowed-hosts",
    "*",
    "--cors-allowed-origins",
    "*",
    "--x-frame-options",
    "NONE",
]


def _port_open(port: int) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def mlflow_ui(port: int = 5000, height: int = 720, timeout: float = 120) -> None:
    """Open the MLflow UI for this project's runs.

    - Colab: MLflow runs on the Colab machine, so a localhost link would point at *your*
      computer ("site can't be reached"). The UI is started there and opened through
      Colab's proxy: a new-tab link to the URL from `google.colab.kernel.proxyPort`, plus
      the UI inline.
    - SageMaker / Azure ML: the UI is started on the notebook machine and linked through
      Jupyter's proxy (`<jupyter base url>/proxy/<port>/`).
    - Kaggle exposes no ports: query runs with `mlflow.search_runs` instead.
    - Container (`docker compose up lab mlflow`): the `mlflow` service already serves it.
    - Local: starts `mlflow ui` in the background and links to it.
    """
    import subprocess
    import sys
    import time

    from IPython.display import HTML, Javascript, display

    os.chdir(ROOT)
    where = platform()
    colab = where == "colab"
    proxied = where in ("colab", "sagemaker", "azureml")
    if where == "kaggle":
        display(
            HTML(
                "Kaggle exposes no ports, so the MLflow UI cannot open here. Query the runs:"
                ' <code>mlflow.set_tracking_uri("sqlite:///mlflow.db");'
                " mlflow.search_runs()</code>, or download <code>mlflow.db</code> and run"
                " <code>mlflow ui</code> locally."
            )
        )
        return
    if os.path.exists("/.dockerenv") and not colab:
        display(
            HTML(
                f'<a href="http://localhost:{port}" target="_blank">Open MLflow</a> '
                "(served by the <code>mlflow</code> container)"
            )
        )
        return
    if not _port_open(port):
        uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        cmd = [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            uri,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if proxied:
            cmd += COLAB_MLFLOW_FLAGS
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
        )
        deadline = time.time() + timeout
        while not _port_open(port):
            if time.time() > deadline:
                raise RuntimeError(f"MLflow UI did not start on port {port}")
            time.sleep(1)
    if colab:
        from google.colab import output

        # The proxied URL works in a normal tab; serve_kernel_port_as_window's link only
        # resolves inside Colab's output frame.
        url = output.eval_js(f"google.colab.kernel.proxyPort({port})")
        display(
            HTML(
                f'<a href="{url}" target="_blank" rel="noopener">Open MLflow in a new tab</a>'
                f" <small>({url})</small>"
            )
        )
        output.serve_kernel_port_as_iframe(port, height=height)
    elif proxied:
        display(Javascript(PROXY_LINK_JS.replace("PORT", str(port))))
    else:
        display(
            HTML(
                f'<a href="http://localhost:{port}" target="_blank">'
                f"Open MLflow (http://localhost:{port})</a>"
            )
        )


def show(pipeline: str, model: str | None = None, names: list[str] | None = None, width=560):
    """Display saved figures inline (works in Jupyter and Colab)."""
    from IPython.display import Image, Markdown, display

    for p in sorted(figure_dir(pipeline, model).glob("*.png")):
        if names is None or p.stem in names:
            display(Markdown(f"**{p.stem}**"), Image(filename=str(p), width=width))
