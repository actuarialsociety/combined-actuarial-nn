"""Run and inspect the Kedro pipelines from a notebook (local, container or Colab).

    from deep_actuarial_science.notebook import run, load, show, leaderboard
    run("frequency", models=["glm", "cann"], params={"frequency.data.scale": 0.1})
    leaderboard("frequency", ["glm", "cann"])
    show("frequency", "cann")

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


def figure_dir(pipeline: str, model: str | None = None) -> Path:
    base = ROOT / "data" / "08_reporting" / pipeline
    return base / "comparison" if model is None else base / model / "figures"


# Colab serves kernel ports through its own proxy domains. MLflow (3.x) rejects unknown Host
# headers and cross-origin writes, and forbids framing, so allow exactly those domains.
COLAB_HOSTS = "*.colab.dev,*.colab.dev:*,*.googleusercontent.com,*.googleusercontent.com:*"
COLAB_ORIGINS = (
    "https://*.colab.dev,https://*.googleusercontent.com,https://colab.research.google.com"
)


def _port_open(port: int) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def mlflow_ui(port: int = 5000, height: int = 720, timeout: float = 120) -> None:
    """Open the MLflow UI for this project's runs.

    - Colab: MLflow runs on the Colab machine, so a localhost link would point at *your*
      computer ("site can't be reached"). The UI is started there and opened through
      Colab's proxy: inline, plus a link that opens it in a new tab.
    - Container (`docker compose up lab mlflow`): the `mlflow` service already serves it.
    - Local: starts `mlflow ui` in the background and links to it.
    """
    import subprocess
    import sys
    import time

    from IPython.display import HTML, display

    os.chdir(ROOT)
    colab = "google.colab" in sys.modules
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
        if colab:
            cmd += [
                "--allowed-hosts",
                f"{COLAB_HOSTS},localhost:*,127.0.0.1:*",
                "--cors-allowed-origins",
                COLAB_ORIGINS,
                "--x-frame-options",
                "NONE",
            ]
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

        output.serve_kernel_port_as_window(port, anchor_text="Open MLflow in a new tab")
        output.serve_kernel_port_as_iframe(port, height=height)
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
