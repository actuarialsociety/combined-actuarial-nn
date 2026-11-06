"""Pipelines: `ingest`, `frequency`, `mortality`, `reserving`.

Each model is a namespace inside its pipeline, so you can run one model:

    kedro run --pipeline frequency --namespaces glm,cann
"""

from kedro.pipeline import Pipeline

from deep_actuarial_science.pipelines import frequency, ingest, mortality, reserving


def register_pipelines() -> dict[str, Pipeline]:
    pipes = {
        "ingest": ingest.create_pipeline(),
        "frequency": frequency.create_pipeline(),
        "mortality": mortality.create_pipeline(),
        "reserving": reserving.create_pipeline(),
    }
    pipes["__default__"] = pipes["frequency"] + pipes["mortality"] + pipes["reserving"]
    return pipes
