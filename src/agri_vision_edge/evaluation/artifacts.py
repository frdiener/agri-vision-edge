"""
Benchmark artifact generation.
"""

from __future__ import annotations

from pathlib import Path

from .export import (
    save_json,
)


def save_benchmark_artifacts(
    *,
    output_dir,
    benchmark_result,
    runtime,
    model_name,
    delegate,
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        benchmark_result.predictions,
        output_dir /
        "predictions.json",
    )

    from agri_vision_edge.evaluation.benchmark import latency_summary

    save_json(
        latency_summary(
            benchmark_result
            .latencies_ms
        ),
        output_dir /
        "latency.json",
    )

    save_json(
        {
            "model":
                model_name,

            "delegate":
                delegate,

            "input_details":
                runtime.input_details,

            "output_details":
                runtime.output_details,
        },
        output_dir /
        "runtime.json",
    )


def save_failure_artifact(
    *,
    output_dir,
    exception,
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        {
            "status": "failed",
            "exception":
                type(exception).__name__,
            "message":
                str(exception),
        },
        output_dir / "error.json",
    )
