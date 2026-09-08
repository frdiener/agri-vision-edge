"""
Benchmark artifact generation.
"""

from __future__ import annotations

from pathlib import Path

from .export import (
    save_json,
)
from .integrity import (
    prediction_integrity,
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

    # Record prediction integrity because pycocotools can treat non-finite boxes
    # as matches at every IoU threshold and report inflated AP.
    integrity = prediction_integrity(
        benchmark_result.predictions
    )

    if integrity.corrupt:
        print(
            "[warning] this run produced unusable predictions: "
            f"{integrity.describe()}. The runtime is broken, not the metrics; "
            "on the i.MX8MP this is what the Teflon/NPU delegate does to an "
            "fp32 graph -- re-run fp32 models with --cpu."
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

    # Record the effective delegate so silent CPU fallbacks remain distinguishable
    # from accelerated runs.
    active_delegate = getattr(runtime, "active_delegate", None)

    save_json(
        {
            "model":
                model_name,

            # Kept under the original key for backwards compatibility with
            # already-collected results.
            "delegate":
                delegate,

            "delegate_requested":
                delegate,

            "delegate_active":
                active_delegate,

            "backend":
                "delegate" if active_delegate else "cpu",

            # Which stage of the pipeline produced these predictions. The
            # SavedModel reference runs on the host and its latency is not
            # comparable to a device figure, so the report needs to tell the
            # rungs apart by more than the results directory name.
            "format":
                getattr(runtime, "runtime_format", "tflite"),

            "predictions_integrity":
                integrity.to_dict(),

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
