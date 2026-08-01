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

    # Same reasoning as `delegate_active` below: a run whose runtime produced
    # garbage must not be indistinguishable from a good one in the results
    # tree. Non-finite boxes in particular do NOT make the evaluator fail --
    # pycocotools scores them as a match at every IoU threshold and reports an
    # inflated AP -- so record the counters here, where the predictions were
    # made, and say so on the console while the operator is still watching.
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

    # `delegate` is only what was requested. A delegate that is missing or
    # fails to load falls back to CPU silently, so record what the runtime
    # actually used as well -- otherwise a CPU run is indistinguishable from an
    # accelerated one in the results, and every latency comparison built on
    # them is wrong.
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
