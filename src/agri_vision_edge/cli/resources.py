"""
Sustained-load resource measurement.

An additional measurement path not part of ``ave benchmark``.

``ave resources`` asks **what does this model cost to run**.

Images are decoded **once** into a small in-RAM pool and cycled.

The run is bracketed like this::

    [load + warmup]  chirp  ..gap..  MEASURED LOOP  ..gap..  chirp

The gaps keep the synchronisation chirp out of the
measured window and are themselves idle stretches immediately either side
of the load, which is the local baseline the power analysis subtracts.

Outputs, per run directory:

``resources.csv.gz``    periodic CPU / memory / thermal / frequency samples
``iterations.csv.gz``   per-inference epoch timestamps, latency, phase split,
                        detection count
``run.json``            config, delegate state, phase boundaries, latency
                        breakdown, clock anchors
``resources_meta.json`` sampler health, peak RSS, clock anchors
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def env_var(value: str) -> tuple[str, str]:
    try:
        key, val = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}") from exc

    if not key:
        raise argparse.ArgumentTypeError("environment-variable name cannot be empty")

    return key, val


def collect_images(path: Path, pool_size: int) -> list[Path]:
    """
    Pick a deterministic, evenly-spread pool of images from `path`.
    """

    if path.is_file():
        return [path]

    candidates = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    if not candidates:
        raise SystemExit(f"no images found in {path}")

    if pool_size <= 0 or pool_size >= len(candidates):
        return candidates

    step = len(candidates) / pool_size

    return [candidates[int(index * step)] for index in range(pool_size)]


def load_pool(paths: list[Path]) -> list[Any]:
    import cv2

    pool = []

    for path in paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"[warning] failed to read {path}")
            continue

        pool.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not pool:
        raise SystemExit("no images could be decoded")

    return pool


def check_output(detections) -> dict[str, Any]:
    """
    Is this runtime actually computing anything?
    """

    from agri_vision_edge.evaluation.integrity import prediction_integrity

    integrity = prediction_integrity(
        [{"bbox": d.bbox, "score": d.score} for d in detections]
    )

    result = integrity.to_dict()
    result.pop("degenerate_boxes", None)

    return result


def latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {}

    ordered = sorted(latencies_ms)
    count = len(ordered)

    def percentile(fraction: float) -> float:
        return ordered[min(count - 1, int(fraction * count))]

    mean = statistics.mean(ordered)

    return {
        "count": count,
        "mean_latency_ms": mean,
        "median_latency_ms": statistics.median(ordered),
        "min_latency_ms": ordered[0],
        "max_latency_ms": ordered[-1],
        "p95_latency_ms": percentile(0.95),
        "p99_latency_ms": percentile(0.99),
        "stdev_latency_ms": statistics.stdev(ordered) if count > 1 else 0.0,
        "throughput_fps": 1000.0 / mean if mean > 0 else 0.0,
    }


#: Phases a runtime may report, in execution order. ``postprocess`` is derived
#: rather than measured -- it is whatever the call spent outside the three
#: timed regions, which is exactly the detection decode -- and ``resize`` is a
#: *part of* ``preprocess``, not a sibling of it. Both facts matter when
#: reading the summary: the four do not sum to the total.
PHASES = ("resize", "preprocess", "invoke", "postprocess")


def write_iterations(path: Path, rows: list[tuple[Any, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(str(path), "wt", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        # Phase columns are appended, never inserted: the first six are what
        # every previously collected sweep has, and power_report.py reads them
        # positionally.
        writer.writerow(
            (
                "iteration",
                "image_index",
                "t_start_s",
                "t_end_s",
                "latency_ms",
                "detections",
                *(f"{phase}_ms" for phase in PHASES),
            )
        )
        writer.writerows(rows)


def phase_breakdown(
    latencies_ms: list[float],
    phase_samples: dict[str, list[float]],
) -> dict[str, Any]:
    """
    Per-phase order statistics, plus the latency net of the resize.

    ``net_of_resize`` is computed per iteration and only then summarised, not
    as a difference of two medians. The two are not the same number and the
    per-iteration one is the honest one: a run whose resize happened to be slow
    on the same iterations its inference was slow would otherwise have the
    covariance quietly averaged away.
    """

    if not latencies_ms or not phase_samples.get("resize"):
        return {}

    resize = phase_samples["resize"]

    net = [
        latency - cost
        for latency, cost in zip(latencies_ms, resize, strict=False)
        # A negative net is not a slow resize, it is a mismatched pair; there
        # is no iteration in which the resize is not part of the latency.
        if latency >= cost
    ]

    breakdown = {}

    for phase, samples in phase_samples.items():
        if not samples:
            continue

        stats = latency_stats(samples)
        # A phase has a duration, not a frame rate. "4144 fps" for the resize
        # step is arithmetically true and means nothing anyone would want.
        stats.pop("throughput_fps", None)
        breakdown[phase] = stats

    # This one keeps it: it is a rate the machine could actually sustain if the
    # frames arrived at the model's input size, which is the question a camera
    # pipeline with a hardware scaler in front of it is asking.
    breakdown["net_of_resize"] = latency_stats(net)

    if latencies_ms:
        mean_total = statistics.mean(latencies_ms)
        breakdown["resize_share"] = (
            statistics.mean(resize) / mean_total if mean_total > 0 else None
        )

    return breakdown


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ave resources",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("model", help="Single .tflite model (or SavedModel dir)")
    parser.add_argument("images", help="Image file or directory to draw the pool from")

    parser.add_argument("--output-dir", default="resource_results")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Prefix added to the run directory name inside --output-dir",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Run directory name inside --output-dir (default: the model stem). "
            "Needed when the same model is run more than once in a sweep -- "
            "under different delegate environments, say -- since the default "
            "name would make the second run overwrite the first"
        ),
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=120.0,
        help=(
            "Duration of the measured loop. Long enough for current, clock and "
            "die temperature to settle (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Run exactly N inferences instead of a fixed duration",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Discarded iterations before the measured loop (default: %(default)s)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=16,
        help=(
            "Images decoded into RAM and cycled. Full-resolution frames are "
            "~3 MB each (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.2,
        help="Resource sampling period in seconds (default: %(default)s)",
    )

    parser.add_argument(
        "--chirp-seconds",
        type=float,
        default=1.0,
        help=(
            "All-core burst bracketing the measured loop, stamping a "
            "recognisable step into the off-board power trace so the "
            "host-to-host clock offset can be verified against the data. "
            "0 disables (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=5.0,
        help=(
            "Idle stretch between chirp and measured loop. Doubles as the "
            "local idle-power baseline (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--delegate",
        default="/usr/lib/libteflon.so",
        help=(
            "Path to the TFLite delegate, or 'none' for CPU "
            "(use for fp32 models — the NPU delegate is for INT8)"
        ),
    )
    parser.add_argument(
        "--label",
        default="",
        help="Free-form label copied into run.json",
    )
    parser.add_argument(
        "--no-phases",
        action="store_true",
        help=(
            "Do not break the latency down into resize / preprocess / invoke. "
            "The breakdown costs a few sub-microsecond clock reads per "
            "inference, which is nothing against a millisecond-scale one -- "
            "use this only to prove that, or to reproduce a pre-breakdown run "
            "exactly"
        ),
    )
    parser.add_argument(
        "-e",
        "--env",
        metavar="KEY=VALUE",
        action="append",
        type=env_var,
        default=[],
        help="Set an environment variable; may be passed multiple times",
    )

    args = parser.parse_args(argv)

    env = dict(args.env)
    os.environ.update(env)

    delegate = args.delegate

    if delegate is not None and delegate.strip().lower() in ("", "none"):
        delegate = None

    model_path = Path(args.model)
    run_name = args.output_name or model_path.stem
    output_dir = Path(args.output_dir) / f"{args.output_prefix}{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    from agri_vision_edge.evaluation.resources import (
        chirp,
        spawn_sampler,
        system_info,
    )

    # Sampler first, and before the interpreter is built: the memory peak of a
    # run lands during model load and delegate warm-up, and a sampler started
    # afterwards reports a peak RSS that never includes it. Forking now also
    # keeps the child small, before the interpreter is resident.
    sampler = spawn_sampler(
        csv_path=output_dir / "resources.csv.gz",
        meta_path=output_dir / "resources_meta.json",
        interval=args.sample_interval,
    )

    run: dict[str, Any] = {
        "schema": 1,
        "label": args.label,
        "run_name": run_name,
        "model": model_path.name,
        "model_path": str(model_path),
        "model_bytes": model_path.stat().st_size if model_path.is_file() else None,
        "delegate_requested": str(delegate) if delegate else None,
        "environment": env,
        "config": {
            "seconds": args.seconds,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "pool_size": args.pool_size,
            "sample_interval": args.sample_interval,
            "chirp_seconds": args.chirp_seconds,
            "gap_seconds": args.gap_seconds,
            "phases": not args.no_phases,
        },
        "system": system_info(),
        "clock": {
            "t_start_s": "time.time(), UTC seconds",
            "note": (
                "All timestamps here are this device's epoch clock. Power is "
                "recorded on another host against its CLOCK_MONOTONIC and is "
                "joined afterwards; see chirps[] for the alignment markers."
            ),
        },
        "phases": {},
        "chirps": [],
    }

    cpu_count = os.cpu_count() or 1

    try:
        image_paths = collect_images(Path(args.images), args.pool_size)

        run["phases"]["pool_decode_start"] = time.time()
        pool = load_pool(image_paths)
        run["phases"]["pool_decode_end"] = time.time()
        run["images"] = [str(p) for p in image_paths]

        print(f"pool: {len(pool)} image(s) from {args.images}")

        from agri_vision_edge.runtime.inference.factory import build_runtime

        run["phases"]["model_load_start"] = time.time()

        # score_threshold pinned to 0.0 exactly as `ave benchmark` does, so the
        # post-NMS filtering loop does the same amount of work and the latency
        # here stays comparable with latency.json.
        runtime = build_runtime(
            model_path=model_path,
            delegate_path=delegate,
            score_threshold=0.0,
        )

        run["phases"]["model_load_end"] = time.time()

        # Enabled before the warm-up, so the first *measured* iteration is not
        # also the first one that ever took the timing branch.
        if not args.no_phases:
            runtime.enable_phase_timing()

        run["delegate_active"] = getattr(runtime, "active_delegate", None)
        run["backend"] = "delegate" if run["delegate_active"] else "cpu"
        run["runtime_format"] = getattr(runtime, "runtime_format", "tflite")
        run["input_details"] = _jsonable(getattr(runtime, "input_details", None))
        run["output_details"] = _jsonable(getattr(runtime, "output_details", None))

        run["phases"]["warmup_start"] = time.time()

        warmup_detections = []

        for index in range(args.warmup):
            warmup_detections = runtime.predict(pool[index % len(pool)])

        run["phases"]["warmup_end"] = time.time()

        if args.warmup > 0:
            run["output_integrity"] = check_output(warmup_detections)

            if run["output_integrity"]["corrupt"]:
                print(
                    "[warning] this runtime is producing unusable output "
                    "(non-finite boxes or out-of-range scores). The power and "
                    "latency below are real, but they measure a computation "
                    "that is not happening — on the i.MX8MP this is what the "
                    "Teflon delegate does to an fp32 graph. Re-run with "
                    "--delegate none for a meaningful fp32 figure."
                )

        # ---- chirp / gap / measured loop / gap / chirp ----

        if args.chirp_seconds > 0:
            start, end = chirp(args.chirp_seconds, cpu_count)
            run["chirps"].append(
                {"phase": "pre", "start": start, "end": end, "workers": cpu_count}
            )

        run["phases"]["idle_pre_start"] = time.time()
        time.sleep(args.gap_seconds)
        run["phases"]["idle_pre_end"] = time.time()

        rows: list[tuple[Any, ...]] = []
        latencies_ms: list[float] = []
        phase_samples: dict[str, list[float]] = {phase: [] for phase in PHASES}

        run["phases"]["measure_start"] = time.time()
        loop_start = time.perf_counter()

        iteration = 0

        while True:
            image_index = iteration % len(pool)

            t_start = time.time()
            start_perf = time.perf_counter()

            detections = runtime.predict(pool[image_index])

            end_perf = time.perf_counter()

            latency_ms = (end_perf - start_perf) * 1000.0
            latencies_ms.append(latency_ms)

            # Read, not kept: the runtime overwrites this dict on the next
            # call. Empty when --no-phases, or for a runtime that does not
            # instrument itself (the SavedModel one has no cv2 resize to
            # report), in which case the columns stay blank rather than zero --
            # zero would read as "free".
            timings = runtime.phase_timings_ms

            resize_ms = timings.get("resize")
            preprocess_ms = timings.get("preprocess")
            invoke_ms = timings.get("invoke")

            if preprocess_ms is not None and invoke_ms is not None:
                postprocess_ms = max(0.0, latency_ms - preprocess_ms - invoke_ms)
            else:
                postprocess_ms = None

            for phase, value in (
                ("resize", resize_ms),
                ("preprocess", preprocess_ms),
                ("invoke", invoke_ms),
                ("postprocess", postprocess_ms),
            ):
                if value is not None:
                    phase_samples[phase].append(value)

            rows.append(
                (
                    iteration,
                    image_index,
                    f"{t_start:.6f}",
                    "%.6f" % (t_start + (end_perf - start_perf)),
                    f"{latency_ms:.4f}",
                    len(detections),
                    *(
                        "" if value is None else f"{value:.4f}"
                        for value in (
                            resize_ms,
                            preprocess_ms,
                            invoke_ms,
                            postprocess_ms,
                        )
                    ),
                )
            )

            iteration += 1

            if args.iterations > 0:
                if iteration >= args.iterations:
                    break
            elif time.perf_counter() - loop_start >= args.seconds:
                break

        run["phases"]["measure_end"] = time.time()

        run["phases"]["idle_post_start"] = time.time()
        time.sleep(args.gap_seconds)
        run["phases"]["idle_post_end"] = time.time()

        if args.chirp_seconds > 0:
            start, end = chirp(args.chirp_seconds, cpu_count)
            run["chirps"].append(
                {"phase": "post", "start": start, "end": end, "workers": cpu_count}
            )

        run["latency"] = latency_stats(latencies_ms)
        run["latency_phases"] = phase_breakdown(latencies_ms, phase_samples)
        run["detections_total"] = sum(row[5] for row in rows)
        run["detections_mean"] = run["detections_total"] / len(rows) if rows else 0.0

        write_iterations(output_dir / "iterations.csv.gz", rows)

        run["status"] = "ok"

    except Exception as exc:
        run["status"] = "failed"
        run["exception"] = type(exc).__name__
        run["message"] = str(exc)
        print(f"[error] {type(exc).__name__}: {exc}")
        raise

    finally:
        run["sampler"] = sampler.stop()

        with open(output_dir / "run.json", "w") as handle:
            json.dump(run, handle, indent=2, default=str)
            handle.write("\n")

    stats = run["latency"]

    print(
        f"\n{run['model']}  backend={run['backend']}  "
        f"{stats['count']} iterations  "
        f"mean {stats['mean_latency_ms']:.2f} ms  "
        f"p95 {stats['p95_latency_ms']:.2f} ms  "
        f"{stats['throughput_fps']:.1f} fps"
    )

    breakdown = run.get("latency_phases") or {}

    if breakdown:
        net = breakdown["net_of_resize"]
        share = breakdown.get("resize_share")

        print(
            "  of which: "
            + "  ".join(
                f"{phase} {breakdown[phase]['mean_latency_ms']:.2f} ms"
                for phase in PHASES
                if phase in breakdown
            )
        )
        print(
            f"  net of resize: mean {net['mean_latency_ms']:.2f} ms  "
            f"{net['throughput_fps']:.1f} fps"
            + (f"  (resize is {share * 100:.1f} % of the call)" if share else "")
        )

    print(f"artifacts: {output_dir}")

    cadence = run["sampler"].get("cadence", {})

    if cadence.get("samples"):
        print(
            f"resource samples: {cadence['samples']} "
            f"(interval p95 {cadence['interval_p95_s'] * 1000:.0f} ms, "
            f"requested {cadence['requested_interval_s'] * 1000:.0f} ms)"
        )

    return 0


def _jsonable(value):
    """
    Interpreter details carry numpy arrays/dtypes; make them JSON-safe.
    """

    if value is None:
        return None

    try:
        import numpy as np
    except ImportError:
        return value

    def convert(item):
        if isinstance(item, dict):
            return {key: convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, type):
            return getattr(item, "__name__", str(item))
        return item

    return convert(value)


if __name__ == "__main__":
    raise SystemExit(main())
