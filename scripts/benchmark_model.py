#!/usr/bin/env python3
"""
Run ``benchmark_model`` across a directory of ``.tflite`` files and collect
the results into a CSV and a JSONL sidecar.

For each model ``tflite_runtime`` is used to read the actual input tensor
shape, dtype and quantization parameters so the source image can be correctly
preprocessed and fed via ``--input_layer_value_files``.

Delegate flags
--------------
teflon    --external_delegate_path (default /usr/lib/libteflon.so)
xnnpack   --use_xnnpack=true
none      no delegate flags (CPU baseline)

Usage
-----
    scripts/benchmark_model.py artifacts/tflite \\
        --image datasets/test-bundle/images/img.png

    scripts/benchmark_model.py artifacts/tflite \\
        --delegate xnnpack --num-runs 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

NAME_RE = re.compile(
    r"""
    ^
    (?P<arch>[^_]+(?:-[^_]+)*)_
    (?P<classes>[^_]+)_
    (?P<dataset>[^_]+(?:-[^_]+)*)_
    (?P<resolution>\d+)_
    (?P<precision>[^_]+)_
    (?P<quant>[^_]+)_
    (?P<granularity>[^_]+)_
    (?P<nms>[^_.]+)
    \.tflite$
    """,
    re.VERBOSE,
)

NAME_RE_SHORT = re.compile(
    r"""
    ^
    (?P<arch>[^_]+(?:-[^_]+)*)_
    (?P<classes>[^_]+)_
    (?P<dataset>[^_]+(?:-[^_]+)*)_
    (?P<resolution>\d+)_
    (?P<precision>[^_]+)_
    (?P<quant>[^_.]+)
    \.tflite$
    """,
    re.VERBOSE,
)

NAME_RE_NO_GRANULARITY = re.compile(
    r"""
    ^
    (?P<arch>[^_]+(?:-[^_]+)*)_
    (?P<classes>[^_]+)_
    (?P<dataset>[^_]+(?:-[^_]+)*)_
    (?P<resolution>\d+)_
    (?P<precision>[^_]+)_
    (?P<quant>[^_]+)_
    (?P<nms>[^_.]+)
    \.tflite$
    """,
    re.VERBOSE,
)


_EMPTY_NAME: dict[str, str] = {
    "arch": "",
    "classes": "",
    "dataset": "",
    "resolution": "",
    "precision": "",
    "quant": "",
    "granularity": "",
    "nms": "",
}


def parse_name(path: Path) -> dict:
    for regex in (NAME_RE, NAME_RE_NO_GRANULARITY, NAME_RE_SHORT):
        if m := regex.match(path.name):
            return {**_EMPTY_NAME, **m.groupdict()}
    return dict(_EMPTY_NAME)


# ---------------------------------------------------------------------------
# benchmark_model output parsing
# ---------------------------------------------------------------------------

TIMING_RE = re.compile(
    r"Inference timings in us:\s*"
    r"Init:\s*(?P<init>[\d.]+),\s*"
    r"First inference:\s*(?P<first>[\d.]+),\s*"
    r"Warmup \(avg\):\s*(?P<warmup>[\d.]+),\s*"
    r"Inference \(avg\):\s*(?P<inference>[\d.]+)"
)
STATS_RE = re.compile(
    r"count=(?P<count>\d+)\s+first=(?P<first>[\d.]+)\s+curr=(?P<curr>[\d.]+)\s+"
    r"min=(?P<min>[\d.]+)\s+max=(?P<max>[\d.]+)\s+avg=(?P<avg>[\d.]+)\s+"
    r"std=(?P<std>[\d.]+)\s+p5=(?P<p5>[\d.]+)\s+median=(?P<median>[\d.]+)\s+"
    r"p95=(?P<p95>[\d.]+)"
)
MODEL_SIZE_RE = re.compile(r"input model file size \(MB\):\s*([\d.]+)", re.IGNORECASE)
MEMORY_RE = re.compile(r"Memory footprint delta.*?init=([\d.]+)\s+overall=([\d.]+)")
SUBGRAPH_RE = re.compile(r"teflon: ===== subgraph #\d+: (\d+) operations,")
COMPILE_RE = re.compile(r"teflon: compiled graph, took ([\d.]+) ms")
INVOKE_RE = re.compile(r"teflon: invoked graph, took ([\d.]+) ms")


def us_to_ms(x: str | float) -> float:
    return float(x) / 1000.0


def parse_output(text: str) -> dict:
    result: dict = {}

    if m := TIMING_RE.search(text):
        result.update(
            init_ms=us_to_ms(m["init"]),
            first_ms=us_to_ms(m["first"]),
            warmup_avg_ms=us_to_ms(m["warmup"]),
            latency_avg_ms=us_to_ms(m["inference"]),
        )
        result["fps"] = 1000.0 / result["latency_avg_ms"]

    stats = list(STATS_RE.finditer(text))
    if stats:
        m = stats[-1]
        result.update(
            runs=int(m["count"]),
            latency_min_ms=us_to_ms(m["min"]),
            latency_max_ms=us_to_ms(m["max"]),
            latency_std_ms=us_to_ms(m["std"]),
            latency_p5_ms=us_to_ms(m["p5"]),
            latency_median_ms=us_to_ms(m["median"]),
            latency_p95_ms=us_to_ms(m["p95"]),
        )

    if m := MODEL_SIZE_RE.search(text):
        result["model_size_mb"] = float(m.group(1))

    if m := MEMORY_RE.search(text):
        result["memory_init_mb"] = float(m.group(1))
        result["memory_overall_mb"] = float(m.group(2))

    subgraphs = [int(x) for x in SUBGRAPH_RE.findall(text)]
    if subgraphs:
        result["delegate_partitions"] = len(subgraphs)
        result["delegated_ops"] = sum(subgraphs)

    compile_times = [float(x) for x in COMPILE_RE.findall(text)]
    if compile_times:
        result["teflon_compile_ms"] = sum(compile_times)

    support_lines = [ln for ln in text.splitlines() if re.match(r"^\s*\d+\s+op:", ln)]
    if support_lines:
        unsupported = sum(
            bool(re.search(r"\bunsupported\b", ln)) for ln in support_lines
        )
        result["total_ops"] = len(support_lines)
        result["unsupported_ops"] = unsupported
        result["supported_ops"] = len(support_lines) - unsupported

    invokes = [float(x) for x in INVOKE_RE.findall(text)]
    if invokes and result.get("runs"):
        result["teflon_invoke_ms"] = mean(invokes[-result["runs"] :])

    return result


# ---------------------------------------------------------------------------
# Input tensor introspection and preprocessing
# ---------------------------------------------------------------------------


def get_input_spec(model_path: Path) -> dict | None:
    """Return the first input tensor's name, shape, dtype and quant params."""
    try:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore[import]
        except ImportError:
            from tensorflow.lite.python.interpreter import (  # type: ignore[import]
                Interpreter,
            )
        interp = Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        d = interp.get_input_details()[0]
        scale, zero_point = d["quantization"]
        return {
            "name": d["name"],
            "shape": d["shape"].tolist(),
            "dtype": d["dtype"],
            "scale": float(scale),
            "zero_point": int(zero_point),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not inspect {model_path.name}: {exc}")
        return None


def prepare_input(image_path: Path, spec: dict, tmp_dir: Path) -> Path:
    """Preprocess image to a raw binary tensor matching the model's input spec."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    _, h, w, _ = spec["shape"]
    dtype = spec["dtype"]
    scale = spec["scale"]
    zero_point = spec["zero_point"]

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"cv2.imread could not decode {image_path}")

    tensor = (
        cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        / 127.5
        - 1.0
    )

    if dtype is not np.float32:
        q = (
            np.round(tensor / scale + zero_point)
            if scale > 0
            else np.round(tensor * 127.5)
        )
        info = np.iinfo(dtype)
        tensor = np.clip(q, info.min, info.max).astype(dtype)

    out_path = tmp_dir / f"input_{h}x{w}_{getattr(dtype, '__name__', dtype)}.bin"
    out_path.write_bytes(tensor[np.newaxis].tobytes())
    return out_path


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def benchmark(
    model: Path,
    image_path: Path | None,
    benchmark_model_bin: str,
    delegate: str,
    delegate_path: str,
    num_runs: int,
    warmup_runs: int,
    min_secs: float,
    logs_dir: Path,
    tmp_dir: Path,
) -> dict:
    name_fields = parse_name(model)

    cmd: list[str] = [
        benchmark_model_bin,
        f"--graph={model}",
        f"--num_runs={num_runs}",
        f"--min_secs={min_secs}",
        "--max_secs=600",
        f"--warmup_runs={warmup_runs}",
        "--warmup_min_secs=0",
        "--report_peak_memory_footprint=true",
    ]

    if delegate == "teflon":
        cmd.append(f"--external_delegate_path={delegate_path}")
    elif delegate == "xnnpack":
        cmd.append("--use_xnnpack=true")

    input_fed = False
    if image_path is not None:
        spec = get_input_spec(model)
        if spec is not None:
            try:
                bin_path = prepare_input(image_path, spec, tmp_dir)
                # benchmark_model uses ',' to separate dimensions within a shape
                # and ':' to separate multiple inputs.  Tensor names often
                # contain ':' (e.g. "serving_default_input:0") which this build
                # does not escape correctly in --input_layer_value_files.  Using
                # a simple alias avoids the issue; benchmark_model maps it
                # positionally to tensor 0 with a harmless warning.
                shape_str = ",".join(str(d) for d in spec["shape"])
                cmd += [
                    "--input_layer=img",
                    f"--input_layer_shape={shape_str}",
                    f"--input_layer_value_files=img:{bin_path}",
                ]
                input_fed = True
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] could not prepare input: {exc}")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    text = proc.stdout or ""

    log_path = logs_dir / f"{model.stem}_{delegate}.log"
    log_path.write_text(text)

    return {
        "model": model.name,
        "delegate": delegate,
        **name_fields,
        "input_fed": input_fed,
        "returncode": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "failed",
        **parse_output(text),
        "log": str(log_path),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict], csv_path: Path) -> None:
    fields = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "models", type=Path, help="Directory of .tflite files or a single .tflite path."
    )
    ap.add_argument(
        "--image",
        type=Path,
        default=None,
        metavar="PATH",
        help="Source image; preprocessed and fed to each model.",
    )
    ap.add_argument("--benchmark-model", default="benchmark_model", metavar="PATH")
    ap.add_argument(
        "--delegate", default="teflon", choices=["teflon", "xnnpack", "none"]
    )
    ap.add_argument("--delegate-path", default="/usr/lib/libteflon.so", metavar="PATH")
    ap.add_argument("--num-runs", type=int, default=50, metavar="N")
    ap.add_argument("--warmup-runs", type=int, default=5, metavar="N")
    ap.add_argument("--min-secs", type=float, default=0.0, metavar="S")
    ap.add_argument(
        "--output", type=Path, default=Path("benchmark-results"), metavar="DIR"
    )
    ap.add_argument(
        "--override",
        action="store_true",
        help="Re-run models that already have a log file.",
    )
    ap.add_argument("--glob", default="*.tflite", metavar="PATTERN")
    args = ap.parse_args(argv)

    p = args.models.resolve()
    if p.is_dir():
        models = sorted(p.glob(args.glob))
    elif p.is_file() and p.suffix == ".tflite":
        models = [p]
    else:
        print(
            f"[error] {args.models} is not a directory or .tflite file", file=sys.stderr
        )
        return 1

    if not models:
        print(f"[error] no .tflite files found under {args.models}", file=sys.stderr)
        return 1

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs_dir = out / "logs"
    logs_dir.mkdir(exist_ok=True)
    jsonl_path = out / "results.jsonl"
    csv_path = out / "benchmark_model.csv"

    print(f"models     : {len(models)}")
    print(f"delegate   : {args.delegate}")
    if args.delegate == "teflon":
        print(f"  path     : {args.delegate_path}")
    print(f"num_runs   : {args.num_runs}")
    print(f"warmup_runs: {args.warmup_runs}")
    print(f"image      : {args.image or '(random noise)'}")
    print(f"output     : {out}")
    print()

    rows: list[dict] = []
    ran = skipped = failed = 0

    with tempfile.TemporaryDirectory(prefix="bm_inputs_") as tmp:
        tmp_dir = Path(tmp)
        for i, model in enumerate(models, 1):
            log_path = logs_dir / f"{model.stem}_{args.delegate}.log"
            if log_path.exists() and not args.override:
                print(f"[{i:3}/{len(models)}] {model.name}  [skip]")
                skipped += 1
                continue

            print(f"[{i:3}/{len(models)}] {model.name}", flush=True)
            row = benchmark(
                model=model,
                image_path=args.image,
                benchmark_model_bin=args.benchmark_model,
                delegate=args.delegate,
                delegate_path=args.delegate_path,
                num_runs=args.num_runs,
                warmup_runs=args.warmup_runs,
                min_secs=args.min_secs,
                logs_dir=logs_dir,
                tmp_dir=tmp_dir,
            )
            rows.append(row)

            with jsonl_path.open("a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")

            latency = row.get("latency_avg_ms")
            status = row.get("status", "?")
            fed = "real-image" if row.get("input_fed") else "random-noise"
            if latency is not None:
                print(
                    f"          {latency:9.3f} ms  ({row.get('fps', 0.0):.1f} fps)  [{status}] [{fed}]"
                )
            else:
                print(f"          {'--':>9}         [{status}] [{fed}]")

            if status == "ok":
                ran += 1
            else:
                failed += 1

    if rows:
        write_csv(rows, csv_path)

    print()
    print(f"done: {ran} ok, {skipped} skipped, {failed} failed")
    print(f"JSONL : {jsonl_path}")
    print(f"CSV   : {csv_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
