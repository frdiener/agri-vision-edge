#!/usr/bin/env python3
"""Summarize power, energy, CPU, memory, and temperature for a power sweep.

Meter monotonic timestamps and board epoch timestamps are mapped to the dev
host's epoch using paired SSH clock probes. Probe drift is reported, and
board-generated power chirps provide an independent end-to-end alignment check;
large residuals mark the joined power figures as invalid.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(str(path), "rt", newline="")

    return open(path, newline="")


#: Exclusion padding around known-busy phases, in seconds.
PHASE_PAD_S = 0.5

#: Fraction of chirp amplitude used for rising-edge detection.
CHIRP_THRESHOLD_FRACTION = 0.75

#: Maximum accepted chirp alignment residual, in seconds.
ALIGNMENT_TOLERANCE_S = 2.5

#: Truncated traces already reported, keyed by name, size, and mtime. A sweep
#: may copy or hard-link one trace into every run directory.
_TRUNCATED_REPORTED: set[tuple[str, int, int]] = set()


def _file_identity(path: Path) -> tuple[str, int, int] | None:
    """``(name, size, mtime_ns)``, which copies of one trace share."""
    try:
        info = path.stat()
    except OSError:
        return None

    return (path.name, info.st_size, info.st_mtime_ns)


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read CSV rows, retaining decoded rows from a gzip stream without a footer."""
    rows: list[list[str]] = []
    truncated = False

    try:
        with open_maybe_gzip(path) as stream:
            for row in csv.reader(stream):
                rows.append(row)
    except (EOFError, gzip.BadGzipFile, OSError) as exc:
        truncated = True
        # Warn once for each shared trace.
        identity = _file_identity(path)
        first_time = identity is None or identity not in _TRUNCATED_REPORTED

        if identity is not None:
            _TRUNCATED_REPORTED.add(identity)

        if first_time:
            print(
                f"[warning] {path.name}: {type(exc).__name__} — using the "
                f"{len(rows)} row(s) recovered before the break. The logger was "
                "killed before it closed the gzip stream; the samples are "
                "intact, only the trailer is missing.",
                file=sys.stderr,
            )

    if not rows:
        return [], []

    # Drop an incomplete final row left by termination during a write.
    if truncated and len(rows) > 1 and len(rows[-1]) != len(rows[0]):
        rows.pop()

    return rows[0], rows[1:]


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class Window:
    """A closed time interval on the dev host's epoch clock."""

    __slots__ = ("start", "end")

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    @property
    def duration(self) -> float:
        return self.end - self.start

    def contains(self, t: float) -> bool:
        return self.start <= t <= self.end


def mean_offset(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> tuple[float, float, float]:
    """
    ``(offset, drift, uncertainty)`` from a pair of clock probes.

    ``drift`` is the difference between the two probes, i.e. how far the two
    clocks moved apart over the run; a large value invalidates a single
    constant offset for the whole window.
    """

    probes = [p for p in (before, after) if p]

    if not probes:
        return 0.0, 0.0, float("inf")

    offsets = [p["offset_s"] for p in probes]
    uncertainty = max(p["uncertainty_s"] for p in probes)
    drift = (offsets[-1] - offsets[0]) if len(offsets) > 1 else 0.0

    return statistics.mean(offsets), drift, uncertainty


def load_power(
    path: Path,
    anchor: dict[str, Any],
    offset: float,
) -> list[tuple[float, float]]:
    """
    ``[(t_local_epoch, watts)]`` from the meter trace.

    ``t_ns`` is the lab server's ``CLOCK_MONOTONIC``, which has no epoch
    meaning at all. The anchor pairs one monotonic reading with one epoch
    reading on that host; ``offset`` then moves the result onto this host's
    clock.
    """

    header, rows = read_csv(path)

    if not rows:
        return []

    index_t = header.index("t_ns")
    index_p = header.index("power_W")

    anchor_epoch = anchor["t_epoch_s"]
    anchor_monotonic = anchor["t_monotonic_ns"]

    samples = []

    for row in rows:
        if len(row) <= max(index_t, index_p):
            continue

        t_ns = to_float(row[index_t])
        meter_epoch = anchor_epoch + (t_ns - anchor_monotonic) / 1e9
        samples.append((meter_epoch - offset, to_float(row[index_p])))

    samples.sort(key=lambda item: item[0])

    return samples


def slice_window(samples: list[tuple[float, float]], window: Window):
    return [value for t, value in samples if window.contains(t)]


def integrate(samples: list[tuple[float, float]], window: Window) -> float:
    """
    Trapezoidal energy in joules over `window`.

    Trapezoidal integration weights the FNB58's nonuniform packet intervals by
    their actual widths.
    """

    inside = [(t, value) for t, value in samples if window.contains(t)]

    if len(inside) < 2:
        return 0.0

    total = 0.0

    for (t0, p0), (t1, p1) in zip(inside, inside[1:], strict=False):
        total += (p1 + p0) / 2.0 * (t1 - t0)

    return total


def find_edge(
    samples: list[tuple[float, float]],
    around: float,
    baseline: float,
    search: float = 8.0,
    exclude: Iterable[Window] = (),
    keep: Iterable[Window] = (),
) -> float | None:
    """Find a chirp's fractional-amplitude rising edge near ``around``.

    The search scans backward from the local peak, ignoring ``exclude`` windows
    except regions in ``keep``; it returns ``None`` when no reliable edge exists.
    """

    nearby = [
        (t, value)
        for t, value in samples
        if around - search <= t <= around + search
        and (
            any(window.contains(t) for window in keep)
            or not any(window.contains(t) for window in exclude)
        )
    ]

    if len(nearby) < 4:
        return None

    peak_index = max(range(len(nearby)), key=lambda i: nearby[i][1])
    amplitude = nearby[peak_index][1] - baseline

    # Reject steps too small for reliable identification.
    if amplitude <= 0.05 * max(baseline, 0.1):
        return None

    threshold = baseline + amplitude * CHIRP_THRESHOLD_FRACTION

    for index in range(peak_index, 0, -1):
        (t0, p0), (t1, p1) = nearby[index - 1], nearby[index]
        if p0 < threshold <= p1:
            if p1 == p0:
                return t1

            # Linear interpolation between the straddling samples.
            return t0 + (threshold - p0) / (p1 - p0) * (t1 - t0)

    return None


def summarise_resources(path: Path, window: Window, offset: float) -> dict[str, Any]:
    header, rows = read_csv(path)

    if not rows:
        return {}

    def column(name: str) -> int | None:
        return header.index(name) if name in header else None

    index_t = column("t_epoch_s")

    if index_t is None:
        return {}

    temp_columns = [i for i, name in enumerate(header) if name.startswith("temp_")]
    freq_columns = [i for i, name in enumerate(header) if name.startswith("freq_")]

    index_cpu = column("cpu_pct")
    index_proc_cores = column("proc_cpu_cores")
    index_proc_pct = column("proc_cpu_pct")
    index_rss = column("proc_rss_kb")
    index_hwm = column("proc_hwm_kb")
    index_avail = column("memavailable_kb")

    cpu, proc_cores, proc_pct, rss, hwm, avail = [], [], [], [], [], []
    temps, freqs = [], []

    for row in rows:
        t_local = to_float(row[index_t]) - offset

        if not window.contains(t_local):
            continue

        if index_cpu is not None:
            cpu.append(to_float(row[index_cpu]))
        if index_proc_cores is not None:
            proc_cores.append(to_float(row[index_proc_cores]))
        if index_proc_pct is not None:
            proc_pct.append(to_float(row[index_proc_pct]))
        if index_rss is not None:
            rss.append(to_float(row[index_rss]))
        if index_hwm is not None:
            hwm.append(to_float(row[index_hwm]))
        if index_avail is not None:
            avail.append(to_float(row[index_avail]))

        for i in temp_columns:
            value = to_float(row[i])
            if value:
                temps.append(value)
        for i in freq_columns:
            value = to_float(row[i])
            if value:
                freqs.append(value)

    if not cpu:
        return {}

    result = {
        "samples": len(cpu),
        "cpu_pct_mean": statistics.mean(cpu),
        "cpu_pct_max": max(cpu),
        "proc_cpu_cores_mean": statistics.mean(proc_cores) if proc_cores else None,
        "proc_cpu_pct_mean": statistics.mean(proc_pct) if proc_pct else None,
        "proc_rss_kb_mean": statistics.mean(rss) if rss else None,
        "proc_rss_kb_max": max(rss) if rss else None,
        "proc_hwm_kb_max": max(hwm) if hwm else None,
        "mem_available_kb_min": min(avail) if avail else None,
    }

    if temps:
        # /sys reports millidegrees.
        result["temp_c_max"] = max(temps) / 1000.0
        result["temp_c_mean"] = statistics.mean(temps) / 1000.0

    if freqs:
        result["cpu_khz_mean"] = statistics.mean(freqs)
        result["cpu_khz_max"] = max(freqs)

    return result


def summarise_iterations(path: Path, window: Window, offset: float) -> dict[str, Any]:
    header, rows = read_csv(path)

    if not rows:
        return {}

    index_start = header.index("t_start_s")
    index_latency = header.index("latency_ms")

    latencies = [
        to_float(row[index_latency])
        for row in rows
        if window.contains(to_float(row[index_start]) - offset)
    ]

    if not latencies:
        return {}

    ordered = sorted(latencies)

    return {
        "inferences": len(ordered),
        "mean_latency_ms": statistics.mean(ordered),
        "p95_latency_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def analyse_run(run_dir: Path) -> dict[str, Any] | None:
    run_json = run_dir / "run.json"

    if not run_json.exists():
        return None

    run = json.loads(run_json.read_text())

    if run.get("status") != "ok":
        return {
            "run": run_dir.name,
            "status": run.get("status", "unknown"),
            "message": run.get("message"),
        }

    sweep_run_path = run_dir / "sweep_run.json"
    sweep_run = (
        json.loads(sweep_run_path.read_text()) if sweep_run_path.exists() else {}
    )

    device_offset, device_drift, device_uncertainty = mean_offset(
        sweep_run.get("device_clock_probe_before"),
        sweep_run.get("device_clock_probe_after"),
    )

    phases = run["phases"]

    def local(t: float) -> float:
        return t - device_offset

    measure = Window(local(phases["measure_start"]), local(phases["measure_end"]))
    idle_windows = [
        Window(local(phases["idle_pre_start"]), local(phases["idle_pre_end"])),
        Window(local(phases["idle_post_start"]), local(phases["idle_post_end"])),
    ]

    result: dict[str, Any] = {
        "run": run_dir.name,
        "status": "ok",
        "model": run.get("model"),
        "backend": run.get("backend"),
        "delegate_active": run.get("delegate_active"),
        "device": run.get("system", {}).get("device_tree_model")
        or run.get("system", {}).get("hostname"),
        "measure_seconds": measure.duration,
        "latency": run.get("latency", {}),
        "clock": {
            "device_offset_s": device_offset,
            "device_drift_s": device_drift,
            "device_uncertainty_s": device_uncertainty,
        },
        "sampler_cadence": run.get("sampler", {}).get("cadence"),
        # Carried through so a run that drew power without computing anything
        # cannot be read off the summary as a clean result.
        "output_integrity": run.get("output_integrity"),
    }

    resources_path = run_dir / "resources.csv.gz"

    if resources_path.exists():
        result["resources"] = summarise_resources(
            resources_path, measure, device_offset
        )
        result["resources_idle"] = summarise_resources(
            resources_path, idle_windows[0], device_offset
        )

    iterations_path = run_dir / "iterations.csv.gz"

    if iterations_path.exists():
        result["iterations"] = summarise_iterations(
            iterations_path, measure, device_offset
        )

    # Power
    power_path = run_dir / "power.csv.gz"
    anchor = sweep_run.get("meter_anchor")

    if not power_path.exists() or not anchor:
        result["power"] = None
        return result

    meter_offset, meter_drift, meter_uncertainty = mean_offset(
        sweep_run.get("meter_clock_probe_before"),
        sweep_run.get("meter_clock_probe_after"),
    )

    samples = load_power(power_path, anchor, meter_offset)

    if not samples:
        result["power"] = None
        return result

    result["clock"]["meter_offset_s"] = meter_offset
    result["clock"]["meter_drift_s"] = meter_drift
    result["clock"]["meter_uncertainty_s"] = meter_uncertainty

    idle_values = []

    for window in idle_windows:
        idle_values.extend(slice_window(samples, window))

    idle_mean = statistics.mean(idle_values) if idle_values else 0.0

    measured = slice_window(samples, measure)

    if not measured:
        result["power"] = None
        result["power_error"] = (
            "no power samples inside the measured window — check clock_sync.json"
        )
        return result

    energy = integrate(samples, measure)
    net_energy = energy - idle_mean * measure.duration

    inferences = result.get("iterations", {}).get("inferences") or run.get(
        "latency", {}
    ).get("count")

    power = {
        "samples": len(measured),
        "idle_w": idle_mean,
        "mean_w": statistics.mean(measured),
        "median_w": statistics.median(measured),
        "max_w": max(measured),
        "min_w": min(measured),
        "net_mean_w": statistics.mean(measured) - idle_mean,
        "energy_j": energy,
        "net_energy_j": net_energy,
    }

    if inferences:
        power["energy_per_inference_mj"] = 1000.0 * energy / inferences
        power["net_energy_per_inference_mj"] = 1000.0 * net_energy / inferences

    # Alignment cross-check
    alignment = {}

    # Exclude known busy phases that can draw more power than the chirp.
    busy = [
        Window(local(phases[start]) - PHASE_PAD_S, local(phases[end]) + PHASE_PAD_S)
        for start, end in (
            ("pool_decode_start", "pool_decode_end"),
            ("model_load_start", "model_load_end"),
            ("warmup_start", "warmup_end"),
            ("measure_start", "measure_end"),
        )
        if start in phases and end in phases
    ]

    chirp_windows = [
        Window(local(c["start"]), local(c["end"]))
        for c in run.get("chirps", [])
        if c.get("start") is not None and c.get("end") is not None
    ]

    for chirp in run.get("chirps", []):
        expected = local(chirp["start"])
        detected = find_edge(
            samples, expected, idle_mean, exclude=busy, keep=chirp_windows
        )

        if detected is not None:
            alignment[chirp["phase"]] = {
                "expected_s": expected,
                "detected_s": detected,
                "residual_s": detected - expected,
            }

    if alignment:
        residuals = [entry["residual_s"] for entry in alignment.values()]
        alignment["max_abs_residual_s"] = max(abs(r) for r in residuals)
        aligned = alignment["max_abs_residual_s"] <= ALIGNMENT_TOLERANCE_S
        alignment["state"] = "verified" if aligned else "misaligned"
        alignment["verified"] = aligned
    else:
        alignment = {"state": "unverified", "verified": False}

    result["power"] = power
    result["alignment"] = alignment or None

    return result


def print_table(results: list[dict[str, Any]]) -> None:
    rows = [r for r in results if r.get("status") == "ok"]

    if not rows:
        print("no successful runs")
        return

    header = (
        "run",
        "backend",
        "lat ms",
        "fps",
        "P W",
        "net P W",
        "mJ/inf",
        "CPU %",
        "cores",
        "RSS MiB",
        "degC",
    )

    def cell(result: dict[str, Any]) -> tuple[str, ...]:
        power = result.get("power") or {}
        res = result.get("resources") or {}
        latency = result.get("latency") or {}

        def fmt(value, spec="{:.2f}"):
            return spec.format(value) if isinstance(value, (int, float)) else "-"

        rss = res.get("proc_rss_kb_max")

        # A run whose output is NaN still has entirely real power and latency
        # numbers; they just do not describe a working detector.
        corrupt = (result.get("output_integrity") or {}).get("corrupt")
        name = ("!! " if corrupt else "") + result["run"][:44]

        return (
            name,
            result.get("backend", "-"),
            fmt(latency.get("mean_latency_ms")),
            fmt(latency.get("throughput_fps"), "{:.1f}"),
            fmt(power.get("mean_w")),
            fmt(power.get("net_mean_w")),
            fmt(power.get("net_energy_per_inference_mj"), "{:.1f}"),
            fmt(res.get("cpu_pct_mean"), "{:.0f}"),
            fmt(res.get("proc_cpu_cores_mean")),
            fmt(rss / 1024.0 if rss else None, "{:.0f}"),
            fmt(res.get("temp_c_max"), "{:.1f}"),
        )

    table = [header] + [cell(r) for r in rows]
    widths = [max(len(str(row[i])) for row in table) for i in range(len(header))]

    for index, row in enumerate(table):
        print(
            "| "
            + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row))
            + " |"
        )

        if index == 0:
            print("|-" + "-|-".join("-" * width for width in widths) + "-|")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sweep", type=Path, help="sweep directory")
    parser.add_argument(
        "--output",
        default=None,
        help="summary JSON path (default: <sweep>/power_summary.json)",
    )
    parser.add_argument(
        "--org",
        action="store_true",
        help="print an org-mode table only",
    )

    args = parser.parse_args(argv)

    sweep_dir = args.sweep

    if not sweep_dir.is_dir():
        print(f"not a directory: {sweep_dir}", file=sys.stderr)
        return 1

    run_dirs = sorted(p for p in sweep_dir.iterdir() if (p / "run.json").exists())

    if not run_dirs:
        print(f"no runs found in {sweep_dir}", file=sys.stderr)
        return 1

    results = []

    for run_dir in run_dirs:
        analysed = analyse_run(run_dir)

        if analysed:
            results.append(analysed)

    summary = {
        "schema": 1,
        "sweep": str(sweep_dir),
        "runs": results,
    }

    output = Path(args.output) if args.output else sweep_dir / "power_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")

    print_table(results)

    if args.org:
        return 0

    print()

    def _state(result) -> str:
        return (result.get("alignment") or {}).get("state", "unverified")

    scored = [r for r in results if r.get("status") == "ok" and r.get("power")]
    misaligned = [r for r in scored if _state(r) == "misaligned"]
    unverified = [r for r in scored if _state(r) == "unverified"]

    for result in results:
        alignment = result.get("alignment")

        if alignment and "max_abs_residual_s" in alignment:
            print(
                f"alignment {result['run'][:44]:<46} "
                f"chirp residual {alignment['max_abs_residual_s'] * 1000:7.1f} ms"
            )

    if misaligned:
        print(
            f"\n[error] {len(misaligned)} run(s) MISALIGNED: the chirp was found "
            "somewhere other than where the board says it happened, so the power "
            "figures are joined to the wrong part of the trace."
        )

    if unverified:
        print(
            f"\n[note] {len(unverified)} run(s) unverified: no chirp edge was "
            "detectable, so the join rests on the ssh clock probe alone "
            "(measured uncertainty < 0.1 s, against a 120 s window)."
        )

    corrupt = [r for r in results if (r.get("output_integrity") or {}).get("corrupt")]

    if corrupt:
        print(
            f"\n[warning] {len(corrupt)} run(s) marked '!!' produced non-finite "
            "output. Their power and latency are real measurements, but not of a "
            "working detector — the Teflon delegate on an fp32 graph:"
        )

        for result in corrupt:
            print(f"    {result['run']}  ({result.get('backend')})")

    failed = [r for r in results if r.get("status") != "ok"]

    if failed:
        print(f"\n{len(failed)} failed run(s):")

        for result in failed:
            print(f"  {result['run']}: {result.get('message')}")

    print(f"\nwrote {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
