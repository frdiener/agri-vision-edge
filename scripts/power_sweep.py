#!/usr/bin/env python3
"""
Drive a power / resource sweep across three machines.

Runs on the **dev host**, which reaches the other two by ssh and exports the
repository over NFS::

    dev host  ── ssh ─────────▶  lab server   (FNIRSI FNB58 on USB, inline on
      │                           the board's supply line)
      ├──────── ssh ─────────▶  target board  (runs `ave resources`)
      └──────── NFS ─────────▶  target board  (repo + results, same paths)

Per model it starts the power logger on the lab server, runs one sustained
inference loop on the board, stops the logger, and pulls the trace into the run
directory next to the board's own artifacts. The board writes its CSVs straight
to the NFS mount, so nothing has to be copied back from it.

Clock alignment
---------------
The three hosts share no clock, and the power trace is timestamped in the *lab
server's* ``CLOCK_MONOTONIC``, which has no defined origin at all. Two
independent mechanisms are recorded so the join can be checked rather than
trusted:

1. **Offset probe.** Before and after each run, the lab server's epoch clock is
   sampled over ssh several times; the sample with the smallest round trip wins
   and the offset is its midpoint estimate, with ``rtt / 2`` as the error bound.
   A paired ``(epoch, monotonic)`` reading taken on the lab server maps the
   trace's ``t_ns`` onto that epoch clock. Probing on both sides of the run also
   exposes drift over the run's duration.

2. **Sync chirp.** ``ave resources`` saturates the board's cores for a moment
   either side of the measured loop, which puts an unmistakable step into the
   power trace at a timestamp the board recorded locally. ``power_report.py``
   matches those edges and reports the residual against the probe estimate. If
   the two disagree, the join is wrong and says so.

Nothing here parses the power trace; this script only *collects*, in keeping
with the rest of the pipeline (see ``scripts/benchmark_all.sh``). Run
``scripts/power_report.py`` afterwards to join and summarise.

Example
-------
::

    scripts/power_sweep.py \\
        --device root@192.168.11.131 \\
        --meter-host rlabD-srv \\
        --device-repo /mnt/agri-vision-edge \\
        --seconds 120 --cpu --dry-run
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Reading the lab server's two clocks in one shot. `repr` of a tuple keeps the
#: full float precision that `print` of a bare float would also give, but is
#: unambiguous to parse.
CLOCK_SNIPPET = "import time; print(repr((time.time(), time.monotonic_ns())))"


class Remote:
    """A host reachable over ssh."""

    def __init__(self, target: str, ssh_options: list[str], python: str = "python3"):
        self.target = target
        self.ssh_options = ssh_options
        self.python = python

    @property
    def ssh(self) -> list[str]:
        return ["ssh", *self.ssh_options, self.target]

    def run(
        self,
        command: str,
        *,
        capture: bool = True,
        check: bool = False,
        timeout: float | None = 60.0,
        binary: bool = False,
    ):
        return subprocess.run(
            [*self.ssh, command],
            capture_output=capture,
            text=not binary,
            check=check,
            timeout=timeout,
        )

    def script(self, body: str, *, timeout: float | None = 60.0):
        """Run a multi-line shell script, avoiding a quoting nightmare."""

        return subprocess.run(
            [*self.ssh, "bash", "-s"],
            input=body,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def probe_clock(remote: Remote, samples: int = 7) -> dict[str, Any]:
    """
    Estimate ``remote_epoch - local_epoch`` using the fastest round trip.

    Christian's algorithm: the remote reading is assumed to have happened at
    the midpoint of the local send/receive pair, which is exact only if the
    round trip is symmetric. The fastest sample is used because it is the one
    least polluted by scheduling and queueing, and half of it bounds the error.
    """

    best: dict[str, Any] | None = None

    for _ in range(samples):
        before = time.time()
        result = remote.run(f"{remote.python} -c {shlex.quote(CLOCK_SNIPPET)}")
        after = time.time()

        if result.returncode != 0:
            continue

        try:
            remote_epoch, remote_monotonic = eval(result.stdout.strip())  # noqa: S307
        except Exception:
            continue

        rtt = after - before
        midpoint = (before + after) / 2.0

        if best is None or rtt < best["rtt_s"]:
            best = {
                "rtt_s": rtt,
                "local_epoch_s": midpoint,
                "remote_epoch_s": remote_epoch,
                "remote_monotonic_ns": remote_monotonic,
                # Add this to a local timestamp to get the remote's clock.
                "offset_s": remote_epoch - midpoint,
                "uncertainty_s": rtt / 2.0,
            }

    if best is None:
        raise RuntimeError(f"could not probe clock on {remote.target}")

    return best


#: Named model selections.
#:
#: ``arch-matrix`` is the collapsed set: one representative per
#: (architecture x precision/quantisation), 13 models instead of 43. The two
#: dropped dimensions were checked against the existing ``latency.json`` sweeps
#: on both boards rather than assumed, holding everything else fixed:
#:
#:   dimension varied              median spread   max
#:   sc vs mc                            3.3 %    10.8 %
#:   trained untiled vs tiled            3.4 %    14.2 %
#:   ---- kept ----
#:   ssd-mn2 vs fpnlite                 20.3 %    52.3 %
#:   fp32 vs int8                      166.3 %   174.6 %
#:
#: The artifacts say why. sc and mc differ by +0.34 % (mn2) / +0.03 % (fpnlite)
#: in file size, because only the class predictor changes width --
#: num_anchors x (num_classes + 1), i.e. 2 vs 3 channels per anchor; everything
#: upstream is the same workload. Untiled- and tiled-trained exports differ by
#: -0.02 % / -0.05 %: the *same graph* with different weight values, so there is
#: no mechanism by which their compute could differ.
#:
#: The survivor of each collapse is the reference config, untiled mc.
#:
#: yolov7-tiny is the one exception, and deliberately so: it exists only as a
#: tiled-trained 512 export, so it cannot follow "untiled". Training tiling is
#: precisely the dimension shown above not to affect compute, so that part is
#: free; the 512 input is a genuine difference and is *meant* to be there --
#: input resolution is an architecture property, which is what this preset
#: varies.
#:
#: This is about resource cost only. Accuracy obviously does depend on class
#: count and training data -- `ave benchmark` still needs the full matrix.
PRESETS: dict[str, list[str]] = {
    "arch-matrix": [
        "ssd-mn2_mc_phenobench_320_*",
        "ssd-mn2-fpnlite_mc_phenobench_320_*",
        "yolov7-tiny_*",
    ],
}


def collect_models(models_dir: Path, patterns: list[str] | None) -> list[Path]:
    """
    Resolve model selectors to .tflite paths.

    Every model in ``artifacts/tflite`` sits beside a ``.metadata.json``
    sidecar, so a natural selector like ``ssd-mn2_sc_*_int8_*`` matches twice
    as many files as intended and half of them are not models. The suffix
    filter is applied after globbing rather than expecting the caller to spell
    it out.
    """

    if patterns:
        selected: list[Path] = []

        for pattern in patterns:
            candidate = Path(pattern)

            if candidate.is_file():
                selected.append(candidate)
                continue

            selected.extend(
                sorted(
                    path
                    for path in models_dir.glob(pattern)
                    if path.suffix == ".tflite"
                )
            )

        # Preserve order, drop duplicates.
        seen = set()
        unique = []

        for path in selected:
            if path not in seen:
                seen.add(path)
                unique.append(path)

        return unique

    return sorted(models_dir.glob("*.tflite"))


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h{minutes:02d}m"

    if minutes:
        return f"{minutes}m{secs:02d}s"

    return f"{secs}s"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    hosts = parser.add_argument_group("hosts")
    hosts.add_argument(
        "--device",
        required=True,
        help="ssh target for the board, e.g. root@192.168.11.131",
    )
    hosts.add_argument(
        "--meter-host",
        required=True,
        help="ssh target for the lab server with the FNB58 attached",
    )
    hosts.add_argument(
        "--device-repo",
        default=None,
        help=(
            "Repository path as the board sees it over NFS. Defaults to this "
            "host's repo path, which is correct when the mount point matches"
        ),
    )
    hosts.add_argument(
        "--device-python",
        default="python3",
        help="python on the board (default: %(default)s)",
    )
    hosts.add_argument(
        "--meter-python",
        default="python3",
        help="python on the lab server; needs pyusb (default: %(default)s)",
    )
    hosts.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        metavar="OPT",
        help="extra ssh option, e.g. -oBatchMode=yes (repeatable)",
    )

    selection = parser.add_argument_group("selection")
    chooser = selection.add_mutually_exclusive_group()
    chooser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="GLOB",
        help=(
            "Model paths or globs relative to artifacts/tflite "
            "(default: every .tflite there)"
        ),
    )
    chooser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help=(
            "Named selection. 'arch-matrix' keeps one model per "
            "architecture x precision (13 instead of 43): class count and "
            "training tiling were measured not to affect resource cost — "
            "see PRESETS in this file for the numbers"
        ),
    )
    selection.add_argument(
        "--images",
        default=None,
        help=(
            "Image directory on the board (default: "
            "<device-repo>/datasets/test-bundle/images). Tiling is irrelevant "
            "here — the loop only needs representative inputs"
        ),
    )
    selection.add_argument(
        "--output",
        default=None,
        help=(
            "Sweep directory; must be inside the repo so the board can write "
            "to it over NFS (default: benchmark_results/power/<host>/<stamp>)"
        ),
    )
    selection.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip runs whose run.json already exists",
    )

    load = parser.add_argument_group("load")
    load.add_argument("--seconds", type=float, default=120.0)
    load.add_argument("--warmup", type=int, default=20)
    load.add_argument("--pool-size", type=int, default=16)
    load.add_argument("--sample-interval", type=float, default=0.2)
    load.add_argument("--chirp-seconds", type=float, default=1.0)
    load.add_argument("--gap-seconds", type=float, default=5.0)
    load.add_argument(
        "--cooldown",
        type=float,
        default=30.0,
        help=(
            "Idle seconds between runs, so each starts from a comparable "
            "thermal state (default: %(default)s)"
        ),
    )
    load.add_argument(
        "--delegate",
        default="/usr/lib/libteflon.so",
        help="delegate path on the board (default: %(default)s)",
    )
    load.add_argument(
        "--cpu",
        action="store_true",
        help="disable the delegate; required for trustworthy fp32 numbers",
    )
    load.add_argument(
        "-e",
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Environment variable for the board run, forwarded to "
            "`ave resources` (repeatable). Note that TEFLON_DEBUG=verbose et al "
            "add console I/O to a measured window — useful for a delegation "
            "probe, not for a power number"
        ),
    )
    load.add_argument(
        "--overhead-estimate",
        type=float,
        default=30.0,
        help=(
            "Assumed per-run setup cost (pool decode, model load, delegate "
            "compile) used only for the up-front time estimate "
            "(default: %(default)s)"
        ),
    )

    meter = parser.add_argument_group("meter")
    meter.add_argument(
        "--meter-logger",
        default=str(REPO_ROOT / "scripts" / "fnirsi_logger.py"),
        help="local path to the FNB58 logger, copied to the lab server",
    )
    meter.add_argument(
        "--meter-settle",
        type=float,
        default=3.0,
        help=(
            "Seconds of logging before and after the board run, so the trace "
            "brackets the whole window (default: %(default)s)"
        ),
    )
    meter.add_argument(
        "--no-meter",
        action="store_true",
        help="skip the power meter entirely (resource-only sweep)",
    )

    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    delegate = "none" if args.cpu else args.delegate

    models_dir = REPO_ROOT / "artifacts" / "tflite"
    patterns = PRESETS[args.preset] if args.preset else args.models
    models = collect_models(models_dir, patterns)

    if not models:
        print(f"no models selected in {models_dir}", file=sys.stderr)
        return 1

    # A preset that silently matches nothing would shrink the sweep without
    # saying so, and the gap would only surface once the report came up short.
    if args.preset:
        for pattern in PRESETS[args.preset]:
            if not any(path.match(pattern) for path in models):
                print(
                    f"[warning] preset {args.preset!r} pattern {pattern!r} "
                    f"matched no model in {models_dir}",
                    file=sys.stderr,
                )

    device_repo = Path(args.device_repo) if args.device_repo else REPO_ROOT

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    device_tag = args.device.split("@")[-1].replace(":", "_")

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        suffix = "_cpu" if args.cpu else ""
        output_dir = (
            REPO_ROOT / "benchmark_results" / "power" / f"{device_tag}{suffix}" / stamp
        )

    # The board reaches the results tree only through the NFS mount, so the
    # sweep directory has to be expressible as a repo-relative path.
    try:
        relative_output = output_dir.relative_to(REPO_ROOT)
    except ValueError:
        print(
            f"--output must be inside the repository ({REPO_ROOT}); "
            f"the board writes there over NFS",
            file=sys.stderr,
        )
        return 1

    device_output = device_repo / relative_output
    device_images = (
        Path(args.images)
        if args.images
        else device_repo / "datasets" / "test-bundle" / "images"
    )

    per_run = (
        args.seconds
        + 2 * args.gap_seconds
        + 2 * args.chirp_seconds
        + 2 * args.meter_settle
        + args.overhead_estimate
        + args.cooldown
    )

    print(
        f"models          : {len(models)}"
        + (f"  (preset {args.preset})" if args.preset else "")
    )
    print(f"device          : {args.device}  (repo {device_repo})")
    print(f"meter host      : {'disabled' if args.no_meter else args.meter_host}")
    print(f"delegate        : {delegate}")
    print(f"output          : {output_dir}")
    print(f"device sees     : {device_output}")
    print(
        f"estimated time  : {format_duration(per_run * len(models))} "
        f"({format_duration(per_run)} per run)"
    )
    print()

    device = Remote(args.device, args.ssh_option, args.device_python)
    meter_host = Remote(args.meter_host, args.ssh_option, args.meter_python)

    remote_logger = f"/tmp/fnirsi_logger_{stamp}.py"

    sweep: dict[str, Any] = {
        "schema": 1,
        "started": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "meter_host": None if args.no_meter else args.meter_host,
        "device_repo": str(device_repo),
        "delegate": delegate,
        "config": {
            "seconds": args.seconds,
            "warmup": args.warmup,
            "pool_size": args.pool_size,
            "sample_interval": args.sample_interval,
            "chirp_seconds": args.chirp_seconds,
            "gap_seconds": args.gap_seconds,
            "cooldown": args.cooldown,
            "meter_settle": args.meter_settle,
        },
        "images": str(device_images),
        "runs": [],
    }

    clock_sync: dict[str, Any] = {
        "schema": 1,
        "method": (
            "Christian's algorithm over ssh; fastest of N round trips, "
            "uncertainty = rtt/2. offset_s converts local -> remote "
            "(remote = local + offset_s). Cross-check against the chirp edges "
            "in the power trace via power_report.py."
        ),
        "probes": [],
    }

    if args.dry_run:
        print("--- dry run: commands that would be issued ---\n")

    def device_command(model: Path, run_name: str) -> str:
        env_args: list[str] = []

        for entry in args.env:
            env_args += ["--env", entry]

        return " ".join(
            shlex.quote(part)
            for part in [
                str(device_repo / "scripts" / "ave"),
                "resources",
                str(device_repo / "artifacts" / "tflite" / model.name),
                str(device_images),
                "--output-dir",
                str(device_output),
                "--seconds",
                str(args.seconds),
                "--warmup",
                str(args.warmup),
                "--pool-size",
                str(args.pool_size),
                "--sample-interval",
                str(args.sample_interval),
                "--chirp-seconds",
                str(args.chirp_seconds),
                "--gap-seconds",
                str(args.gap_seconds),
                "--delegate",
                delegate,
                "--label",
                run_name,
                *env_args,
            ]
        )

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_meter:
            # Staged over stdin rather than scp'd: the lab server needs no
            # checkout of this repo, and the logger that runs is by
            # construction the one in this tree.
            logger_source = Path(args.meter_logger).read_text()

            copied = subprocess.run(
                [*meter_host.ssh, f"cat > {shlex.quote(remote_logger)}"],
                input=logger_source,
                capture_output=True,
                text=True,
                timeout=30.0,
            )

            if copied.returncode != 0:
                print(f"failed to copy logger to {args.meter_host}: {copied.stderr}")
                return 1

            print(f"logger staged at {args.meter_host}:{remote_logger}")

        sweep["device_info"] = device.run(
            "uname -a; cat /proc/cpuinfo | head -30"
        ).stdout

    completed = failed = skipped = 0

    try:
        for index, model in enumerate(models, start=1):
            run_name = model.stem
            run_dir = output_dir / run_name
            remote_power = f"/tmp/power_{stamp}_{run_name}.csv.gz"

            header = f"[{index}/{len(models)}] {run_name}"

            if args.skip_existing and (run_dir / "run.json").exists():
                print(f"{header}  [skip] already present")
                skipped += 1
                continue

            command = device_command(model, run_name)

            if args.dry_run:
                print(header)

                if not args.no_meter:
                    print(
                        f"  meter start : ssh {args.meter_host} "
                        f"'setsid nohup {args.meter_python} {remote_logger} "
                        f"-o {remote_power} &'"
                    )

                print(f"  device      : ssh {args.device} {command}")

                if not args.no_meter:
                    print(f"  meter stop  : ssh {args.meter_host} 'kill -INT <pid>'")
                    print(
                        f"  fetch       : ssh {args.meter_host} 'cat {remote_power}' "
                        f"> {run_dir / 'power.csv.gz'}"
                    )

                print(f"  cooldown    : {args.cooldown}s\n")
                continue

            print(header, flush=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            record: dict[str, Any] = {"run": run_name, "model": model.name}
            meter_pid = None

            # The board's clock is probed too, and it is not optional: every
            # phase boundary and resource sample is stamped in the *board's*
            # epoch clock, so without this offset there is nothing to align the
            # power trace to. (The meter offset alone only gets the trace onto
            # this host's clock.)
            device_probe_before = probe_clock(device)
            device_probe_before["phase"] = "before"
            device_probe_before["run"] = run_name
            device_probe_before["host"] = "device"
            clock_sync["probes"].append(device_probe_before)
            record["device_clock_probe_before"] = device_probe_before

            # ---- clock probes + meter start ----
            if not args.no_meter:
                probe_before = probe_clock(meter_host)
                probe_before["phase"] = "before"
                probe_before["run"] = run_name
                probe_before["host"] = "meter"
                clock_sync["probes"].append(probe_before)
                # Recorded here, at the point it is taken, rather than at the
                # end of the run: the tail of this block has several early
                # exits, and a probe stored only on the happy path is a probe
                # that goes missing exactly when a run needs explaining.
                record["meter_clock_probe_before"] = probe_before

                started = meter_host.script(
                    "set -e\n"
                    f"setsid nohup {shlex.quote(args.meter_python)} "
                    f"{shlex.quote(remote_logger)} -o {shlex.quote(remote_power)} "
                    f"> /tmp/meter_{stamp}.log 2>&1 < /dev/null &\n"
                    "echo PID:$!\n"
                    f"{shlex.quote(args.meter_python)} -c {shlex.quote(CLOCK_SNIPPET)}\n",
                    timeout=30.0,
                )

                for line in started.stdout.splitlines():
                    if line.startswith("PID:"):
                        meter_pid = int(line[4:])
                    elif line.startswith("("):
                        epoch, monotonic = eval(line)  # noqa: S307
                        record["meter_anchor"] = {
                            "t_epoch_s": epoch,
                            "t_monotonic_ns": monotonic,
                        }

                if meter_pid is None:
                    print(f"  [error] meter did not start: {started.stderr.strip()}")
                    failed += 1
                    continue

                record["meter_pid"] = meter_pid
                print(f"  meter running (pid {meter_pid}), settling…", flush=True)
                time.sleep(args.meter_settle)

            # ---- the run itself ----
            record["device_start_local_s"] = time.time()

            result = subprocess.run(
                [*device.ssh, command],
                capture_output=True,
                text=True,
                timeout=None,
            )

            record["device_end_local_s"] = time.time()
            record["returncode"] = result.returncode

            device_probe_after = probe_clock(device)
            device_probe_after["phase"] = "after"
            device_probe_after["run"] = run_name
            device_probe_after["host"] = "device"
            clock_sync["probes"].append(device_probe_after)
            record["device_clock_probe_after"] = device_probe_after

            (run_dir / "console.log").write_text(
                f"$ ssh {args.device} {command}\n\n{result.stdout}\n{result.stderr}"
            )

            if result.returncode != 0:
                print(f"  [error] device run failed (rc={result.returncode})")
                print("  " + result.stderr.strip()[-500:].replace("\n", "\n  "))
                failed += 1
            else:
                tail = [
                    line for line in result.stdout.splitlines() if "iterations" in line
                ]
                print("  " + (tail[-1] if tail else "done"))
                completed += 1

            # ---- meter stop + fetch ----
            if not args.no_meter and meter_pid is not None:
                time.sleep(args.meter_settle)

                probe_after = probe_clock(meter_host)
                probe_after["phase"] = "after"
                probe_after["run"] = run_name
                probe_after["host"] = "meter"
                clock_sync["probes"].append(probe_after)
                record["meter_clock_probe_after"] = probe_after

                # SIGINT, not SIGTERM: the logger traps KeyboardInterrupt to
                # flush and close its stream cleanly.
                meter_host.run(f"kill -INT {meter_pid}", timeout=30.0)
                time.sleep(1.0)

                fetched = subprocess.run(
                    [*meter_host.ssh, f"cat {shlex.quote(remote_power)}"],
                    capture_output=True,
                    timeout=300.0,
                )

                if fetched.returncode == 0 and fetched.stdout:
                    (run_dir / "power.csv.gz").write_bytes(fetched.stdout)
                    record["power_bytes"] = len(fetched.stdout)
                    print(f"  power trace: {len(fetched.stdout) / 1024:.0f} KiB")
                else:
                    print("  [warning] no power trace retrieved")
                    record["power_bytes"] = 0

                meter_host.run(f"rm -f {shlex.quote(remote_power)}", timeout=30.0)

            sweep["runs"].append(record)

            (run_dir / "sweep_run.json").write_text(json.dumps(record, indent=2) + "\n")

            if index < len(models) and args.cooldown > 0:
                print(f"  cooldown {args.cooldown:.0f}s", flush=True)
                time.sleep(args.cooldown)

    except KeyboardInterrupt:
        print("\ninterrupted — stopping meter and saving what we have")

    finally:
        if not args.dry_run:
            if not args.no_meter:
                # Belt and braces: a logger orphaned by an interrupt would keep
                # the meter busy and silently corrupt the next sweep.
                meter_host.run(
                    f"pkill -INT -f {shlex.quote(remote_logger)} || true", timeout=30.0
                )
                meter_host.run(f"rm -f {shlex.quote(remote_logger)}", timeout=30.0)

            sweep["finished"] = datetime.now(timezone.utc).isoformat()
            sweep["completed"] = completed
            sweep["failed"] = failed
            sweep["skipped"] = skipped

            (output_dir / "sweep.json").write_text(json.dumps(sweep, indent=2) + "\n")
            (output_dir / "clock_sync.json").write_text(
                json.dumps(clock_sync, indent=2) + "\n"
            )

    if args.dry_run:
        print(f"dry run: {len(models)} run(s) planned")
        return 0

    print(f"\ndone: {completed} completed, {failed} failed, {skipped} skipped")
    print(f"artifacts: {output_dir}")
    print(f"next: scripts/power_report.py {output_dir}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
