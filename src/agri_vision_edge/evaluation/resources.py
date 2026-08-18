"""Device-side CPU, memory, thermal, and frequency sampling.

Uses only ``/proc``, ``/sys``, and the standard library. Samples are buffered
until stop to avoid measurement-side I/O, and process CPU accounting carries
exited children forward.
"""

from __future__ import annotations

import csv
import gzip
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

CLOCK_TICKS = os.sysconf("SC_CLK_TCK")

#: ``/proc/stat`` cpu-line fields, in kernel order.
CPU_FIELDS = (
    "user",
    "nice",
    "system",
    "idle",
    "iowait",
    "irq",
    "softirq",
    "steal",
)

#: ``/proc/meminfo`` keys retained, in output order.
MEM_FIELDS = (
    "MemTotal",
    "MemFree",
    "MemAvailable",
    "Buffers",
    "Cached",
    "SwapTotal",
    "SwapFree",
)


def read_text(path: str) -> str | None:
    try:
        with open(path) as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        return None


def read_int_file(path: str) -> int | str:
    content = read_text(path)

    if content is None:
        return ""

    try:
        return int(content.strip())
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# system-wide counters
# ---------------------------------------------------------------------------


def read_cpu_times() -> dict[str, list[int]]:
    content = read_text("/proc/stat")

    if content is None:
        return {}

    result: dict[str, list[int]] = {}

    for line in content.splitlines():
        if not line.startswith("cpu"):
            continue

        parts = line.split()
        result[parts[0]] = [int(value) for value in parts[1:]]

    return result


def cpu_deltas(
    previous: dict[str, list[int]],
    current: dict[str, list[int]],
) -> dict[str, tuple[float, dict[str, float]]]:
    """Return CPU busy/state fractions; ``iowait`` counts as idle."""

    result: dict[str, tuple[float, dict[str, float]]] = {}

    for name, now in current.items():
        before = previous.get(name)

        if before is None:
            continue

        deltas = [a - b for a, b in zip(now, before, strict=False)]
        total = sum(deltas)

        if total <= 0:
            result[name] = (0.0, dict.fromkeys(CPU_FIELDS, 0.0))
            continue

        states = {
            field: (deltas[index] / total if index < len(deltas) else 0.0)
            for index, field in enumerate(CPU_FIELDS)
        }

        idle = states.get("idle", 0.0) + states.get("iowait", 0.0)
        result[name] = (1.0 - idle, states)

    return result


def read_meminfo() -> dict[str, int]:
    content = read_text("/proc/meminfo")

    if content is None:
        return {}

    wanted = set(MEM_FIELDS)
    result: dict[str, int] = {}

    for line in content.splitlines():
        key, _, rest = line.partition(":")

        if key in wanted:
            result[key] = int(rest.split()[0])

    return result


def read_loadavg() -> float:
    content = read_text("/proc/loadavg")

    if content is None:
        return 0.0

    return float(content.split()[0])


# ---------------------------------------------------------------------------
# thermal / frequency
# ---------------------------------------------------------------------------


def discover_thermal_zones() -> list[tuple[str, str]]:
    zones: list[tuple[str, str]] = []
    base = Path("/sys/class/thermal")

    if not base.is_dir():
        return zones

    for zone in sorted(base.glob("thermal_zone*")):
        temp = zone / "temp"

        if not temp.exists():
            continue

        label = (read_text(str(zone / "type")) or zone.name).strip()
        label = "".join(c if c.isalnum() else "_" for c in label)
        zones.append((f"{zone.name}_{label}", str(temp)))

    return zones


def discover_cpufreq() -> list[tuple[str, str]]:
    freqs: list[tuple[str, str]] = []
    base = Path("/sys/devices/system/cpu")

    if not base.is_dir():
        return freqs

    for cpu in sorted(base.glob("cpu[0-9]*")):
        path = cpu / "cpufreq" / "scaling_cur_freq"

        if path.exists():
            freqs.append((cpu.name, str(path)))

    return freqs


# ---------------------------------------------------------------------------
# process tree
# ---------------------------------------------------------------------------


def parse_proc_stat(pid: int) -> tuple[int, int, int] | None:
    """Return ``(ppid, CPU jiffies, num_threads)``, or ``None`` if gone."""

    content = read_text(f"/proc/{pid}/stat")

    if content is None:
        return None

    # /proc/<pid>/stat comm may contain spaces/parentheses; split at the last ')'.
    close = content.rfind(")")

    if close < 0:
        return None

    # Fields after comm: [0] state, [1] ppid, [11] utime, [12] stime,
    # [17] num_threads.
    fields = content[close + 2 :].split()

    try:
        return (
            int(fields[1]),
            int(fields[11]) + int(fields[12]),
            int(fields[17]),
        )
    except (IndexError, ValueError):
        return None


def parse_proc_status(pid: int) -> tuple[int, int, int]:
    """Return ``(VmRSS, VmHWM, VmSize)`` in kB."""

    content = read_text(f"/proc/{pid}/status")

    if content is None:
        return 0, 0, 0

    rss = hwm = vsize = 0

    for line in content.splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1])
        elif line.startswith("VmHWM:"):
            hwm = int(line.split()[1])
        elif line.startswith("VmSize:"):
            vsize = int(line.split()[1])

    return rss, hwm, vsize


def descendants(root_pid: int, exclude: set[int] | None = None) -> set[int]:
    """Return the live process tree rooted at ``root_pid``."""

    exclude = exclude or set()
    parents: dict[int, int] = {}

    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue

        pid = int(entry)

        if pid in exclude:
            continue

        info = parse_proc_stat(pid)

        if info is not None:
            parents[pid] = info[0]

    tree = {root_pid}

    changed = True

    while changed:
        changed = False

        for pid, ppid in parents.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True

    return tree


class ProcessTreeSampler:
    """Aggregate CPU and memory over a process tree."""

    def __init__(
        self,
        root_pid: int,
        refresh_seconds: float = 1.0,
        exclude: set[int] | None = None,
    ):
        self.root_pid = root_pid
        self.refresh_seconds = refresh_seconds
        self.exclude = exclude or set()
        self.tracked = {root_pid}
        self.last_refresh = 0.0

        # Per-PID ticks let exited children remain in cumulative CPU accounting.
        self.previous_ticks: dict[int, int] = {}
        self.exited_ticks = 0
        self.previous_total = 0

        self.peak_rss_kb = 0
        self.peak_hwm_kb = 0

    def sample(
        self,
        now_monotonic: float,
        interval_seconds: float,
        cpu_count: int,
    ) -> dict[str, float]:
        if now_monotonic - self.last_refresh >= self.refresh_seconds:
            try:
                self.tracked = descendants(self.root_pid, self.exclude)
            except OSError:
                pass

            self.last_refresh = now_monotonic

        current_ticks: dict[int, int] = {}
        rss_kb = hwm_kb = vsize_kb = 0
        threads = 0
        alive = 0

        for pid in self.tracked:
            info = parse_proc_stat(pid)

            if info is None:
                continue

            _, ticks, thread_count = info
            current_ticks[pid] = ticks
            threads += thread_count
            alive += 1

            pid_rss, pid_hwm, pid_vsize = parse_proc_status(pid)
            rss_kb += pid_rss
            hwm_kb += pid_hwm
            vsize_kb += pid_vsize

        # Preserve final ticks for children that exited between samples.
        for pid, ticks in self.previous_ticks.items():
            if pid not in current_ticks:
                self.exited_ticks += ticks

        had_previous = bool(self.previous_ticks)
        self.previous_ticks = current_ticks

        # Keep the total monotonic: live ticks plus final ticks of exited PIDs.
        current_total = sum(current_ticks.values()) + self.exited_ticks

        delta_ticks = current_total - self.previous_total if had_previous else 0

        self.previous_total = current_total

        if interval_seconds > 0 and delta_ticks > 0:
            core_fraction = (delta_ticks / CLOCK_TICKS) / interval_seconds
        else:
            core_fraction = 0.0

        self.peak_rss_kb = max(self.peak_rss_kb, rss_kb)
        self.peak_hwm_kb = max(self.peak_hwm_kb, hwm_kb)

        return {
            # Core fraction is comparable across different CPU counts.
            "proc_cpu_cores": core_fraction,
            "proc_cpu_pct": 100.0 * core_fraction / cpu_count if cpu_count else 0.0,
            "proc_rss_kb": rss_kb,
            "proc_hwm_kb": hwm_kb,
            "proc_vsize_kb": vsize_kb,
            "proc_threads": threads,
            "proc_count": alive,
        }


# ---------------------------------------------------------------------------
# sync chirp
# ---------------------------------------------------------------------------


def chirp(duration_seconds: float, workers: int) -> tuple[float, float]:
    """Burn ``workers`` cores briefly and return epoch start/end timestamps."""

    start = time.time()
    deadline = time.monotonic() + duration_seconds

    children: list[int] = []

    # fork avoids multiprocessing's POSIX semaphore dependency on minimal targets.
    for _ in range(max(1, workers) - 1):
        try:
            pid = os.fork()
        except OSError:
            break

        if pid == 0:
            try:
                _burn(deadline)
            finally:
                os._exit(0)

        children.append(pid)

    _burn(deadline)

    for pid in children:
        try:
            os.waitpid(pid, 0)
        except OSError:
            pass

    return start, time.time()


def _burn(deadline: float) -> None:
    value = 0

    while time.monotonic() < deadline:
        for _ in range(10000):
            value += 1


# ---------------------------------------------------------------------------
# static system description
# ---------------------------------------------------------------------------


def system_info() -> dict[str, Any]:
    """Return a one-shot description of the target system."""

    uname = os.uname()
    model = read_text("/sys/firmware/devicetree/base/model")
    governor = read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")

    return {
        "hostname": uname.nodename,
        "kernel": uname.release,
        "machine": uname.machine,
        # Hostnames are image-defined; the DT model identifies the board.
        "device_tree_model": model.strip("\x00\n") if model else None,
        "cpu_count": os.cpu_count(),
        "clock_ticks_per_second": CLOCK_TICKS,
        "scaling_governor": governor.strip() if governor else None,
        "mem_total_kb": read_meminfo().get("MemTotal"),
        "python": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# sampler thread
# ---------------------------------------------------------------------------


class ResourceSampler:
    """Sample system-wide and process-tree resources at a fixed cadence."""

    def __init__(
        self,
        *,
        interval: float = 0.2,
        root_pid: int | None = None,
        tree_refresh: float = 1.0,
        exclude: set[int] | None = None,
    ):
        self.interval = interval
        self.root_pid = root_pid if root_pid is not None else os.getpid()
        self.cpu_count = os.cpu_count() or 1

        self.thermal_zones = discover_thermal_zones()
        self.cpufreqs = discover_cpufreq()
        self.cpu_names = sorted(read_cpu_times().keys())

        self._tree = ProcessTreeSampler(
            self.root_pid,
            refresh_seconds=tree_refresh,
            exclude=exclude,
        )
        self._rows: list[list[object]] = []
        self._intervals: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.anchor_start: dict[str, float] | None = None
        self.anchor_end: dict[str, float] | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self.anchor_start = _anchor()
        self._thread = threading.Thread(
            target=self._run,
            name="resource-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        self.anchor_end = _anchor()

    def request_stop(self) -> None:
        """Request sampler shutdown."""

        self._stop.set()

    def run_foreground(self) -> None:
        """Sample in the calling thread until stopped."""

        self.anchor_start = _anchor()
        self._run()
        self.anchor_end = _anchor()

    # -- health ------------------------------------------------------------

    def cadence_report(self) -> dict[str, Any]:
        """Summarize the observed sampling cadence."""

        if not self._intervals:
            return {
                "requested_interval_s": self.interval,
                "samples": 0,
            }

        ordered = sorted(self._intervals)
        count = len(ordered)

        return {
            "requested_interval_s": self.interval,
            "samples": count,
            "interval_mean_s": sum(ordered) / count,
            "interval_p50_s": ordered[count // 2],
            "interval_p95_s": ordered[min(count - 1, int(0.95 * count))],
            "interval_max_s": ordered[-1],
        }

    # -- output ------------------------------------------------------------

    @property
    def header(self) -> list[str]:
        header = ["t_epoch_s", "t_monotonic_ns", "cpu_pct"]
        header += [f"cpu_{field}_pct" for field in CPU_FIELDS]
        header += [f"{name}_pct" for name in self.cpu_names if name != "cpu"]
        header += ["loadavg_1"]
        header += [f"{field.lower()}_kb" for field in MEM_FIELDS]
        header += [
            "proc_cpu_pct",
            "proc_cpu_cores",
            "proc_rss_kb",
            "proc_hwm_kb",
            "proc_vsize_kb",
            "proc_threads",
            "proc_count",
        ]
        header += [f"temp_{label}_mc" for label, _ in self.thermal_zones]
        header += [f"freq_{label}_khz" for label, _ in self.cpufreqs]
        return header

    @property
    def sample_count(self) -> int:
        return len(self._rows)

    @property
    def peak_rss_kb(self) -> int:
        return self._tree.peak_rss_kb

    @property
    def peak_hwm_kb(self) -> int:
        return self._tree.peak_hwm_kb

    def write_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.suffix == ".gz":
            stream = gzip.open(str(destination), "wt", newline="")
        else:
            stream = open(str(destination), "w", newline="")

        with stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(self.header)
            writer.writerows(self._rows)

    # -- internals ---------------------------------------------------------

    def _run(self) -> None:
        previous_cpu = read_cpu_times()
        previous_monotonic = time.monotonic()

        # Absolute deadlines avoid work-time drift; missed deadlines are skipped.
        next_at = time.monotonic() + self.interval

        while True:
            delay = next_at - time.monotonic()

            if self._stop.wait(delay if delay > 0 else 0):
                break

            next_at += self.interval

            if next_at < time.monotonic():
                next_at = time.monotonic() + self.interval

            now_monotonic = time.monotonic()
            interval = now_monotonic - previous_monotonic
            previous_monotonic = now_monotonic
            self._intervals.append(interval)

            current_cpu = read_cpu_times()
            deltas = cpu_deltas(previous_cpu, current_cpu)
            previous_cpu = current_cpu

            aggregate, states = deltas.get("cpu", (0.0, {}))
            meminfo = read_meminfo()
            proc = self._tree.sample(now_monotonic, interval, self.cpu_count)

            row: list[object] = [
                f"{time.time():.6f}",
                time.monotonic_ns(),
                "%.4f" % (100.0 * aggregate),
            ]
            row += ["%.4f" % (100.0 * states.get(field, 0.0)) for field in CPU_FIELDS]
            row += [
                "%.4f" % (100.0 * deltas.get(name, (0.0, {}))[0])
                for name in self.cpu_names
                if name != "cpu"
            ]
            row += [f"{read_loadavg():.2f}"]
            row += [meminfo.get(field, "") for field in MEM_FIELDS]
            row += [
                "{:.4f}".format(proc["proc_cpu_pct"]),
                "{:.4f}".format(proc["proc_cpu_cores"]),
                proc["proc_rss_kb"],
                proc["proc_hwm_kb"],
                proc["proc_vsize_kb"],
                proc["proc_threads"],
                proc["proc_count"],
            ]
            row += [read_int_file(path) for _, path in self.thermal_zones]
            row += [read_int_file(path) for _, path in self.cpufreqs]

            self._rows.append(row)


def _anchor() -> dict[str, float]:
    """Return paired epoch and monotonic timestamps."""

    return {
        "t_epoch_s": time.time(),
        "t_monotonic_ns": time.monotonic_ns(),
    }


class SamplerProcess:
    """Handle for a sampler running in a forked child."""

    def __init__(self, pid: int, csv_path: Path, meta_path: Path):
        self.pid = pid
        self.csv_path = csv_path
        self.meta_path = meta_path

    def stop(self, timeout: float = 15.0) -> dict[str, Any]:
        """Stop the sampler process and return its metadata sidecar."""

        import json
        import signal as signal_module

        try:
            os.kill(self.pid, signal_module.SIGTERM)
        except OSError:
            pass

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                waited, _ = os.waitpid(self.pid, os.WNOHANG)
            except OSError:
                break

            if waited == self.pid:
                break

            time.sleep(0.05)
        else:
            try:
                os.kill(self.pid, signal_module.SIGKILL)
                os.waitpid(self.pid, 0)
            except OSError:
                pass

        try:
            with open(self.meta_path) as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}


def spawn_sampler(
    *,
    csv_path: str | Path,
    meta_path: str | Path,
    interval: float = 0.2,
    tree_refresh: float = 1.0,
) -> SamplerProcess:
    """Fork a child sampler for the calling process tree."""

    import json
    import signal as signal_module

    csv_path = Path(csv_path)
    meta_path = Path(meta_path)
    parent_pid = os.getpid()

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Separate process avoids GIL starvation and keeps sampling off the workload.
    pid = os.fork()

    if pid != 0:
        return SamplerProcess(pid, csv_path, meta_path)

    try:
        sampler = ResourceSampler(
            interval=interval,
            root_pid=parent_pid,
            tree_refresh=tree_refresh,
            exclude={os.getpid()},
        )

        signal_module.signal(
            signal_module.SIGTERM,
            lambda *_: sampler.request_stop(),
        )
        signal_module.signal(
            signal_module.SIGINT,
            lambda *_: sampler.request_stop(),
        )

        sampler.run_foreground()
        sampler.write_csv(csv_path)

        with open(meta_path, "w") as handle:
            json.dump(
                {
                    "sampler_pid": os.getpid(),
                    "measured_pid": parent_pid,
                    "anchor_start": sampler.anchor_start,
                    "anchor_end": sampler.anchor_end,
                    "cadence": sampler.cadence_report(),
                    "peak_proc_rss_kb": sampler.peak_rss_kb,
                    "peak_proc_hwm_kb": sampler.peak_hwm_kb,
                    "thermal_zones": [label for label, _ in sampler.thermal_zones],
                    "cpufreq": [label for label, _ in sampler.cpufreqs],
                },
                handle,
                indent=2,
            )
            handle.write("\n")
    finally:
        os._exit(0)
