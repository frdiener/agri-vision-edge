"""Parse delegation continuity from ``delegate_debug.log`` files.

Teflon selects supported operations; TensorFlow Lite groups them into maximal
dependency-preserving regions. The parser accepts current Etnaviv subgraph lines
and older Ethos-U ``compiling graph`` lines, making partition counts comparable
across backends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

#: ``Teflon delegate: loaded <name> driver``
_BACKEND_RE = re.compile(r"loaded\s+(\w+)\s+driver")

#: One operation, either dialect. ``op: N`` and the glyph are Etnaviv-only.
#: ``kind`` accepts lower case on purpose. Etnaviv prints the operator's name
#: only when it recognises the builtin code, and writes ``unknown`` otherwise.
#: Every SSD export sends builtin 32, ``CUSTOM``, through this path for
#: ``TFLite_Detection_PostProcess``. Requiring uppercase previously omitted the
#: operator from totals and the rejection list.
_OP_RE = re.compile(
    r"^\s*(?P<idx>\d+)\s+"
    r"(?:op:\s*(?P<code>\d+)\s+)?"
    r"(?P<kind>[A-Za-z0-9_]+)\s+"
    r"v(?P<ver>\d+)\s+"
    r"(?:\S\s*)?"
    r"(?P<status>supported|unsupported)\b"
)

#: TFLite ``BuiltinOperator.CUSTOM``.
_CUSTOM_BUILTIN_CODE = "32"

#: ``teflon: ===== subgraph #0: 94 operations, 255 tensors =====`` (Etnaviv)
_SUBGRAPH_RE = re.compile(r"subgraph\s+#(?P<idx>\d+):\s*(?P<ops>\d+)\s+operations")

#: ``teflon: compiling graph: 255 tensors 98 operations`` (Ethos-U)
_COMPILE_RE = re.compile(
    r"compiling graph:\s*(?P<tensors>\d+)\s+tensors\s+(?P<ops>\d+)\s+operations"
)


@dataclass
class Delegation:
    """Partition structure recovered from one ``delegate_debug.log``."""

    backend: str | None = None
    ops_total: int = 0
    ops_delegated: int = 0
    ops_rejected: int = 0
    partitions: list[int] = field(default_factory=list)
    rejected_kinds: dict[str, int] = field(default_factory=dict)

    @property
    def k(self) -> int | None:
        """Number of delegated regions, or ``None`` when nothing was delegated.

        Each backend prints every region once, so ``K`` counts those lines. A
        value of 1 records a contiguous accepted set. Ethos-U reports two
        regions for YOLOv7-tiny exports.
        """
        return len(self.partitions) or None

    @property
    def largest(self) -> int | None:
        return max(self.partitions, default=None)

    @property
    def r_ops(self) -> float | None:
        """Delegated share of all operations the delegate was shown."""
        return self.ops_delegated / self.ops_total if self.ops_total else None

    @property
    def r_largest(self) -> float | None:
        """Share of the delegated operations sitting in the largest partition.

        The denominator includes only delegated operations. This distinguishes
        a small contiguous accepted region from a scattered one.
        """
        if not self.ops_delegated or self.largest is None:
            return None
        return self.largest / self.ops_delegated


def parse_delegate_log(path: str | Path) -> Delegation:
    """Recover the partition structure from one delegate debug log."""
    out = Delegation()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if out.backend is None:
            m = _BACKEND_RE.search(line)
            if m:
                out.backend = m.group(1)
                continue

        m = _SUBGRAPH_RE.search(line)
        if m:
            out.partitions.append(int(m.group("ops")))
            continue

        m = _COMPILE_RE.search(line)
        if m:
            # The i.MX93 image uses this older wording once per region.
            out.partitions.append(int(m.group("ops")))
            continue

        m = _OP_RE.match(line)
        if m:
            out.ops_total += 1
            if m.group("status") == "supported":
                out.ops_delegated += 1
            else:
                out.ops_rejected += 1
                kind = m.group("kind")
                # Name the operator the log could not: a bare "unknown" in the
                # rejection list tells a reader nothing.
                if kind.lower() == "unknown":
                    kind = (
                        "CUSTOM"
                        if m.group("code") == _CUSTOM_BUILTIN_CODE
                        else kind.upper()
                    )
                out.rejected_kinds[kind] = out.rejected_kinds.get(kind, 0) + 1
    return out


def load_delegation(benchmark_root: str | Path) -> pd.DataFrame:
    """One row per run that carries a delegate debug log.

    Joins on ``platform``/``run`` so the frame can be merged with the result of
    :func:`agri_vision_edge.evaluation.benchmark_report.load_benchmark_results`.
    """
    root = Path(benchmark_root)
    rows = []
    for log in sorted(root.glob("*/*/delegate_debug.log")):
        d = parse_delegate_log(log)
        rows.append(
            {
                "platform": log.parent.parent.name,
                "run": log.parent.name,
                "backend_driver": d.backend,
                "ops_total": d.ops_total,
                "ops_delegated": d.ops_delegated,
                "ops_rejected": d.ops_rejected,
                "K": d.k,
                "largest_partition": d.largest,
                "r_ops": d.r_ops,
                "r_largest": d.r_largest,
                "partitions": d.partitions,
                "top_rejected": ", ".join(
                    f"{k}x{v}"
                    for k, v in sorted(d.rejected_kinds.items(), key=lambda kv: -kv[1])[
                        :3
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def continuity_table(
    df: pd.DataFrame,
    *,
    platform: str,
    eval_tiling: str = "untiled",
    percent: bool = True,
) -> pd.DataFrame:
    """Continuity per export scheme for **one** platform.

    ``platform`` is required to keep each table to one board and omit a
    repetitive platform column.
    """
    from agri_vision_edge.evaluation.benchmark_report import parse_run_name

    sel = df[df["platform"] == platform].copy()
    if sel.empty:
        return pd.DataFrame()

    meta = sel["run"].map(parse_run_name)
    sel = sel[meta.notna()].copy()
    meta = meta[meta.notna()]
    for key in ("arch_label", "eval_tiling", "classes", "dataset", "nms"):
        sel[key] = [m.get(key) for m in meta]
    # Build locally so parsed logs do not require a benchmark-results frame.
    sel["scheme"] = [
        "_".join(
            [str(m.get("precision")), str(m.get("quant"))]
            + ([str(m["granularity"])] if m.get("granularity") else [])
        )
        for m in meta
    ]

    sel = sel[sel["eval_tiling"] == eval_tiling]
    if sel.empty:
        return pd.DataFrame()

    scale = 100.0 if percent else 1.0
    out = pd.DataFrame(
        {
            "Architecture": sel["arch_label"],
            "Scheme": sel["scheme"],
            "Ops": sel["ops_total"],
            "Delegated": sel["ops_delegated"],
            "R_ops": (sel["r_ops"] * scale).round(1),
            "K": sel["K"],
            "Largest": sel["largest_partition"],
            "R_largest": (sel["r_largest"] * scale).round(1),
            "Top rejected": sel["top_rejected"],
        }
    )
    return out.sort_values(["Architecture", "Scheme"]).reset_index(drop=True)
