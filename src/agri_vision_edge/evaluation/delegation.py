"""
Parse delegation continuity from `ave benchmark` `delegate_debug.log` files.

The logs provide the quantities used here: accepted operation count, delegated
partition count, and the fraction of accepted operations in the largest
partition.

Teflon uses TensorFlow Lite's `ReplaceNodeSubsetsWithDelegateKernels` to group
accepted operations into maximal dependency-preserving regions. The backend
therefore determines which operations are supported, while TensorFlow Lite
determines the resulting partitions. Partition counts are directly comparable
between Etnaviv and Ethos-U.

Current patched builds emit one line per delegated region as::

```
teflon: ===== subgraph #0: 94 operations, 255 tensors =====
```

Older unpatched builds use::

```
teflon: compiling graph: 255 tensors 98 operations
```

Both forms are accepted. Operation lines also differ slightly between Etnaviv
and Ethos-U; the parser accepts both dialects.

`continuity_table` processes one platform at a time for presentation only;
the resulting continuity metrics are defined identically on both platforms.
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
#: The one operator that reaches this path in every SSD export is builtin 32,
#: ``CUSTOM``, which carries ``TFLite_Detection_PostProcess``. An upper-case
#: class silently dropped that line, so the operator vanished from the totals
#: as well as from the rejection list -- the graph looked two operators short
#: and the detection post-processing looked delegated when nothing accepts it.
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

        Measured on both backends: each region is printed once, so ``K`` is the
        count of those lines. A ``K`` of 1 therefore records a contiguous
        accepted set rather than an inability to split -- the Ethos-U path
        reports two regions for the YOLOv7-tiny exports.
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

        This is the continuity measure. It is deliberately quoted against the
        *delegated* operations rather than against the whole graph, so that a
        model with a small accepted region that is nonetheless contiguous is not
        confused with one whose accepted region is scattered.
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
            # Older wording for the same per-region print, carried on the i.MX93
            # image. Appending is correct rather than a fallback: a graph split
            # into several regions emits this line once per region, as the
            # two-region YOLOv7-tiny runs on that board show.
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

    ``platform`` is required rather than optional, for readability rather than
    because the columns are incommensurable: one table per board keeps the
    schemes on a page without a platform column repeating down every row.
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
    # Built here rather than via ``scheme_name`` so this module stays usable on a
    # frame of parsed logs alone, without a benchmark-results frame to attach to.
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
