"""
Tests for the delegate-rebuild delta table.

Two failure modes this table has to stay clear of, both of which produce a
plausible-looking number rather than an error:

* **Pooling architectures.** The plain head and FPNLite do not respond to a
  delegate rebuild alike, so a median taken across both belongs to neither --
  the same trap ``nms_latency_tradeoff_table`` already documents.
* **Unpaired latency.** The older tree does not contain every configuration the
  newer one does. Differencing two medians taken over *different* config sets
  reports a change that is really a change of subject, so the pairing is per
  configuration and its width is reported.
"""

from __future__ import annotations

import pandas as pd

from agri_vision_edge.evaluation.benchmark_report import (
    CPU_REFERENCE_PLATFORM,
    DEFAULT_NMS,
    delegate_build_table,
)

NPU = "frdm-imx8mp"
OLD = "frdm-imx8mp_unpatched"


def _run(platform, ap, *, arch="SSD MobileNetV2", size="320", latency=None):
    stem = "ssd-mn2-fpnlite" if "FPNLite" in arch else "ssd-mn2"
    return {
        "run": f"untiled_{stem}_mc_phenobench_{size}_int8_ptq_per-tensor_{DEFAULT_NMS}",
        "platform": platform,
        "arch_label": arch,
        "classes": "mc",
        "dataset": "phenobench",
        "size": size,
        "eval_tiling": "untiled",
        "precision": "int8",
        "quant": "ptq",
        "granularity": "per-tensor",
        "nms": DEFAULT_NMS,
        "AP": ap,
        "median_latency_ms": latency,
    }


def _table(rows, **kwargs):
    return delegate_build_table(
        pd.DataFrame(rows), (), eval_tiling="untiled", nms=DEFAULT_NMS, **kwargs
    )


def _frame():
    """A rebuild that fixes both architectures, but by very different amounts."""
    rows = []
    for arch, was, now in (
        ("SSD MobileNetV2", 43.9, 30.5),
        ("SSD MobileNetV2 FPNLite", 123.9, 48.8),
    ):
        rows += [
            _run(CPU_REFERENCE_PLATFORM, 0.38, arch=arch),
            _run(OLD, 0.0002, arch=arch, latency=was),  # collapsed
            _run(NPU, 0.379, arch=arch, latency=now),  # ok
        ]
    return rows


def test_rows_are_resolved_per_architecture():
    table = _table(_frame(), latency_scope="paired")

    assert set(table["Arch"]) == {"SSD MobileNetV2", "SSD MobileNetV2 FPNLite"}

    by_arch = table.set_index("Arch")["d ms %"]
    assert by_arch["SSD MobileNetV2"] == -30.5
    assert by_arch["SSD MobileNetV2 FPNLite"] == -60.6


def test_pooling_architectures_reports_a_median_belonging_to_neither():
    pooled = _table(_frame(), by_arch=False, latency_scope="paired")

    assert len(pooled) == 1
    # The two real answers are -30.5 and -60.6; the pooled figure is neither.
    assert pooled["d ms %"].iloc[0] == -52.8


def test_paired_scope_keeps_the_latency_of_a_collapsed_run():
    """
    A collapsed run's clock still measures dispatch and CPU<->NPU movement,
    which is the cost the rebuild shifts. ``ok-both`` drops it; ``paired``
    keeps it, and both agree it was never scored as correct.
    """
    rows = _frame()

    ok_both = _table(rows, latency_scope="ok-both")
    assert ok_both["ok both"].sum() == 0
    assert ok_both["ms before"].isna().all()

    paired = _table(rows, latency_scope="paired")
    assert paired["ok both"].sum() == 0  # unchanged: this is not a verdict
    assert paired["ms before"].notna().all()
    assert set(paired["Before"]) == {"collapsed"}
    assert set(paired["After"]) == {"ok"}


def test_latency_is_paired_per_configuration_not_pooled_per_tree():
    """
    The older tree was only swept at 320. The 512 rung it never ran must not
    enter either median, or the difference compares two different config sets.
    """
    rows = _frame()
    rows += [
        _run(CPU_REFERENCE_PLATFORM, 0.40, size="512"),
        _run(NPU, 0.399, size="512", latency=86.2),  # no `_unpatched` counterpart
    ]

    table = _table(rows, latency_scope="paired").set_index("Arch")
    row = table.loc["SSD MobileNetV2"]

    assert row["Configs"] == 2  # 320 and 512 are both in scope
    assert row["timed"] == 1  # but only 320 is paired
    assert row["ms after"] == 30.5  # the 512 run's 86.2 ms is excluded
    assert row["d ms %"] == -30.5


def test_no_shared_timed_configuration_leaves_the_latency_empty():
    rows = [
        _run(CPU_REFERENCE_PLATFORM, 0.38),
        _run(NPU, 0.379, latency=30.5),
        _run(OLD, 0.0002, size="512", latency=43.9),
    ]

    table = _table(rows, latency_scope="paired")

    assert (table["timed"] == 0).all()
    assert table["ms before"].isna().all()
    assert table["d ms %"].isna().all()
