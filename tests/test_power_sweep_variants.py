"""
Tests for the power sweep's env-variant mechanism.

The Teflon delegate is configurable at run time -- ``TEFLON_UNSUPPORTED_NODES``
and ``TEFLON_UNSUPPORTED_OPS`` push individual nodes or whole operators back
onto the CPU -- and pricing one of those switches means running the *same file*
twice under different environments. Two things then have to hold, and both fail
silently if they do not:

* the two runs must not share a directory name, or the second overwrites the
  first and the comparison is between a run and itself;
* the environment must be *recorded*, not just applied. These trees are read
  months later by something that was not at the console, and a power figure
  whose delegate configuration is unknown is a number about an unknown machine.

``scripts/power_sweep.py`` is a script rather than a package module, so it is
loaded by path here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_power_sweep():
    spec = importlib.util.spec_from_file_location(
        "power_sweep", REPO_ROOT / "scripts" / "power_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


power_sweep = _load_power_sweep()

MODEL = Path("artifacts/tflite/ssd-mn2_mc_phenobench_320_int8_ptq_per-tensor.tflite")


def test_a_variant_names_its_own_run_directory():
    variant = power_sweep.parse_variant(["no-pack", "TEFLON_UNSUPPORTED_OPS=83"], {})

    assert variant.run_name(MODEL) == (
        "ssd-mn2_mc_phenobench_320_int8_ptq_per-tensor__no-pack"
    )


def test_the_unnamed_variant_leaves_run_names_exactly_as_before():
    """
    Backwards compatibility is the point: every sweep collected before variants
    existed named its directories after the model stem, and `--skip-existing`
    has to keep matching them.
    """
    variant = power_sweep.Variant("", {})

    assert variant.run_name(MODEL) == MODEL.stem
    assert variant.suffix == ""


def test_variant_values_may_contain_commas():
    """
    The switch this exists for takes a comma list of its own, which is why the
    syntax is `--variant NAME KEY=VALUE ...` over argv rather than any
    comma-separated spelling.
    """
    variant = power_sweep.parse_variant(
        ["nodes", "TEFLON_UNSUPPORTED_NODES=66,68,69,71-73,75"], {}
    )

    assert variant.env["TEFLON_UNSUPPORTED_NODES"] == "66,68,69,71-73,75"


def test_values_may_contain_equals_signs():
    variant = power_sweep.parse_variant(["v", "ETNA_MESA_DEBUG=a=b"], {})

    assert variant.env["ETNA_MESA_DEBUG"] == "a=b"


def test_base_env_is_the_baseline_and_the_variant_may_override_one_key():
    base = {"TEFLON_DEBUG": "quiet", "OTHER": "keep"}

    variant = power_sweep.parse_variant(["loud", "TEFLON_DEBUG=verbose"], base)

    assert variant.env == {"TEFLON_DEBUG": "verbose", "OTHER": "keep"}
    # ...without mutating the sweep-wide baseline out from under the next one.
    assert base["TEFLON_DEBUG"] == "quiet"


def test_a_malformed_setting_is_refused_rather_than_ignored():
    """
    A dropped variable would produce a run that looks like the variant it
    claims to be and behaves like the baseline.
    """
    with pytest.raises(SystemExit):
        power_sweep.parse_variant(["v", "TEFLON_UNSUPPORTED_OPS"], {})

    with pytest.raises(SystemExit):
        power_sweep.parse_variant(["v", "=83"], {})


def test_a_variant_needs_a_name():
    with pytest.raises(Exception):  # noqa: B017
        power_sweep.parse_variant([], {})


def test_describe_names_the_environment_it_will_run_under():
    variant = power_sweep.parse_variant(["no-pack", "TEFLON_UNSUPPORTED_OPS=83"], {})

    assert "no-pack" in variant.describe()
    assert "TEFLON_UNSUPPORTED_OPS=83" in variant.describe()

    assert "clean environment" in power_sweep.Variant("baseline", {}).describe()


def test_the_resolution_ladder_preset_is_confined_to_untiled_training():
    """
    512 and 1024 exist only as untiled-trained exports, and evaluation tiling
    would move the source resolution together with the input size -- which is
    the axis the preset varies. Both reasons are in PRESETS; this pins the
    patterns against them.
    """
    patterns = power_sweep.PRESETS["res-ladder"]

    assert not any("phenobench-tiled" in pattern for pattern in patterns)
    assert all("_mc_" in pattern for pattern in patterns)
    assert all("fastnms" in pattern for pattern in patterns)
    assert {"320", "512", "1024"} == {pattern.split("_")[3] for pattern in patterns}
