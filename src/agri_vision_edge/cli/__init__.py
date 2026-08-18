"""
agri-vision-edge command-line interface (``ave``).

A single entry point with subcommands. Subcommand modules are imported lazily
on dispatch, so e.g. ``ave infer`` (which needs ``tflite_runtime``, the
``device`` extra) does not force that import when running ``ave benchmark`` on a
host that only has ``tf.lite``.

Runnable three ways, all reaching the same ``main``:

  * installed:      ``ave <command> ...``        (the ``[project.scripts]`` entry)
  * bare checkout:  ``PYTHONPATH=src python -m agri_vision_edge.cli <command> ...``
  * via shim:       ``scripts/ave <command> ...``
"""

from __future__ import annotations

import argparse
import importlib
import sys

# command name -> (module path, one-line description)
_COMMANDS: dict[str, tuple[str, str]] = {
    "benchmark": (
        "agri_vision_edge.cli.benchmark",
        "Benchmark TFLite model(s) on a COCO dataset",
    ),
    "convert": (
        "agri_vision_edge.cli.convert",
        "Convert TF model variant(s) to int8/fp32 TFLite + metadata",
    ),
    "evaluate": (
        "agri_vision_edge.cli.evaluate",
        "Evaluate COCO predictions / benchmark_results",
    ),
    "infer": (
        "agri_vision_edge.cli.infer",
        "Run a TFLite detector on images and draw boxes (on-device)",
    ),
    "resources": (
        "agri_vision_edge.cli.resources",
        "Sustained inference loop measuring CPU/memory/thermals (on-device)",
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ave",
        description="agri-vision-edge command-line interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="commands:\n"
        + "\n".join(f"  {name:<11} {desc}" for name, (_, desc) in _COMMANDS.items())
        + "\n\nrun `ave <command> -h` for command-specific options.",
    )
    parser.add_argument(
        "command",
        choices=list(_COMMANDS),
        metavar="<command>",
        help="one of: " + ", ".join(_COMMANDS),
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        metavar="...",
        help="arguments forwarded to <command>",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = _build_parser()

    if not argv:
        parser.print_help()
        return 1

    ns = parser.parse_args(argv)

    module = importlib.import_module(_COMMANDS[ns.command][0])
    rc = module.main(ns.args)
    return 0 if rc is None else int(rc)
