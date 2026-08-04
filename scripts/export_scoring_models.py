#!/usr/bin/env python3
"""
Re-export every training stage's SavedModel without the NMS score floor.

The stage exports bake ``batch_non_max_suppression.score_threshold`` (0.05) into
the graph, which truncates the precision/recall curve COCO AP integrates over.
That makes the stock export useless as the *reference* rung of the degradation
ladder -- it would score below the TFLite export it is meant to bound. See
:func:`agri_vision_edge.tfod_trainer.export.export_scoring_saved_model`.

Writes ``<variant>/<stage>/saved_model_nms0/`` for each stage, leaving the
stock ``saved_model/`` (which the TFLite conversion traces from) untouched.

Usage
-----
    python scripts/export_scoring_models.py [--artifacts artifacts/tf]
                                            [--stages ptq qat_per-tensor ...]
                                            [--variant NAME ...] [--override]
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# `agri_vision_edge.tfod_trainer` imports `object_detection` at module scope, and
# object_detection only resolves once the vendored copy is on sys.path -- so the
# path injection has to happen before the import, not inside it.
from agri_vision_edge.third_party import setup_tensorflow_models  # noqa: E402

setup_tensorflow_models()

from agri_vision_edge.tfod_trainer.export import (  # noqa: E402
    SCORING_EXPORT_NAME,
    export_scoring_saved_model,
)

DEFAULT_STAGES = ("finetune", "ptq", "qat_per-tensor", "qat_per-channel")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "tf"
    )
    parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    parser.add_argument("--variant", nargs="+", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--override",
        action="store_true",
        help="Re-export stages that already have a scoring export.",
    )
    args = parser.parse_args(argv)

    variants = sorted(p for p in args.artifacts.iterdir() if p.is_dir())
    if args.variant:
        wanted = set(args.variant)
        variants = [v for v in variants if v.name in wanted]

    done = skipped = failed = 0

    for variant in variants:
        for stage_name in args.stages:
            stage = variant / stage_name
            if not (stage / "checkpoint").is_dir():
                continue

            out = stage / SCORING_EXPORT_NAME
            if out.exists() and not args.override:
                print(f"[skip] {variant.name}/{stage_name} (already exported)")
                skipped += 1
                continue

            print(f"[export] {variant.name}/{stage_name}")
            try:
                export_scoring_saved_model(stage, score_threshold=args.score_threshold)
                done += 1
            except Exception as exc:  # keep the sweep going
                print(f"[error] {variant.name}/{stage_name}: {exc}", file=sys.stderr)
                traceback.print_exc()
                failed += 1

    print(f"\ndone: {done} exported, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
