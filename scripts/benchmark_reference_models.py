#!/usr/bin/env python3
"""
Score the pre-conversion SavedModel reference for every training stage.

This is the rung above TFLite in the deployment chain: without it, the
conversion loss (TFOD post-processing -> TFLite_Detection_PostProcess) and the
quantization loss are folded together and both get attributed to quantization.

Runs each stage's SavedModel through the *same* ``ave benchmark`` path the
device runs use, so ``predictions.json`` is produced by identical code against
identical annotations and is directly comparable.

Both exports of each stage are scored, into separate results trees:

    benchmark_results/tf-savedmodel/       saved_model/       score floor 0.05
    benchmark_results/tf-savedmodel-nms0/  saved_model_nms0/  score floor 0

The floored tree is the apples-to-apples rung -- the TFLite exports bake the
same 0.05 floor into their graphs. The floor-free tree is the true ceiling, and
the difference between the two says what the floor costs, which is what decides
whether re-converting and re-benchmarking everything without it is worth the
device time. The floor binds only for the single-class models; the multi-class
ones hit the 100-detection cap first.

Which stages are scored follows ``ConversionTarget.stage_candidates``, so each
reference is the checkpoint its INT8 exports were actually converted from:

    ptq             -> fp32_ptq, int8_ptq_per-tensor, int8_ptq_per-channel
    qat_per-tensor  -> int8_qat_per-tensor, int8_qat_per-channel

Two stages are deliberately not scored. ``finetune`` is the source of no
conversion target. ``qat_per-channel`` is preferred by nothing: granularity is a
conversion-time choice that does not change the QAT training graph, so
``int8_qat_per-channel`` resolves to ``qat_per-tensor`` whenever it exists --
which it does for every variant. (Those directories are also stale: rebuilding
``ssd-mn2-fpnlite_mc_phenobench-tiled_320/qat_per-channel`` fails with 246
weight fake-quant variables absent from its checkpoint, i.e. it was trained
against a since-changed graph. Nothing deployed derives from it.)

The reference run names therefore carry no granularity token -- a float
checkpoint has no weight granularity.

Usage
-----
    python scripts/benchmark_reference_models.py [--variant NAME ...]
                                                 [--stages ptq ...]
                                                 [--override]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: stage directory -> the (precision, quant[, granularity]) tokens that name the
#: run, mirroring how the TFLite exports are named. Every stage is float here,
#: so the precision is always fp32; the quant tokens identify which checkpoint
#: the corresponding INT8 exports came from.
STAGE_TOKENS = {
    "ptq": "fp32_ptq",
    "qat_per-tensor": "fp32_qat",
}

#: export sub-directory -> results tree.
EXPORTS = {
    "saved_model": "tf-savedmodel",
    "saved_model_nms0": "tf-savedmodel-nms0",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "tf")
    parser.add_argument("--bundle", type=Path, default=REPO_ROOT / "datasets" / "test-bundle")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "benchmark_results")
    parser.add_argument("--stages", nargs="+", default=list(STAGE_TOKENS))
    parser.add_argument("--variant", nargs="+", default=None)
    parser.add_argument("--exports", nargs="+", default=list(EXPORTS))
    parser.add_argument("--eval-tiling", default="untiled", choices=["untiled", "tiled"])
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args(argv)

    if args.eval_tiling == "tiled":
        images = args.bundle / "images_tiled"
        ann_suffix = "_tiled"
    else:
        images = args.bundle / "images"
        ann_suffix = ""

    variants = sorted(p for p in args.artifacts.iterdir() if p.is_dir())
    if args.variant:
        wanted = set(args.variant)
        variants = [v for v in variants if v.name in wanted]

    ran = skipped = failed = 0

    for variant in variants:
        cls = "mc" if "_mc_" in variant.name else "sc"
        annotations = args.bundle / f"annotations_{cls}{ann_suffix}.json"

        for stage_name in args.stages:
            tokens = STAGE_TOKENS.get(stage_name)
            if tokens is None:
                continue

            for export_name in args.exports:
                model = variant / stage_name / export_name
                if not (model / "saved_model.pb").exists():
                    continue

                platform = EXPORTS[export_name]
                run_name = f"{args.eval_tiling}_{variant.name}_{tokens}"
                output_root = args.results / platform
                final_dir = output_root / run_name

                if (final_dir / "latency.json").exists() and not args.override:
                    print(f"[skip] {platform}/{run_name}")
                    skipped += 1
                    continue

                print(f"[run]  {platform}/{run_name}")

                # `ave benchmark` names the output directory after the model
                # stem, which is `saved_model[_nms0]` for every stage -- so the
                # run is produced under that name and renamed afterwards.
                staged = output_root / model.name

                result = subprocess.run(
                    [
                        str(REPO_ROOT / "scripts" / "ave"),
                        "benchmark",
                        str(model),
                        str(images),
                        "--annotations",
                        str(annotations),
                        "--output-dir",
                        str(output_root),
                        "--delegate",
                        "none",
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0 or not (staged / "latency.json").exists():
                    print(f"[error] {platform}/{run_name}", file=sys.stderr)
                    print(result.stdout[-2000:], file=sys.stderr)
                    print(result.stderr[-2000:], file=sys.stderr)
                    failed += 1
                    continue

                if final_dir.exists():
                    for leftover in final_dir.iterdir():
                        leftover.unlink()
                    final_dir.rmdir()
                staged.rename(final_dir)
                ran += 1

    print(f"\ndone: {ran} run, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
