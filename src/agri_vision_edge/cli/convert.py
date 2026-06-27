"""
Batch-convert trained TF model variants to deployable TFLite models.

For each variant under ``artifacts/tf/`` (or a single named variant), build the
standard int8/fp32 TFLite models -- one per training stage that is present
(``ptq`` -> int8/int8 per-channel/fp32, ``qat`` -> int8, ``qat_per-channel`` ->
int8 per-channel) -- with the default IoU threshold and fast NMS, embedding
ObjectDetector metadata. Conversion only; no evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ave convert",
        description=(
            "Convert TF model variant(s) under artifacts/tf/ to int8/fp32 TFLite "
            "models with embedded metadata (fast NMS, default IoU threshold)."
        ),
    )
    parser.add_argument(
        "variant",
        nargs="?",
        help="variant dir name under --artifacts (default: convert all variants)",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/tf"),
        help="root holding the TF model variants (default: artifacts/tf)",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        default=Path("datasets"),
        help="datasets root (representative dataset + label maps; default: datasets)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/tflite"),
        help="output directory for the .tflite models (default: artifacts/tflite)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold baked into the post-processing op (default: 0.5)",
    )
    parser.add_argument(
        "--no-native-resize",
        dest="native_resize",
        action="store_false",
        help=(
            "build FPN models with the legacy PACK/reshape upsample instead of the "
            "default NPU-delegatable RESIZE_NEAREST_NEIGHBOR op (no-op for non-FPN "
            "models)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reconvert even if the output .tflite already exists",
    )

    args = parser.parse_args(argv)

    if not args.artifacts.is_dir():
        parser.error(f"artifacts dir not found: {args.artifacts}")

    if args.variant:
        # Accept either a variant name under --artifacts or a direct path.
        variant_dir = args.artifacts / args.variant
        if not variant_dir.is_dir():
            variant_dir = Path(args.variant)
        if not variant_dir.is_dir():
            parser.error(f"variant not found: {args.variant}")
        variant_dirs = [variant_dir]
    else:
        variant_dirs = sorted(p for p in args.artifacts.iterdir() if p.is_dir())
        if not variant_dirs:
            parser.error(f"no variants found under {args.artifacts}")

    # Imported here so other `ave` subcommands don't pull in TensorFlow.
    from agri_vision_edge.conversion.tflite import convert_variant

    total = 0
    for variant_dir in variant_dirs:
        print(f"\n{variant_dir.name}")
        written = convert_variant(
            variant_dir,
            datasets_dir=args.datasets,
            out_dir=args.out,
            iou_threshold=args.iou,
            native_resize=args.native_resize,
            overwrite=args.overwrite,
        )
        total += len(written)

    print(f"\ncompleted: {total} model(s) written to {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
