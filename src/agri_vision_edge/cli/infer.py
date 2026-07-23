"""
Run a TFLite detector on images and draw the predicted boxes.

Two detector families are supported, auto-detected from the model's output
tensors (via :func:`agri_vision_edge.runtime.inference.factory.build_runtime`),
so the same command runs either — exactly like ``ave benchmark``:

- **SSD MobileNetV2** (primary) — four post-NMS outputs
  (boxes / scores / classes / count): ``TFLiteRuntime``. Input 320×320, ``[-1, 1]``.
- **YOLOv7-tiny** — three raw grid outputs decoded in ``YoloTFLiteRuntime``
  (sigmoid + anchor/grid reconstruction + per-class NMS). Input 512×512, ``[0, 1]``.

Both the input resolution and the normalization are read from the model, so the
320-vs-512 difference needs no per-model flag; ``--size`` is only an override.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from agri_vision_edge.runtime.inference.factory import build_runtime
from agri_vision_edge.runtime.inference.tflite import DEFAULT_TEFLON_LIB

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# BGR palette cycled by category_id for box drawing (crop green, weed red first).
COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 128, 0),
    (0, 200, 255),
    (255, 0, 255),
]


def color_for(category_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color for a 1-based category id."""

    return COLORS[(category_id - 1) % len(COLORS)]


def build_detector(model_path, *, delegate, threshold, iou, size):
    """Build the detector runtime; note when ``--size`` is moot for fixed SSD."""

    runtime = build_runtime(
        model_path=model_path,
        delegate_path=delegate,
        score_threshold=threshold,
        iou_threshold=iou,
        size=size,
    )

    if size is not None and size != runtime.input_size:
        print(
            f"[infer] --size {size} ignored: input is model-fixed at "
            f"{runtime.input_size}"
        )

    return runtime


def draw_detections(image_bgr, detections, output_path, labels):
    """Render detections (already score-filtered) onto a copy and save it."""

    h, w = image_bgr.shape[:2]

    for det in detections:
        ymin, xmin, ymax, xmax = det.bbox

        x1 = max(0, min(w - 1, int(xmin * w)))
        y1 = max(0, min(h - 1, int(ymin * h)))
        x2 = max(0, min(w - 1, int(xmax * w)))
        y2 = max(0, min(h - 1, int(ymax * h)))

        color = color_for(det.category_id)
        name = labels.get(det.category_id, f"class-{det.category_id}")

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            image_bgr,
            f"{name} {det.score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        print(f"{name:<5} score={det.score:.3f} box=[{x1}, {y1}, {x2}, {y2}]")

    cv2.imwrite(str(output_path), image_bgr)

    print(f"Rendered {len(detections)} detection(s) -> {output_path}")


def collect_images(source: Path) -> list[Path]:
    """Resolve a file or directory argument into a list of image paths."""

    if not source.exists():
        raise FileNotFoundError(f"Image source not found: {source}")

    if source.is_file():
        return [source]

    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    if not images:
        raise RuntimeError(f"No images found in {source}")

    return images


def main(argv=None):

    parser = argparse.ArgumentParser(
        prog="ave infer",
        description="Run a TFLite detector (SSD or YOLOv7-tiny) on images.",
    )

    parser.add_argument(
        "model",
        help="Path to the .tflite model",
    )

    parser.add_argument(
        "images",
        help="Image file or directory of images to run on",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help=(
            "Score threshold for kept detections "
            "(default: the model's embedded score_threshold)"
        ),
    )

    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help=(
            "Override the input resolution (default: the model's own input "
            "size — 320 for SSD, 512 for YOLOv7-tiny)"
        ),
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help=(
            "IoU threshold for YOLO NMS (ignored for SSD) "
            "(default: the model's embedded iou_threshold)"
        ),
    )

    parser.add_argument(
        "--delegate",
        default=DEFAULT_TEFLON_LIB,
        help=(
            "Path to the TFLite delegate, or 'none' to run on CPU "
            "(use 'none' for fp32 models — the NPU delegate is for INT8)"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ./<model-stem>-output)",
    )

    args = parser.parse_args(argv)

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    delegate = args.delegate
    if delegate is not None and delegate.strip().lower() in ("", "none"):
        delegate = None

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"./{model_path.stem}-output")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(Path(args.images))

    print(f"Found {len(image_paths)} image(s)")

    detector = build_detector(
        model_path,
        delegate=delegate,
        threshold=args.threshold,
        iou=args.iou,
        size=args.size,
    )

    print(f"[infer] input size: {detector.input_size}")

    # Class names come from the model's embedded metadata; models without it
    # (e.g. the separately-exported YOLO artifacts) fall back to "class-<id>".
    labels = getattr(detector, "labels", {})

    for image_path in image_paths:
        print(f"\n=== {image_path.name} ===")

        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            print(f"[skip] failed to read {image_path}")
            continue

        try:
            detections = detector.predict(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

            draw_detections(
                image_bgr,
                detections,
                output_dir / image_path.name,
                labels,
            )

        except Exception as e:
            print(f"[FAIL] {image_path.name}: {type(e).__name__}: {e}")

    print(f"\nDone. Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
