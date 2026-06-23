"""Run a TFLite detector on images and draw the predicted boxes (on-device)."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tflite_runtime.interpreter import (
    Interpreter,
    load_delegate,
)

#
# Optional delegate
#

TEFLON_LIB = "/usr/lib/libteflon.so"

delegate = None

if Path(TEFLON_LIB).exists():
    try:
        loaded_delegate = load_delegate(TEFLON_LIB)

        delegate = loaded_delegate

        print(f"[runtime] loaded delegate: {TEFLON_LIB}")

    except Exception as e:
        print(f"[runtime] failed to load delegate: {TEFLON_LIB}")

        print("[runtime] falling back to CPU")

        print(f"[runtime] reason: {e}")

else:
    print(f"[runtime] delegate not found: {TEFLON_LIB}")

    print("[runtime] using CPU runtime")


INPUT_SIZE = 320

LABELS = {
    0: "crop",
    1: "weed",
}

COLORS = {
    1: (0, 255, 0),  # green
    2: (0, 0, 255),  # red
}


def load_image_int8(path):

    img_np = cv2.imread(str(path))

    if img_np is None:
        raise RuntimeError(f"Failed to load image: {path}")

    img_np = cv2.cvtColor(
        img_np,
        cv2.COLOR_BGR2RGB,
    )

    img_np = cv2.resize(
        img_np,
        (INPUT_SIZE, INPUT_SIZE),
    )

    #
    # Quantize exactly like TF Lite export:
    #
    # q = x / scale + zero_point
    #
    # scale = 1.0
    # zero_point = -128
    #

    img_np = img_np.astype(np.float32)

    img_np = img_np - 128.0

    img_np = np.round(img_np)

    img_np = np.clip(
        img_np,
        -128,
        127,
    )

    img_np = img_np.astype(np.int8)

    img_np = np.expand_dims(
        img_np,
        axis=0,
    )

    return img_np


def dequantize(arr, quantization):

    scale, zero_point = quantization

    return scale * (arr.astype(np.float32) - zero_point)


def draw_boxes(
    image_path,
    output_path,
    boxes,
    classes,
    scores,
    score_threshold=0.1,
):

    img = cv2.imread(str(image_path))

    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    h, w = img.shape[:2]

    detections = 0

    for box, cls_id, score in zip(
        boxes,
        classes,
        scores,
        strict=False,
    ):
        if score < score_threshold:
            continue

        cls_id = int(cls_id)

        ymin, xmin, ymax, xmax = box

        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        #
        # Clamp
        #

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        color = COLORS.get(
            cls_id,
            (255, 255, 255),
        )

        label_name = LABELS.get(
            cls_id,
            f"class-{cls_id}",
        )

        label = f"{label_name} {score:.2f}"

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        print(f"{label_name:<5} score={score:.3f} box=[{x1}, {y1}, {x2}, {y2}]")

        detections += 1

    cv2.imwrite(
        str(output_path),
        img,
    )

    print(f"\nRendered {detections} detections")

    print(f"Saved: {output_path}")


def process_image(
    interpreter,
    input_details,
    output_details,
    image_path,
    output_dir,
    float_output=False,
    threshold=0.2,
):

    input_tensor = load_image_int8(image_path)

    print(
        "input tensor:",
        input_tensor.min(),
        input_tensor.max(),
        input_tensor.dtype,
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        input_tensor,
    )

    interpreter.invoke()

    #
    # Output ordering:
    #
    # 0 -> scores
    # 1 -> boxes
    # 2 -> num detections
    # 3 -> classes
    #

    raw_scores = interpreter.get_tensor(output_details[0]["index"])

    raw_boxes = interpreter.get_tensor(output_details[1]["index"])

    raw_num = interpreter.get_tensor(output_details[2]["index"])

    raw_classes = interpreter.get_tensor(output_details[3]["index"])

    #
    # Float-output compatibility mode
    #

    if float_output:
        print("[runtime] float output mode enabled")

        scores = raw_scores[0]

        boxes = raw_boxes[0]

        classes = np.round(raw_classes[0]).astype(np.int32)

        num = int(np.squeeze(raw_num))

    #
    # Standard quantized-output mode
    #

    else:
        scores = dequantize(
            raw_scores,
            output_details[0]["quantization"],
        )[0]

        boxes = dequantize(
            raw_boxes,
            output_details[1]["quantization"],
        )[0]

        classes = dequantize(
            raw_classes,
            output_details[3]["quantization"],
        )[0]

        classes = np.round(classes).astype(np.int32)

        num = int(
            np.squeeze(
                dequantize(
                    raw_num,
                    output_details[2]["quantization"],
                )
            )
        )

    print("\n=== DETECTION DEBUG ===")

    print("num:", num)

    print("scores:", scores[:10])

    print("classes:", classes[:10])

    valid = scores > threshold

    print(
        "valid detections:",
        np.sum(valid),
    )

    output_path = output_dir / image_path.name

    draw_boxes(
        image_path=image_path,
        output_path=output_path,
        boxes=boxes[valid],
        classes=classes[valid],
        scores=scores[valid],
        score_threshold=threshold,
    )


def get_image_paths(args):

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    #
    # Single image mode
    #

    if args.image:
        image_path = Path(args.image)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        return [image_path]

    #
    # Directory mode
    #

    src_dir = Path(args.src)

    if not src_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {src_dir}")

    image_paths = sorted(
        [p for p in src_dir.iterdir() if p.suffix.lower() in image_extensions]
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {src_dir}")

    return image_paths


def create_interpreter(model_path):

    if delegate is not None:
        print("[runtime] using TFLite with delegate")

        interpreter = Interpreter(
            model_path=str(model_path),
            experimental_delegates=[delegate],
        )

    else:
        print("[runtime] using TFLite CPU runtime")

        interpreter = Interpreter(
            model_path=str(model_path),
        )

    interpreter.allocate_tensors()

    return interpreter


def main(argv=None):

    parser = argparse.ArgumentParser(prog="ave infer")

    parser.add_argument(
        "model",
        help="Path to .tflite model",
    )

    parser.add_argument(
        "--src",
        default="./images",
        help="Source image directory",
    )

    parser.add_argument(
        "--image",
        help="Optional single image path",
    )

    parser.add_argument(
        "--float-output",
        action="store_true",
        help=("Model outputs are already float32"),
    )

    args = parser.parse_args(argv)

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir = Path(f"./{model_path.stem}-output")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = get_image_paths(args)

    print(f"Found {len(image_paths)} image(s)")

    interpreter = create_interpreter(model_path)

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    print("\n=== INPUT DETAILS ===")

    for detail in input_details:
        print(detail)

    print("\n=== OUTPUT DETAILS ===")

    for detail in output_details:
        print(detail)

    for image_path in image_paths:
        print(f"\n=== Processing: {image_path.name} ===")

        try:
            process_image(
                interpreter=interpreter,
                input_details=input_details,
                output_details=output_details,
                image_path=image_path,
                output_dir=output_dir,
                float_output=args.float_output,
            )

        except Exception as e:
            print(f"[FAIL] {image_path.name}: {e}")

    print("\nDone.")

    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
