#!/usr/bin/env python3

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

MODEL_PATH = "detect.tflite"
SCORE_THRESHOLD = 0.4
TEFLON_LIB = "/usr/lib/libteflon.so"

# Kiosk display resolution. getWindowImageRect is unreliable under
# Weston/Wayland (it reports the source frame size, not the screen), which
# caused the square to be padded to the video size first and then scaled
# again by the compositor -> double bars. We instead build a canvas that
# already matches the panel, so the compositor adds no bars of its own.
DISPLAY_W = 1920
DISPLAY_H = 1080



# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

interpreter = Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=(
	    load_delegate(TEFLON_LIB),
    ),
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_idx = input_details[0]["index"]

_, INPUT_H, INPUT_W, _ = input_details[0]["shape"]

scale, zero_point = input_details[0]["quantization"]

print("Input:", input_details[0]["shape"])
print("Quantization:", scale, zero_point)

for i, out in enumerate(output_details):
    print(i, out["name"], out["shape"])

# ---------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------

cap = cv2.VideoCapture(0)

WINDOW = "SSD-MobileNetV2"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN,
)


def letterbox(img, target_w, target_h):
    """
    Scale img to fit target while preserving aspect ratio, padding the
    remainder with black bars (letterbox / pillarbox).
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------------------------------------
    # Center-crop to a square so the image is not distorted before
    # inference, and the displayed boxes line up with what the model saw.
    # -------------------------------------------------------------

    h, w = frame.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    square = frame[y0:y0 + side, x0:x0 + side]

    vis = square.copy()

    # -------------------------------------------------------------
    # Preprocess
    # -------------------------------------------------------------

    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(rgb, (INPUT_W, INPUT_H))

    # uint8 image -> int8 model input
    inp = resized.astype(np.float32)
    # inp = inp / scale + zero_point
    inp -= 128.0
    inp = np.clip(inp, -128, 127).astype(np.int8)

    interpreter.set_tensor(
        input_idx,
        inp[np.newaxis, ...],
    )

    # -------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------

    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]["index"])[0]
    classes = interpreter.get_tensor(output_details[3]["index"])[0]
    scores = interpreter.get_tensor(output_details[0]["index"])[0]
    count = int(interpreter.get_tensor(output_details[2]["index"])[0])

    # -------------------------------------------------------------
    # Draw detections (normalized coords map onto the square crop)
    # -------------------------------------------------------------

    for box, cls, score in zip(
        boxes[:count],
        classes[:count],
        scores[:count],
    ):

        if score < SCORE_THRESHOLD:
            continue

        # print(f"{score}:{box}")

        ymin, xmin, ymax, xmax = box

        x1 = int(xmin * side)
        y1 = int(ymin * side)
        x2 = int(xmax * side)
        y2 = int(ymax * side)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            vis,
            f"{int(cls)} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # -------------------------------------------------------------
    # Display: pad the square straight to the panel resolution so the
    # only bars are the black ones we draw (no compositor scaling bars).
    # -------------------------------------------------------------

    display = letterbox(vis, DISPLAY_W, DISPLAY_H)

    cv2.imshow(WINDOW, display)

    key = cv2.waitKey(1)
    if key in (27, ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
