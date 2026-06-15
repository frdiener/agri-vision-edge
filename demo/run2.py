#!/usr/bin/env python3

import threading
import time
from collections import deque

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

MODEL_PATH = "detect.tflite"
SCORE_THRESHOLD = 0.4
TEFLON_LIB = "/usr/lib/libteflon.so"

# Window (seconds) over which FPS / inference time are averaged.
HUD_WINDOW_SEC = 2.0

# Render at the PANEL's native resolution. cv2's fullscreen imshow scales the
# image to the window on the CPU, so a sub-panel canvas is actually slower
# (it forces an upscale); matching the panel makes the blit a plain copy.
DISPLAY_W = 1920
DISPLAY_H = 1080

# Capture settings. Many webcams default to uncompressed YUYV, which is
# USB-bandwidth-limited to ~15 fps and makes cap.read() block the loop.
# Forcing MJPG lets the camera deliver compressed frames at full rate.
CAP_W = 1280
CAP_H = 720
CAP_FPS = 30



# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
#
# NOTE: the interpreter is intentionally NOT built here. The Teflon NPU
# delegate binds its device context and DMA-mapped tensor buffers to the
# thread that calls allocate_tensors(); invoking from another thread faults
# with SIGBUS. Since only inference_worker() touches the interpreter, it is
# constructed there so construction, allocation and invoke share one thread.

# ---------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------

cap = cv2.VideoCapture(0)
# Request MJPG + resolution + rate BEFORE the first read so the camera
# negotiates a fast (compressed) mode instead of bandwidth-limited YUYV.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
# Keep only the freshest frame so we never display a stale, buffered one.
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(
    "Capture:",
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "x",
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    "@",
    cap.get(cv2.CAP_PROP_FPS),
    "fps",
)


class FrameGrabber:
    """
    Continuously pulls frames from the camera on a background thread so the
    camera's per-frame latency overlaps with inference + display instead of
    serializing in the main loop. read() returns the most recent frame.
    """

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            # cap.read() returns a fresh array each call, so swapping the
            # reference is enough; any frame the main loop already holds
            # stays valid.
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


grabber = FrameGrabber(cap)

# Wait for the first frame before entering the loop.
while grabber.read() is None and grabber.running:
    time.sleep(0.005)

WINDOW = "SSD-MobileNetV2"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN,
)

HUD_H = 170


def draw_hud(canvas, lines):
    """
    Draw HUD text lines (top-left) with a dark outline so they stay
    readable over both the black bars and the live image.
    """
    x, y, step = 16, 34, 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, text in enumerate(lines):
        org = (x, y + i * step)
        cv2.putText(canvas, text, org, font, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, org, font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)


def _prune(buf, now):
    while buf and now - buf[0][0] > HUD_WINDOW_SEC:
        buf.popleft()


def _avg(buf):
    return sum(v for _, v in buf) / len(buf) if buf else 0.0


class _Shared:
    """Hand-off between the worker (producer) and the main display thread."""

    def __init__(self):
        # Latest fully-rendered frame. Each is a fresh, immutable canvas, so
        # publishing is a single atomic reference assignment - no lock needed.
        self.canvas = None
        # Display cost (imshow + event pump), smoothed; written by main.
        self.disp_ms = 0.0


shared = _Shared()
stop_event = threading.Event()


def inference_worker():
    """
    Producer thread: grab the freshest frame, run inference, and render a
    finished display canvas. Runs in parallel with the main display thread,
    so inference overlaps with imshow instead of serializing.
    """
    # Build the interpreter on THIS thread so the Teflon NPU delegate's
    # thread-affine device context / DMA buffers are allocated and invoked
    # from the same thread (cross-thread use faults with SIGBUS).
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

    # The hot-loop preprocess maps a uint8 pixel straight to int8 via the
    # branch-free `rgb ^ 0x80` (== pixel - 128). That is only equivalent to the
    # proper `quantize(pixel/127.5 - 1)` when the input tensor is int8 with
    # scale ~= 2/255 and zero_point == 0. Assert it once here so a re-converted
    # detect.tflite with a different input quantization fails loudly instead of
    # silently skewing every frame.
    assert input_details[0]["dtype"] == np.int8, (
        f"expected int8 input, got {input_details[0]['dtype']}"
    )
    assert abs(scale - 2 / 255) < 1e-3 and zero_point == 0, (
        f"detect.tflite input quant changed ({scale}, {zero_point}); "
        "the `rgb ^ 0x80` preprocess is no longer valid"
    )

    frame_stamps = deque()
    infer_stamps = deque()
    disp_side = 0
    off_x = off_y = 0
    last_log = 0.0

    while not stop_event.is_set():
        frame = grabber.read()
        if frame is None:
            if not grabber.running:
                stop_event.set()
            continue

        # Center-crop to a square (view into frame, no copy).
        h, w = frame.shape[:2]
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        square = frame[y0:y0 + side, x0:x0 + side]

        if disp_side == 0:
            disp_side = min(DISPLAY_W, DISPLAY_H)
            off_x = (DISPLAY_W - disp_side) // 2
            off_y = (DISPLAY_H - disp_side) // 2

        # Preprocess: downscale, convert the small image, then map
        # uint8 [0,255] -> int8 [-128,127] via a branch-free sign-bit flip
        # (v ^ 0x80 viewed as int8 == v - 128). No float math, no clip.
        small = cv2.resize(square, (INPUT_W, INPUT_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        inp = (rgb ^ np.uint8(0x80)).view(np.int8)
        interpreter.set_tensor(input_idx, inp[np.newaxis, ...])

        t_infer = time.perf_counter()
        interpreter.invoke()
        infer_ms = (time.perf_counter() - t_infer) * 1000.0

        boxes = interpreter.get_tensor(output_details[1]["index"])[0]
        classes = interpreter.get_tensor(output_details[3]["index"])[0]
        scores = interpreter.get_tensor(output_details[0]["index"])[0]
        count = int(interpreter.get_tensor(output_details[2]["index"])[0])

        # Render onto a FRESH canvas: the black bars come free from zeros, and
        # because each published canvas is never mutated again the hand-off to
        # the display thread needs no lock.
        canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
        roi = canvas[off_y:off_y + disp_side, off_x:off_x + disp_side]
        if square.shape[0] == disp_side:
            roi[:] = square
        else:
            cv2.resize(
                square,
                (disp_side, disp_side),
                dst=roi,
                interpolation=cv2.INTER_NEAREST,
            )

        num_objects = 0
        top_score = 0.0

        for box, cls, score in zip(
            boxes[:count],
            classes[:count],
            scores[:count],
        ):

            if score < SCORE_THRESHOLD:
                continue

            num_objects += 1
            top_score = max(top_score, float(score))

            ymin, xmin, ymax, xmax = box

            x1 = off_x + int(xmin * disp_side)
            y1 = off_y + int(ymin * disp_side)
            x2 = off_x + int(xmax * disp_side)
            y2 = off_y + int(ymax * disp_side)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                canvas,
                f"{int(cls)} {score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )

        # Metrics. FPS here is the detection (production) rate; inf + oth is
        # the worker's per-frame time, while disp runs in parallel on main.
        now = time.perf_counter()
        frame_stamps.append((now, None))
        infer_stamps.append((now, infer_ms))
        _prune(frame_stamps, now)
        _prune(infer_stamps, now)

        span = frame_stamps[-1][0] - frame_stamps[0][0]
        fps = (len(frame_stamps) - 1) / span if span > 0 else 0.0
        inf_avg = _avg(infer_stamps)
        frame_ms = 1000.0 / fps if fps > 0 else 0.0
        oth_avg = max(0.0, frame_ms - inf_avg)

        draw_hud(
            canvas,
            [
                f"FPS: {fps:5.1f}",
                f"inf {inf_avg:4.1f} | disp {shared.disp_ms:4.1f}"
                f" | oth {oth_avg:4.1f} ms",
                f"Objects: {num_objects}  Top: {top_score:.2f}",
            ],
        )

        # Periodic stdout log so the metrics are visible headlessly.
        if now - last_log >= 2.0:
            print(
                f"FPS {fps:5.1f} | inf {inf_avg:5.1f} | disp "
                f"{shared.disp_ms:5.1f} | oth {oth_avg:5.1f} ms | obj {num_objects}",
                flush=True,
            )
            last_log = now

        shared.canvas = canvas


worker = threading.Thread(target=inference_worker, daemon=True)
worker.start()

# Main thread: only display. Blit each newly produced canvas; otherwise just
# pump GUI events so the window stays responsive.
last_shown = None
while not stop_event.is_set():
    c = shared.canvas
    if c is not None and c is not last_shown:
        t_disp = time.perf_counter()
        cv2.imshow(WINDOW, c)
        key = cv2.waitKey(1)
        d = (time.perf_counter() - t_disp) * 1000.0
        shared.disp_ms = 0.9 * shared.disp_ms + 0.1 * d
        last_shown = c
    else:
        key = cv2.waitKey(1)

    if key in (27, ord("q")):
        stop_event.set()

stop_event.set()
worker.join(timeout=1.0)
grabber.stop()
cap.release()
cv2.destroyAllWindows()
