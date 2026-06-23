# agri-vision-edge

**Evaluation of Quantized Lightweight Object Detection Architectures on Embedded NPUs for Agricultural Applications**

An end-to-end pipeline for fine-tuning, quantizing, deploying, and benchmarking lightweight object detectors on embedded Neural Processing Units (NPUs). The project accompanies a bachelor thesis investigating how feature-fusion complexity and INT8 quantization affect *delegated execution continuity*, graph partitioning, and runtime latency on heterogeneous edge accelerators — using a fully open-source TensorFlow Lite / Mesa-Teflon deployment stack.

> Author: Freimut Diener · License: MIT

---

## Motivation

Object detection research is usually optimized for benchmark accuracy on the desktop. On embedded systems the picture is different: complex feature-fusion structures, tensor reshaping, and quantization-sensitive operators can quietly push work off the NPU and back onto the CPU, destroying the latency, thermal, and energy budget that low-power agricultural and robotic platforms depend on.

The concrete application is **precision agriculture**. Weed management today often relies on blanket herbicide application. Real-time, on-device weed detection enables *site-specific* treatment — less chemical use, lower cost, and a path toward autonomous field robotics — but only if the detector runs continuously and predictably on a constrained accelerator.

## Research focus

The thesis studies two TensorFlow Lite–compatible detectors:

- **SSD MobileNetV2** — a contiguous, convolution-heavy SSD baseline.
- **SSD MobileNetV2 FPNLite** — the same backbone with a lightweight feature-pyramid fusion head.

…across two embedded NPU design philosophies:

| Platform | NPU | Delegate stack |
| --- | --- | --- |
| **NXP i.MX8M Plus** | VeriSilicon / Vivante | TFLite + Mesa **Teflon** delegate |
| **NXP i.MX93** | Arm **Ethos-U65** | TFLite + Vela compiler |

Both are fine-tuned on the **PhenoBench** weed-detection dataset and exported to INT8 via **post-training quantization (PTQ)** and **quantization-aware training (QAT)**. The guiding research questions:

1. How does lightweight feature-fusion complexity affect delegated INT8 execution continuity on embedded NPUs?
2. How do PTQ and QAT influence quantized deployment behavior and delegation continuity?
3. How do heterogeneous embedded NPUs differ in graph partitioning and CPU fallback behavior?
4. What deployment tradeoffs emerge between contiguous SSD-style architectures and feature-fusion variants?
5. Which architectural and quantization characteristics improve deployment efficiency under TFLite delegated execution?

See [`docs/thesis_proposal.org`](docs/thesis_proposal.org) for context.

## Pipeline overview

```
PhenoBench  ──►  fine-tune (TFOD)  ──►  PTQ / QAT  ──►  INT8 TFLite  ──►  on-device benchmark  ──►  COCO eval
   data            Kaggle GPU          tf.lite          export           i.MX8MP / i.MX93        AP / latency
```

1. **Data** — PhenoBench is converted to TFRecord, optionally **tiled** 2x2 to recover small-object recall, and reduced to single-class (`sc`, weed) or multi-class (`mc`) label maps.
2. **Fine-tuning** — SSD MobileNetV2 / FPNLite are fine-tuned with the tfod_trainer module, using (vendored) TensorFlow Object Detection API libraries.. Training is GPU-bound and runs on **Kaggle** notebooks; outputs are pulled back and merged locally.
3. **Quantization** — INT8 export via `tf.lite` using either PTQ (with a correctly-normalized representative dataset) or QAT (annotated weights / full scheme), plus FP32 baselines.
4. **Deployment & benchmarking** — TFLite models run on the target boards under the matching NPU delegate; latency and per-detection outputs are captured.
5. **Evaluation** — predictions are scored with COCO metrics (AP / AP50 / AP75, size-stratified AP/AR) per platform and configuration.

## Repository layout

```
src/agri_vision_edge/
├── cli/           `ave` console entry point: benchmark / evaluate / infer subcommands
├── data/          PhenoBench loading, TFRecord, tiling, label maps, representative dataset
├── tfod/          TFOD-based training, QAT, BN-folding, export, inference
├── tfod_trainer/  model_lib_v2 training loop (fine-tune / EMA / QAT / export)
├── conversion/    TFLite conversion metadata
├── evaluation/    COCO scoring, benchmark reports, curves, export helpers
├── experiment/    experiment manifests, environment capture, Kaggle integration
├── runtime/       device-side TFLite inference (tflite_runtime ↔ tf.lite shim)
└── third_party/   vendored tensorflow/models (object_detection)

scripts/            shell orchestration + dev/maintenance helpers (run as files)
  ├── benchmark_all.sh / evaluate_all.sh        sweep `ave benchmark` / `ave evaluate`
  ├── sync_kaggle_runs.py                       merge Kaggle finetune + QAT outputs
  ├── prepare_dataset.py                        TFRecord / tiling / label maps
  ├── compile_protos.sh                         regenerate TFOD protobuf stubs
  ├── run_remote.sh                             push & run a model on a board
  └── ave                                       bare-checkout shim for the CLI

notebooks/          Kaggle training/export notebooks (.ipynb) + marimo dev notebooks (.py)
benchmark_results/  per-host results: benchmark_results/<hostname>/<model-stem>/
artifacts/tflite/   exported INT8/FP32 TFLite models + conversion metadata
docs/               thesis proposal, thesis sources (org/tex/pdf)
```

## Environment

This project uses **`uv`** (not pip/poetry) with Python pinned to **3.10** and **TensorFlow 2.12.0** — the embedded TFLite stack is sensitive to these versions. A **Nix flake** provides the dev shell and auto-initializes `.venv`.

```bash
# with Nix + direnv: `direnv allow` initializes the venv automatically
uv sync                       # full preparation/training/conversion env
uv run <cmd>                  # run inside the venv
```

Dependencies are split so the *preparation* and *device* environments stay minimal and disjoint:

- **Core** — `numpy`, `opencv-python`, `pyyaml` (no TFLite interpreter at all).
- **`prep` group** (default) — full `tensorflow` + TFOD stack for training and `tf.lite` conversion.
- **`device` extra** — only `tflite-runtime` for on-device inference:
  ```bash
  pip install agri-vision-edge[device]    # on the board: no tensorflow
  ```

`runtime/inference` prefers `tflite_runtime` and falls back to `tf.lite` only on `ImportError`, so the same code runs in both environments.

## Usage

The pipeline commands are subcommands of a single `ave` CLI. When the package is
installed (`uv sync` / `pip install`), `ave` is on `PATH`. On a bare rsync'd
checkout with no install, use `scripts/ave` (or `PYTHONPATH=src python -m
agri_vision_edge.cli`) — same entry point.

```bash
# Prepare the dataset (TFRecord / tiling / label maps)
uv run python scripts/prepare_dataset.py

# Train / fine-tune / export  →  Kaggle notebooks under notebooks/*.ipynb
uv run python scripts/sync_kaggle_runs.py <config>   # pull + merge Kaggle outputs

# Single-image inference (on-device; needs the `device` extra)
uv run ave infer --model artifacts/tflite/<model>.tflite --image <img>
# bare board:  scripts/ave infer --model <model>.tflite --image <img>

# Benchmark every TFLite model on the current device (wraps `ave benchmark`)
scripts/benchmark_all.sh             # → benchmark_results/<hostname>/<model-stem>/

# Score benchmarked predictions with COCO metrics (wraps `ave evaluate`)
scripts/evaluate_all.sh benchmark_results/<hostname>
```

Model/result naming encodes the configuration, e.g.
`ssd-mn2-fpnlite_mc_phenobench-tiled_320_int8_ptq_fastnms`
→ *FPNLite · multi-class · tiled PhenoBench · 320 px · INT8 · PTQ · fast-NMS*.

## Tooling

| Task | Command |
| --- | --- |
| Lint / format | `uv run ruff check` / `uv run ruff format` |
| Type-check | `uv run basedpyright` |
| Protobufs (TFOD) | `scripts/compile_protos.sh` |

## Notes & gotchas

- **Vendored `object_detection`** — call `setup_tensorflow_models()` before importing anything from `object_detection`; the pypi version does not work, but alternatively, the original repo may be cloned and installed while ignoring dependencies. Also protos need to be compiled then.
- **INT8 calibration** — the representative dataset must feed images in the normalization the model expects (e.g. `[-1, 1]`, not `[0, 255]`); the wrong form silently miscalibrates and collapses scores.
- **Delegate / precision pairing** — INT8 models go to the NPU delegate; FP32 models should run on CPU (the delegate reports float-conv support but silently degrades accuracy).

## License

MIT. Third-party components (notably the vendored TensorFlow Models / Object Detection API) retain their own licenses — see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
