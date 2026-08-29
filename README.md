# agri-vision-edge

Evaluation of quantized lightweight object detection architectures on embedded
NPUs for agricultural applications.

This repository contains the code, experiment definitions and measurement
tooling for my bachelor thesis. It provides an end-to-end pipeline that
fine-tunes lightweight object detectors on a weed-detection dataset, exports
them to INT8 TensorFlow Lite, deploys them to embedded NPU boards, and measures
the accuracy, latency and power cost of each step.

Author: Freimut Diener. License: MIT.

## Scope

The thesis asks how architectural structure and quantization method affect
*delegated execution* on embedded NPUs: which parts of a graph the NPU delegate
accepts, where it falls back to the CPU, and what accuracy, latency and energy
follow from that. Two detectors are studied:

- SSD MobileNetV2 — a contiguous, convolution-only SSD baseline.
- SSD MobileNetV2 FPNLite — the same backbone with a lightweight feature-pyramid
  fusion head.

on two platforms, both driven through the same open-source TensorFlow Lite
stack with the Mesa/Teflon delegate:

| Platform        | NPU                     |
| --------------- | ----------------------- |
| NXP i.MX8M Plus | VeriSilicon/Vivante     |
| NXP i.MX93      | Arm Ethos-U65           |

Models are fine-tuned on [PhenoBench](https://www.phenobench.org/) and exported
with post-training quantization (PTQ) and quantization-aware training (QAT).
YOLOv7-tiny and YOLOX-Nano exports are included as external reference points.
`docs/thesis_proposal.org` states the research questions; `docs/thesis/` holds
the thesis sources.

## Pipeline

```
PhenoBench -> fine-tune (TFOD) -> PTQ / QAT -> INT8 TFLite -> on-device run -> COCO eval
   data          Kaggle GPU        tf.lite      conversion    i.MX8MP/i.MX93   AP, latency, power
```

1. Dataset export. PhenoBench is converted to TFRecord and COCO annotations,
   either as full 1024 px frames or as 512 px tiles (3x3 grid, 0.5 overlap), in
   single-class (weed) or multi-class (crop/weed) form. Plants marked partially
   visible are carried through as do-not-care annotations, matching the official
   benchmark protocol.
2. Fine-tuning. Training uses the TensorFlow Object Detection API through the
   `tfod_trainer` module. It is GPU-bound and runs on Kaggle notebooks
   (`notebooks/*.ipynb`); outputs are merged back locally.
3. Quantization and conversion. `ave convert` exports FP32 and INT8 TFLite
   models from PTQ or QAT checkpoints, in per-tensor and per-channel weight
   granularity, and for both NMS variants of the fused post-processing operator.
4. Deployment and measurement. `ave benchmark` records predictions and latency
   on the target board; `ave resources` runs a sustained inference loop for
   CPU, memory, thermal and power measurement.
5. Evaluation. `ave evaluate` scores predictions with COCO metrics; an optional
   path reproduces the official PhenoBench evaluator for comparability.

The deployment chain is measured at every rung, so the loss attributable to
conversion, post-processing substitution, quantization and delegation can be
separated rather than conflated.

## Repository layout

```
src/agri_vision_edge/
  cli/           `ave` entry point: benchmark, convert, evaluate, infer, resources
  data/          PhenoBench loading, TFRecord/COCO export, tiling, label maps
  tfod/          TFOD model building, QAT, BN folding, TFLite conversion
  tfod_trainer/  training loop (fine-tune, EMA, QAT, export)
  conversion/    conversion targets and metadata
  evaluation/    COCO scoring, delegation analysis, benchmark reports
  experiment/    experiment manifests, environment capture, Kaggle integration
  runtime/       device-side inference (tflite_runtime / tf.lite shim)
  third_party/   vendored tensorflow/models (object_detection)

notebooks/          Kaggle dataset-export and training notebooks; marimo analysis notebooks
scripts/            sweep drivers, power logging and reporting, notebook generators
artifacts/tflite/   exported TFLite models and conversion metadata
benchmark_results/  per-host accuracy and latency results
resource_results/   per-host resource and power measurements
docs/               thesis proposal and thesis sources
```

## Environment

Dependencies are managed with `uv`. Python is pinned to 3.10 and TensorFlow to
2.12.0; the embedded TFLite stack is sensitive to both. A Nix flake provides the
development shell and initializes `.venv`.

```bash
uv sync        # preparation environment: training and conversion
uv run <cmd>   # run inside the venv
```

The preparation and device environments are kept disjoint. The core package
requires only `numpy`, `opencv-python` and `pyyaml`; the default `prep` group
adds TensorFlow and the TFOD stack; the `device` extra adds only
`tflite-runtime`:

```bash
pip install agri-vision-edge[device]   # on the board, no TensorFlow
```

## Usage

`ave` is installed with the package. On a bare checkout use `scripts/ave`.

```bash
ave convert ssd-mn2_mc_phenobench_320              # build FP32/INT8 TFLite models
ave infer artifacts/tflite/<model>.tflite <image>  # single-image inference on the board
scripts/benchmark_all.sh                          # sweep all models on the current device
scripts/evaluate_all.sh benchmark_results/<host>  # score the sweep with COCO metrics
ave resources <model> <images> --seconds 120      # sustained loop for resource/power
```

Model and result names encode the configuration, for example
`ssd-mn2-fpnlite_mc_phenobench-tiled_320_int8_ptq_fastnms`: FPNLite, multi-class,
tiled PhenoBench, 320 px input, INT8, PTQ, fast NMS.

Lint and format with `uv run ruff`, type-check with `uv run basedpyright`, test
with `uv run pytest`.

## Notes

- The vendored `object_detection` package must be set up with
  `setup_tensorflow_models()` before any `object_detection` import; the
  PyPI copy does not work.
- INT8 calibration must feed the representative dataset in the normalization the
  model expects (for example `[-1, 1]`, not `[0, 255]`). The wrong form silently
  miscalibrates and collapses detection scores.
- FP32 models must be benchmarked on the CPU. The delegate reports support for
  float convolutions but returns non-finite outputs, which COCO scoring accepts
  and reports as an implausibly high AP.

## License

MIT. Vendored third-party components, notably the TensorFlow Models Object
Detection API, retain their own licenses; see `THIRD_PARTY_NOTICES.md`.
