This directory contains vendored third-party code.

- tensorflow_models: copied from https://github.com/tensorflow/models
  - License: Apache 2.0 (see `LICENSE`)
  - **No local source modifications.** This is stock upstream `object_detection`
    + `slim`. Earlier local patches to `model_lib_v2.py`, `model_main_tf2.py`,
    `exporter_lib_v2.py` and `export_tflite_graph_lib_tf2.py` were reverted to
    upstream `master`; the custom training / eval / QAT-export logic now lives
    in `agri_vision_edge.tfod_trainer`, which uses only stock-upstream symbols.

Why this is still vendored (rather than the PyPI package):

- The site-packages `object_detection` install is broken, so
  `setup_tensorflow_models()` injects `research/` + `research/slim/` instead.
- The compiled protobuf stubs (`object_detection/protos/*_pb2.py`) are checked
  in here. They are the one thing this tree adds on top of upstream (upstream
  ships only the `.proto` sources). Regenerate them with
  `scripts/compile_protos.sh` if you bump the vendored `.proto` files.
