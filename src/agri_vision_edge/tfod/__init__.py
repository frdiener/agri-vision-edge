from .common import (
    get_tf_models_research_dir,
)

from .config import (
    load_pipeline_config,
    save_pipeline_config,
    configure_ssd_pipeline,
)

from .train import (
    launch_training,
)

from .eval import (
    launch_eval,
)

from .export import (
    export_saved_model,
    export_tflite_graph,
    export_all,
)

from .mobilenetv2_bn_folding import (
    fold_mobilenetv2_backbone
)

# NOTE: the QAT helpers (quantize_backbone / quantize_detection_head) live in
# .qat, which imports tensorflow_model_optimization at module load. They are
# exposed lazily via __getattr__ below so that importing `agri_vision_edge.tfod`
# for a plain (non-QAT) workflow does not require tfmot -- it is only pulled in
# when a QAT helper is actually accessed.


__all__ = [
    #
    # Common
    #
    "get_tf_models_research_dir",

    #
    # Config
    #
    "load_pipeline_config",
    "save_pipeline_config",
    "configure_ssd_pipeline",

    #
    # Training
    #
    "launch_training",

    #
    # Evaluation
    #
    "launch_eval",

    #
    # Export
    #
    "export_saved_model",
    "export_tflite_graph",
    "export_all",

    #
    # QAT
    #
    "fold_mobilenetv2_backbone",
    "quantize_backbone",
    "quantize_detection_head",
]


# Lazy access to the QAT helpers (see note above): only imports .qat -- and thus
# tensorflow_model_optimization -- when one of these names is actually used.
_QAT_LAZY = {"quantize_backbone", "quantize_detection_head"}


def __getattr__(name):
    if name in _QAT_LAZY:
        from . import qat

        return getattr(qat, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
