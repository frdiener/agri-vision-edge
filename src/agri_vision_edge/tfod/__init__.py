from .common import (
    fpn_native_resize_upsampling,
    get_tf_models_research_dir,
)
from .config import (
    configure_ssd_pipeline,
    load_pipeline_config,
    save_pipeline_config,
)
from .folding import fold_mobilenetv2_backbone

# Import QAT helpers lazily because .qat loads tensorflow_model_optimization at
# module import time. Plain workflows do not require tfmot.


__all__ = [
    #
    # Common
    #
    "get_tf_models_research_dir",
    "fpn_native_resize_upsampling",
    #
    # Config
    #
    "load_pipeline_config",
    "save_pipeline_config",
    "configure_ssd_pipeline",
    #
    # QAT
    #
    "fold_mobilenetv2_backbone",
    "quantize_backbone",
    "quantize_detection_model",
]


# Names that trigger the lazy .qat import.
_QAT_LAZY = {"quantize_backbone", "quantize_detection_model"}


def __getattr__(name):
    if name in _QAT_LAZY:
        from . import qat

        return getattr(qat, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
