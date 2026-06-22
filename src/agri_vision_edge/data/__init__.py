"""
Dataset utilities.

Provides:

- TFRecord export
- COCO export
- representative datasets
- preprocessing
- label maps
"""

from phenobench import PhenoBench

from .categories import (
    build_category_map,
    build_class_names,
)
from .coco import (
    export_coco_annotations,
)
from .datasets import (
    PHENOBENCH_MULTICLASS,
    PHENOBENCH_WEED_ONLY,
    DatasetDefinition,
)
from .label_map import (
    write_label_map,
)
from .preprocessing import (
    normalize_boxes,
    resize_image_and_boxes,
    split_indices,
)
from .rep_dataset import (
    build_rep_indices,
    normalized_representative_dataset,
    representative_dataset,
)
from .tfrecord import (
    build_record,
)
from .yolo import (
    export_yolo_split,
    write_data_yaml,
    yolo_class_names,
)

__all__ = [
    "PhenoBench",
    "DatasetDefinition",
    "PHENOBENCH_MULTICLASS",
    "PHENOBENCH_WEED_ONLY",
    "build_category_map",
    "build_class_names",
    "resize_image_and_boxes",
    "normalize_boxes",
    "split_indices",
    "export_coco_annotations",
    "build_record",
    "export_yolo_split",
    "write_data_yaml",
    "yolo_class_names",
    "representative_dataset",
    "normalized_representative_dataset",
    "build_rep_indices",
    "write_label_map",
]
