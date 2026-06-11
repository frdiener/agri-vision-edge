"""
Dataset utilities.

Provides:

- TFRecord export
- COCO export
- representative datasets
- preprocessing
- label maps
"""

from ..third_party.phenobench import (
    PhenoBench,
)

from .datasets import (
    DatasetDefinition,
    PHENOBENCH_MULTICLASS,
    PHENOBENCH_WEED_ONLY,
)

from .categories import (
    build_category_map,
    build_class_names,
)

from .preprocessing import (
    resize_image_and_boxes,
    normalize_boxes,
    split_indices,
)

from .coco import (
    export_coco_annotations,
)

from .tfrecord import (
    build_record,
)

from .rep_dataset import (
    representative_dataset,
    normalized_representative_dataset,
    build_rep_indices,
)

from .label_map import (
    write_label_map,
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
    "representative_dataset",
    "normalized_representative_dataset",
    "build_rep_indices",
    "write_label_map",
]
