"""
Canonical dataset definitions.

Dataset definitions centralize dataset semantics used across:

- TFRecord export
- COCO export
- runtime metadata
- visualization
- evaluation

All exported category IDs follow COCO conventions:

    1-based contiguous integer IDs

Dataset definitions additionally support:

- semantic label remapping
- binary detection variants
- merged-category experiments
- framework-independent evaluation

The `label_mapping` field maps upstream dataset
labels to exported category IDs.

Example:

    upstream PhenoBench labels:
        1 -> crop
        2 -> weed

    weed-only export:
        2 -> 1
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetDefinition:
    """
    Canonical dataset definition.

    Attributes
    ----------
    name:
        Human-readable dataset definition name.

    categories:
        Exported COCO-compatible categories.

    label_mapping:
        Mapping from upstream dataset labels
        to exported category IDs.

        Labels not present in the mapping
        are ignored during export.
    """
    name: str
    categories: list[dict]
    label_mapping: dict[int, int]

#
# Full multiclass detection
#
# upstream:
#   1 -> crop
#   2 -> weed
#

PHENOBENCH_MULTICLASS = DatasetDefinition(
    name="phenobench_multiclass",
    categories=[
        {
            "id": 1,
            "name": "crop",
        },
        {
            "id": 2,
            "name": "weed",
        },
    ],
    label_mapping={
        1: 1,
        2: 2,
    },
)


#
# Binary weed-only detection
#
# upstream:
#   2 -> weed
#
# exported:
#   1 -> weed
#

PHENOBENCH_WEED_ONLY = DatasetDefinition(
    name="phenobench_weed_only",
    categories=[
        {
            "id": 1,
            "name": "weed",
        },
    ],
    label_mapping={
        2: 1,
    },
)
