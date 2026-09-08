"""Canonical dataset semantics and upstream-to-COCO label mappings.

Exported category IDs are contiguous and 1-based. Labels absent from a
``label_mapping`` are excluded from that dataset variant.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetDefinition:
    """COCO categories and their upstream label mapping."""
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
