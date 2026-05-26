"""
Canonical dataset definitions.

Dataset definitions centralize category semantics used across:

- TFRecord export
- COCO export
- runtime metadata
- visualization
- evaluation

All category IDs follow COCO conventions:

    1-based contiguous integer IDs
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetDefinition:
    """
    Canonical dataset definition.
    """

    name: str

    categories: list[dict]


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
)


PHENOBENCH_WEED_ONLY = DatasetDefinition(

    name="phenobench_weed_only",

    categories=[

        {
            "id": 1,
            "name": "weed",
        },
    ],
)
