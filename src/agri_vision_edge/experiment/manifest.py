from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STRUCTURE = {
    "metadata": {},
    "environment": {},
    "inputs": {},
    "training": {},
    "metrics": {},
    "results": {},
    "artifacts": {},
}


@dataclass
class ExperimentManifest:
    """
    Lightweight experiment tracking manifest.

    Stores:
    - experiment metadata
    - environment information
    - training configuration
    - metrics
    - artifacts
    - evaluation results

    The manifest is intentionally framework-agnostic.
    """

    name: str
    task: str | None = None

    data: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_STRUCTURE.copy()
    )

    def __post_init__(self) -> None:

        self.data["metadata"] = {
            "name": self.name,
            "task": self.task,
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    # =========================================================
    # Generic insertion helpers
    # =========================================================

    def set_section(
        self,
        section: str,
        values: dict[str, Any],
    ) -> None:
        """
        Replace an entire top-level section.
        """

        self._validate_section(section)

        self.data[section] = values

    def update_section(
        self,
        section: str,
        values: dict[str, Any],
    ) -> None:
        """
        Merge values into a section.
        """

        self._validate_section(section)

        self.data[section].update(values)

    def add_metric(
        self,
        name: str,
        value: Any,
    ) -> None:
        """
        Add a scalar metric.
        """

        self.data["metrics"][name] = value

    def add_artifact(
        self,
        path: str | Path,
        artifact_type: str | None = None,
    ) -> None:
        """
        Register an artifact file.
        """

        artifact = {
            "path": str(path),
        }

        if artifact_type:
            artifact["type"] = artifact_type

        self.data["artifacts"].setdefault(
            "files",
            []
        ).append(artifact)

    # =========================================================
    # Serialization
    # =========================================================

    def to_dict(self) -> dict[str, Any]:
        return self.data

    def save(
        self,
        path: str | Path,
        indent: int = 2,
    ) -> None:

        path = Path(path)

        with open(path, "w") as f:
            json.dump(
                self.data,
                f,
                indent=indent,
                default=str,
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "ExperimentManifest":

        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        metadata = data.get("metadata", {})

        manifest = cls(
            name=metadata.get(
                "name",
                "experiment",
            ),
            task=metadata.get("task"),
        )

        manifest.data = data

        return manifest

    # =========================================================
    # Internal helpers
    # =========================================================

    @staticmethod
    def _validate_section(section: str) -> None:

        if section not in DEFAULT_STRUCTURE:
            raise ValueError(
                f"Unknown manifest section: {section}"
            )
