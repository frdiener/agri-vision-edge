from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_STRUCTURE = {
    "metadata": {},
    "environment": {},
    "checkpoint": {},
    "dataset": {},
    "stages": {},
    "results": {},
    "artifacts": {},
}


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


@dataclass
class ExperimentManifest:
    """
    Lightweight framework-agnostic experiment manifest.

    Designed for:
    - multi-stage ML pipelines
    - Kaggle notebook modular workflows
    - edge deployment benchmarking
    - quantization studies
    - artifact lineage tracking

    Stages are immutable snapshots of:
    - config
    - metrics
    - artifacts
    - runtime/environment information
    """

    name: str
    task: str | None = None

    experiment_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    parent_experiment_id: str | None = None

    data: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(
            DEFAULT_STRUCTURE
        )
    )

    # =========================================================
    # Initialization
    # =========================================================

    def __post_init__(self) -> None:

        self.data["metadata"] = {
            "experiment_id": self.experiment_id,
            "parent_experiment_id": self.parent_experiment_id,
            "name": self.name,
            "task": self.task,
            "created_at_utc": utc_now_iso(),
            "schema_version": "2.0",
        }

    # =========================================================
    # Generic section helpers
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
        Merge values into a top-level section.
        """

        self._validate_section(section)

        self.data[section].update(values)

    # =========================================================
    # Environment / Dataset / Checkpoint
    # =========================================================

    def set_environment(
        self,
        **kwargs: Any,
    ) -> None:

        self.data["environment"].update(kwargs)

    def set_dataset(
        self,
        **kwargs: Any,
    ) -> None:

        self.data["dataset"].update(kwargs)

    def set_checkpoint(
        self,
        **kwargs: Any,
    ) -> None:

        self.data["checkpoint"].update(kwargs)

    # =========================================================
    # Stage handling
    # =========================================================

    def add_stage(
        self,
        stage_name: str,
        *,
        config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        notes: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a pipeline stage snapshot.

        Example stages:
        - finetune
        - qat
        - conversion
        - runtime_eval
        """

        if stage_name in self.data["stages"]:
            raise ValueError(
                f"Stage already exists: {stage_name}"
            )

        self.data["stages"][stage_name] = {
            "created_at_utc": utc_now_iso(),
            "config": config or {},
            "metrics": metrics or {},
            "artifacts": artifacts or {},
            "runtime": runtime or {},
            "notes": notes or {},
        }

    def update_stage(
        self,
        stage_name: str,
        *,
        config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        notes: dict[str, Any] | None = None,
    ) -> None:
        """
        Update existing stage contents.
        """

        if stage_name not in self.data["stages"]:
            raise ValueError(
                f"Unknown stage: {stage_name}"
            )

        stage = self.data["stages"][stage_name]

        if config:
            stage["config"].update(config)

        if metrics:
            stage["metrics"].update(metrics)

        if artifacts:
            stage["artifacts"].update(artifacts)

        if runtime:
            stage["runtime"].update(runtime)

        if notes:
            stage["notes"].update(notes)

    # =========================================================
    # Results
    # =========================================================

    def add_result(
        self,
        name: str,
        value: Any,
    ) -> None:

        self.data["results"][name] = value

    # =========================================================
    # Global artifacts
    # =========================================================

    def add_artifact(
        self,
        path: str | Path,
        *,
        artifact_type: str | None = None,
        stage: str | None = None,
        description: str | None = None,
    ) -> None:
        """
        Register a global artifact.
        """

        artifact = {
            "path": str(path),
        }

        if artifact_type:
            artifact["type"] = artifact_type

        if stage:
            artifact["stage"] = stage

        if description:
            artifact["description"] = description

        self.data["artifacts"].setdefault(
            "files",
            []
        ).append(artifact)

    # =========================================================
    # Merge support
    # =========================================================

    def merge(
        self,
        other: ExperimentManifest,
    ) -> None:
        """
        Merge another manifest into this one.

        Intended for:
        - modular notebook workflows
        - stage aggregation
        - distributed experiment execution
        """

        # Merge stages
        for stage_name, stage_data in other.data[
            "stages"
        ].items():

            if stage_name in self.data["stages"]:
                raise ValueError(
                    f"Duplicate stage during merge: "
                    f"{stage_name}"
                )

            self.data["stages"][
                stage_name
            ] = stage_data

        # Merge artifacts
        self.data["artifacts"].setdefault(
            "files",
            []
        ).extend(
            other.data["artifacts"].get(
                "files",
                []
            )
        )

        # Merge results
        self.data["results"].update(
            other.data["results"]
        )

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
    ) -> ExperimentManifest:

        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        metadata = data.get(
            "metadata",
            {},
        )

        manifest = cls(
            name=metadata.get(
                "name",
                "experiment",
            ),
            task=metadata.get("task"),
            experiment_id=metadata.get(
                "experiment_id",
                str(uuid.uuid4()),
            ),
            parent_experiment_id=metadata.get(
                "parent_experiment_id"
            ),
        )

        manifest.data = data

        return manifest

    # =========================================================
    # Internal helpers
    # =========================================================

    @staticmethod
    def _validate_section(
        section: str,
    ) -> None:

        if section not in DEFAULT_STRUCTURE:
            raise ValueError(
                f"Unknown manifest section: "
                f"{section}"
            )
