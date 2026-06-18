"""Embed TFLite ObjectDetector metadata into the exported model."""

from pathlib import Path
import re
import tempfile

from tflite_support import metadata as metadata_api
from tflite_support.metadata_writers import object_detector, writer_utils


def read_labels(label_map_path: Path, expected_classes: int) -> list[str]:
    """Read labels from a TFOD label_map.pbtxt in ascending ID order."""
    text = label_map_path.read_text(encoding="utf-8")

    items: list[tuple[int, str]] = []

    for block in re.findall(r"\bitem\s*\{(.*?)\}", text, flags=re.DOTALL):
        id_match = re.search(r"\bid\s*:\s*(\d+)", block)
        name_match = re.search(
            r'\b(?:display_name|name)\s*:\s*["\']([^"\']+)["\']',
            block,
        )

        if id_match and name_match:
            items.append((int(id_match.group(1)), name_match.group(1)))

    if not items:
        raise ValueError(f"No labels found in {label_map_path}")

    items.sort(key=lambda item: item[0])

    ids = [label_id for label_id, _ in items]
    labels = [label for _, label in items]

    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate label IDs in {label_map_path}: {ids}")

    if len(labels) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} classes, but found "
            f"{len(labels)} in {label_map_path}: {labels}"
        )

    return labels


def write_object_detector_metadata(
    model_path: Path,
    label_map_path: Path,
    num_classes: int,
) -> Path:
    """
    Embed ObjectDetector metadata and labels into a TFLite model.

    The model expects normalized input:

        normalized = (pixel - 127.5) / 127.5

    The normalization metadata remains the same for float and quantized
    models. Quantized models additionally carry their tensor quantization
    parameters in the TFLite graph.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    if not label_map_path.is_file():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")

    labels = read_labels(label_map_path, num_classes)

    # MetadataWriter requires labels as a file path. populate() embeds that
    # file into the TFLite model, so the temporary file is not retained.
    with tempfile.TemporaryDirectory(prefix="tflite-metadata-") as temp_dir:
        labels_path = Path(temp_dir) / "labels.txt"
        labels_path.write_text(
            "".join(f"{label}\n" for label in labels),
            encoding="utf-8",
        )

        writer = object_detector.MetadataWriter.create_for_inference(
            writer_utils.load_file(str(model_path)),
            input_norm_mean=[127.5],
            input_norm_std=[127.5],
            label_file_paths=[str(labels_path)],
        )

        populated_model = writer.populate()

    # The labels file has now been embedded in populated_model.
    writer_utils.save_file(populated_model, str(model_path))

    metadata_json_path = model_path.with_suffix(".metadata.json")
    metadata_json = metadata_api.MetadataDisplayer.with_model_file(
        str(model_path)
    ).get_metadata_json()

    metadata_json_path.write_text(
        metadata_json,
        encoding="utf-8",
    )

    print(f"Metadata written: {model_path.name}")
    print(f"Embedded labels:  {labels}")
    print(f"Metadata JSON:    {metadata_json_path.name}")

    return metadata_json_path
