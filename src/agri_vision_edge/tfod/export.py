"""
TensorFlow Object Detection export utilities.

Provides helpers for exporting TF-OD checkpoints
to TensorFlow SavedModel and TFLite-compatible
SavedModel formats.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import shutil
import tempfile

from .common import (
    get_tf_models_research_dir,
    run_tfod_command,
)


PathLike = Union[str, Path]


def _prepare_checkpoint_dir(
    trained_checkpoint_dir: PathLike,
    checkpoint_path: Optional[PathLike] = None,
) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """
    Prepare checkpoint directory for TF-OD exporters.

    TF-OD exporters always export the latest checkpoint
    in a checkpoint directory. To export a specific
    checkpoint, this helper creates a temporary checkpoint
    directory containing only the requested checkpoint.

    Args:
        trained_checkpoint_dir:
            Directory containing checkpoints.
        checkpoint_path:
            Optional specific checkpoint path.

    Returns:
        Tuple:
            (
                prepared_checkpoint_dir,
                temporary_directory_handle_or_None
            )
    """
    trained_checkpoint_dir = Path(
        trained_checkpoint_dir
    )

    #
    # Default: use original checkpoint directory
    #

    if checkpoint_path is None:
        return trained_checkpoint_dir, None

    checkpoint_path = Path(checkpoint_path)

    #
    # Create temporary checkpoint directory
    #

    tmpdir_obj = tempfile.TemporaryDirectory()

    tmpdir = Path(tmpdir_obj.name)

    #
    # Copy checkpoint shard files
    #

    for suffix in [
        ".index",
        ".data-00000-of-00001",
    ]:
        src = checkpoint_path.with_suffix(
            suffix
        )

        if not src.exists():
            raise FileNotFoundError(
                f"Missing checkpoint file: {src}"
            )

        dst = tmpdir / src.name

        shutil.copy2(src, dst)

    #
    # Write TF checkpoint state file
    #

    (tmpdir / "checkpoint").write_text(
        (
            f'model_checkpoint_path: '
            f'"{checkpoint_path.name}"\n'
        )
    )

    return tmpdir, tmpdir_obj


def export_saved_model(
    pipeline_config_path: PathLike,
    trained_checkpoint_dir: PathLike,
    output_directory: PathLike,
    checkpoint_path: Optional[PathLike] = None,
    input_type: str = "image_tensor",
    log_file: Optional[PathLike] = None,
):
    """
    Export a TensorFlow Object Detection model
    to TensorFlow SavedModel format.

    This uses TF-OD's generic exporter and produces
    a standard TensorFlow SavedModel suitable for:

    - TensorFlow inference
    - further graph manipulation
    - generic SavedModel workflows

    Args:
        pipeline_config_path:
            Path to pipeline.config.
        trained_checkpoint_dir:
            Directory containing training checkpoints.
        output_directory:
            Export destination directory.
        checkpoint_path:
            Optional specific checkpoint path
            (e.g. ckpt-12).
        input_type:
            TF-OD exporter input type.
        log_file:
            Optional export log file.

    Returns:
        Completed subprocess handle.
    """
    research_dir = get_tf_models_research_dir()

    script = (
        research_dir
        / "object_detection"
        / "exporter_main_v2.py"
    )

    checkpoint_dir, tmpdir_obj = (
        _prepare_checkpoint_dir(
            trained_checkpoint_dir,
            checkpoint_path,
        )
    )

    args = [
        "python",
        str(script),
        "--input_type",
        input_type,
        "--pipeline_config_path",
        str(pipeline_config_path),
        "--trained_checkpoint_dir",
        str(checkpoint_dir),
        "--output_directory",
        str(output_directory),
    ]

    try:
        return run_tfod_command(
            args,
            log_file=log_file,
            background=False,
        )

    finally:
        #
        # Cleanup temporary checkpoint dir
        #

        if tmpdir_obj is not None:
            tmpdir_obj.cleanup()


def export_tflite_graph(
    pipeline_config_path: PathLike,
    trained_checkpoint_dir: PathLike,
    output_directory: PathLike,
    checkpoint_path: Optional[PathLike] = None,
    max_detections: int = 100,
    use_regular_nms: bool = False,
    log_file: Optional[PathLike] = None,
):
    """
    Export a TF-OD model using the dedicated
    TensorFlow Lite export pipeline.

    This exporter rewrites the graph specifically
    for TFLite compatibility and should be preferred
    when the final deployment target is:

    - TensorFlow Lite
    - embedded inference
    - NPU delegates
    - Edge accelerators

    Compared to the generic SavedModel exporter,
    this export path typically produces graphs with:

    - fewer dynamic ops
    - fewer TensorList ops
    - reduced control flow
    - better quantization compatibility

    Args:
        pipeline_config_path:
            Path to pipeline.config.
        trained_checkpoint_dir:
            Directory containing training checkpoints.
        output_directory:
            Export destination directory.
        checkpoint_path:
            Optional specific checkpoint path.
        max_detections:
            Maximum detections per image.
        use_regular_nms:
            Use regular NMS instead of fast NMS.
        log_file:
            Optional export log file.

    Returns:
        Completed subprocess handle.
    """
    research_dir = get_tf_models_research_dir()

    script = (
        research_dir
        / "object_detection"
        / "export_tflite_graph_tf2.py"
    )

    checkpoint_dir, tmpdir_obj = (
        _prepare_checkpoint_dir(
            trained_checkpoint_dir,
            checkpoint_path,
        )
    )

    args = [
        "python",
        str(script),
        "--pipeline_config_path",
        str(pipeline_config_path),
        "--trained_checkpoint_dir",
        str(checkpoint_dir),
        "--output_directory",
        str(output_directory),
        "--max_detections",
        str(max_detections),
    ]

    if use_regular_nms:
        args.append("--use_regular_nms")

    try:
        return run_tfod_command(
            args,
            log_file=log_file,
            background=False,
        )

    finally:
        #
        # Cleanup temporary checkpoint dir
        #

        if tmpdir_obj is not None:
            tmpdir_obj.cleanup()
