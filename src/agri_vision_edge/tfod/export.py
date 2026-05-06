"""
TensorFlow Object Detection export utilities.

Provides helpers for exporting TF-OD checkpoints
to TensorFlow SavedModel format.
"""

from pathlib import Path
from typing import Optional, Union
import shutil
import tempfile

from .common import (
    get_tf_models_research_dir,
    run_tfod_command,
)


PathLike = Union[str, Path]


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

    By default, TensorFlow Object Detection exports the
    latest checkpoint found inside `trained_checkpoint_dir`.

    Optionally, a specific checkpoint can be exported
    by providing `checkpoint_path`.

    Since TF-OD's exporter does not support exporting
    arbitrary checkpoints directly, this helper creates
    a temporary checkpoint directory containing only the
    requested checkpoint and its checkpoint state file.

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

    trained_checkpoint_dir = Path(
        trained_checkpoint_dir
    )

    #
    # Optional specific checkpoint export
    #
    tmpdir_obj = None

    if checkpoint_path is not None:
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

        trained_checkpoint_dir = tmpdir

    args = [
        "python",
        str(script),
        "--input_type",
        input_type,
        "--pipeline_config_path",
        str(pipeline_config_path),
        "--trained_checkpoint_dir",
        str(trained_checkpoint_dir),
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
