"""
Inference utilities for TensorFlow Object Detection models.

Provides helpers for:
- loading exported SavedModels
- running inference
- visualizing detections
"""

from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import tensorflow as tf

from PIL import Image

from object_detection.utils import (
    label_map_util,
    visualization_utils as viz_utils,
)


PathLike = Union[str, Path]


def load_saved_model(
    model_dir: PathLike,
):
    """
    Load a TensorFlow SavedModel exported by TF-OD.

    Args:
        model_dir:
            Path to exported_model/saved_model.

    Returns:
        Loaded TF SavedModel callable.
    """
    return tf.saved_model.load(str(model_dir))


def load_label_map(
    label_map_path: PathLike,
):
    """
    Load TF-OD label map.

    Args:
        label_map_path:
            Path to label_map.pbtxt.

    Returns:
        category_index dictionary.
    """
    return label_map_util.create_category_index_from_labelmap(
        str(label_map_path),
        use_display_name=True,
    )


def preprocess_image(
    image: np.ndarray,
    image_size: Optional[int] = 320,
) -> np.ndarray:
    """
    Preprocess image for TF-OD inference.

    Args:
        image:
            RGB image.
        image_size:
            Optional resize target.

    Returns:
        Preprocessed RGB image.
    """
    if image_size is not None:
        image = cv2.resize(
            image,
            (image_size, image_size),
        )

    return image


def run_inference(
    detect_fn,
    image: np.ndarray,
) -> Dict:
    """
    Run TF-OD inference on an RGB image.

    Args:
        detect_fn:
            Loaded TF SavedModel callable.
        image:
            RGB image.

    Returns:
        Detection dictionary.
    """
    input_tensor = tf.convert_to_tensor(
        image[tf.newaxis, ...]
    )

    detections = detect_fn(input_tensor)

    return detections


def visualize_detections(
    image: np.ndarray,
    detections: Dict,
    category_index: Dict,
    score_threshold: float = 0.05,
    max_boxes: int = 100,
) -> Image.Image:
    """
    Visualize TF-OD detections.

    Args:
        image:
            RGB image.
        detections:
            TF-OD detection output.
        category_index:
            TF-OD label map dictionary.
        score_threshold:
            Minimum confidence threshold.
        max_boxes:
            Maximum boxes to draw.

    Returns:
        PIL image with rendered detections.
    """
    image_vis = image.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_vis,
        detections["detection_boxes"][0].numpy(),
        detections["detection_classes"][0].numpy().astype(np.int32),
        detections["detection_scores"][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=max_boxes,
        min_score_thresh=score_threshold,
        agnostic_mode=False,
    )

    return Image.fromarray(image_vis)


def detect_image(
    detect_fn,
    image_path: PathLike,
    category_index: Dict,
    image_size: Optional[int] = 320,
    score_threshold: float = 0.05,
    max_boxes: int = 100,
):
    """
    Run full TF-OD inference pipeline on an image.

    Args:
        detect_fn:
            Loaded TF SavedModel callable.
        image_path:
            Path to input image.
        category_index:
            TF-OD label map dictionary.
        image_size:
            Optional resize target.
        score_threshold:
            Visualization confidence threshold.
        max_boxes:
            Maximum boxes to draw.

    Returns:
        Tuple:
            PIL visualization image,
            raw detections
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = preprocess_image(
        image,
        image_size=image_size,
    )

    detections = run_inference(
        detect_fn,
        image,
    )

    visualization = visualize_detections(
        image=image,
        detections=detections,
        category_index=category_index,
        score_threshold=score_threshold,
        max_boxes=max_boxes,
    )

    return visualization, detections
