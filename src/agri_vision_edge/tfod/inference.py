"""
Inference utilities for TensorFlow Object Detection (TF-OD) models.

This module provides a clean interface for:
- loading exported SavedModels
- preprocessing input images
- running inference
- applying optional Non-Maximum Suppression (NMS)
- visualizing detections

Typical usage:

    detect_fn = load_saved_model("/path/to/saved_model")
    category_index = load_label_map("label_map.pbtxt")

    vis, detections = detect_image(
        detect_fn,
        image_path="image.png",
        category_index=category_index,
    )
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from object_detection.utils import (
    label_map_util,
    visualization_utils as viz_utils,
)

# Type alias
PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------

def load_saved_model(model_dir: PathLike):
    """
    Load a TensorFlow SavedModel exported by TF-OD.

    Args:
        model_dir:
            Path to exported_model/saved_model directory.

    Returns:
        Callable TensorFlow detection function.
    """
    return tf.saved_model.load(str(model_dir))


def load_label_map(label_map_path: PathLike) -> Dict:
    """
    Load TF-OD label map.

    Args:
        label_map_path:
            Path to label_map.pbtxt.

    Returns:
        category_index dictionary used by visualization.
    """
    return label_map_util.create_category_index_from_labelmap(
        str(label_map_path),
        use_display_name=True,
    )


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def preprocess_image(
    image: np.ndarray,
    image_size: Optional[int] = 320,
) -> np.ndarray:
    """
    Preprocess image for TF-OD inference.

    Args:
        image:
            RGB image as NumPy array (H, W, 3).
        image_size:
            Optional target size (square resize). If None, no resizing.

    Returns:
        Preprocessed RGB image.
    """
    if image_size is not None:
        image = cv2.resize(image, (image_size, image_size))

    return image


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

def run_inference(
    detect_fn,
    image: np.ndarray,
) -> Dict[str, tf.Tensor]:
    """
    Run TF-OD inference on an RGB image.

    Args:
        detect_fn:
            Loaded TF SavedModel callable.
        image:
            RGB image (H, W, 3).

    Returns:
        Detection dictionary containing:
            - detection_boxes
            - detection_scores
            - detection_classes
    """
    input_tensor = tf.convert_to_tensor(image[tf.newaxis, ...])
    detections = detect_fn(input_tensor)
    return detections


# ---------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------

def apply_nms(
    detections: Dict[str, tf.Tensor],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    max_detections: int = 50,
) -> Dict[str, tf.Tensor]:
    """
    Apply Non-Maximum Suppression (NMS) to TF-OD detections.

    This reduces overlapping bounding boxes by keeping only the highest
    scoring boxes per region.

    Args:
        detections:
            Raw TF-OD detection dictionary.
        iou_threshold:
            Intersection-over-Union threshold for suppression.
        score_threshold:
            Minimum score for a box to be considered.
        max_detections:
            Maximum number of boxes to keep.

    Returns:
        Filtered detection dictionary with same structure.
    """
    boxes = detections["detection_boxes"][0]
    scores = detections["detection_scores"][0]
    classes = detections["detection_classes"][0]

    selected_indices = tf.image.non_max_suppression(
        boxes,
        scores,
        max_output_size=max_detections,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )

    return {
        "detection_boxes": tf.gather(boxes, selected_indices)[tf.newaxis, ...],
        "detection_scores": tf.gather(scores, selected_indices)[tf.newaxis, ...],
        "detection_classes": tf.gather(classes, selected_indices)[tf.newaxis, ...],
    }


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def visualize_detections(
    image: np.ndarray,
    detections: Dict[str, tf.Tensor],
    category_index: Dict,
    score_threshold: float = 0.0,
    max_boxes: int = 50,
) -> Image.Image:
    """
    Visualize TF-OD detections on an image.

    Args:
        image:
            RGB image (H, W, 3).
        detections:
            Detection dictionary (optionally NMS-filtered).
        category_index:
            Label map dictionary.
        score_threshold:
            Minimum score for visualization (typically 0.0 if NMS applied).
        max_boxes:
            Maximum boxes to draw.

    Returns:
        PIL Image with rendered bounding boxes.
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


# ---------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------

def detect_image(
    detect_fn,
    image_path: PathLike,
    category_index: Dict,
    image_size: Optional[int] = 320,
    score_threshold: float = 0.05,
    max_boxes: int = 50,
    apply_nms_flag: bool = True,
    nms_iou_threshold: float = 0.5,
) -> Tuple[Image.Image, Dict[str, tf.Tensor]]:
    """
    Run full TF-OD inference pipeline on an image.

    Steps:
        1. Load image from disk
        2. Convert to RGB
        3. Resize (optional)
        4. Run inference
        5. Apply NMS (optional)
        6. Visualize detections

    Args:
        detect_fn:
            Loaded TF SavedModel callable.
        image_path:
            Path to input image.
        category_index:
            Label map dictionary.
        image_size:
            Optional resize target (must match training if used).
        score_threshold:
            Score threshold for NMS (and fallback visualization).
        max_boxes:
            Maximum number of detections to keep/draw.
        apply_nms_flag:
            Whether to apply NMS postprocessing.
        nms_iou_threshold:
            IoU threshold for NMS.

    Returns:
        Tuple:
            - PIL Image with detections
            - Raw or filtered detection dictionary
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = preprocess_image(
        image,
        image_size=image_size,
    )

    detections = run_inference(
        detect_fn,
        image,
    )

    if apply_nms_flag:
        detections = apply_nms(
            detections,
            iou_threshold=nms_iou_threshold,
            score_threshold=score_threshold,
            max_detections=max_boxes,
        )
        vis_threshold = 0.0  # already filtered
    else:
        vis_threshold = score_threshold

    visualization = visualize_detections(
        image=image,
        detections=detections,
        category_index=category_index,
        score_threshold=vis_threshold,
        max_boxes=max_boxes,
    )

    return visualization, detections
