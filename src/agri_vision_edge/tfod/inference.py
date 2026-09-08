"""Inference and visualization helpers for TFOD SavedModels."""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import (
    label_map_util,
)
from object_detection.utils import (
    visualization_utils as viz_utils,
)
from PIL import Image

# Type alias
PathLike = str | Path


# Loading utilities

def load_saved_model(model_dir: PathLike):
    """Load a TFOD SavedModel detection function."""
    return tf.saved_model.load(str(model_dir))


def load_label_map(label_map_path: PathLike) -> dict:
    """Load a TFOD label map as a visualization category index."""
    return label_map_util.create_category_index_from_labelmap(
        str(label_map_path),
        use_display_name=True,
    )


# Preprocessing

def preprocess_image(
    image: np.ndarray,
    image_size: int | None = 320,
) -> np.ndarray:
    """Optionally resize an RGB image to a square TFOD input."""
    if image_size is not None:
        image = cv2.resize(image, (image_size, image_size))

    return image


# Inference

def run_inference(
    detect_fn,
    image: np.ndarray,
) -> dict[str, tf.Tensor]:
    """Run a TFOD detection function on one RGB image."""
    input_tensor = tf.convert_to_tensor(image[tf.newaxis, ...])
    detections = detect_fn(input_tensor)
    return detections


# Postprocessing

def apply_nms(
    detections: dict[str, tf.Tensor],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    max_detections: int = 50,
) -> dict[str, tf.Tensor]:
    """Apply NMS and return a detection dictionary with a batch dimension."""
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


# Visualization

def visualize_detections(
    image: np.ndarray,
    detections: dict[str, tf.Tensor],
    category_index: dict,
    score_threshold: float = 0.0,
    max_boxes: int = 50,
) -> Image.Image:
    """Render TFOD detections on an RGB image."""
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


# High-level API

def detect_image(
    detect_fn,
    image_path: PathLike,
    category_index: dict,
    image_size: int | None = 320,
    score_threshold: float = 0.05,
    max_boxes: int = 50,
    apply_nms_flag: bool = True,
    nms_iou_threshold: float = 0.5,
) -> tuple[Image.Image, dict[str, tf.Tensor]]:
    """Load, optionally resize, infer, filter, and annotate one RGB image.

    ``image_size`` must match the model's training size when set. Returns the
    annotated PIL image and the raw or NMS-filtered detections.
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
