"""
Unit tests for the numpy port of PhenoBench's partial-plant filter rule.

These pin the ported behaviour to the upstream semantics in
``phenobench.evaluation.auxiliary.filter.filter_partials_boxes``: a prediction
whose own area is more than ``threshold`` contained inside a partial ground-truth
box is dropped, and partial ground-truth boxes are split out of scoring.
"""

from __future__ import annotations

import numpy as np

from agri_vision_edge.evaluation.partials import (
    containment_fractions,
    filter_predictions_against_partials,
    is_partial_annotation,
    partial_prediction_mask,
    split_annotations_by_partial,
    xywh_to_xyxy,
)


def test_xywh_to_xyxy():
    boxes = np.array([[10.0, 20.0, 5.0, 7.0]])
    np.testing.assert_allclose(xywh_to_xyxy(boxes), [[10.0, 20.0, 15.0, 27.0]])


def test_xywh_to_xyxy_empty():
    assert xywh_to_xyxy(np.zeros((0, 4))).shape == (0, 4)


def test_containment_fraction_fully_inside():
    # Prediction fully inside the GT -> fraction 1.0.
    pred = xywh_to_xyxy(np.array([[10.0, 10.0, 10.0, 10.0]]))
    gt = xywh_to_xyxy(np.array([[0.0, 0.0, 100.0, 100.0]]))
    frac = containment_fractions(pred, gt)
    assert frac.shape == (1, 1)
    np.testing.assert_allclose(frac[0, 0], 1.0)


def test_containment_fraction_half_inside():
    # Half of the prediction's area overlaps the GT -> fraction 0.5.
    pred = xywh_to_xyxy(np.array([[0.0, 0.0, 10.0, 10.0]]))
    gt = xywh_to_xyxy(np.array([[5.0, 0.0, 100.0, 10.0]]))
    frac = containment_fractions(pred, gt)
    np.testing.assert_allclose(frac[0, 0], 0.5)


def test_containment_zero_area_prediction():
    pred = xywh_to_xyxy(np.array([[0.0, 0.0, 0.0, 0.0]]))
    gt = xywh_to_xyxy(np.array([[0.0, 0.0, 10.0, 10.0]]))
    frac = containment_fractions(pred, gt)
    np.testing.assert_allclose(frac[0, 0], 0.0)


def test_partial_mask_drops_only_over_threshold():
    partials = np.array([[0.0, 0.0, 100.0, 100.0]])
    preds = np.array(
        [
            [10.0, 10.0, 10.0, 10.0],  # fully inside -> drop
            [95.0, 95.0, 10.0, 10.0],  # 25% inside (5x5 of 10x10) -> keep
        ]
    )
    mask = partial_prediction_mask(preds, partials, threshold=0.5)
    assert mask.tolist() == [True, False]


def test_partial_mask_boundary_is_strict():
    # Exactly 0.5 must NOT be dropped (upstream uses score > 0.5).
    preds = np.array([[0.0, 0.0, 10.0, 10.0]])
    partials = np.array([[5.0, 0.0, 100.0, 10.0]])  # exactly half
    mask = partial_prediction_mask(preds, partials, threshold=0.5)
    assert mask.tolist() == [False]


def test_partial_mask_no_partials():
    preds = np.array([[0.0, 0.0, 10.0, 10.0]])
    mask = partial_prediction_mask(preds, np.zeros((0, 4)), threshold=0.5)
    assert mask.tolist() == [False]


def test_is_partial_annotation_variants():
    assert is_partial_annotation({"partial": 1})
    assert is_partial_annotation({"visibility": 0.3})
    assert not is_partial_annotation({"visibility": 0.9})
    assert is_partial_annotation({"ignore": 1})
    assert not is_partial_annotation({})
    # visibility exactly at the threshold counts as partial (<=).
    assert is_partial_annotation({"visibility": 0.5})


def test_split_annotations_by_partial():
    anns = [
        {"id": 1, "visibility": 0.9},
        {"id": 2, "visibility": 0.2},
        {"id": 3, "partial": 1},
        {"id": 4},
    ]
    scored, partial = split_annotations_by_partial(anns)
    assert [a["id"] for a in scored] == [1, 4]
    assert [a["id"] for a in partial] == [2, 3]


def test_filter_predictions_against_partials_per_image():
    partials = [
        {"image_id": 1, "bbox": [0.0, 0.0, 100.0, 100.0]},
    ]
    predictions = [
        {"image_id": 1, "bbox": [10.0, 10.0, 10.0, 10.0], "score": 0.9},  # drop
        {"image_id": 1, "bbox": [200.0, 200.0, 10.0, 10.0], "score": 0.8},  # keep
        {"image_id": 2, "bbox": [10.0, 10.0, 10.0, 10.0], "score": 0.7},  # keep (no partial)
    ]
    kept = filter_predictions_against_partials(predictions, partials)
    kept_scores = sorted(p["score"] for p in kept)
    assert kept_scores == [0.7, 0.8]


def test_filter_predictions_no_partials_passthrough():
    predictions = [{"image_id": 1, "bbox": [0.0, 0.0, 1.0, 1.0], "score": 0.5}]
    kept = filter_predictions_against_partials(predictions, [])
    assert kept == predictions
