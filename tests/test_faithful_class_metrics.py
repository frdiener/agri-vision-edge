"""
Tests for how upstream's ``mAP`` / ``mAP_cls`` are reported back.

Upstream computes ``mAP`` as the unweighted mean over *whichever* classes it
happened to score, and that set is not stable:

* ``cvt_gt_to_bbox_map`` labels instances with their raw ``semantics`` value and
  never calls ``convert_partial_semantics``, so PhenoBench's partial ids ``3``
  and ``4`` survive as extra classes;
* ``filter_partials_boxes`` nests its ground-truth removal inside the
  per-prediction loop, so an image with **zero** predictions keeps its partial
  ground truth -- and with it those extra classes.

Measured on the i.MX8MP sweep, the extra-class count tracked the number of
prediction-less images exactly (0 -> 2 classes, a handful -> 3, hundreds -> 4),
and the reported ``mAP`` equalled ``mean(mAP_cls)`` in every single run. Reading
``mAP_cls[1]`` as "weed" therefore only works by accident, and ``mAP`` is not
comparable between runs at all.
"""

from __future__ import annotations

import pytest

from agri_vision_edge.evaluation.faithful import annotate_class_metrics


def test_multiclass_two_classes_is_the_plain_mean():
    out = annotate_class_metrics(
        {"mAP": 32.95, "mAP_cls": [40.71, 25.2]}, ["crop", "weed"], 0
    )

    assert out["ap_per_class"] == {"crop": 40.71, "weed": 25.2}
    assert out["ap_partial_classes"] == []
    assert out["mAP_plants"] == pytest.approx(32.96, abs=0.01)
    assert out["class_names"] == ["crop", "weed"]
    assert out["upstream_class_count"] == 2


def test_phantom_partial_class_dilutes_upstream_map_but_not_map_plants():
    # Real run: tiled fpnlite mc per-tensor. Upstream reported mAP 28.11, which
    # is (54.06 + 30.26 + 0) / 3 -- the third entry is partial-crop.
    out = annotate_class_metrics(
        {"mAP": 28.11, "mAP_cls": [54.06, 30.26, 0.0]}, ["crop", "weed"], 1
    )

    assert out["ap_per_class"] == {"crop": 54.06, "weed": 30.26}
    assert out["ap_partial_classes"] == [0.0]
    # The comparable number ignores the class no model can predict.
    assert out["mAP_plants"] == pytest.approx(42.16, abs=0.01)
    assert out["mAP"] == 28.11, "the verbatim upstream value must be preserved"
    assert out["class_names"] == ["crop", "weed", "partial-crop"]
    assert out["images_without_predictions"] == 1


def test_both_partial_classes_are_named():
    out = annotate_class_metrics(
        {"mAP": 0.05, "mAP_cls": [0.2, 0.0, 0.0, 0.0]}, ["crop", "weed"], 306
    )

    assert out["class_names"] == ["crop", "weed", "partial-crop", "partial-weed"]
    assert out["ap_partial_classes"] == [0.0, 0.0]
    assert out["upstream_class_count"] == 4


def test_weed_only_model_reports_the_weed_entry_not_the_average():
    # Real run: tiled fpnlite sc per-tensor. Upstream mAP 10.28 = (0+30.83+0)/3,
    # but the model can only ever predict weed, whose AP is 30.83 -- which is
    # what the pycocotools AP (31.33) is comparable to.
    out = annotate_class_metrics(
        {"mAP": 10.28, "mAP_cls": [0.0, 30.83, 0.0]}, ["weed"], 7
    )

    assert out["ap_per_class"] == {"crop": 0.0, "weed": 30.83}
    assert out["mAP_plants"] == pytest.approx(30.83, abs=0.01)
    assert out["predicted_classes"] == ["weed"]


def test_phantom_classes_are_flagged_on_stderr(capsys):
    annotate_class_metrics(
        {"mAP": 28.11, "mAP_cls": [54.06, 30.26, 0.0]}, ["crop", "weed"], 1
    )

    err = capsys.readouterr().err
    assert "mAP_plants" in err
    assert "no predictions" in err


def test_no_phantom_classes_means_no_warning(capsys):
    annotate_class_metrics(
        {"mAP": 32.95, "mAP_cls": [40.71, 25.2]}, ["crop", "weed"], 0
    )

    assert capsys.readouterr().err == ""


def test_missing_per_class_data_is_tolerated():
    out = annotate_class_metrics({"mAP": 0.0}, ["crop", "weed"], 0)

    assert out["ap_per_class"] == {}
    assert out["mAP_plants"] is None
    assert out["upstream_class_count"] == 0
