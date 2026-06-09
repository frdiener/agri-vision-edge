"""
Evaluation logic.
"""

from __future__ import annotations

import collections

import tensorflow as tf

from object_detection import model_lib
from object_detection import inputs
from object_detection.core import standard_fields as fields

from object_detection.model_lib_v2 import (
    _compute_losses_and_predictions_dicts
        as compute_losses_and_predictions_dicts,
    prepare_eval_dict,
)


def create_eval_dataset(
    detection_model,
    configs,
):
    return inputs.eval_input(
        eval_config=configs["eval_config"],
        eval_input_config=
            configs["eval_input_configs"][0],
        model_config=configs["model"],
        model=detection_model,
    )


def evaluate(
    detection_model,
    eval_dataset,
    runtime,
):
    """
    Run one complete evaluation pass.
    """

    detection_model._is_training = False

    for evaluator in runtime.evaluators:
        evaluator.clear()

    losses = collections.defaultdict(list)

    for features, labels in eval_dataset:

        labels_unstacked = (
            model_lib.unstack_batch(
                labels,
                unpad_groundtruth_tensors=
                    runtime.unpad_groundtruth_tensors,
            )
        )

        losses_dict, prediction_dict = (
            compute_losses_and_predictions_dicts(
                detection_model,
                features,
                labels_unstacked,
                training_step=None,
                add_regularization_loss=
                    runtime.add_regularization_loss,
            )
        )

        prediction_dict = (
            detection_model.postprocess(
                prediction_dict,
                features[
                    fields.InputDataFields
                    .true_image_shape
                ],
            )
        )

        eval_dict, _ = prepare_eval_dict(
            prediction_dict,
            labels,
            features,
        )

        for evaluator in runtime.evaluators:
            evaluator.add_eval_dict(eval_dict)

        for k, v in losses_dict.items():
            losses[k].append(v)

    metrics = {}

    for evaluator in runtime.evaluators:
        metrics.update(
            evaluator.evaluate()
        )

    for name, values in losses.items():
        metrics[name] = (
            tf.reduce_mean(values)
        )

    detection_model._is_training = True

    return metrics
