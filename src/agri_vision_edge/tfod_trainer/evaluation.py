"""
Evaluation logic.
"""

from __future__ import annotations

import collections

import tensorflow as tf
from object_detection import inputs, model_lib
from object_detection.core import standard_fields as fields
from object_detection.model_lib_v2 import (
    _compute_losses_and_predictions_dicts as compute_losses_and_predictions_dicts,
)
from object_detection.model_lib_v2 import (
    prepare_eval_dict,
)


def create_eval_dataset(
    detection_model,
    configs,
    cache=True,
):
    """
    Build the TFOD evaluation dataset.

    When cache=True the fully preprocessed dataset is cached after
    the first pass, which is beneficial when evaluating many
    checkpoints on the same validation set.
    """

    dataset = inputs.eval_input(
        eval_config=configs["eval_config"],
        eval_input_config=
            configs["eval_input_configs"][0],
        model_config=configs["model"],
        model=detection_model,
    )

    if cache:
        dataset = dataset.cache()

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def materialize_eval_dataset(
    eval_dataset,
):
    """
    Convert the dataset into an in-memory list.

    Useful when repeatedly evaluating many checkpoints against the
    same validation set.

    Returns:
        list[(features, labels)]
    """

    return list(eval_dataset)


@tf.function(
    reduce_retracing=True,
)
def _eval_step(
    detection_model,
    features,
    labels_unstacked,
    add_regularization_loss,
):
    """
    Compiled evaluation step.
    """

    losses_dict, prediction_dict = (
        compute_losses_and_predictions_dicts(
            detection_model,
            features,
            labels_unstacked,
            training_step=None,
            add_regularization_loss=
                add_regularization_loss,
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

    return (
        losses_dict,
        prediction_dict,
    )


def evaluate(
    detection_model,
    eval_dataset,
    runtime,
):
    """
    Run one complete evaluation pass.

    eval_dataset may be either:
        - tf.data.Dataset
        - materialized list returned by
          materialize_eval_dataset()
    """

    # Put BatchNorm into inference mode for eval. Setting `_is_training` alone
    # is NOT enough: object_detection's FreezableBatchNorm reads the *global
    # Keras learning phase* at call time for non-frozen layers, and
    # eager_train_step leaves it set to True. Without resetting it, eval runs
    # BatchNorm in training mode and updates its moving statistics from the
    # (unaugmented, full-frame) eval batches -- which overflows the smallest
    # feature map's moving_variance to NaN and corrupts the model. Mirrors
    # object_detection.model_lib_v2.eager_eval_loop.
    detection_model._is_training = False
    tf.keras.backend.set_learning_phase(False)

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
            _eval_step(
                detection_model,
                features,
                labels_unstacked,
                runtime.add_regularization_loss,
            )
        )

        eval_dict, _ = prepare_eval_dict(
            prediction_dict,
            labels,
            features,
        )

        for evaluator in runtime.evaluators:
            evaluator.add_eval_dict(
                eval_dict
            )

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

    # Restore training mode for the subsequent train steps (eager_train_step
    # would set it too, but keep train/eval state symmetric).
    detection_model._is_training = True
    tf.keras.backend.set_learning_phase(True)

    return metrics
