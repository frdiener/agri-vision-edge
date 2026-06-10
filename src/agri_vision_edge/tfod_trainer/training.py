"""
Training loop.
"""

from __future__ import annotations

import time

import tensorflow as tf

from object_detection import inputs

from object_detection.model_lib_v2 import (
    eager_train_step,
)

from .state import TrainerState
from .evaluation import (
    evaluate,
    create_eval_dataset,
)
from .utils import (
    current_learning_rate,
    metrics_to_float,
    write_json,
)


def make_train_step(runtime):

    @tf.function
    def train_step(
        detection_model,
        iterator,
    ):
        features, labels = next(iterator)

        losses = eager_train_step(
            detection_model,
            features,
            labels,
            runtime.unpad_groundtruth_tensors,
            runtime.optimizer,
            training_step=runtime.global_step,
            add_regularization_loss=
                runtime.add_regularization_loss,
            clip_gradients_value=
                runtime.clip_gradients_value,
            num_replicas=1,
        )

        runtime.global_step.assign_add(1)

        return losses

    return train_step


def create_train_dataset(
    detection_model,
    configs,
):
    return inputs.train_input(
        train_config=
            configs["train_config"],
        train_input_config=
            configs["train_input_config"],
        model_config=
            configs["model"],
        model=detection_model,
    ).repeat()


def train(
    detection_model,
    runtime,
    trainer_cfg,
):
    state = TrainerState()

    train_ds = create_train_dataset(
        detection_model,
        runtime.configs,
    )

    iterator = iter(train_ds)

    train_steps = (
        runtime.configs["train_config"]
        .num_steps
    )

    for _ in range(
        int(runtime.global_step.numpy()),
        train_steps,
    ):

        start = time.time()

        print("Making trainstep_fn...")
        train_step_fn = make_train_step(runtime)

        print("Running train step...")
        losses = train_step_fn(
            detection_model,
            iterator,
        )
        print("completed train step")

        duration = time.time() - start

        current_step = int(
            runtime.global_step.numpy()
        )

        if (
            current_step
            % trainer_cfg.log_every
            != 0
        ):
            continue

        train_metrics = (
            metrics_to_float(losses)
        )

        train_metrics[
            "learning_rate"
        ] = float(
            current_learning_rate(
                runtime.learning_rate
            )
        )

        train_metrics[
            "steps_per_sec"
        ] = 1.0 / duration

        metrics = evaluate(
            detection_model,
            create_eval_dataset(
                detection_model,
                runtime.configs,
            ),
            runtime,
        )

        metric_value = float(
            metrics[
                trainer_cfg.metric_name
            ]
        )

        if metric_value > state.best_metric:

            state.best_metric = (
                metric_value
            )

            state.patience_counter = 0

            checkpoint_path = (
                runtime.manager.save()
            )

            write_json(
                trainer_cfg.best_metric_path,
                {
                    "step": current_step,
                    "metric":
                        metric_value,
                    "checkpoint":
                        checkpoint_path,
                },
            )

        else:
            state.patience_counter += 1

        if (
            state.patience_counter
            >= trainer_cfg
                .early_stopping_patience
        ):
            break
