"""
Training loop.
"""

from __future__ import annotations

import time

import tensorflow as tf
from object_detection import inputs
from object_detection.model_lib_v2 import (
    eager_eval_loop,
    eager_train_step,
)

from agri_vision_edge.tfod_trainer.setup import (
    apply_graph_modifications,
    restore_weights,
)

from .evaluation import (
    create_eval_dataset,
    evaluate,
)
from .state import TrainerState
from .utils import (
    current_learning_rate,
    metrics_to_float,
    write_json,
)


def make_train_step(runtime, detection_model):

    # `detection_model` is captured by closure (not passed as a tf.function
    # argument); this mirrors object_detection.model_lib_v2.train_loop. Passing
    # a Keras model as a tf.function argument is a known footgun for stateful
    # side-effects (e.g. BatchNorm moving-statistic updates).
    @tf.function
    def train_step(
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


def run_evaluation(detection_model, runtime):
    """
    Run one evaluation pass, transparently swapping EMA weights in/out.

    With EMA enabled the moving-average weights are evaluated (swapped in, then
    the raw training weights swapped back), mirroring
    ``object_detection.model_lib_v2.train_loop``.
    """
    if runtime.use_moving_average:
        runtime.optimizer.swap_weights()

    metrics = evaluate(
        detection_model,
        create_eval_dataset(
            detection_model,
            runtime.configs,
        ),
        runtime,
    )

    if runtime.use_moving_average:
        runtime.optimizer.swap_weights()

    return metrics


def save_best_checkpoint(
    runtime,
    trainer_cfg,
    state,
    current_step,
    metric_value,
):
    """
    Record ``metric_value`` as the new best, checkpoint the weights and write
    ``best_metric.json``.

    Tracks the true strict maximum only -- it does NOT touch the early-stopping
    or plateau stall counters, which advance on their own delta-gated references
    (see ``train``). This decoupling lets the export keep the genuine best while
    a jittery-but-slightly-improving run can still trigger LR drops / early stop.
    """
    state.best_metric = metric_value

    checkpoint_path = runtime.manager.save()

    write_json(
        trainer_cfg.best_metric_path,
        {
            "step": current_step,
            "metric_name": trainer_cfg.metric_name,
            "metric_value": metric_value,
            "checkpoint": checkpoint_path,
        },
    )

    return checkpoint_path


def apply_lr_warmup(runtime, step):
    """
    Linearly ramp ``runtime.lr_var`` from the warmup LR to the base LR over the
    first ``lr_warmup_steps`` steps, then leave it at the base value.

    No-op unless the plateau schedule is active (``lr_var`` is set) and a warmup
    is configured. Once warmup is done the plateau logic (in ``train``) owns the
    LR, so this only touches ``lr_var`` for ``step <= lr_warmup_steps``.
    """
    if runtime.lr_var is None or runtime.lr_warmup_steps <= 0:
        return

    if step > runtime.lr_warmup_steps:
        return

    frac = min(step / float(runtime.lr_warmup_steps), 1.0)
    warmed = runtime.lr_warmup + (runtime.lr_base - runtime.lr_warmup) * frac
    runtime.lr_var.assign(warmed)


def maybe_reduce_lr_on_plateau(
    detection_model,
    runtime,
    trainer_cfg,
    state,
    current_step,
):
    """
    ReduceLROnPlateau step, called on every *non-improving* evaluation.

    Counts consecutive non-improving evals; once the count reaches
    ``lr_plateau_patience`` (and no cooldown is active) it multiplies ``lr_var``
    by ``lr_plateau_factor`` (floored at ``lr_plateau_min_lr``) and opens a
    ``lr_plateau_cooldown`` grace window. When ``lr_plateau_restore_best`` is set
    the best checkpoint is restored first (warm restart: best weights + optimizer
    slots, current step count preserved) before the lower LR is applied.

    No-op unless the plateau schedule is active. Independent of the
    early-stopping patience counter, so LR drops happen before early stopping.

    Returns:
        bool: True when the LR schedule is exhausted -- i.e. the plateau logic
        has hit the ``lr_plateau_min_lr`` floor for ``lr_plateau_exhausted_patience``
        stalls and the caller should stop training.
    """
    if not trainer_cfg.lr_plateau or runtime.lr_var is None:
        return False

    # Still warming up: don't count plateaus against the ramp.
    if current_step <= runtime.lr_warmup_steps:
        return False

    if state.cooldown_counter > 0:
        state.cooldown_counter -= 1
        state.plateau_counter = 0
        return False

    state.plateau_counter += 1

    if state.plateau_counter < trainer_cfg.lr_plateau_patience:
        return False

    old_lr = float(runtime.lr_var.numpy())
    new_lr = max(
        old_lr * trainer_cfg.lr_plateau_factor,
        trainer_cfg.lr_plateau_min_lr,
    )

    # Reset the counter and open the cooldown window regardless of whether we
    # can still reduce (so we don't re-trigger every eval at the LR floor).
    state.plateau_counter = 0
    state.cooldown_counter = trainer_cfg.lr_plateau_cooldown

    if new_lr >= old_lr:
        # Floored stall: the LR is already at the min and further reductions
        # cannot help. Count it; once we have exhausted our patience the run has
        # nothing left to try, so signal the caller to stop.
        state.lr_floored = True
        state.min_lr_stall_counter += 1
        print(
            f"LR plateau: already at min_lr ({old_lr:.3e}); floored stall "
            f"{state.min_lr_stall_counter}/{trainer_cfg.lr_plateau_exhausted_patience} "
            f"(step {current_step})."
        )
        return (
            trainer_cfg.lr_plateau_exhausted_patience > 0
            and state.min_lr_stall_counter
            >= trainer_cfg.lr_plateau_exhausted_patience
        )

    # Warm restart: rewind to the best weights/optimizer state before dropping
    # the LR. Restoring the checkpoint also rewinds global_step, so save and
    # re-assign it to keep the training horizon intact.
    if trainer_cfg.lr_plateau_restore_best and runtime.manager.latest_checkpoint:
        best_path = runtime.manager.latest_checkpoint
        saved_step = int(runtime.global_step.numpy())
        runtime.ckpt.restore(best_path).expect_partial()
        runtime.global_step.assign(saved_step)
        print(
            f"LR plateau: restored best checkpoint ({best_path}) for warm "
            f"restart."
        )

    # Assign AFTER the restore, so a checkpoint-restored LR can't clobber it.
    runtime.lr_var.assign(new_lr)

    # Latch the floored flag if this reduction landed exactly on the min LR.
    if new_lr <= trainer_cfg.lr_plateau_min_lr:
        state.lr_floored = True

    print(
        f"LR plateau: reduced learning rate {old_lr:.3e} -> {new_lr:.3e} "
        f"at step {current_step} (cooldown {trainer_cfg.lr_plateau_cooldown})."
    )
    return False


def assert_finite_model(detection_model, step):
    """
    Abort training if any model weight is non-finite.

    A BatchNorm moving_variance can overflow to NaN/Inf on a transient
    activation spike without the (batch-statistic) training loss ever showing
    it -- but eval and the exported SavedModel use the moving statistics, so a
    single poisoned BN silently turns the model to garbage. Fail loudly here,
    before a corrupted checkpoint is saved or exported, rather than shipping a
    broken `ptq/`.
    """
    # Only float variables support tf.math.is_finite; skip int counters etc.
    bad = [
        v.name
        for v in detection_model.variables
        if v.dtype.is_floating
        and not bool(tf.reduce_all(tf.math.is_finite(v)))
    ]
    if bad:
        raise FloatingPointError(
            f"Non-finite values in {len(bad)} model variable(s) at step "
            f"{step} (e.g. {bad[:3]}). Training diverged; aborting before a "
            f"corrupted checkpoint is saved. Lower learning_rate_base or "
            f"gradient_clipping_by_norm."
        )


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

    # Restore weights before training (resume, fine-tune, and EMA setup).
    restore_weights(
        detection_model,
        runtime,
        train_ds,
    )

    # Apply optimizer reset / BN folding / backbone QAT before the train
    # step is traced, so the modified optimizer and backbone are captured.
    graph_modified = apply_graph_modifications(
        detection_model,
        runtime,
        trainer_cfg,
        train_ds,
    )

    # When we are going to seed the best-metric tracker with a scored baseline
    # eval below, skip this purely-informational (unscored) eval to avoid
    # evaluating the same initial weights twice.
    if graph_modified and not trainer_cfg.initial_eval_checkpoint:
        print(
            "\nEvaluating initial modified "
            "configuration..."
        )
        eval_input = inputs.eval_input(
            eval_config=runtime.configs['eval_config'],
            eval_input_config=runtime.configs['eval_input_config'],
            model_config=runtime.configs['model'],
            model=detection_model,
        )
        eager_eval_loop(
            detection_model,
            runtime.configs,
            eval_input,
            use_tpu=False,
            global_step=runtime.global_step,
        )

    # Seed the best-metric tracker with the restored (baseline) weights, and
    # checkpoint them, before the first train step. This guarantees the exported
    # "best" checkpoint is never worse than the starting point: a reduced
    # schedule that only ever regresses (e.g. a PTQ float base resuming an
    # already-converged finetune) will export the baseline itself.
    if trainer_cfg.initial_eval_checkpoint:
        current_step = int(runtime.global_step.numpy())

        print(
            "\nEvaluating initial weights to seed the best-metric baseline..."
        )

        metrics = run_evaluation(detection_model, runtime)

        record = {"step": current_step}
        record.update(metrics_to_float(metrics))
        state.metrics_history.append(record)

        if trainer_cfg.save_metrics_history:
            write_json(
                trainer_cfg.history_path,
                state.metrics_history,
            )

        metric_value = float(
            metrics[trainer_cfg.metric_name]
        )

        save_best_checkpoint(
            runtime,
            trainer_cfg,
            state,
            current_step,
            metric_value,
        )

        # Seed the decoupled stall references to the baseline too, so the first
        # training evals are measured against it (an initial dip below baseline
        # is then correctly a non-improvement rather than a reset from -inf).
        state.es_ref = metric_value
        state.plateau_ref = metric_value

        print(
            f"Initial baseline {trainer_cfg.metric_name}={metric_value}; "
            "checkpoint saved. Training must beat it to overwrite."
        )

    iterator = iter(train_ds)

    train_steps = (
        runtime.configs["train_config"]
        .num_steps
    )

    print("Making trainstep_fn...")
    train_step_fn = make_train_step(runtime, detection_model)

    for _ in range(
        int(runtime.global_step.numpy()),
        train_steps,
    ):

        start = time.time()

        losses = train_step_fn(
            iterator,
        )

        duration = time.time() - start

        current_step = int(
            runtime.global_step.numpy()
        )

        # Warmup ramp for the plateau schedule (no-op otherwise). Runs every
        # step so the LR variable tracks the ramp before plateau reductions.
        apply_lr_warmup(runtime, current_step)

        if (
            current_step
            % trainer_cfg.log_every
            != 0
        ):
            continue

        # Catch a diverged BatchNorm (NaN/Inf moving stats) as soon as it
        # appears, before it is evaluated, checkpointed or exported.
        assert_finite_model(detection_model, current_step)

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

        # Learning rate at scientific precision so plateau reductions down to
        # `lr_plateau_min_lr` (e.g. 1e-6) stay legible; other scalars at 4dp.
        metric_parts = [
            f"{k}={v:.3e}" if k == "learning_rate" else f"{k}={v:.4f}"
            for k, v in train_metrics.items()
        ]

        # Scheduler / tracker state as of the last completed eval: the running
        # best, the early-stopping patience, and -- when the plateau schedule is
        # active -- its stall counter and cooldown window.
        best_tag = trainer_cfg.metric_name.split("/")[-1]
        best_str = (
            "n/a"
            if state.best_metric == float("-inf")
            else f"{state.best_metric:.4f}"
        )
        sched_parts = [
            f"best_{best_tag}={best_str}",
            f"patience={state.patience_counter}"
            f"/{trainer_cfg.early_stopping_patience}",
        ]
        if trainer_cfg.lr_plateau:
            sched_parts.append(
                f"plateau={state.plateau_counter}"
                f"/{trainer_cfg.lr_plateau_patience}"
            )
            sched_parts.append(f"cooldown={state.cooldown_counter}")
            if state.lr_floored:
                sched_parts.append(
                    f"min_lr_stall={state.min_lr_stall_counter}"
                    f"/{trainer_cfg.lr_plateau_exhausted_patience}"
                )

        print(
            f"Step {current_step}: "
            + " | ".join(metric_parts + sched_parts)
        )

        # Evaluate (swapping EMA weights in/out when enabled, matching
        # object_detection.model_lib_v2.train_loop).
        metrics = run_evaluation(detection_model, runtime)

        # Record this step's metrics for later plotting. Eval metrics go in
        # first so the training Loss/* values (and LR / throughput) from
        # train_metrics shadow the eval losses under the same keys, matching
        # what agri_vision_edge.evaluation.curves expects.
        record = {"step": current_step}
        record.update(metrics_to_float(metrics))
        record.update(train_metrics)
        state.metrics_history.append(record)

        if trainer_cfg.save_metrics_history:
            write_json(
                trainer_cfg.history_path,
                state.metrics_history,
            )

        metric_value = float(
            metrics[
                trainer_cfg.metric_name
            ]
        )

        # 1) Checkpointing tracks the TRUE strict best, so the export never
        #    misses a genuinely better model -- even a microscopic gain.
        if metric_value > state.best_metric:
            print(
                f"New best {trainer_cfg.metric_name}: {metric_value:.5f} "
                f"(prev {state.best_metric:.5f}); saving checkpoint..."
            )
            save_best_checkpoint(
                runtime,
                trainer_cfg,
                state,
                current_step,
                metric_value,
            )

        # 2) Early-stopping counter, delta-gated against its own reference so
        #    sub-noise improvements don't keep the run alive indefinitely.
        if (
            metric_value
            > state.es_ref + trainer_cfg.early_stopping_min_delta
        ):
            state.es_ref = metric_value
            state.patience_counter = 0
        else:
            state.patience_counter += 1
            print(
                f"No early-stop improvement (> {trainer_cfg.early_stopping_min_delta}) "
                f"over {state.es_ref:.5f} | patience "
                f"{state.patience_counter}/{trainer_cfg.early_stopping_patience}"
            )

        # 3) Plateau counter, delta-gated against its own reference; on a stall
        #    it drives the LR reduction (no-op unless lr_plateau is enabled).
        #    A True return means the LR schedule is exhausted (floored for
        #    `lr_plateau_exhausted_patience` stalls) -> stop.
        if trainer_cfg.lr_plateau:
            if (
                metric_value
                > state.plateau_ref + trainer_cfg.lr_plateau_min_delta
            ):
                state.plateau_ref = metric_value
                state.plateau_counter = 0
                state.min_lr_stall_counter = 0
            else:
                lr_exhausted = maybe_reduce_lr_on_plateau(
                    detection_model,
                    runtime,
                    trainer_cfg,
                    state,
                    current_step,
                )
                if lr_exhausted:
                    print(
                        f"Stopping at step {current_step}: LR schedule exhausted "
                        f"(at min_lr {trainer_cfg.lr_plateau_min_lr:.3e} with no "
                        f"improvement for {state.min_lr_stall_counter} floored "
                        f"stalls)."
                    )
                    break

        if (
            state.patience_counter
            >= trainer_cfg
                .early_stopping_patience
        ):
            print(
                f"Stopping at step {current_step}: early-stopping patience "
                f"{state.patience_counter}/{trainer_cfg.early_stopping_patience} "
                f"reached."
            )
            break
