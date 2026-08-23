"""
Training loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

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


def ensure_optimizer_state_created(runtime, detection_model):
    """Create optimizer/EMA slot variables without taking a training step.

    Keras optimizers generally create momentum/variance slots lazily on their
    first ``apply_gradients`` call.  A checkpoint written before that point can
    restore model weights but cannot rewind optimizer state, which turns a
    later plateau restart into a model-only rewind with stale momentum.

    This must run after graph modifications, because BN folding or QAT may
    replace the model's trainable variables.  ``_create_all_weights`` is the
    TensorFlow 2.12/legacy-optimizer path used by TFOD and also lets optimizer
    wrappers such as MovingAverage create their shadow/slot variables.  The
    public ``build`` path covers newer Keras optimizers.
    """
    variables = list(detection_model.trainable_variables)
    if not variables:
        raise RuntimeError(
            "Cannot initialize optimizer state: the detection model has no "
            "trainable variables. Ensure the model is built first."
        )

    optimizer = runtime.optimizer

    create_all_weights = getattr(optimizer, "_create_all_weights", None)
    if callable(create_all_weights):
        create_all_weights(variables)
    else:
        build = getattr(optimizer, "build", None)
        if callable(build):
            build(variables)
        else:
            create_slots = getattr(optimizer, "_create_slots", None)
            if not callable(create_slots):
                raise RuntimeError(
                    "Cannot initialize optimizer state for "
                    f"{type(optimizer).__name__}: no supported build/slot "
                    "creation API was found."
                )
            create_slots(variables)

    optimizer_variables = getattr(optimizer, "variables", None)
    if callable(optimizer_variables):
        optimizer_variables = optimizer_variables()
    elif optimizer_variables is None:
        optimizer_variables = getattr(optimizer, "weights", ())

    print(
        "Initialized optimizer checkpoint state: "
        f"{len(tuple(optimizer_variables))} optimizer/EMA variable(s) for "
        f"{len(variables)} trainable model variable(s)."
    )


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
    state.best_checkpoint_path = checkpoint_path

    write_json(
        trainer_cfg.best_metric_path,
        {
            "step": current_step,
            "metric_name": trainer_cfg.control.metric_name,
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
    metric_value,
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
    if not trainer_cfg.control.lr_plateau or runtime.lr_var is None:
        return False

    # Still warming up: don't count plateaus against the ramp.
    if current_step <= runtime.lr_warmup_steps:
        return False

    # Only anneal once the metric has cleared 0 + tolerance (reusing the plateau
    # min_delta as the tolerance). Early on the model detects nothing and the
    # metric sits at ~0; "no improvement" there is meaningless, so annealing
    # would needlessly kill the LR before the model gets going.
    if metric_value <= trainer_cfg.control.lr_plateau_min_delta:
        return False

    if state.cooldown_counter > 0:
        state.cooldown_counter -= 1
        state.plateau_counter = 0
        return False

    state.plateau_counter += 1

    if state.plateau_counter < trainer_cfg.control.lr_plateau_patience:
        return False

    old_lr = float(runtime.lr_var.numpy())
    new_lr = max(
        old_lr * trainer_cfg.control.lr_plateau_factor,
        trainer_cfg.control.lr_plateau_min_lr,
    )

    # Reset the counter and open the cooldown window regardless of whether we
    # can still reduce (so we don't re-trigger every eval at the LR floor).
    state.plateau_counter = 0
    state.cooldown_counter = trainer_cfg.control.lr_plateau_cooldown

    if new_lr >= old_lr:
        # Floored stall: the LR is already at the min and further reductions
        # cannot help. Count it; once we have exhausted our patience the run has
        # nothing left to try, so signal the caller to stop.
        state.lr_floored = True
        state.min_lr_stall_counter += 1
        print(
            f"LR plateau: already at min_lr ({old_lr:.3e}); floored stall "
            f"{state.min_lr_stall_counter}/{trainer_cfg.control.lr_plateau_exhausted_patience} "
            f"(step {current_step})."
        )
        return (
            trainer_cfg.control.lr_plateau_exhausted_patience > 0
            and state.min_lr_stall_counter
            >= trainer_cfg.control.lr_plateau_exhausted_patience
        )

    # Warm restart: rewind to the exact best weights and optimizer/EMA state
    # before dropping the LR. Restoring also rewinds global_step, so preserve the current training
    # horizon explicitly.
    best_path = getattr(state, "best_checkpoint_path", None)
    if trainer_cfg.control.lr_plateau_restore_best and best_path:
        saved_step = int(runtime.global_step.numpy())

        restore_status = runtime.ckpt.restore(best_path)
        restore_status.assert_existing_objects_matched()

        runtime.global_step.assign(saved_step)
        print(
            f"LR plateau: restored best checkpoint ({best_path}) for warm "
            f"restart; preserved global_step={saved_step}."
        )
    elif trainer_cfg.control.lr_plateau_restore_best:
        print(
            "LR plateau: warm restart requested, but no best checkpoint has "
            "been recorded; reducing the LR without restoring."
        )

    # Assign AFTER the restore, so a checkpoint-restored LR can't clobber it.
    runtime.lr_var.assign(new_lr)

    # Latch the floored flag if this reduction landed exactly on the min LR.
    if new_lr <= trainer_cfg.control.lr_plateau_min_lr:
        state.lr_floored = True

    print(
        f"LR plateau: reduced learning rate {old_lr:.3e} -> {new_lr:.3e} "
        f"at step {current_step} (cooldown {trainer_cfg.control.lr_plateau_cooldown})."
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


@dataclass
class TrainOutcome:
    """
    Why the training loop returned.

    ``budget_exhausted`` is the one the caller must not ignore: it means the run
    stopped on the clock rather than on a stopping rule, so the model is
    *unconverged* and the train dir has to be carried into another session. Every
    other outcome (a metric-driven stop, or reaching the step horizon) means the
    run is done and the train dir is disposable.
    """

    stop_reason: str | None = None
    budget_exhausted: bool = False
    final_step: int = 0

    @property
    def converged(self) -> bool:
        return not self.budget_exhausted


def train(
    detection_model,
    runtime,
    trainer_cfg,
    steps_per_epoch=None,
):
    """
    Run the custom training loop.

    ``steps_per_epoch`` (``ceil(train_samples / batch_size)``, supplied by
    ``run_finetune`` from the bundle metadata) turns evaluation, checkpointing
    and the early-stopping / plateau bookkeeping onto an epoch cadence
    (``control.eval_every_epochs``) and enables the optional ``control.max_epochs``
    cap. When it is ``None`` (sample count unavailable) the loop falls back to the
    legacy step cadence (``control.log_every``).
    """
    state = TrainerState()

    train_ds = create_train_dataset(
        detection_model,
        runtime.configs,
    )

    # Restore weights before training (resume, fine-tune, and EMA setup).
    resumed = restore_weights(
        detection_model,
        runtime,
        train_ds,
    )

    # The checkpoint restored weights, optimizer slots and the global step; the
    # trainer's own bookkeeping lives outside it and has to be reloaded
    # explicitly, or the run would restart its schedule from scratch (best
    # metric back to -inf, plateau patience back to zero, curves truncated to
    # this session). Only trusted on an actual resume: a state file left in a
    # train dir whose checkpoints are gone describes weights we no longer have.
    if resumed and trainer_cfg.state_path.is_file():
        state = TrainerState.load(
            trainer_cfg.state_path,
            train_dir=trainer_cfg.train_dir,
            history_path=trainer_cfg.history_path,
        )
        print(
            f"Resumed trainer state: best={state.best_metric:.5f} "
            f"plateau={state.plateau_counter}/{trainer_cfg.control.lr_plateau_patience} "
            f"cooldown={state.cooldown_counter} "
            f"floored_stalls={state.min_lr_stall_counter} "
            f"history={len(state.metrics_history)} record(s)."
        )
    elif resumed:
        print(
            "Resumed from a checkpoint with no trainer_state.json alongside it: "
            "schedule bookkeeping restarts from scratch (pre-resume evals are "
            "not in the history and the best-metric tracker is re-seeded)."
        )

    # Apply optimizer reset / BN folding / backbone QAT before the train
    # step is traced, so the modified optimizer and backbone are captured.
    graph_modified = apply_graph_modifications(
        detection_model,
        runtime,
        trainer_cfg,
        train_ds,
    )

    # Materialize optimizer slots and (when enabled) EMA shadow variables before
    # any baseline checkpoint is written. Otherwise an initial checkpoint can
    # only rewind model weights while leaving later optimizer momentum intact.
    ensure_optimizer_state_created(runtime, detection_model)

    # When we are going to seed the best-metric tracker with a scored baseline
    # eval below, skip this purely-informational (unscored) eval to avoid
    # evaluating the same initial weights twice.
    if graph_modified and not trainer_cfg.control.initial_eval_checkpoint:
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
    # Skipped on a resume: the tracker is already seeded from the restored
    # state, and re-seeding it here would overwrite a genuinely better earlier
    # best with whatever the mid-run weights happen to score.
    if trainer_cfg.control.initial_eval_checkpoint and not resumed:
        current_step = int(runtime.global_step.numpy())

        print(
            "\nEvaluating initial weights to seed the best-metric baseline..."
        )

        metrics = run_evaluation(detection_model, runtime)

        record = {"step": current_step}
        if steps_per_epoch and steps_per_epoch > 0:
            record["epoch"] = current_step / steps_per_epoch
        record.update(metrics_to_float(metrics))
        state.metrics_history.append(record)

        if trainer_cfg.control.save_metrics_history:
            write_json(
                trainer_cfg.history_path,
                state.metrics_history,
            )

        metric_value = float(
            metrics[trainer_cfg.control.metric_name]
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
            f"Initial baseline {trainer_cfg.control.metric_name}={metric_value}; "
            "checkpoint saved. Training must beat it to overwrite."
        )

    iterator = iter(train_ds)

    control = trainer_cfg.control

    num_steps = int(
        runtime.configs["train_config"].num_steps
    )

    # Resolve the evaluation cadence and training horizon. With a known
    # steps_per_epoch we run on an epoch cadence (eval every `eval_every_epochs`
    # epochs) and honour the optional `max_epochs` cap; otherwise we fall back to
    # the legacy step cadence (`log_every`).
    if steps_per_epoch and steps_per_epoch > 0:
        eval_interval = max(
            1, round(control.eval_every_epochs * steps_per_epoch)
        )
        # `max_epochs`, when set, OVERRIDES the pipeline's num_steps entirely:
        # the horizon becomes exactly that many epochs, so num_steps can stay at
        # its large default and be ignored. Without it, num_steps is the horizon.
        if control.max_epochs is not None:
            train_steps = control.max_epochs * steps_per_epoch
        else:
            train_steps = num_steps
        geometry = (
            f"Epoch geometry: steps_per_epoch={steps_per_epoch}; evaluating "
            f"every {control.eval_every_epochs} epoch(s) (= {eval_interval} "
            f"steps); training to step {train_steps} "
            f"(~{train_steps / steps_per_epoch:.2f} epochs)"
        )
        if control.max_epochs is not None:
            geometry += f"; horizon set by max_epochs={control.max_epochs}"
        print(geometry + ".")
    else:
        eval_interval = control.log_every
        train_steps = num_steps
        print(
            "Epoch geometry unavailable (no train_samples in bundle metadata); "
            f"falling back to eval every {eval_interval} steps."
        )

    # Wall-clock budget, measured from here so it covers training and the
    # evaluations interleaved with it -- at high resolution the evals are the
    # larger half, and a budget that ignored them would overshoot badly.
    wall_clock_start = time.time()
    runtime_budget_seconds = (
        control.max_runtime_hours * 3600
        if control.max_runtime_hours is not None
        else None
    )
    if runtime_budget_seconds is not None:
        print(
            f"Wall-clock budget: {control.max_runtime_hours:.2f} h "
            "(checked at evaluation boundaries; the run stops gracefully so it "
            "can export and be resumed)."
        )

    outcome = TrainOutcome()

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

        # Evaluate on the (epoch or step) cadence, and always on the final step
        # so the last -- possibly partial -- epoch is scored and checkpointed.
        is_final_step = current_step >= train_steps
        if current_step % eval_interval != 0 and not is_final_step:
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
        best_tag = trainer_cfg.control.metric_name.split("/")[-1]
        best_str = (
            "n/a"
            if state.best_metric == float("-inf")
            else f"{state.best_metric:.4f}"
        )
        # early_stopping_patience == 0 means the stop is disabled; the counter
        # is still tracked, so show its limit as "off" rather than "/0".
        es_patience = trainer_cfg.control.early_stopping_patience
        es_limit = es_patience if es_patience else "off"
        sched_parts = [
            f"best_{best_tag}={best_str}",
            f"patience={state.patience_counter}/{es_limit}",
        ]
        if trainer_cfg.control.lr_plateau:
            sched_parts.append(
                f"plateau={state.plateau_counter}"
                f"/{trainer_cfg.control.lr_plateau_patience}"
            )
            sched_parts.append(f"cooldown={state.cooldown_counter}")
            if state.lr_floored:
                sched_parts.append(
                    f"min_lr_stall={state.min_lr_stall_counter}"
                    f"/{trainer_cfg.control.lr_plateau_exhausted_patience}"
                )

        step_label = f"Step {current_step}"
        if steps_per_epoch and steps_per_epoch > 0:
            step_label += f" (epoch {current_step / steps_per_epoch:.2f})"
        print(
            step_label + ": "
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
        if steps_per_epoch and steps_per_epoch > 0:
            record["epoch"] = current_step / steps_per_epoch
        record.update(metrics_to_float(metrics))
        record.update(train_metrics)
        state.metrics_history.append(record)

        if trainer_cfg.control.save_metrics_history:
            write_json(
                trainer_cfg.history_path,
                state.metrics_history,
            )

        metric_value = float(
            metrics[
                trainer_cfg.control.metric_name
            ]
        )

        # 1) Checkpointing tracks the TRUE strict best, so the export never
        #    misses a genuinely better model -- even a microscopic gain.
        if metric_value > state.best_metric:
            print(
                f"New best {trainer_cfg.control.metric_name}: {metric_value:.5f} "
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
            > state.es_ref + trainer_cfg.control.early_stopping_min_delta
        ):
            state.es_ref = metric_value
            state.patience_counter = 0
        else:
            state.patience_counter += 1
            es_patience = trainer_cfg.control.early_stopping_patience
            es_limit = es_patience if es_patience else "off"
            print(
                f"No early-stop improvement (> {trainer_cfg.control.early_stopping_min_delta}) "
                f"over {state.es_ref:.5f} | patience "
                f"{state.patience_counter}/{es_limit}"
            )

        # The stop conditions below record their reason rather than breaking
        # immediately, so the trainer state can be persisted once, after the
        # counters settle, on every path out of the eval block.
        stop_reason = None

        # 0) Wall-clock budget. Checked first and independently of the metric:
        #    this is not a statement about convergence but about the session
        #    being about to end, and stopping here is what lets the run export
        #    and be resumed rather than be killed mid-step.
        if runtime_budget_seconds is not None:
            elapsed = time.time() - wall_clock_start
            if elapsed >= runtime_budget_seconds:
                stop_reason = (
                    f"Stopping at step {current_step}: wall-clock budget spent "
                    f"({elapsed / 3600:.2f} h of "
                    f"{runtime_budget_seconds / 3600:.2f} h). The run has NOT "
                    "converged -- resume it in a new session with this train "
                    "dir attached."
                )
                outcome.budget_exhausted = True

        # 3) Plateau counter, delta-gated against its own reference; on a stall
        #    it drives the LR reduction (no-op unless lr_plateau is enabled).
        #    A True return means the LR schedule is exhausted (floored for
        #    `lr_plateau_exhausted_patience` stalls) -> stop.
        #    Skipped once we are stopping on the clock: a plateau trigger can
        #    warm-restart to the best checkpoint, and rewinding the weights in
        #    the same breath as snapshotting them for resume would hand the next
        #    session an older model than the one it just trained.
        if not stop_reason and trainer_cfg.control.lr_plateau:
            if (
                metric_value
                > state.plateau_ref + trainer_cfg.control.lr_plateau_min_delta
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
                    metric_value,
                )
                if lr_exhausted:
                    stop_reason = (
                        f"Stopping at step {current_step}: LR schedule exhausted "
                        f"(at min_lr {trainer_cfg.control.lr_plateau_min_lr:.3e} with no "
                        f"improvement for {state.min_lr_stall_counter} floored "
                        f"stalls)."
                    )

        # early_stopping_patience == 0 disables the stop entirely (the counter
        # above is still advanced + logged for diagnostics); the LR-plateau
        # schedule then owns termination.
        if (
            not stop_reason
            and trainer_cfg.control.early_stopping_patience
            and state.patience_counter
            >= trainer_cfg.control.early_stopping_patience
        ):
            stop_reason = (
                f"Stopping at step {current_step}: early-stopping patience "
                f"{state.patience_counter}/{trainer_cfg.control.early_stopping_patience} "
                f"reached."
            )

        # Persist the resume point now that this eval's counters have settled
        # (and after any LR reduction), so both artifacts describe a *completed*
        # evaluation. The checkpoint carries weights/optimizer/step, the state
        # file carries everything else; restoring one without the other is what
        # makes a resumed run diverge from an uninterrupted one. Deliberately
        # written before the stop, so a graceful end is also recorded.
        runtime.last_manager.save()
        state.save(trainer_cfg.state_path)

        if stop_reason:
            print(stop_reason)
            outcome.stop_reason = stop_reason
            outcome.final_step = current_step
            break

    outcome.final_step = int(runtime.global_step.numpy())
    return outcome
