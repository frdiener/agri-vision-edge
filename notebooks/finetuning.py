import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Finetune / QAT")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Heavy imports. setup_tensorflow_models() must run before anything pulls
    # in `object_detection`, so do it here at the top.
    import json
    import sys

    from agri_vision_edge.third_party import setup_tensorflow_models

    setup_tensorflow_models()

    from agri_vision_edge.experiment import FineTuneConfig
    from agri_vision_edge.tfod_trainer import (
        FinetuneRunConfig,
        export_run,
        run_finetune,
    )

    return (
        FineTuneConfig,
        FinetuneRunConfig,
        export_run,
        json,
        sys,
        run_finetune,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Finetune / QAT

    Configure a run below and press **Train**, or drive the exact same
    logic head-less from Python:

    ```python
    from agri_vision_edge.tfod_trainer import FinetuneRunConfig, run_finetune

    run_finetune(FinetuneRunConfig(
        model_path="models/ssd_mobilenet_v2_320x320_coco17_tpu-8",
        dataset_bundle_path="datasets/phenobench_sc_tiled",
        num_classes=1,
        output_dir="runs/finetune",
        qat=True,   # omit for a plain finetune (-> PTQ at conversion)
    ))
    ```

    To run *this notebook* against a preset config (e.g. as a script), pass the
    training config as an argument. This bypasses the form and the Train button.
    """)
    return


@app.cell
def _(json, sys):
    # Set to a FinetuneRunConfig or dict to bypass the UI; None = interactive.
    OVERRIDE = None
    try:
        if len(sys.argv) > 1:
            OVERRIDE = json.loads(sys.argv[1])
    except Exception:
        pass
    return (OVERRIDE,)


@app.cell(hide_code=True)
def _(mo):
    form = (
        mo.md(
            """
            ## Run configuration

            **Inputs**

            - {model_path}
            - {dataset_bundle_path}
            - {num_classes}
            - {output_dir}

            **Training**

            - {batch_size} {num_steps}
            - {learning_rate_base} {image_size}
            - {early_stopping_patience}

            **Quantization-aware training** (leave off for a plain finetune)

            - {qat} {qat_per_channel}
            - {reset_optimizer}
            """
        )
        .batch(
            model_path=mo.ui.text(
                value="models/ssd_mobilenet_v2_320x320_coco17_tpu-8",
                label="Model path",
                full_width=True,
            ),
            dataset_bundle_path=mo.ui.text(
                value="datasets/phenobench_sc_tiled",
                label="Dataset bundle",
                full_width=True,
            ),
            num_classes=mo.ui.number(value=1, start=1, stop=1000, label="Classes"),
            output_dir=mo.ui.text(
                value="runs/finetune",
                label="Output dir",
                full_width=True,
            ),
            batch_size=mo.ui.number(value=16, start=1, stop=512, label="Batch size"),
            num_steps=mo.ui.number(
                value=20000, start=1, stop=10_000_000, label="Steps"
            ),
            learning_rate_base=mo.ui.number(
                value=0.004, start=0.0, stop=1.0, step=0.0001, label="LR base"
            ),
            image_size=mo.ui.number(value=320, start=64, stop=2048, label="Image size"),
            early_stopping_patience=mo.ui.number(
                value=50, start=1, stop=10000, label="Early-stop patience"
            ),
            qat=mo.ui.switch(
                value=False,
                label="QAT (full int8: fold BN + fake-quant backbone + head)",
            ),
            qat_per_channel=mo.ui.switch(
                value=False,
                label="Per-channel weights (i.MX93 Ethos-U; off = per-tensor, i.MX8M Plus)",
            ),
            reset_optimizer=mo.ui.dropdown(
                options={"Auto (on for QAT)": None, "On": True, "Off": False},
                value="Auto (on for QAT)",
                label="Reset optimizer",
            ),
        )
        .form(submit_button_label="Build config")
    )
    form
    return (form,)


@app.cell
def _(FineTuneConfig, FinetuneRunConfig, OVERRIDE, form, mo):
    if OVERRIDE is not None:
        run_config = (
            OVERRIDE
            if isinstance(OVERRIDE, FinetuneRunConfig)
            else FinetuneRunConfig.from_mapping(OVERRIDE)
        )
    else:
        mo.stop(
            form.value is None,
            mo.md("_Fill in the form above and click **Build config**._"),
        )
        v = form.value
        run_config = FinetuneRunConfig(
            model_path=v["model_path"],
            dataset_bundle_path=v["dataset_bundle_path"],
            num_classes=int(v["num_classes"]),
            output_dir=v["output_dir"],
            finetune=FineTuneConfig(
                batch_size=int(v["batch_size"]),
                num_steps=int(v["num_steps"]),
                learning_rate_base=float(v["learning_rate_base"]),
                image_size=int(v["image_size"]),
                early_stopping_patience=int(v["early_stopping_patience"]),
            ),
            qat=bool(v["qat"]),
            qat_per_channel=bool(v["qat_per_channel"]),
            reset_optimizer=v["reset_optimizer"],
        )
    return (run_config,)


@app.cell(hide_code=True)
def _(mo, run_config):
    mo.md(f"""
    ### Resolved run

    | Setting | Value |
    |---|---|
    | Model | `{run_config.model_path}` |
    | Dataset bundle | `{run_config.dataset_bundle_path}` |
    | Classes | {run_config.num_classes} |
    | Output | `{run_config.output_dir}` |
    | Steps | {run_config.finetune.num_steps} |
    | Batch | {run_config.finetune.batch_size} |
    | Image size | {run_config.finetune.image_size} |
    | QAT | {run_config.qat} |
    | Per-channel | {run_config.qat_per_channel} |
    | Pipeline → | `{run_config.pipeline_config_path}` |
    | Train dir → | `{run_config.train_dir}` |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    train_button = mo.ui.run_button(label="Train")
    train_button
    return (train_button,)


@app.cell
def _(OVERRIDE, mo, run_config, run_finetune, train_button):
    should_run = (OVERRIDE is not None) or train_button.value
    mo.stop(
        not should_run,
        mo.md("_Press **Train** to start (or set `OVERRIDE` for head-less runs)._"),
    )
    result = run_finetune(run_config)
    return (result,)


@app.cell(hide_code=True)
def _(json, mo, result):
    best = (
        json.loads(result.best_metric_path.read_text())
        if result.best_metric_path.exists()
        else None
    )
    mo.md(
        f"""
        ## Result

        - Pipeline: `{result.pipeline_config}`
        - Checkpoints: `{result.train_dir}`
        - Best metric: **{best["metric_value"]:.5f}** ({best["metric_name"]})
          at step {best["step"]}
        """
        if best
        else f"""
        ## Result

        - Pipeline: `{result.pipeline_config}`
        - Checkpoints: `{result.train_dir}`
        - No `best_metric.json` written yet.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, result):
    # Training curves, built from the plain metrics_history.json the trainer
    # writes (no TensorBoard) via the same curves.py plotters as before.
    from agri_vision_edge.evaluation.curves import (
        load_history_scalars,
        plot_learning_rate,
        plot_loss_curves,
        plot_map_curves,
        plot_recall_curves,
    )

    mo.stop(
        not result.history_path.exists(),
        mo.md("_No `metrics_history.json` written yet._"),
    )

    history_df = load_history_scalars(result.history_path)

    loss_fig, _ = plot_loss_curves(history_df)
    map_fig, _ = plot_map_curves(history_df)
    recall_fig, _ = plot_recall_curves(history_df)
    lr_fig, _ = plot_learning_rate(history_df)

    mo.vstack(
        [
            mo.md("## Training curves"),
            mo.as_html(loss_fig),
            mo.as_html(map_fig),
            mo.as_html(recall_fig),
            mo.as_html(lr_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Export

    Export the best checkpoint to the TF model-zoo layout
    (`checkpoint/ckpt-0` + `pipeline.config` + `saved_model/`). The SavedModel
    is an fp32 graph for test inference; the checkpoint is reusable as the
    `model_path` of a follow-up run (e.g. to **resume QAT from this finetune**,
    exactly like the initial finetune resumes from the COCO17 checkpoint).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    export_button = mo.ui.run_button(label="Export best checkpoint + SavedModel")
    export_button
    return (export_button,)


@app.cell
def _(OVERRIDE, export_button, export_run, mo, result, run_config):
    should_export = (OVERRIDE is not None) or export_button.value
    mo.stop(
        not should_export,
        mo.md("_Press **Export** to write the best checkpoint and SavedModel._"),
    )
    # Depend on `result` so export only runs after training populated train_dir.
    _ = result
    export_result = export_run(run_config)
    return (export_result,)


@app.cell(hide_code=True)
def _(export_result, mo, run_config):
    mo.md(f"""
    ### Exported

    - Checkpoint: `{export_result.checkpoint}`
    - SavedModel: `{export_result.saved_model_dir}`
    - Pipeline: `{export_result.pipeline_config}`

    Resume QAT from this finetune (same mechanism as finetuning from COCO17):

    ```python
    run_finetune(FinetuneRunConfig(
        model_path="{export_result.export_dir}",
        dataset_bundle_path="{run_config.dataset_bundle_path}",
        num_classes={run_config.num_classes},
        output_dir="runs/qat",
        qat=True,
    ))
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    test_image = mo.ui.text(
        label="Test image for fp32 inference (optional)",
        full_width=True,
    )
    test_image
    return (test_image,)


@app.cell
def _(export_result, mo, run_config, test_image):
    mo.stop(
        not test_image.value,
        mo.md("_Set an image path above to run fp32 inference on the export._"),
    )

    from agri_vision_edge.tfod.inference import (
        detect_image,
        load_label_map,
        load_saved_model,
    )

    detect_fn = load_saved_model(export_result.saved_model_dir)
    category_index = load_label_map(run_config.label_map)

    vis, _ = detect_image(
        detect_fn,
        image_path=test_image.value,
        category_index=category_index,
        image_size=run_config.finetune.image_size,
    )
    vis
    return


if __name__ == "__main__":
    app.run()
