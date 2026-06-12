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

    from agri_vision_edge.third_party import setup_tensorflow_models

    setup_tensorflow_models()

    from agri_vision_edge.experiment import FineTuneConfig
    from agri_vision_edge.tfod_trainer import (
        FinetuneRunConfig,
        run_finetune,
    )

    return FineTuneConfig, FinetuneRunConfig, json, run_finetune


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
        qat_scheme="full",   # omit / None for a plain finetune
    ))
    ```

    To run *this notebook* against a preset config (e.g. as a script), set
    `OVERRIDE` in the next cell to a `FinetuneRunConfig` or a dict; that
    bypasses the form and the Train button.
    """)
    return


@app.cell
def _():
    # Set to a FinetuneRunConfig or dict to bypass the UI; None = interactive.
    OVERRIDE = None
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

            **Quantization-aware training** (leave scheme empty for a plain finetune)

            - {qat_scheme}
            - {fold_bn} {reset_optimizer}
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
            qat_scheme=mo.ui.dropdown(
                options={
                    "None (plain finetune)": None,
                    "full (int8 QAT)": "full",
                    "weights only": "weights",
                },
                value="None (plain finetune)",
                label="QAT scheme",
            ),
            fold_bn=mo.ui.switch(value=False, label="Fold BatchNorm"),
            reset_optimizer=mo.ui.switch(value=False, label="Reset optimizer"),
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
            qat_scheme=v["qat_scheme"],
            fold_bn=v["fold_bn"],
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
    | QAT scheme | {run_config.qat_scheme} |
    | Fold BN | {run_config.fold_bn} |
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
        - Best metric: **{best['metric_value']:.5f}** ({best['metric_name']})
          at step {best['step']}
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


if __name__ == "__main__":
    app.run()
