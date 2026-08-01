import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        r"""
        # Dataset Conversion

        Local counterpart of the Kaggle export notebooks
        `01`–`04_export_{mc,sc}_phenobench[-tiled]_no-partials.ipynb`.
        **Those notebooks are authoritative** — training runs on their output —
        so the defaults here mirror them and any change there has to be mirrored
        back into this notebook.

        Two bundle types:

        * **Prepared Dataset** — TFRecords + COCO annotations + label map +
          representative-dataset indices, i.e. `datasets/phenobench_*_no-partials`.
        * **Source Dataset** — a second *raw* PhenoBench tree cut into tiles,
          i.e. `datasets/phenobench_raw_tiled`. Nothing trains on it; it exists
          because `ave benchmark` needs tile **images** on disk and
          `ave evaluate --faithful` needs tile **masks** on disk. Both join it to
          the prepared bundle *by file name*, so its grid must match the prepared
          bundle's — see the warning under "Tiling".

        ## Partial (do-not-care) plants

        Per the upstream PhenoBench protocol, a plant with visibility
        `<= partial_threshold` is *partial*. Since the preprocessing overhaul it
        is no longer dropped at load time (`ignore_partial`); instead it is
        **tagged** and the policy is applied per artifact:

        | artifact | partials |
        |---|---|
        | `train.record`, `train_annotations.json` | dropped (`include_partials=False`) |
        | `val` / `test` / `true_eval` records + annotations | kept, flagged `ignore` / `is_partial` (`include_partials=True`) |

        For tiled bundles the criterion is the plant's *effective* visibility —
        upstream frame visibility × the fraction surviving the tile cut — so a
        plant sliced by a tile border is treated exactly like an
        originally-partial one.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    dataset = mo.ui.dropdown(
        options={"PhenoBench": "phenobench"},
        value="PhenoBench",
    )

    bundle_type = mo.ui.radio(
        options=["Source Dataset", "Prepared Dataset"],
        value="Prepared Dataset",
    )

    tiling = mo.ui.checkbox(label="Enable tiling")
    resize = mo.ui.checkbox(label="Resize images")

    label_mode = mo.ui.radio(
        options={
            "Single-class": "sc",
            "Multi-class": "mc",
        },
        value="Single-class",
    )

    # Defaults are the geometry of notebooks 03/04: a 3x3 grid with half-tile
    # overlap, which on a 1024 frame yields uniform 512 px tiles (same tile size
    # as the old 2x2 grid, but tile-boundary plants are recovered by the
    # overlap).
    tile_rows = mo.ui.number(
        start=1,
        stop=10,
        value=3,
        label="Tile rows",
    )

    tile_cols = mo.ui.number(
        start=1,
        stop=10,
        value=3,
        label="Tile columns",
    )

    # FRACTION of the tile size, not pixels: `compute_tiles` solves
    # tile = width / ((cols - 1) * (1 - overlap) + 1) and requires [0, 1).
    tile_overlap = mo.ui.number(
        start=0.0,
        stop=0.95,
        step=0.05,
        value=0.5,
        label="Tile overlap (fraction of tile size)",
    )

    # Upstream do-not-care criterion: effective visibility <= this ratio.
    partial_threshold = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.5,
        label="Partial (do-not-care) visibility threshold",
    )

    resolution = mo.ui.dropdown(
        options=["320", "512"],
        value="320",
    )

    run = mo.ui.run_button(label="Generate Dataset")
    return (
        bundle_type,
        dataset,
        label_mode,
        partial_threshold,
        resize,
        resolution,
        run,
        tile_cols,
        tile_overlap,
        tile_rows,
        tiling,
    )


@app.cell(hide_code=True)
def _(
    bundle_type,
    dataset,
    label_mode,
    mo,
    partial_threshold,
    resize,
    resolution,
    run,
    tile_cols,
    tile_overlap,
    tile_rows,
    tiling,
):
    prepared = bundle_type.value == "Prepared Dataset"

    label_section = (
        mo.vstack(
            [
                mo.md("### Label Configuration"),
                label_mode,
            ]
        )
        if prepared
        else mo.md("")
    )

    tiling_section = (
        mo.vstack(
            [
                mo.md("### Tiling"),
                tile_rows,
                tile_cols,
                tile_overlap,
                mo.callout(
                    mo.md(
                        "A materialized tiled tree and a prepared bundle are "
                        "joined **by file name** (`{stem}_tile{i}.png`). A 2x2 "
                        "and a 3x3 tree share `_tile0..3` while those denote "
                        "*different crops*, so a mismatched grid evaluates "
                        "against the wrong ground truth **without raising**. "
                        "Keep this identical to notebooks `03`/`04`."
                    ),
                    kind="warn",
                ),
            ]
        )
        if tiling.value
        else mo.md("")
    )

    partial_section = (
        mo.vstack(
            [
                mo.md("### Partials"),
                partial_threshold,
            ]
        )
        if prepared
        else mo.md("")
    )

    resolution_section = (
        mo.vstack(
            [
                mo.md("### Target Resolution"),
                resolution,
                mo.callout(
                    mo.md(
                        "Notebooks `01`–`04` store images at their **native** "
                        "resolution (1024 frames / 512 tiles) and let the "
                        "trainer's `fixed_shape_resizer` downsize at runtime, "
                        "so TFOD's crop/zoom augmentations still see full "
                        "detail. Leave this off to match them."
                    ),
                    kind="warn",
                ),
            ]
        )
        if resize.value
        else mo.md("")
    )

    summary = mo.md(f"""
    ### Export Summary

    | Setting | Value |
    |----------|----------|
    | Dataset | {dataset.value} |
    | Bundle | {bundle_type.value} |
    | Labels | {label_mode.value if prepared else "N/A"} |
    | Tiling | {f"{tile_rows.value}x{tile_cols.value} @ overlap {tile_overlap.value}" if tiling.value else "Disabled"} |
    | Partial threshold | {partial_threshold.value if prepared else "N/A"} |
    | Resize | {resolution.value if resize.value else "Native resolution"} |
    """)

    mo.hstack(
        [
            mo.vstack(
                [
                    dataset,
                    bundle_type,
                    label_section,
                    tiling,
                    tiling_section,
                    partial_section,
                    resize,
                    resolution_section,
                ]
            ),
            mo.vstack(
                [
                    summary,
                    run,
                ]
            ),
        ],
        widths=[2, 1],
    )
    return


@app.cell
def _(
    bundle_type,
    dataset,
    label_mode,
    mo,
    partial_threshold,
    resize,
    resolution,
    run,
    tile_cols,
    tile_overlap,
    tile_rows,
    tiling,
):
    from pprint import pprint

    mo.stop(not run.value, mo.md("Press **Generate Dataset** to start."))

    config = {
        "dataset": dataset.value,
        "bundle_type": bundle_type.value,
        "label_mode": (
            label_mode.value if bundle_type.value == "Prepared Dataset" else None
        ),
        "tiling": tiling.value,
        "tile_rows": tile_rows.value if tiling.value else None,
        "tile_cols": tile_cols.value if tiling.value else None,
        # Fraction of the tile size, see `compute_tiles`.
        "tile_overlap": float(tile_overlap.value) if tiling.value else None,
        "partial_threshold": float(partial_threshold.value),
        "resize": resize.value,
        "resolution": int(resolution.value) if resize.value else None,
    }

    pprint(config)
    return (config,)


@app.cell
def _(config):
    import json
    from pathlib import Path

    from phenobench import PhenoBench
    from tqdm.auto import tqdm

    from agri_vision_edge.data import (
        PHENOBENCH_MULTICLASS,
        PHENOBENCH_WEED_ONLY,
        build_record,
        build_rep_indices,
        export_coco_annotations,
        split_indices,
        write_label_map,
    )
    from agri_vision_edge.data.plant_boxes import PartialAwarePhenoBench
    from agri_vision_edge.data.raw_tiling import materialize_tiled_dataset
    from agri_vision_edge.data.tiling import TiledPhenoBench

    SEED = 42

    PREPARED = config["bundle_type"] == "Prepared Dataset"

    DATASET_DEFINITION = (
        PHENOBENCH_MULTICLASS if config["label_mode"] == "mc" else PHENOBENCH_WEED_ONLY
    )

    SOURCE_ROOT = Path("datasets") / (config["dataset"] + "_raw_full")

    if PREPARED:
        # Matches the Kaggle bundles the trainer/converter look for:
        # phenobench_{sc,mc}[_tiled]_no-partials.
        _suffix = (
            config["label_mode"]
            + ("_tiled" if config["tiling"] else "")
            + "_no-partials"
        )
    elif config["tiling"]:
        _suffix = "raw_tiled"
    else:
        # A "Source Dataset" bundle is only ever a re-cut of the raw tree; with
        # tiling off there is nothing to produce (SOURCE_ROOT already is it).
        raise ValueError(
            "Source Dataset export needs tiling enabled — without it the "
            f"result would be a copy of {SOURCE_ROOT}."
        )

    DEST_ROOT = Path("datasets") / (config["dataset"] + "_" + _suffix)

    if DEST_ROOT.exists():
        raise FileExistsError(
            f"Destination path {DEST_ROOT} already exists. "
            "Please remove or rename it before running the export."
        )

    assert SOURCE_ROOT.exists(), f"Missing source dataset: {SOURCE_ROOT}"

    DEST_ROOT.mkdir(parents=True)

    DEST_ROOT  # noqa: B018 -- bare expression = this marimo cell's rendered output
    return (
        DATASET_DEFINITION,
        DEST_ROOT,
        PREPARED,
        Path,
        PartialAwarePhenoBench,
        PhenoBench,
        SEED,
        SOURCE_ROOT,
        TiledPhenoBench,
        build_record,
        build_rep_indices,
        export_coco_annotations,
        json,
        materialize_tiled_dataset,
        split_indices,
        tqdm,
        write_label_map,
    )


@app.cell
def _(DEST_ROOT, PREPARED, SOURCE_ROOT, config, materialize_tiled_dataset, tqdm):
    # --- Source Dataset: materialize the tiled RAW tree ------------------------
    #
    # Cuts every split/sub-directory of SOURCE_ROOT with the same
    # `compute_tiles` geometry the in-memory `TiledPhenoBench` uses, and names
    # the tiles `{stem}_tile{i}.png` starting at i=0 exactly like that wrapper —
    # which is what makes the prepared bundle's annotations resolve against
    # these files. The grid is recorded in `tiling_config.json`.
    tiling_stats = None

    if not PREPARED:
        tiling_stats = materialize_tiled_dataset(
            SOURCE_ROOT,
            DEST_ROOT,
            rows=config["tile_rows"],
            cols=config["tile_cols"],
            overlap=config["tile_overlap"],
            workers=None,
            progress=lambda iterable, desc="": tqdm(iterable, desc=desc),
            exist_ok=True,
        )

    tiling_stats  # noqa: B018 -- bare expression = this marimo cell's rendered output
    return


@app.cell
def _(
    PREPARED,
    PartialAwarePhenoBench,
    PhenoBench,
    SOURCE_ROOT,
    TiledPhenoBench,
    config,
    mo,
):
    mo.stop(
        not PREPARED,
        mo.md(
            "**Source Dataset** bundle complete — the remaining cells build a "
            "*Prepared Dataset* (TFRecords / COCO) and are skipped."
        ),
    )

    def _raw(split):
        # `ignore_partial=False`: partials are NOT dropped at load time any
        # more. They are tagged below and filtered per artifact via
        # `include_partials`, which is what lets the eval records carry them as
        # do-not-care. `plant_visibility` is the upstream partiality source.
        return PhenoBench(
            root=SOURCE_ROOT,
            split=split,
            target_types=[
                "semantics",
                "plant_instances",
                "plant_visibility",
            ],
            ignore_partial=False,
        )

    def _wrap(raw):
        if config["tiling"]:
            return TiledPhenoBench(
                raw,
                rows=config["tile_rows"],
                cols=config["tile_cols"],
                overlap=config["tile_overlap"],
                partial_threshold=config["partial_threshold"],
            )

        return PartialAwarePhenoBench(
            raw,
            partial_threshold=config["partial_threshold"],
        )

    train_dataset = _wrap(_raw("train"))
    val_dataset = _wrap(_raw("val"))

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    return train_dataset, val_dataset


@app.cell
def _(config, train_dataset):
    # Native resolution unless explicitly resized: 1024 for full frames, the
    # actual tile size for tiled bundles (derived from the computed grid rather
    # than assumed, so a non-default grid stays correct).
    if config["resize"]:
        IMAGE_SIZE = config["resolution"]
    elif config["tiling"]:
        _tile = train_dataset.tiles[0]
        assert _tile.width == _tile.height, (
            f"Non-square tile {_tile.width}x{_tile.height}: the exporter "
            "resizes each tile to a square, which would distort it."
        )
        IMAGE_SIZE = _tile.width
    else:
        IMAGE_SIZE = 1024

    print("Image size:", IMAGE_SIZE)
    return (IMAGE_SIZE,)


@app.cell
def _(DEST_ROOT, SEED, json, split_indices, val_dataset):
    # The official val split is halved into val + held-out test (deterministic).
    val_idx, test_idx = split_indices(
        len(val_dataset),
        val_ratio=0.5,
        seed=SEED,
    )

    (DEST_ROOT / "val_test_split.json").write_text(
        json.dumps(
            {
                "val": val_idx,
                "test": test_idx,
            },
            indent=2,
        )
    )
    return test_idx, val_idx


@app.cell
def _(
    DATASET_DEFINITION,
    DEST_ROOT,
    IMAGE_SIZE,
    build_record,
    test_idx,
    train_dataset,
    val_dataset,
    val_idx,
):
    train_stats = build_record(
        DEST_ROOT / "train.record",
        train_dataset,
        dataset_definition=DATASET_DEFINITION,
        target_size=IMAGE_SIZE,
        skip_negatives=False,
        include_partials=False,  # do not train on partials
    )

    true_eval_stats = build_record(
        DEST_ROOT / "true_eval.record",
        val_dataset,
        dataset_definition=DATASET_DEFINITION,
        target_size=IMAGE_SIZE,
        skip_negatives=False,
        include_partials=True,  # flag partials as do-not-care
    )

    val_stats = build_record(
        DEST_ROOT / "val.record",
        val_dataset,
        dataset_definition=DATASET_DEFINITION,
        indices=val_idx,
        target_size=IMAGE_SIZE,
        skip_negatives=False,
        include_partials=True,
    )

    test_stats = build_record(
        DEST_ROOT / "test.record",
        val_dataset,
        dataset_definition=DATASET_DEFINITION,
        indices=test_idx,
        target_size=IMAGE_SIZE,
        skip_negatives=False,
        include_partials=True,
    )
    return test_stats, train_stats, true_eval_stats, val_stats


@app.cell
def _(
    DATASET_DEFINITION,
    DEST_ROOT,
    export_coco_annotations,
    test_idx,
    train_dataset,
    val_dataset,
    val_idx,
):
    # Consumed by `ave evaluate` (`--ignore-partials` reads the partial flags).
    export_coco_annotations(
        DEST_ROOT / "train_annotations.json",
        train_dataset,
        dataset_definition=DATASET_DEFINITION,
        include_partials=False,
    )

    export_coco_annotations(
        DEST_ROOT / "true_eval_annotations.json",
        val_dataset,
        dataset_definition=DATASET_DEFINITION,
        include_partials=True,
    )

    export_coco_annotations(
        DEST_ROOT / "val_annotations.json",
        val_dataset,
        dataset_definition=DATASET_DEFINITION,
        indices=val_idx,
        include_partials=True,
    )

    export_coco_annotations(
        DEST_ROOT / "test_annotations.json",
        val_dataset,
        dataset_definition=DATASET_DEFINITION,
        indices=test_idx,
        include_partials=True,
    )
    return


@app.cell
def _(DEST_ROOT, SEED, build_rep_indices, json, train_dataset):
    # INT8 calibration indices. These are *positions* into the train split, so
    # they are only valid for a dataset rebuilt with this exact geometry.
    rep_indices = build_rep_indices(
        dataset=train_dataset,
        num_samples=200,
        seed=SEED,
    )

    (DEST_ROOT / "rep_dataset.json").write_text(json.dumps(rep_indices))
    return (rep_indices,)


@app.cell
def _(DATASET_DEFINITION, DEST_ROOT, write_label_map):
    write_label_map(
        DEST_ROOT / "label_map.pbtxt",
        dataset_definition=DATASET_DEFINITION,
    )
    return


@app.cell
def _(
    DATASET_DEFINITION,
    DEST_ROOT,
    IMAGE_SIZE,
    SEED,
    config,
    json,
    rep_indices,
    test_stats,
    train_dataset,
    train_stats,
    true_eval_stats,
    val_dataset,
    val_stats,
):
    metadata = {
        "dataset_definition": {
            "name": DATASET_DEFINITION.name,
            "categories": DATASET_DEFINITION.categories,
        },
        "image_size": IMAGE_SIZE,
        # Recorded so the converter can rebuild the calibration dataset with the
        # geometry this bundle's rep_dataset.json indices address.
        "tiling": (
            {
                "rows": config["tile_rows"],
                "cols": config["tile_cols"],
                "overlap": config["tile_overlap"],
            }
            if config["tiling"]
            else None
        ),
        "partial_threshold": config["partial_threshold"],
        "partials_policy": "drop in train, do-not-care in eval",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "rep_samples": len(rep_indices),
        "split_seed": SEED,
        "train_stats": train_stats,
        "true_eval_stats": true_eval_stats,
        "val_stats": val_stats,
        "test_stats": test_stats,
    }

    (DEST_ROOT / "dataset_metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )
    return (metadata,)


@app.cell
def _(DEST_ROOT):
    artifacts = [
        "train.record",
        "val.record",
        "test.record",
        "true_eval.record",
        "train_annotations.json",
        "true_eval_annotations.json",
        "val_annotations.json",
        "test_annotations.json",
        "label_map.pbtxt",
        "rep_dataset.json",
        "val_test_split.json",
        "dataset_metadata.json",
    ]

    missing = [p for p in artifacts if not (DEST_ROOT / p).exists()]

    assert not missing, f"Missing artifacts: {missing}"

    print("All artifacts generated successfully.")

    for artifact in artifacts:
        print(artifact)
    return


@app.cell
def _(json, metadata):
    print(json.dumps(metadata, indent=2))
    return


if __name__ == "__main__":
    app.run()
