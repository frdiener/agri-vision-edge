import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

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

    tile_rows = mo.ui.number(
        start=1,
        stop=10,
        value=2,
        label="Tile rows",
    )

    tile_cols = mo.ui.number(
        start=1,
        stop=10,
        value=2,
        label="Tile columns",
    )

    tile_overlap = mo.ui.number(
        start=0,
        stop=512,
        value=0,
        label="Tile overlap (pixels)",
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
        mo,
        resize,
        resolution,
        run,
        tile_cols,
        tile_overlap,
        tile_rows,
        tiling,
    )


@app.cell(hide_code=True)
def _(mo, tiling):
    # Drop PhenoBench's partial (border) plants? True matches the official
    # PhenoBench bbox protocol (its evaluator filters partials) and the Kaggle
    # tiled bundles. Default tracks tiling: on for tiled, off otherwise.
    ignore_partial = mo.ui.checkbox(
        value=tiling.value,
        label="Ignore partial (border) plants",
    )
    return (ignore_partial,)


@app.cell(hide_code=True)
def _(
    bundle_type,
    dataset,
    ignore_partial,
    label_mode,
    mo,
    resize,
    resolution,
    run,
    tile_cols,
    tile_overlap,
    tile_rows,
    tiling,
):
    label_section = (
        mo.vstack(
            [
                mo.md("### Label Configuration"),
                label_mode,
            ]
        )
        if bundle_type.value == "Prepared Dataset"
        else mo.md("")
    )

    tiling_section = (
        mo.vstack(
            [
                mo.md("### Tiling"),
                tile_rows,
                tile_cols,
                tile_overlap,
            ]
        )
        if tiling.value
        else mo.md("")
    )

    resolution_section = (
        mo.vstack(
            [
                mo.md("### Target Resolution"),
                resolution,
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
    | Labels | {label_mode.value if bundle_type.value == "Prepared Dataset" else "N/A"} |
    | Tiling | {"Enabled" if tiling.value else "Disabled"} |
    | Ignore partials | {ignore_partial.value} |
    | Resize | {"Enabled" if resize.value else "Disabled"} |
    """)

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("# Dataset Conversion"),
                    dataset,
                    bundle_type,
                    label_section,
                    tiling,
                    tiling_section,
                    ignore_partial,
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
    ignore_partial,
    label_mode,
    resize,
    resolution,
    run,
    tile_cols,
    tile_overlap,
    tile_rows,
    tiling,
):
    from pprint import pprint

    if run.value:
        config = {
            "dataset": dataset.value,
            "bundle_type": bundle_type.value,
            "label_mode": (
                label_mode.value if bundle_type.value == "Prepared Dataset" else None
            ),
            "tiling": tiling.value,
            "tile_rows": tile_rows.value if tiling.value else None,
            "tile_cols": tile_cols.value if tiling.value else None,
            "tile_overlap": tile_overlap.value if tiling.value else None,
            "ignore_partial": ignore_partial.value,
            "resize": resize.value,
            "resolution": resolution.value if resize.value else None,
        }

        pprint(config)
    return (config,)


@app.cell
def _(config):
    from pathlib import Path
    import json

    import numpy as np
    from PIL import Image
    from tqdm.auto import tqdm

    from phenobench import PhenoBench

    from agri_vision_edge.data.tiling import (
        TiledPhenoBench,
        FilterConfig,
        compute_tiles,
        crop_array,
    )
    from agri_vision_edge.data import (
        PHENOBENCH_MULTICLASS,
        PHENOBENCH_WEED_ONLY,
        split_indices,
        build_record,
        build_rep_indices,
        write_label_map,
        export_coco_annotations,
    )

    SEED = 42

    IMAGE_SIZE = (
        config["resolution"] if config["resize"] else 512 if config["tiling"] else 1024
    )

    DATASET_DEFINITION = (
        PHENOBENCH_MULTICLASS if config["label_mode"] == "mc" else PHENOBENCH_WEED_ONLY
    )

    SOURCE_ROOT = Path("datasets") / (config["dataset"] + "_raw_full")

    DEST_ROOT = Path("datasets") / (
        config["dataset"]
        + "_"
        + (
            config["label_mode"] + "_tiled"
            if config["tiling"] and config["bundle_type"] == "Prepared Dataset"
            else "raw_tiled"
            if ["tiling"]
            else f"raw_{config['resolution']}"
            if config["resize"]
            else "raw_full"
        )
    )

    if DEST_ROOT.exists():
        raise FileExistsError(
            f"Destination path {DEST_ROOT} already exists. "
            "Please remove it before running the export."
        )

    assert SOURCE_ROOT.exists()
    DEST_ROOT.mkdir(parents=True)
    DEST_ROOT
    return (
        DATASET_DEFINITION,
        DEST_ROOT,
        FilterConfig,
        IMAGE_SIZE,
        Image,
        Path,
        PhenoBench,
        SEED,
        SOURCE_ROOT,
        TiledPhenoBench,
        build_record,
        build_rep_indices,
        compute_tiles,
        crop_array,
        export_coco_annotations,
        json,
        np,
        split_indices,
        tqdm,
        write_label_map,
    )


@app.cell
def _(
    DEST_ROOT,
    Image,
    Path,
    SOURCE_ROOT,
    compute_tiles,
    config,
    crop_array,
    json,
    np,
    tqdm,
):
    TARGET_DIRS = [
        "images",
        "semantics",
        "plant_instances",
        "leaf_instances",
        "plant_visibility",
        "leaf_visibility",
    ]

    def tile_file(
        src: Path,
        dst_dir: Path,
    ):
        array = np.array(Image.open(src))

        h, w = array.shape[:2]

        tiles = compute_tiles(
            width=w,
            height=h,
            rows=config["tile_rows"],
            cols=config["tile_cols"],
            overlap=config["tile_overlap"],
        )

        for tile_idx, tile in enumerate(
            tiles,
            start=1,
        ):
            tile_array = crop_array(
                array,
                tile,
            )

            out_path = dst_dir / f"{src.stem}_tile{tile_idx}{src.suffix}"

            Image.fromarray(tile_array).save(out_path)

    if config["tiling"] and config["bundle_type"] == "Source Dataset":
        for split in [
            "train",
            "val",
            "test",
        ]:
            print(f"Processing {split}...")

            image_dir = SOURCE_ROOT / split / "images"

            filenames = sorted(image_dir.glob("*.png"))

            for filename in tqdm(
                filenames,
                desc=split,
            ):
                for target_dir in TARGET_DIRS:
                    src = SOURCE_ROOT / split / target_dir / filename.name

                    if not src.exists():
                        continue

                    dst_dir = DEST_ROOT / split / target_dir

                    dst_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    tile_file(
                        src,
                        dst_dir,
                    )

        print("Done.")

        json.dump(
            {
                "rows": config["tile_rows"],
                "cols": config["tile_cols"],
                "overlap": config["tile_overlap"],
                "source_dataset": str(SOURCE_ROOT),
            },
            (DEST_ROOT / "tiling_config.json").open("w"),
            indent=2,
        )
    return


@app.cell
def _(FilterConfig, PhenoBench, SOURCE_ROOT, TiledPhenoBench, config, mo):
    mo.stop(
        config["bundle_type"] == "Source Dataset",
        mo.md(
            "Further cells will be executed only if "
            "'Prepared Dataset' bundle type is selected."
        ),
    )

    train_dataset = PhenoBench(
        root=SOURCE_ROOT,
        split="train",
        target_types=[
            "semantics",
            "plant_instances",
        ],
        ignore_partial=config["ignore_partial"],
    )

    val_dataset = PhenoBench(
        root=SOURCE_ROOT,
        split="val",
        target_types=[
            "semantics",
            "plant_instances",
        ],
        ignore_partial=config["ignore_partial"],
    )

    if config["tiling"]:
        filter_config = FilterConfig(
            min_instance_pixels=32,
            min_bbox_width=4,
            min_bbox_height=4,
            min_bbox_area=32,
            min_visible_fraction=0.7,
        )

        train_dataset = TiledPhenoBench(
            train_dataset,
            rows=2,
            cols=2,
            overlap=0.0,
            filter_config=filter_config,
        )

        val_dataset = TiledPhenoBench(
            val_dataset,
            rows=2,
            cols=2,
            overlap=0.0,
            filter_config=filter_config,
        )

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    return train_dataset, val_dataset


@app.cell
def _(DEST_ROOT, SEED, json, split_indices, val_dataset):
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
    )

    true_eval_stats = build_record(
        DEST_ROOT / "true_eval.record",
        val_dataset,
        dataset_definition=DATASET_DEFINITION,
        target_size=IMAGE_SIZE,
        skip_negatives=False,
    )

    val_stats = build_record(
        DEST_ROOT / "val.record",
        val_dataset,
        dataset_definition=DATASET_DEFINITION,
        indices=val_idx,
        target_size=IMAGE_SIZE,
        skip_negatives=False,
    )

    test_stats = build_record(
        DEST_ROOT / "test.record",
        val_dataset,
        dataset_definition=DATASET_DEFINITION,
        indices=test_idx,
        target_size=IMAGE_SIZE,
        skip_negatives=False,
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
    export_coco_annotations(
        DEST_ROOT / "train_annotations.json",
        train_dataset,
        dataset_definition=DATASET_DEFINITION,
    )

    export_coco_annotations(
        DEST_ROOT / "true_eval_annotations.json",
        val_dataset,
        dataset_definition=DATASET_DEFINITION,
    )

    export_coco_annotations(
        DEST_ROOT / "val_annotations.json",
        val_dataset,
        dataset_definition=DATASET_DEFINITION,
        indices=val_idx,
    )

    export_coco_annotations(
        DEST_ROOT / "test_annotations.json",
        val_dataset,
        dataset_definition=DATASET_DEFINITION,
        indices=test_idx,
    )
    return


@app.cell
def _(DEST_ROOT, SEED, build_rep_indices, json, train_dataset):
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
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "rep_samples": len(rep_indices),
        "ignore_partial": config["ignore_partial"],
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
    return (artifacts,)


@app.cell
def _(artifacts):
    for artifact in artifacts:
        print(artifact)
    return


@app.cell
def _(json, metadata):
    print(json.dumps(metadata, indent=2))
    return


if __name__ == "__main__":
    app.run()
