import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # PhenoBench SSD/FPN Analysis

    Explore object sizes, anchor matching,
    SSD assignment and FPN assignment.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    import altair as alt

    alt.data_transformers.enable("vegafusion")
    import numpy as np
    import pandas as pd

    from tqdm.auto import tqdm

    from phenobench import PhenoBench

    return PhenoBench, alt, mo, np, pd, tqdm


@app.cell(hide_code=True)
def _(PhenoBench):
    phenobench_train_dataset = PhenoBench(
        root="datasets/phenobench_raw",
        split="train",
        target_types=[
            "semantics",
            "plant_instances",
        ],
    )
    return (phenobench_train_dataset,)


@app.cell(hide_code=True)
def _(np, pd, phenobench_train_dataset, tqdm):
    weed_rows = []

    image_rows = []

    for sample in tqdm(phenobench_train_dataset):
        sample_semantics = sample["semantics"]
        sample_instances = sample["plant_instances"]

        image_height, image_width = sample_semantics.shape

        weed_mask = sample_semantics == 2

        weed_ids = np.unique(sample_instances[weed_mask])

        weed_ids = weed_ids[weed_ids > 0]

        image_rows.append(
            {
                "image_width": image_width,
                "image_height": image_height,
                "weed_count": len(weed_ids),
            }
        )

        for weed_id in weed_ids:
            weed_instance_mask = (sample_instances == weed_id) & weed_mask

            ys, xs = np.where(weed_instance_mask)

            xmin = xs.min()
            xmax = xs.max()

            ymin = ys.min()
            ymax = ys.max()

            bbox_width = xmax - xmin + 1
            bbox_height = ymax - ymin + 1

            bbox_area = bbox_width * bbox_height

            mask_area = weed_instance_mask.sum()

            weed_rows.append(
                {
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "bbox_area": bbox_area,
                    "mask_area": mask_area,
                    "aspect_ratio": (bbox_width / bbox_height),
                    "fill_ratio": (mask_area / bbox_area),
                    "width_frac": (bbox_width / image_width),
                    "height_frac": (bbox_height / image_height),
                    "area_frac": (bbox_area / (image_width * image_height)),
                }
            )

    weed_stats_df = pd.DataFrame(weed_rows)

    image_stats_df = pd.DataFrame(image_rows)
    return image_stats_df, weed_stats_df


@app.cell(hide_code=True)
def _(mo):
    target_resolution_ui = mo.ui.slider(
        start=128,
        stop=1024,
        step=32,
        value=320,
    )

    crop_area_ui = mo.ui.slider(
        start=0.2,
        stop=1.0,
        step=0.05,
        value=0.6,
    )
    return crop_area_ui, target_resolution_ui


@app.cell(hide_code=True)
def _(crop_area_ui, mo, target_resolution_ui):
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.md("**Target resolution**"),
                    target_resolution_ui,
                    mo.md(f"`{target_resolution_ui.value}px`"),
                ]
            ),
            mo.hstack(
                [
                    mo.md("**Crop min area**"),
                    crop_area_ui,
                    mo.md(f"`{crop_area_ui.value:.2f}`"),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, weed_stats_df):
    median_width = weed_stats_df.bbox_width.median()

    median_height = weed_stats_df.bbox_height.median()

    mo.hstack(
        [
            mo.stat(
                label="Weed instances",
                value=f"{len(weed_stats_df):,}",
            ),
            mo.stat(
                label="Median aspect ratio",
                value=f"{weed_stats_df.aspect_ratio.median():.2f}",
            ),
            mo.stat(
                label="Median width",
                value=f"{median_width:.1f}px",
            ),
            mo.stat(
                label="Median height",
                value=f"{median_height:.1f}px",
            ),
            mo.stat(
                label="Median fill ratio",
                value=f"{weed_stats_df.fill_ratio.median():.2f}",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, np, pd, weed_stats_df):
    ssd_scale_df = pd.DataFrame({"ssd_scale": np.sqrt(weed_stats_df.area_frac)})

    anchor_min_scale_ui = mo.ui.slider(
        start=0.005,
        stop=0.25,
        step=0.005,
        value=0.01,
        label="Anchor min scale",
    )

    anchor_max_scale_ui = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.01,
        value=0.25,
        label="Anchor max scale",
    )
    return anchor_max_scale_ui, anchor_min_scale_ui, ssd_scale_df


@app.cell(hide_code=True)
def _(anchor_max_scale_ui, anchor_min_scale_ui, mo):
    mo.vstack(
        [
            mo.hstack(
                [
                    anchor_min_scale_ui,
                    mo.md(f"**{anchor_min_scale_ui.value:.3f}**"),
                ]
            ),
            mo.hstack(
                [
                    anchor_max_scale_ui,
                    mo.md(f"**{anchor_max_scale_ui.value:.3f}**"),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, anchor_max_scale_ui, anchor_min_scale_ui, pd, ssd_scale_df):
    anchor_range_df = pd.DataFrame(
        {
            "x": [
                anchor_min_scale_ui.value,
                anchor_max_scale_ui.value,
            ]
        }
    )

    histogram_chart = (
        alt.Chart(ssd_scale_df)
        .mark_bar()
        .encode(
            alt.X(
                "ssd_scale:Q",
                bin=alt.Bin(maxbins=120),
            ),
            y="count()",
        )
        .properties(
            width=600,
            title="Weed Objects within scale Limits",
        )
    )

    rule_chart = alt.Chart(anchor_range_df).mark_rule(size=3).encode(x="x:Q")

    histogram_chart + rule_chart
    return


@app.cell(hide_code=True)
def _(crop_area_ui, np, pd, target_resolution_ui, weed_stats_df):
    target_resolution_value = target_resolution_ui.value

    resized_weed_stats_df = pd.DataFrame(
        {
            **weed_stats_df,
            "bbox_width_target": weed_stats_df.width_frac
            * target_resolution_value,
            "bbox_height_target": weed_stats_df.height_frac
            * target_resolution_value,
        }
    )

    crop_scale_factor = np.sqrt(crop_area_ui.value)

    cropped_weed_stats_df = pd.DataFrame(
        {
            **resized_weed_stats_df,
            "bbox_width_crop": resized_weed_stats_df.bbox_width_target
            / crop_scale_factor,
            "bbox_height_crop": resized_weed_stats_df.bbox_height_target
            / crop_scale_factor,
        }
    )
    return (cropped_weed_stats_df,)


@app.cell(hide_code=True)
def _(alt, cropped_weed_stats_df):
    chart = (
        alt.Chart(cropped_weed_stats_df)
        .mark_bar()
        .encode(
            alt.X(
                "bbox_width_crop:Q",
                bin=alt.Bin(maxbins=80),
                title="Object width (px)",
            ),
            alt.Y(
                "count()",
                title="Objects",
            ),
        )
        .properties(
            title="Weed width after crop + resize",
        )
    )

    chart
    return


@app.cell
def _(alt, cropped_weed_stats_df):
    alt.Chart(cropped_weed_stats_df).mark_circle(
        opacity=0.05,
        size=20,
    ).encode(
        x="bbox_width_crop:Q",
        y="bbox_height_crop:Q",
    ).properties(
        title="Weed width vs height",
    ).interactive()
    return


@app.cell(hide_code=True)
def _(alt, weed_stats_df):

    alt.Chart(
        weed_stats_df
    ).mark_bar().encode(
        alt.X(
            "aspect_ratio:Q",
            bin=alt.Bin(maxbins=100),
        ),
        y="count()",
    ).properties(
        width=600,
    )
    return


@app.cell(hide_code=True)
def _(alt, image_stats_df):

    alt.Chart(
        image_stats_df
    ).mark_bar().encode(
        alt.X(
            "weed_count:Q",
            bin=alt.Bin(maxbins=50),
        ),
        y="count()",
    ).properties(
        width=600,
    )
    return


@app.cell(hide_code=True)
def _(mo, phenobench_train_dataset):
    image_index_ui = mo.ui.slider(
        start=0,
        stop=len(
            phenobench_train_dataset
        ) - 1,
        value=0,
    )
    return (image_index_ui,)


@app.cell(hide_code=True)
def _(image_index_ui, mo):
    mo.vstack([
        mo.hstack([
            "Image Index in Train Set",
            mo.md(f"**{image_index_ui.value}**"),
            image_index_ui,
        ])
    ])
    return


@app.cell
def _(image_index_ui, phenobench_train_dataset):

    inspection_sample = (
        phenobench_train_dataset[
            image_index_ui.value
        ]
    )

    inspection_sample["image"]
    return


@app.cell(hide_code=True)
def _(np, pd):
    def box_iou_wh(
        box_width,
        box_height,
        anchor_width,
        anchor_height,
    ):
        """
        IoU assuming same center.
        Only width/height matter.
        """

        intersection_width = min(
            box_width,
            anchor_width,
        )

        intersection_height = min(
            box_height,
            anchor_height,
        )

        intersection_area = intersection_width * intersection_height

        box_area = box_width * box_height

        anchor_area = anchor_width * anchor_height

        union_area = box_area + anchor_area - intersection_area

        return intersection_area / union_area


    def generate_ssd_anchor_shapes(
        image_size,
        min_scale,
        max_scale,
        num_layers,
        aspect_ratios,
    ):
        """
        Approximate TF SSDAnchorGenerator.
        """

        scales = np.linspace(
            min_scale,
            max_scale,
            num_layers,
        )

        anchor_rows = []

        for layer_index, scale in enumerate(scales):
            for aspect_ratio in aspect_ratios:
                anchor_width = image_size * scale * np.sqrt(aspect_ratio)

                anchor_height = image_size * scale / np.sqrt(aspect_ratio)

                anchor_rows.append(
                    {
                        "generator": "ssd",
                        "layer": layer_index,
                        "scale": scale,
                        "aspect_ratio": aspect_ratio,
                        "anchor_width": anchor_width,
                        "anchor_height": anchor_height,
                    }
                )

        return pd.DataFrame(anchor_rows)


    def generate_fpn_anchor_shapes(
        image_size,
        min_level,
        max_level,
        anchor_scale,
        aspect_ratios,
        scales_per_octave,
    ):
        """
        Approximate TF MultiscaleAnchorGenerator.
        """

        anchor_rows = []

        for level in range(
            min_level,
            max_level + 1,
        ):
            stride = 2**level

            for octave_scale_index in range(scales_per_octave):
                octave_scale = 2 ** (octave_scale_index / scales_per_octave)

                base_size = stride * anchor_scale * octave_scale

                for aspect_ratio in aspect_ratios:
                    anchor_width = base_size * np.sqrt(aspect_ratio)

                    anchor_height = base_size / np.sqrt(aspect_ratio)

                    anchor_rows.append(
                        {
                            "generator": "fpn",
                            "level": level,
                            "stride": stride,
                            "aspect_ratio": aspect_ratio,
                            "anchor_width": anchor_width,
                            "anchor_height": anchor_height,
                        }
                    )

        return pd.DataFrame(anchor_rows)


    def compute_anchor_matches(
        object_stats_df,
        anchor_shapes_df,
        width_column,
        height_column,
    ):

        match_rows = []

        anchor_widths = anchor_shapes_df["anchor_width"].to_numpy()

        anchor_heights = anchor_shapes_df["anchor_height"].to_numpy()

        for row_index, row in object_stats_df.iterrows():
            box_width = row[width_column]

            box_height = row[height_column]

            ious = np.array(
                [
                    box_iou_wh(
                        box_width,
                        box_height,
                        anchor_width,
                        anchor_height,
                    )
                    for (
                        anchor_width,
                        anchor_height,
                    ) in zip(
                        anchor_widths,
                        anchor_heights,
                    )
                ]
            )

            best_anchor_index = np.argmax(ious)

            best_anchor = anchor_shapes_df.iloc[best_anchor_index]

            match_rows.append(
                {
                    "object_index": row_index,
                    "box_width": box_width,
                    "box_height": box_height,
                    "best_iou": float(ious[best_anchor_index]),
                    "best_anchor_width": best_anchor["anchor_width"],
                    "best_anchor_height": best_anchor["anchor_height"],
                    "best_aspect_ratio": best_anchor["aspect_ratio"],
                    "best_layer": best_anchor.get(
                        "layer",
                        np.nan,
                    ),
                    "best_level": best_anchor.get(
                        "level",
                        np.nan,
                    ),
                    "best_stride": best_anchor.get(
                        "stride",
                        np.nan,
                    ),
                }
            )

        return pd.DataFrame(match_rows)


    def compute_coverage(
        match_df,
    ):
        thresholds = [
            0.3,
            0.4,
            0.5,
            0.6,
            0.75,
        ]

        coverage_rows = []

        for threshold in thresholds:
            coverage_rows.append(
                {
                    "threshold": threshold,
                    "coverage": (match_df.best_iou >= threshold).mean(),
                }
            )

        return pd.DataFrame(coverage_rows)


    def compute_feature_cell_occupancy(
        match_df,
    ):

        occupancy_df = match_df.copy()

        occupancy_df["width_cells"] = (
            occupancy_df.box_width / occupancy_df.best_stride
        )

        occupancy_df["height_cells"] = (
            occupancy_df.box_height / occupancy_df.best_stride
        )

        occupancy_df["area_cells"] = (
            occupancy_df.width_cells * occupancy_df.height_cells
        )

        return occupancy_df

    return (
        compute_anchor_matches,
        compute_coverage,
        generate_fpn_anchor_shapes,
        generate_ssd_anchor_shapes,
    )


@app.cell
def _(
    compute_anchor_matches,
    compute_coverage,
    cropped_weed_stats_df,
    generate_ssd_anchor_shapes,
):
    ssd_anchor_shapes_df = generate_ssd_anchor_shapes(
        image_size=320,
        min_scale=0.01,
        max_scale=0.25,
        num_layers=6,
        aspect_ratios=[
            1.0,
            2.0,
            0.5,
        ],
    )

    ssd_match_df = compute_anchor_matches(
        cropped_weed_stats_df,
        ssd_anchor_shapes_df,
        width_column="bbox_width_crop",
        height_column="bbox_height_crop",
    )

    ssd_coverage_df = compute_coverage(
        ssd_match_df
    )
    return


@app.cell
def _(
    compute_anchor_matches,
    compute_coverage,
    cropped_weed_stats_df,
    generate_fpn_anchor_shapes,
):
    fpn_anchor_shapes_df = (
        generate_fpn_anchor_shapes(
            image_size=320,
            min_level=3,
            max_level=7,
            anchor_scale=1.5,
            aspect_ratios=[
                1.0,
                2.0,
                0.5,
            ],
            scales_per_octave=1,
        )
    )

    fpn_match_df = compute_anchor_matches(
        cropped_weed_stats_df,
        fpn_anchor_shapes_df,
        width_column="bbox_width_crop",
        height_column="bbox_height_crop",
    )

    fpn_coverage_df = compute_coverage(
        fpn_match_df
    )
    return


@app.cell
def _(fpn_cov50, pd, ssd_cov50):
    comparison_df = pd.DataFrame(
        [
            {
                "model": "SSD",
                "coverage": ssd_cov50,
            },
            {
                "model": "FPN",
                "coverage": fpn_cov50,
            },
        ]
    )
    return


@app.cell
def _(
    fpn_iou_chart,
    fpn_summary,
    mo,
    scatter_chart,
    ssd_iou_chart,
    ssd_summary,
    summary_cards,
    width_chart,
):
    dataset_tab = mo.vstack([
        summary_cards,
        width_chart,
        scatter_chart,
    ])

    ssd_tab = mo.vstack([
        ssd_summary,
        ssd_iou_chart,
    ])

    fpn_tab = mo.vstack([
        fpn_summary,
        fpn_iou_chart,
    ])

    mo.ui.tabs(
        {
            "Dataset": dataset_tab,
            "SSD": ssd_tab,
            "FPN": fpn_tab,
        }
    )
    return


if __name__ == "__main__":
    app.run()
