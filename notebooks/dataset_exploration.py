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

    for image_index, sample in tqdm(
        enumerate(phenobench_train_dataset),
        total=len(phenobench_train_dataset),
        desc="Extracting weed statistics",
    ):
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
                    "image_index": image_index,
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "bbox_area": bbox_area,
                    "mask_area": mask_area,
                    "aspect_ratio": (bbox_width / bbox_height),
                    "fill_ratio": (mask_area / bbox_area),
                    "width_frac": (bbox_width / image_width),
                    "height_frac": (bbox_height / image_height),
                    "area_frac": (bbox_area / (image_width * image_height)),
                    "elongation": np.maximum(
                        (bbox_width / bbox_height),
                        1 / (bbox_width / bbox_height),
                    ),
                }
            )

    weed_stats_df = pd.DataFrame(weed_rows)

    image_stats_df = pd.DataFrame(image_rows)
    return image_stats_df, weed_stats_df


@app.cell(hide_code=True)
def _(mo):
    target_resolution_ui = mo.ui.number(
        start=128,
        stop=1024,
        step=32,
        value=320,
    )

    crop_area_ui = mo.ui.number(
        start=0.2,
        stop=1.0,
        step=0.05,
        value=0.6,
    )
    return crop_area_ui, target_resolution_ui


@app.cell(hide_code=True)
def _(mo, weed_stats_df):
    median_width = weed_stats_df.bbox_width.median()

    median_height = weed_stats_df.bbox_height.median()

    summary_cards = mo.hstack(
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
    return (summary_cards,)


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
            "bbox_area_target": weed_stats_df.area_frac
            * target_resolution_value**2,
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
            "bbox_area_crop": resized_weed_stats_df.bbox_area_target
            / crop_area_ui.value,
            "area_frac_crop": weed_stats_df.area_frac / crop_area_ui.value,
            "ssd_scale_crop": np.sqrt(
                weed_stats_df.area_frac / crop_area_ui.value
            ),
        }
    )

    cropped_weed_stats_df
    return cropped_weed_stats_df, target_resolution_value


@app.cell(hide_code=True)
def _(alt, cropped_weed_stats_df):
    weed_width = (
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
    return


@app.cell
def _(alt, cropped_weed_stats_df):
    width_chart = (
        alt.Chart(
            cropped_weed_stats_df
        )
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
    return (width_chart,)


@app.cell(hide_code=True)
def _(alt, cropped_weed_stats_df):
    scatter_chart = (
        alt.Chart(
            cropped_weed_stats_df
        )
        .mark_circle(
            opacity=0.05,
            size=20,
        )
        .encode(
            x="bbox_width_crop:Q",
            y="bbox_height_crop:Q",
        )
        .properties(
            title="Weed width vs height",
        )
        .interactive()
    )
    return (scatter_chart,)


@app.cell
def _(alt, weed_stats_df):

    weed_per_aspect_ratio = alt.Chart(
        weed_stats_df
    ).mark_bar().encode(
        alt.X(
            "aspect_ratio:Q",
            bin=alt.Bin(maxbins=100),
        ),
        y="count()",
    )
    return (weed_per_aspect_ratio,)


@app.cell
def _(alt, image_stats_df):
    weed_count_per_image = alt.Chart(
        image_stats_df
    ).mark_bar().encode(
        alt.X(
            "weed_count:Q",
            bin=alt.Bin(maxbins=50),
        ),
        y="count()",
    )
    return (weed_count_per_image,)


@app.cell(hide_code=True)
def _(mo, phenobench_train_dataset):
    image_index_ui = mo.ui.number(
        start=0,
        stop=len(
            phenobench_train_dataset
        ) - 1,
        value=0,
    )
    return (image_index_ui,)


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


@app.cell(hide_code=True)
def _(mo):
    ssd_min_scale_ui = mo.ui.number(
        start=0.005,
        stop=0.1,
        step=0.005,
        value=0.01,
        label="Min scale",
    )

    ssd_max_scale_ui = mo.ui.number(
        start=0.05,
        stop=0.5,
        step=0.01,
        value=0.25,
        label="Max scale",
    )

    ssd_num_layers_ui = mo.ui.number(
        start=3,
        stop=8,
        step=1,
        value=6,
        label="Layers",
    )

    ssd_aspect_ratios_ui = mo.ui.multiselect(
        options=[
            0.33,
            0.5,
            0.75,
            1.0,
            1.5,
            2.0,
            3.0,
        ],
        value=[
            1.0,
            2.0,
            0.5,
        ],
        label="Aspect ratios",
    )
    return (
        ssd_aspect_ratios_ui,
        ssd_max_scale_ui,
        ssd_min_scale_ui,
        ssd_num_layers_ui,
    )


@app.cell
def _(
    mo,
    ssd_aspect_ratios_ui,
    ssd_max_scale_ui,
    ssd_min_scale_ui,
    ssd_num_layers_ui,
):
    ssd_anchor_controls = mo.vstack([
        mo.md("## SSD Anchor Configuration"),

        mo.hstack([
            ssd_min_scale_ui,
            mo.md(f"`{ssd_min_scale_ui.value:.3f}`"),
        ]),

        mo.hstack([
            ssd_max_scale_ui,
            mo.md(f"`{ssd_max_scale_ui.value:.3f}`"),
        ]),

        mo.hstack([
            ssd_num_layers_ui,
            mo.md(f"`{ssd_num_layers_ui.value}`"),
        ]),

        ssd_aspect_ratios_ui,

        mo.hstack([
            mo.stat(
                str(
                    len(ssd_aspect_ratios_ui.value)
                    * ssd_num_layers_ui.value
                ),
                "Anchor count",
            ),
            mo.stat(
                (
                    f"{ssd_min_scale_ui.value:.3f}"
                    " → "
                    f"{ssd_max_scale_ui.value:.3f}"
                ),
                "Scale range",
            ),
        ]),
    ])
    return (ssd_anchor_controls,)


@app.cell
def _(
    compute_anchor_matches,
    compute_coverage,
    cropped_weed_stats_df,
    generate_ssd_anchor_shapes,
    np,
    pd,
    ssd_aspect_ratios_ui,
    ssd_max_scale_ui,
    ssd_min_scale_ui,
    ssd_num_layers_ui,
    target_resolution_value,
):
    ssd_anchor_shapes_df = generate_ssd_anchor_shapes(
        image_size=target_resolution_value,
        min_scale=ssd_min_scale_ui.value,
        max_scale=ssd_max_scale_ui.value,
        num_layers=ssd_num_layers_ui.value,
        aspect_ratios=sorted(
            ssd_aspect_ratios_ui.value
        ),
    )

    ssd_match_df = compute_anchor_matches(
        cropped_weed_stats_df,
        ssd_anchor_shapes_df,
        width_column="bbox_width_crop",
        height_column="bbox_height_crop",
    )

    ssd_match_df = pd.DataFrame(
        {
            **ssd_match_df,
            "object_aspect_ratio":
                cropped_weed_stats_df.bbox_width_crop
                / cropped_weed_stats_df.bbox_height_crop,
            "ssd_scale_crop": np.sqrt(cropped_weed_stats_df.area_frac_crop),
        }
    )

    ssd_coverage_df = compute_coverage(
        ssd_match_df
    )
    return ssd_anchor_shapes_df, ssd_coverage_df, ssd_match_df


@app.cell(hide_code=True)
def _(alt, pd, ssd_match_df, ssd_max_scale_ui, ssd_min_scale_ui):
    "Weed Objects within scale Limits"

    anchor_range_df = pd.DataFrame(
        {
            "x": [
                ssd_min_scale_ui.value,
                ssd_max_scale_ui.value,
            ]
        }
    )

    histogram_chart = (
        alt.Chart(ssd_match_df)
        .mark_bar()
        .encode(
            alt.X(
                "ssd_scale_crop:Q",
                bin=alt.Bin(
                    step=0.001
                ),
                scale=alt.Scale(
                    domain=[0, 0.4]
                ),
            ),
            y="count()",
        )
        .properties(
            width=600,
            title="Weed Objects within scale Limits",
        )
    )

    rule_chart = alt.Chart(anchor_range_df).mark_rule(size=3).encode(x="x:Q")

    weeds_with_limits = histogram_chart + rule_chart
    return (weeds_with_limits,)


@app.cell
def _(mo):
    fpn_min_level_ui = mo.ui.number(
        start=1,
        stop=8,
        step=1,
        value=3,
        label="Min level",
    )

    fpn_max_level_ui = mo.ui.number(
        start=1,
        stop=8,
        step=1,
        value=7,
        label="Max level",
    )

    fpn_anchor_scale_ui = mo.ui.number(
        start=0.25,
        stop=8.0,
        step=0.25,
        value=1.5,
        label="Anchor scale",
    )

    fpn_scales_per_octave_ui = mo.ui.number(
        start=1,
        stop=4,
        step=1,
        value=1,
        label="Scales / octave",
    )

    fpn_aspect_ratios_ui = mo.ui.multiselect(
        options=[
            0.33,
            0.5,
            0.75,
            1.0,
            1.5,
            2.0,
            3.0,
        ],
        value=[
            1.0,
            2.0,
            0.5,
        ],
        label="Aspect ratios",
    )
    return (
        fpn_anchor_scale_ui,
        fpn_aspect_ratios_ui,
        fpn_max_level_ui,
        fpn_min_level_ui,
        fpn_scales_per_octave_ui,
    )


@app.cell
def _(
    fpn_anchor_scale_ui,
    fpn_aspect_ratios_ui,
    fpn_max_level_ui,
    fpn_min_level_ui,
    fpn_scales_per_octave_ui,
    mo,
):
    anchor_count = (
        (fpn_max_level_ui.value
         - fpn_min_level_ui.value
         + 1)
        * len(fpn_aspect_ratios_ui.value)
        * fpn_scales_per_octave_ui.value
    )

    fpn_anchor_controls = mo.vstack(
        [
            mo.md("## FPN Anchor Configuration"),
            mo.hstack(
                [
                    fpn_min_level_ui,
                    fpn_max_level_ui,
                ]
            ),
            mo.hstack(
                [
                    fpn_anchor_scale_ui,
                    fpn_scales_per_octave_ui,
                ]
            ),
            fpn_aspect_ratios_ui,
            mo.hstack(
                [
                    mo.stat(
                        str(anchor_count),
                        "Total anchor shapes",
                    ),
                    mo.stat(
                        str(fpn_max_level_ui.value - fpn_min_level_ui.value + 1),
                        "Levels",
                    ),
                    mo.stat(
                        f"{fpn_anchor_scale_ui.value:.2f}",
                        "Anchor scale",
                    ),
                    mo.stat(
                        str(len(fpn_aspect_ratios_ui.value)),
                        "Aspect ratios",
                    ),
                ]
            ),
        ]
    )
    return (fpn_anchor_controls,)


@app.cell
def _(
    compute_anchor_matches,
    compute_coverage,
    cropped_weed_stats_df,
    fpn_anchor_scale_ui,
    fpn_aspect_ratios_ui,
    fpn_max_level_ui,
    fpn_min_level_ui,
    fpn_scales_per_octave_ui,
    generate_fpn_anchor_shapes,
    target_resolution_value,
):
    fpn_anchor_shapes_df = (
        generate_fpn_anchor_shapes(
            image_size=target_resolution_value,
            min_level=fpn_min_level_ui.value,
            max_level=fpn_max_level_ui.value,
            anchor_scale=fpn_anchor_scale_ui.value,
            aspect_ratios=sorted(
                fpn_aspect_ratios_ui.value
            ),
            scales_per_octave=(
                fpn_scales_per_octave_ui.value
            ),
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
    return fpn_coverage_df, fpn_match_df


@app.cell(hide_code=True)
def _(mo, ssd_coverage_df, ssd_match_df):
    ssd_cov50 = (
        ssd_coverage_df
        .query("threshold == 0.5")
        .coverage
        .iloc[0]
    )

    ssd_summary = mo.hstack([
        mo.stat(
            f"{ssd_cov50:.1%}",
            "Coverage @0.5",
        ),
        mo.stat(
            f"{ssd_match_df.best_iou.median():.2f}",
            "Median IoU",
        ),
        mo.stat(
            f"{ssd_match_df.best_iou.quantile(0.9):.2f}",
            "P90 IoU",
        ),
    ])
    return ssd_cov50, ssd_summary


@app.cell(hide_code=True)
def _(alt, ssd_match_df):
    ssd_iou_chart = (
        alt.Chart(
            ssd_match_df
        )
        .mark_bar()
        .encode(
            alt.X(
                "best_iou:Q",
                bin=alt.Bin(maxbins=50),
            ),
            y="count()",
        )
        .properties(
            title="SSD Best Anchor IoU",
        )
    )
    return (ssd_iou_chart,)


@app.cell(hide_code=True)
def _(fpn_coverage_df, fpn_match_df, mo):
    fpn_cov50 = (
        fpn_coverage_df
        .query("threshold == 0.5")
        .coverage
        .iloc[0]
    )

    fpn_summary = mo.hstack([
        mo.stat(
            f"{fpn_cov50:.1%}",
            "Coverage @0.5",
        ),
        mo.stat(
            f"{fpn_match_df.best_iou.median():.2f}",
            "Median IoU",
        ),
        mo.stat(
            f"{fpn_match_df.best_iou.quantile(0.9):.2f}",
            "P90 IoU",
        ),
    ])
    return fpn_cov50, fpn_summary


@app.cell(hide_code=True)
def _(alt, fpn_match_df):
    fpn_iou_chart = (
        alt.Chart(
            fpn_match_df
        )
        .mark_bar()
        .encode(
            alt.X(
                "best_iou:Q",
                bin=alt.Bin(maxbins=50),
            ),
            y="count()",
        )
        .properties(
            title="FPN Best Anchor IoU",
        )
    )
    return (fpn_iou_chart,)


@app.cell(hide_code=True)
def _(alt, fpn_cov50, pd, ssd_cov50):
    comparison_df = pd.DataFrame([
        {
            "Model": "SSD",
            "Coverage": ssd_cov50,
        },
        {
            "Model": "FPN",
            "Coverage": fpn_cov50,
        },
    ])

    coverage_comparison_chart = (
        alt.Chart(
            comparison_df
        )
        .mark_bar()
        .encode(
            x="Model:N",
            y="Coverage:Q",
        )
        .properties(
            title="Coverage @ IoU 0.5",
        )
    )
    return (coverage_comparison_chart,)


@app.cell(hide_code=True)
def _(alt, cropped_weed_stats_df, ssd_anchor_shapes_df):
    "samples vs anchors"
    weed_chart = (
        alt.Chart(cropped_weed_stats_df)
        .mark_rect()
        .encode(
            x=alt.X(
                "bbox_width_crop:Q",
                bin=alt.Bin(step=1),
            ),
            y=alt.Y(
                "bbox_height_crop:Q",
                bin=alt.Bin(step=1),
            ),
            color="count():Q",
        )
    )

    anchor_chart = (
        alt.Chart(
            ssd_anchor_shapes_df
        )
        .mark_square(
            size=200,
        )
        .encode(
            x=alt.X(
                "anchor_width:Q",
                title="Anchor width (px)",
            ),
            y=alt.Y(
                "anchor_height:Q",
                title="Anchor height (px)",
            ),
            color="layer:N",
            tooltip=[
                "layer",
                "scale",
                "aspect_ratio",
            ],
        )
    )

    samples_vs_anchors = weed_chart + anchor_chart
    return (samples_vs_anchors,)


@app.cell
def _(
    coverage_comparison_chart,
    fpn_anchor_controls,
    fpn_iou_chart,
    fpn_summary,
    mo,
    samples_vs_anchors,
    scatter_chart,
    ssd_anchor_controls,
    ssd_iou_chart,
    ssd_summary,
    summary_cards,
    weed_count_per_image,
    weed_per_aspect_ratio,
    weeds_with_limits,
    width_chart,
):
    dataset_tab = mo.vstack([
        summary_cards,
        mo.hstack([
            weed_count_per_image,
            weed_per_aspect_ratio
        ]),
        mo.hstack([
            width_chart,
            scatter_chart,  
        ]),
    ])

    ssd_tab = mo.vstack([
        ssd_anchor_controls,
        ssd_summary,
        mo.hstack([
            weeds_with_limits
        ]),
        mo.hstack([
            samples_vs_anchors,
            ssd_iou_chart,
        ]),
    ])

    fpn_tab = mo.vstack([
        fpn_anchor_controls,
        fpn_summary,
        fpn_iou_chart,
    ])

    comparison_tab = mo.vstack([
        coverage_comparison_chart,
    ])

    dashboard = mo.ui.tabs(
        {
            "Dataset": dataset_tab,
            "SSD": ssd_tab,
            "FPN": fpn_tab,
            "Comparison": comparison_tab,
        }
    )
    return (dashboard,)


@app.cell(hide_code=True)
def _(alt, mo, pd, ssd_match_df):
    analysis_df = ssd_match_df.copy()

    # --------------------------------------------------
    # Quantile buckets
    # --------------------------------------------------

    analysis_df["size_bucket"] = pd.qcut(
        analysis_df["ssd_scale_crop"],
        q=[
            0.0,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00,
        ],
        labels=[
            "P0-50",
            "P50-75",
            "P75-90",
            "P90-95",
            "P95-99",
            "P99-100",
        ],
    )

    # --------------------------------------------------
    # Summary table
    # --------------------------------------------------

    bucket_summary_df = (
        analysis_df
        .groupby("size_bucket")
        .agg(
            count=("best_iou", "size"),
            median_iou=("best_iou", "median"),
            mean_iou=("best_iou", "mean"),
            coverage_05=(
                "best_iou",
                lambda x: (x >= 0.5).mean(),
            ),
            coverage_075=(
                "best_iou",
                lambda x: (x >= 0.75).mean(),
            ),
            median_scale=(
                "ssd_scale_crop",
                "median",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Scatter
    # --------------------------------------------------

    scatter_df = (
        analysis_df.sample(
            min(
                5000,
                len(analysis_df),
            ),
            random_state=42,
        )
    )

    size_vs_iou_chart = (
        alt.Chart(scatter_df)
        .mark_circle(
            opacity=0.15,
            size=20,
        )
        .encode(
            x=alt.X(
                "ssd_scale_crop:Q",
                title="Object scale",
            ),
            y=alt.Y(
                "best_iou:Q",
                title="Best anchor IoU",
            ),
            color=alt.Color(
                "size_bucket:N",
                legend=None,
            ),
            tooltip=[
                "ssd_scale_crop",
                "best_iou",
            ],
        )
        .properties(
            title="Object size vs best anchor IoU",
            width="container",
            height=400,
        )
    )

    # --------------------------------------------------
    # Coverage by bucket
    # --------------------------------------------------

    coverage_chart = (
        alt.Chart(bucket_summary_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "size_bucket:N",
                title="Size percentile bucket",
            ),
            y=alt.Y(
                "coverage_05:Q",
                title="Coverage @ 0.5",
            ),
            tooltip=[
                "count",
                alt.Tooltip(
                    "coverage_05:Q",
                    format=".1%",
                ),
                alt.Tooltip(
                    "coverage_075:Q",
                    format=".1%",
                ),
                "median_iou",
            ],
        )
        .properties(
            title="Coverage by object size",
            width="container",
            height=300,
        )
    )

    # --------------------------------------------------
    # Summary cards
    # --------------------------------------------------

    largest_5pct_df = (
        analysis_df[
            analysis_df.ssd_scale_crop
            >= analysis_df.ssd_scale_crop.quantile(
                0.95
            )
        ]
    )

    summary_cards_sl = mo.hstack([
        mo.stat(
            f"{(analysis_df.best_iou >= 0.5).mean():.1%}",
            "Overall Coverage @0.5",
        ),
        mo.stat(
            f"{(largest_5pct_df.best_iou >= 0.5).mean():.1%}",
            "Largest 5% Coverage",
        ),
        mo.stat(
            f"{largest_5pct_df.best_iou.median():.2f}",
            "Largest 5% Median IoU",
        ),
    ])

    mo.vstack([
        summary_cards_sl,
        coverage_chart,
        size_vs_iou_chart,
        bucket_summary_df,
    ])
    return


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


@app.cell
def _(dashboard):
    dashboard
    return


@app.cell
def _(alt, ssd_match_df):
    aspect_ratio_usage_df = (
        ssd_match_df
        .best_aspect_ratio
        .value_counts(normalize=True)
        .rename("fraction")
        .reset_index()
        .rename(
            columns={
                "index": "aspect_ratio",
            }
        )
    )

    aspect_ratio_usage_chart = (
        alt.Chart(
            aspect_ratio_usage_df
        )
        .mark_bar()
        .encode(
            x=alt.X(
                "best_aspect_ratio:N",
                title="Winning aspect ratio",
            ),
            y=alt.Y(
                "fraction:Q",
                axis=alt.Axis(format="%"),
                title="Fraction of weeds",
            ),
            tooltip=[
                "best_aspect_ratio",
                alt.Tooltip(
                    "fraction:Q",
                    format=".1%",
                ),
            ],
        )
        .properties(
            title="Anchor Aspect Ratio Usage",
            width="container",
            height=400,
        )
    )

    aspect_ratio_usage_chart
    return


@app.cell
def _(ssd_match_df):
    ssd_match_df.groupby(
        "best_aspect_ratio"
    )["best_iou"].agg(
        ["count", "median", "mean"]
    )
    return


@app.cell
def _(alt, ssd_match_df):
    aspect_ratio_chart = (
        alt.Chart(
            ssd_match_df.sample(
                min(
                    10000,
                    len(ssd_match_df),
                ),
                random_state=42,
            )
        )
        .mark_circle(
            opacity=0.2,
            size=20,
        )
        .encode(
            x=alt.X(
                "object_aspect_ratio:Q",
                title="Object aspect ratio",
            ),
            y=alt.Y(
                "best_iou:Q",
                title="Best IoU",
            ),
            color=alt.Color(
                "best_aspect_ratio:N",
                title="Winning anchor ratio",
            ),
            tooltip=[
                "object_aspect_ratio",
                "best_aspect_ratio",
                "best_iou",
            ],
        )
        .interactive()
    )

    aspect_ratio_chart
    return


@app.cell
def _(alt, ssd_match_df):
    aspect_ratio_heatmap = (
        alt.Chart(ssd_match_df)
        .mark_rect()
        .encode(
            x=alt.X(
                "object_aspect_ratio:Q",
                bin=alt.Bin(step=0.05),
                title="Object aspect ratio",
            ),
            y=alt.Y(
                "best_aspect_ratio:N",
                title="Winning anchor ratio",
            ),
            color=alt.Color(
                "count():Q",
                title="Count",
            ),
        )
        .properties(
            title="Which anchor ratio wins for which object ratio",
            width="container",
            height=300,
        )
    )

    aspect_ratio_heatmap
    return


@app.cell
def _(weed_stats_df):
    most_elongated_weeds_df = (
        weed_stats_df
        .sort_values(
            "elongation",
            ascending=False,
        )
        .head(50)
    )

    most_elongated_weeds_df[
        [
            "image_index",
            "bbox_width",
            "bbox_height",
            "aspect_ratio",
            "elongation",
        ]]
    return


@app.cell(hide_code=True)
def _(image_index_ui, mo, np, phenobench_train_dataset):
    from PIL import ImageDraw

    def draw_weed_boxes(
        sample,
        color="red",
        width=3,
    ):
        image = sample["image"].copy()

        draw = ImageDraw.Draw(image)

        semantics = sample["semantics"]
        instances = sample["plant_instances"]

        weed_mask = semantics == 2

        weed_ids = np.unique(
            instances[weed_mask]
        )

        weed_ids = weed_ids[weed_ids > 0]

        for weed_id in weed_ids:

            instance_mask = (
                (instances == weed_id)
                & weed_mask
            )

            ys, xs = np.where(
                instance_mask
            )

            draw.rectangle(
                [
                    (
                        xs.min(),
                        ys.min(),
                    ),
                    (
                        xs.max(),
                        ys.max(),
                    ),
                ],
                outline=color,
                width=width,
            )

        return image

    inspection_sample = (
        phenobench_train_dataset[
            image_index_ui.value
        ]
    )

    mo.vstack([
        mo.hstack([
            "Image Index in Train Set",
            mo.md(f"**{image_index_ui.value}**"),
            image_index_ui,
        ]),
        draw_weed_boxes(
            inspection_sample
        )
    ])
    return


@app.cell
def _(cropped_weed_stats_df):
    cropped_weed_stats_df[
        [
            "bbox_width_crop",
            "bbox_height_crop",
        ]
    ].describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.50,
            0.90,
            0.95,
            0.99,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
