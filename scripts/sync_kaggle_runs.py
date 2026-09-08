#!/usr/bin/env python3

"""Download Kaggle stage outputs and merge their experiment manifests."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agri_vision_edge.experiment import ExperimentManifest  # noqa: E402

DEFAULT_OWNER = "freimutdiener"
# finetune publishes the fp32 base under finetune/; ptq is the QAT-parity float
# run; qat_per-tensor / qat_per-channel are the per-tensor (i.MX8M Plus) and
# per-channel (i.MX93 Ethos-U) int8 runs. Pulling all four by default collects
# the whole config in one go; a stage whose kernel is not published yet is just
# skipped with a warning.
DEFAULT_STAGES = ["finetune", "ptq", "qat_per-tensor", "qat_per-channel"]
ARTIFACTS_TF = Path(__file__).resolve().parent.parent / "artifacts" / "tf"

# Default SSD matrix: {plain SSD, FPNLite} × {sc, mc} × {untiled, tiled}.
SSD_CONFIGS = [
    "ssd-mn2_sc_phenobench_320",
    "ssd-mn2_mc_phenobench_320",
    "ssd-mn2_sc_phenobench-tiled_320",
    "ssd-mn2_mc_phenobench-tiled_320",
    "ssd-mn2-fpnlite_sc_phenobench_320",
    "ssd-mn2-fpnlite_mc_phenobench_320",
    "ssd-mn2-fpnlite_sc_phenobench-tiled_320",
    "ssd-mn2-fpnlite_mc_phenobench-tiled_320",
]


def stage_manifest_name(stage: str) -> str:
    """Fragment filename a stage's notebook publishes."""
    return "manifest.json" if stage == "finetune" else f"manifest.{stage}.json"


# Kaggle caps a notebook's title (hence its slug body) at 50 characters. When the
# full ``...-qat-per-tensor`` / ``...-qat-per-channel`` body would exceed that,
# the kernel is titled with the abbreviated ``...-qat-pt`` / ``...-qat-pc``
# suffix instead. Internal stage, manifest, and artifact names remain unchanged.
KAGGLE_SLUG_MAX = 50

# Long QAT stage suffix -> abbreviation used only when the slug body would exceed
# ``KAGGLE_SLUG_MAX`` (the config prefix is otherwise short enough to keep the
# full, self-describing stage name).
_SLUG_ABBREVIATIONS = {
    "-qat-per-tensor": "-qat-pt",
    "-qat-per-channel": "-qat-pc",
}


def kernel_slug(owner: str, config: str, stage: str) -> str:
    """``<owner>/<config-and-stage as a kaggle slug>``.

    Kaggle slugs are lowercase with hyphens. This includes the QAT stage tokens
    ``qat_per-tensor`` and ``qat_per-channel``. If
    that body would exceed Kaggle's 50-char title cap, the long suffix is
    abbreviated (``-qat-pt`` / ``-qat-pc``) to match how the kernel had to be
    titled (e.g. the FPNLite tiled configs).
    """
    body = f"{config}-{stage}".replace("_", "-").lower()
    if len(body) > KAGGLE_SLUG_MAX:
        for suffix, abbr in _SLUG_ABBREVIATIONS.items():
            if body.endswith(suffix):
                body = body[: -len(suffix)] + abbr
                break
    return f"{owner}/{body}"


# Download


def download_kernel(slug: str, dest: Path, *, force: bool, quiet: bool) -> bool:
    """
    Download one kernel's output into ``dest`` (merged into the tree).

    ``kaggle kernels output`` downloads the output files individually (it does
    not zip, unlike ``kaggle datasets download``), but we still detect and
    extract a ``*.zip`` defensively in case that ever changes. The kernel
    ``*.log`` is skipped. Returns False with a warning when the kernel cannot be
    fetched.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cmd = ["kaggle", "kernels", "output", slug, "-p", str(tmp_dir)]
        if force:
            cmd.append("-o")
        if quiet:
            cmd.append("-q")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            msg = (result.stderr or result.stdout).strip().splitlines()
            print(f"  ! skip {slug}: {msg[-1] if msg else 'download failed'}")
            return False

        for zip_path in tmp_dir.rglob("*.zip"):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)
            zip_path.unlink()

        copied = 0
        for src in tmp_dir.rglob("*"):
            if src.is_dir() or src.suffix == ".log":
                continue
            rel = src.relative_to(tmp_dir)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied += 1

        print(f"  + {slug}: {copied} files -> {dest}")
        return True


# Merge


def merge_fragments(dest: Path, stages: list[str]) -> ExperimentManifest:
    """
    Fold every present ptq/qat fragment into the finetune ``manifest.json``.

    Mirrors the stage, artifact, and result handling in
    ``ExperimentManifest.merge``. Existing stages are overwritten and artifact
    files are deduplicated, making repeated runs idempotent.
    """
    base_path = dest / "manifest.json"
    if not base_path.exists():
        raise FileNotFoundError(
            f"no finetune manifest at {base_path} -- download the finetune "
            "output first (drop --no-download)"
        )

    base = ExperimentManifest.load(base_path)
    base_files = base.data.setdefault("artifacts", {}).setdefault("files", [])
    seen = {(f.get("path"), f.get("stage")) for f in base_files}

    for stage in stages:
        if stage == "finetune":
            continue
        frag_path = dest / stage_manifest_name(stage)
        if not frag_path.exists():
            continue

        frag = ExperimentManifest.load(frag_path)

        for name, data in frag.data.get("stages", {}).items():
            base.data["stages"][name] = data  # idempotent overwrite

        for artifact in frag.data.get("artifacts", {}).get("files", []):
            key = (artifact.get("path"), artifact.get("stage"))
            if key not in seen:
                base_files.append(artifact)
                seen.add(key)

        base.data.setdefault("results", {}).update(frag.data.get("results", {}))
        print(f"  merged {frag_path.name}: stage(s) {list(frag.data['stages'])}")

    base.save(base_path)
    return base


def print_summary(manifest: ExperimentManifest) -> None:
    stages = manifest.data.get("stages", {})
    print(f"\n  manifest: {len(stages)} stage(s)")
    for name, stage in stages.items():
        best = stage.get("metrics", {}).get("best_metric", {})
        if best:
            print(
                f"    - {name:9s} {best.get('metric_name', '?')}="
                f"{best.get('metric_value', float('nan')):.5f} "
                f"@ step {best.get('step', '?')}"
            )
        else:
            print(f"    - {name}")
    n_files = len(manifest.data.get("artifacts", {}).get("files", []))
    print(f"  artifacts: {n_files} file(s) registered")


# CLI


def sync_config(config: str, args: argparse.Namespace) -> None:
    dest = Path(args.dest) if args.dest else ARTIFACTS_TF / config
    dest.mkdir(parents=True, exist_ok=True)

    slug_overrides = dict(args.slug or [])
    stages = args.stages

    print(f"== {config} -> {dest}")

    if not args.no_download:
        for stage in stages:
            slug = slug_overrides.get(stage) or kernel_slug(args.owner, config, stage)
            download_kernel(slug, dest, force=args.force, quiet=args.quiet)

    manifest = merge_fragments(dest, stages)

    if not args.keep_fragments:
        for stage in stages:
            if stage == "finetune":
                continue
            frag = dest / stage_manifest_name(stage)
            if frag.exists():
                frag.unlink()

    print_summary(manifest)


def parse_slug(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--slug expects stage=owner/slug, got {value!r}"
        )
    stage, slug = value.split("=", 1)
    return stage, slug


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync + merge a config's finetune/PTQ/QAT Kaggle outputs.",
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help=(
            "config slug(s), e.g. ssd-mn2_sc_phenobench_320; "
            "omit to sync all eight SSD configs"
        ),
    )
    parser.add_argument(
        "--owner", default=DEFAULT_OWNER, help="Kaggle owner (default: %(default)s)"
    )
    parser.add_argument(
        "--dest",
        help="local dir (default: artifacts/tf/<config>); only valid with one config",
    )
    parser.add_argument(
        "--stages",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=DEFAULT_STAGES,
        help=(
            "comma-separated stages "
            "(default: finetune,ptq,qat_per-tensor,qat_per-channel)"
        ),
    )
    parser.add_argument(
        "--slug",
        type=parse_slug,
        action="append",
        metavar="STAGE=OWNER/SLUG",
        help="override a stage's kernel slug (repeatable)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="skip Kaggle download; only re-merge what is already on disk",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="pass -o to kaggle (re-download even if up to date)",
    )
    parser.add_argument(
        "--keep-fragments",
        action="store_true",
        help="keep the manifest.<stage>.json fragments after merging",
    )
    parser.add_argument("--quiet", action="store_true", help="quiet kaggle download")
    args = parser.parse_args(argv)

    # No slugs given -> sync the whole eight-config SSD matrix.
    configs = args.configs or SSD_CONFIGS

    if args.dest and len(configs) > 1:
        parser.error("--dest is only valid with a single config")

    for config in configs:
        sync_config(config, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
