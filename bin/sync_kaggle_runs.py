#!/usr/bin/env python3

"""
Pull a config's finetune + QAT Kaggle kernel outputs and merge their manifests.

Each config is trained as independent Kaggle notebooks that each publish only
their own slice of the experiment:

    finetune  ->  manifest.json        + ptq/
    qat0      ->  manifest.qat0.json   + qat0/
    qat1      ->  manifest.qat1.json   + qat1/
    qat2      ->  manifest.qat2.json   + qat2/
    qat3      ->  manifest.qat3.json   + qat3/

This script downloads each kernel's output into one local config directory (the
finetune's ``manifest.json`` and the qatN ``manifest.qatN.json`` fragments don't
collide, and ``ptq/`` / ``qatN/`` land side by side), then folds every fragment
into the finetune's full ``manifest.json`` -- the same stage/artifact/result
merge as ``ExperimentManifest.merge``, but idempotent (re-merging an already
-merged stage overwrites instead of raising), so it is safe to re-run.

Kaggle slug convention (override per stage with --slug if a kernel was titled
differently): ``<owner>/<config '_'->'-' lowercased>-<stage>``, e.g.
``freimutdiener/ssd-mn2-sc-phenobench-320-qat0``.

Usage:
    bin/sync_kaggle_runs.py ssd-mn2_sc_phenobench_320
    bin/sync_kaggle_runs.py ssd-mn2_sc_phenobench_320 --dest artifacts/tf/ssd-mn2_sc_phenobench_320
    bin/sync_kaggle_runs.py <config> --no-download        # re-merge what's on disk
    bin/sync_kaggle_runs.py <config> --stages finetune,qat0,qat2
    bin/sync_kaggle_runs.py <config> --per-channel        # the qatN_per-channel kernels
"""

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
DEFAULT_STAGES = ["finetune", "qat0", "qat1", "qat2", "qat3"]
ARTIFACTS_TF = Path(__file__).resolve().parent.parent / "artifacts" / "tf"


def stage_manifest_name(stage: str) -> str:
    """Fragment filename a stage's notebook publishes."""
    return "manifest.json" if stage == "finetune" else f"manifest.{stage}.json"


def kernel_slug(owner: str, config: str, stage: str) -> str:
    """``<owner>/<config-and-stage as a kaggle slug>``.

    Kaggle slugs are lowercase with hyphens only, so the whole ``config-stage``
    body is hyphenated -- including a per-channel stage like ``qat2_per-channel``
    -> ``...-qat2-per-channel``.
    """
    body = f"{config}-{stage}".replace("_", "-").lower()
    return f"{owner}/{body}"


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def download_kernel(slug: str, dest: Path, *, force: bool, quiet: bool) -> bool:
    """
    Download one kernel's output into ``dest`` (merged into the tree).

    ``kaggle kernels output`` downloads the output files individually (it does
    not zip, unlike ``kaggle datasets download``), but we still detect and
    extract a ``*.zip`` defensively in case that ever changes. The kernel
    ``*.log`` is skipped. Returns False (with a warning) if the kernel can't be
    fetched -- e.g. a qatN run that hasn't been published yet.
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


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------


def merge_fragments(dest: Path, stages: list[str]) -> ExperimentManifest:
    """
    Fold every present qatN fragment into the finetune ``manifest.json``.

    Mirrors ``ExperimentManifest.merge`` (stages + artifact files + results) but
    is idempotent: an already-merged stage is overwritten rather than raising,
    and artifact files are de-duplicated, so re-running after a fresh download
    -- or with --no-download -- converges to the same result.
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def sync_config(config: str, args: argparse.Namespace) -> None:
    dest = Path(args.dest) if args.dest else ARTIFACTS_TF / config
    dest.mkdir(parents=True, exist_ok=True)

    slug_overrides = dict(args.slug or [])
    stages = args.stages

    if args.per_channel:
        # Target the per-channel QAT kernels/folders: each non-finetune stage
        # gets a "_per-channel" suffix (separate Kaggle notebooks publish
        # qatN_per-channel/ + manifest.qatN_per-channel.json). finetune is the
        # shared fp32 base, so it stays as-is. Run once without and once with
        # --per-channel into the same dest to collect both into one manifest.
        stages = [
            s if (s == "finetune" or s.endswith("_per-channel")) else f"{s}_per-channel"
            for s in stages
        ]

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
        description="Sync + merge a config's finetune/QAT Kaggle outputs.",
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="config slug(s), e.g. ssd-mn2_sc_phenobench-320",
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
        help="comma-separated stages (default: finetune,qat0,qat1,qat2,qat3)",
    )
    parser.add_argument(
        "--slug",
        type=parse_slug,
        action="append",
        metavar="STAGE=OWNER/SLUG",
        help="override a stage's kernel slug (repeatable)",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help=(
            "target the per-channel QAT kernels: suffix each non-finetune stage "
            "with '_per-channel' (qatN -> qatN_per-channel). Run with and without "
            "to collect both per-tensor and per-channel into one manifest."
        ),
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
        help="keep the manifest.qatN.json fragments after merging",
    )
    parser.add_argument("--quiet", action="store_true", help="quiet kaggle download")
    args = parser.parse_args(argv)

    if args.dest and len(args.configs) > 1:
        parser.error("--dest is only valid with a single config")

    for config in args.configs:
        sync_config(config, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
