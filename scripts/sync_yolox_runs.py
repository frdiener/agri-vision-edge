#!/usr/bin/env python3
"""Download YOLOX Kaggle outputs and normalize them under ``artifacts/onnx``."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiKernelSessionOutputFile,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ONNX = REPO_ROOT / "artifacts" / "onnx"

DEFAULT_OWNER = "freimutdiener"

#: These runs have exactly one stage. There is no PTQ/QAT arm: quantization
#: happens at ONNX -> TFLite conversion time, not during training.
STAGE = "finetune"

#: Kaggle caps notebook titles and slug bodies at 50 characters. The longest
#: YOLOX body is 43 characters, so no abbreviation table is needed.
KAGGLE_SLUG_MAX = 50

#: Mirrors ``gen_yolox_finetune_notebooks.YOLOX_COMMIT``. The kernel log records
#: the commit the run actually cloned; ``sync.json`` reports whether the two
#: agree instead of taking the pin on faith.
YOLOX_COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"

#: The four notebooks from scripts/gen_yolox_finetune_notebooks.py, named in the
#: repository config convention with hyphenated architectures and
#: underscore-separated fields. ``parse_run_name`` and ``benchmark_all.sh``
#: consume this form, so downstream TFLite files must preserve it.
CONFIGS = [
    "yolox-nano_mc_phenobench_320",
    "yolox-nano_sc_phenobench_320",
    "yolox-nano_mc_phenobench-tiled_320",
    "yolox-nano_sc_phenobench-tiled_320",
]


class SyncError(RuntimeError):
    """A config could not be synced; reported per config, never fatal."""


# Kaggle


def kernel_slug(owner: str, config: str) -> str:
    """``<owner>/<config-and-stage as a kaggle slug>``."""

    body = f"{config}-{STAGE}".replace("_", "-").lower()

    assert len(body) <= KAGGLE_SLUG_MAX, (
        f"slug body {body!r} is {len(body)} chars, over Kaggle's "
        f"{KAGGLE_SLUG_MAX}-char title cap -- the kernel must have been titled "
        "with an abbreviation, which this script does not model"
    )

    return f"{owner}/{body}"


def build_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    return api


def list_session_output(
    api,
    owner: str,
    slug: str,
    *,
    max_pages: int = 50,
) -> tuple[list[ApiKernelSessionOutputFile], str]:
    """
    ``(files, log)`` for a kernel's last session.

    Follow ``next_page_token`` because ``KaggleApi.kernels_output`` reads only
    the first page. Smaller page sizes would otherwise hide published ONNX files.
    """

    from kagglesdk.kernels.types.kernels_api_service import (
        ApiListKernelSessionOutputRequest,
    )

    files: list[ApiKernelSessionOutputFile] = []
    log = ""
    token = None

    with api.build_kaggle_client() as client:
        for _ in range(max_pages):
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            request.page_size = 500

            if token:
                request.page_token = token

            response = client.kernels.kernels_api_client.list_kernel_session_output(
                request
            )

            files.extend(response.files)

            if not log:
                log = response.log or ""

            token = response.next_page_token

            if not token:
                break
        else:
            raise SyncError(
                f"{slug}: still paging after {max_pages} pages; refusing to loop"
            )

    if not files:
        raise SyncError(
            f"{slug}: session published no output files (has it been run?)"
        )

    return files, log


# Selection


def select_files(
    files: list[ApiKernelSessionOutputFile],
    config: str,
) -> tuple[str, dict[str, ApiKernelSessionOutputFile]]:
    """
    ``(exp_name, {destination relative path: file entry})``.

    The ONNX filename supplies YOLOX's ``EXP_NAME``. Deriving the name supports
    notebook changes and identifies the experiment file among stock files in
    ``exps/example/custom/``.
    """

    by_name = {entry.file_name: entry for entry in files}

    graphs = [
        name
        for name in by_name
        if name.endswith(".onnx") and "/outputs/" in f"/{name}"
    ]

    if len(graphs) != 1:
        raise SyncError(
            f"expected exactly one outputs/*.onnx, found {len(graphs)}: "
            f"{sorted(graphs)[:5]}"
        )

    onnx_name = graphs[0]
    exp_name = Path(onnx_name).stem

    selected: dict[str, ApiKernelSessionOutputFile] = {
        onnx_destination_name(config): by_name[onnx_name]
    }

    run_prefix = f"YOLOX_outputs/{exp_name}/"

    required = {
        "best_ckpt.pth": f"{run_prefix}best_ckpt.pth",
        "exp.py": f"exps/example/custom/{exp_name}.py",
    }
    optional = {
        "train_log.txt": f"{run_prefix}train_log.txt",
        "val_log.txt": f"{run_prefix}val_log.txt",
    }

    for destination, needle in {**required, **optional}.items():
        matches = [name for name in by_name if name.endswith(needle)]

        if len(matches) == 1:
            selected[destination] = by_name[matches[0]]
        elif destination in required:
            raise SyncError(
                f"expected exactly one {needle!r}, found {len(matches)}"
            )

    # TensorBoard event files: however many the run wrote, flattened one level.
    for name, entry in by_name.items():
        if f"{run_prefix}tensorboard/" in name:
            selected[f"tensorboard/{Path(name).name}"] = entry

    return exp_name, selected


def onnx_destination_name(config: str) -> str:
    """Return the repository-convention ONNX filename for ``config``."""

    return f"{config}.onnx"


# Log-derived manifest data

_BEST_AP = re.compile(r"best AP is\s+([\d.]+)")
_AP_ALL = re.compile(
    r"IoU=0\.50:0\.95 \| area=\s*all \| maxDets=100 \] = ([-\d.]+)"
)
_LOGGED_COMMIT = re.compile(r"YOLOX:.*?@\s*([0-9a-f]{7,40})")

#: The notebook prints these banners before each of its three post-training
#: evaluations in section 8. These labels distinguish the three otherwise
#: identical pycocotools dumps.
_EVAL_BANNERS = {
    "val_do_not_care": "val / do-not-care",
    "val_strict": "val / strict",
    "test_do_not_care": "test / do-not-care",
}


def _parse_per_class(text: str, header: str) -> dict[str, float]:
    """
    Parse the last ``per class AP:`` / ``per class AR:`` markdown table.

    Both headers are present and their tables are formatted identically, so the
    header string is matched exactly. Reading the wrong one is easy and silent:
    for the mc untiled run, per-class AP is crop 72.380 / weed 32.585 while
    per-class AR is crop 75.086 / weed 41.335.
    """

    start = text.rfind(header)

    if start < 0:
        return {}

    result: dict[str, float] = {}

    for line in text[start + len(header) :].splitlines()[1:]:
        line = line.strip()

        if not line.startswith("|"):
            if result:
                break
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]

        # Separator row (|:---|:---|); the header row falls out below instead,
        # because float("AP") raises.
        if any(set(cell) <= set("-: ") for cell in cells if cell):
            continue

        for index in range(0, len(cells) - 1, 2):
            try:
                result[cells[index]] = float(cells[index + 1])
            except ValueError:
                pass

    return result


def parse_train_log(text: str) -> dict[str, object]:
    best = _BEST_AP.findall(text)

    return {
        "best_ap": float(best[-1]) if best else None,
        "final_per_class_ap": _parse_per_class(text, "per class AP:"),
        "final_per_class_ar": _parse_per_class(text, "per class AR:"),
    }


def kernel_log_stdout(log_text: str) -> str:
    """
    Reassemble the kernel log's stdout stream.

    The log is a JSON array streamed one record per line, so lines carry a
    leading comma and the outer brackets are not reliably present.
    """

    chunks: list[str] = []

    for line in log_text.splitlines():
        line = line.strip().lstrip(",").rstrip(",")

        if not line.startswith("{"):
            continue

        try:
            record = json.loads(line)
        except ValueError:
            continue

        if isinstance(record, dict) and "data" in record:
            chunks.append(str(record["data"]))

    return "".join(chunks)


@dataclass
class KernelLogFacts:
    """What the kernel's stdout still knows that no manifest recorded."""

    #: Banner label -> COCO AP for each of the notebook's three post-training
    #: evaluations. Empty when the banners are absent (an older notebook run).
    evaluations_ap: dict[str, float]

    #: Abbreviated commit the run actually cloned, or ``None``.
    yolox_commit_logged: str | None


def parse_kernel_log(log_text: str) -> KernelLogFacts:
    """
    Post-training evaluations and the cloned YOLOX commit, best effort.

    Missing supplementary provenance produces ``None`` or ``{}``. The raw log
    is always stored.
    """

    stdout = kernel_log_stdout(log_text)

    evaluations: dict[str, float] = {}

    for key, banner in _EVAL_BANNERS.items():
        index = stdout.find(banner)

        if index < 0:
            continue

        match = _AP_ALL.search(stdout, index)

        if match:
            evaluations[key] = float(match.group(1))

    logged = _LOGGED_COMMIT.search(stdout)

    return KernelLogFacts(
        evaluations_ap=evaluations,
        yolox_commit_logged=logged.group(1) if logged else None,
    )


# Download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def download(url: str, destination: Path) -> int:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    # Written via a temporary neighbour so an interrupted transfer cannot leave
    # a short file that a later --no-force run would accept as complete.
    partial = destination.with_suffix(destination.suffix + ".partial")
    written = 0

    with partial.open("wb") as handle:
        for block in response.iter_content(chunk_size=1 << 20):
            handle.write(block)
            written += len(block)

    partial.replace(destination)

    return written


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024

    return f"{size:.1f} GB"


# Sync


def sync_config(config: str, args: argparse.Namespace) -> bool:
    slug = args.slug_overrides.get(config) or kernel_slug(args.owner, config)
    destination = (Path(args.dest) if args.dest else ARTIFACTS_ONNX) / config

    print(f"== {config}")
    print(f"   kernel: {slug}")

    api = build_api()
    files, log_text = list_session_output(api, *slug.split("/", 1))

    exp_name, selected = select_files(files, config)

    print(f"   exp_name: {exp_name}")
    print(f"   published {len(files)} file(s), selected {len(selected)}")

    if args.dry_run:
        for relative in sorted(selected):
            print(f"     would fetch  {relative:<28} <- {selected[relative].file_name}")
        print(f"     would write  {'sync.json':<28} + kernel_log.jsonl")
        return True

    destination.mkdir(parents=True, exist_ok=True)

    records = []
    total = 0

    for relative in sorted(selected):
        target = destination / relative

        if target.exists() and not args.force:
            size = target.stat().st_size
            print(f"     = {relative:<28} {human(size):>10}  (present)")
        else:
            size = download(selected[relative].url, target)
            total += size
            print(f"     + {relative:<28} {human(size):>10}")

        records.append(
            {
                "path": relative,
                "source": selected[relative].file_name,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )

    # Use JSONL because *.log is ignored and this preserves all evaluations.
    log_path = destination / "kernel_log.jsonl"
    log_path.write_text(log_text, encoding="utf-8")

    train_log = destination / "train_log.txt"
    metrics = (
        parse_train_log(train_log.read_text(encoding="utf-8", errors="replace"))
        if train_log.is_file()
        else {}
    )
    from_log = parse_kernel_log(log_text)

    logged_commit = from_log.yolox_commit_logged

    record = {
        "schema": 1,
        "note": (
            "Provenance reconstructed by scripts/sync_yolox_runs.py from the "
            "kernel's own logs. These notebooks publish no ExperimentManifest, "
            "unlike the SSD arm."
        ),
        "config": config,
        "kernel": slug,
        "exp_name": exp_name,
        "onnx_source_name": f"{exp_name}.onnx",
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "yolox_commit_declared": YOLOX_COMMIT,
        "yolox_commit_logged": logged_commit,
        # The log prints an abbreviated hash, so compare on the shorter prefix.
        "yolox_commit_matches": (
            YOLOX_COMMIT.startswith(logged_commit) if logged_commit else None
        ),
        "metrics": metrics,
        "evaluations_ap": from_log.evaluations_ap,
        "kernel_files_published": len(files),
        "kernel_files_kept": len(selected),
        "files": records,
    }

    (destination / "sync.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )

    print(f"   downloaded {human(total)} -> {destination}")

    if metrics.get("best_ap") is not None:
        print(f"   best AP (training): {metrics['best_ap']}")

    for key, value in from_log.evaluations_ap.items():
        print(f"   eval AP {key:<18} {value}")

    if record["yolox_commit_matches"] is False:
        print(
            f"   ! YOLOX commit mismatch: log says {logged_commit}, "
            f"generator pins {YOLOX_COMMIT[:12]}"
        )

    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sync the YOLOX-Nano finetune kernels' ONNX + checkpoints into "
            "artifacts/onnx/."
        ),
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help=f"config slug(s); omit to sync all {len(CONFIGS)}",
    )
    parser.add_argument(
        "--owner", default=DEFAULT_OWNER, help="Kaggle owner (default: %(default)s)"
    )
    parser.add_argument(
        "--dest",
        help="destination root (default: artifacts/onnx); a <config>/ dir is made under it",
    )
    parser.add_argument(
        "--slug",
        action="append",
        metavar="CONFIG=OWNER/SLUG",
        default=[],
        help="override one config's kernel slug (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be fetched, download nothing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download files that are already present",
    )

    args = parser.parse_args(argv)

    args.slug_overrides = {}

    for override in args.slug:
        if "=" not in override:
            parser.error(f"--slug expects CONFIG=OWNER/SLUG, got {override!r}")

        config, slug = override.split("=", 1)
        args.slug_overrides[config] = slug

    configs = args.configs or CONFIGS

    unknown = [c for c in configs if c not in CONFIGS and c not in args.slug_overrides]

    if unknown:
        parser.error(
            f"unknown config(s): {unknown}\nknown: {CONFIGS}\n"
            "(pass --slug CONFIG=OWNER/SLUG for a kernel not listed here)"
        )

    failed = []

    for config in configs:
        try:
            sync_config(config, args)
        except SyncError as error:
            print(f"   ! skip {config}: {error}")
            failed.append(config)
        except Exception as error:  # noqa: BLE001 - one bad kernel must not abort the sweep
            print(f"   ! skip {config}: {type(error).__name__}: {error}")
            failed.append(config)

        print()

    print(f"done: {len(configs) - len(failed)} synced, {len(failed)} failed")

    if failed:
        print(f"failed: {failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
