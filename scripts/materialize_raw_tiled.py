#!/usr/bin/env python3
"""
Materialize ``datasets/phenobench_raw_tiled`` from ``phenobench_raw_full``.

The tiled raw tree is the *file-level* counterpart of the tiling the export
notebooks apply in memory. ``ave benchmark`` reads its ``val/images`` tiles and
``ave evaluate --faithful`` stages its ``val`` masks as ground truth, both keyed
by file name -- so its geometry has to track notebooks ``03``/``04``
(3x3, ``overlap=0.5``, i.e. uniform 512px tiles). See
:mod:`agri_vision_edge.data.raw_tiling` for why a stale grid fails silently.

Usage
-----
    python scripts/materialize_raw_tiled.py [--rows 3] [--cols 3]
                                            [--overlap 0.5] [--workers N]
                                            [--source DIR] [--dest DIR]
                                            [--splits train val test]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agri_vision_edge.data.raw_tiling import (  # noqa: E402
    DEFAULT_SPLITS,
    RAW_SUBDIRS,
    materialize_tiled_dataset,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--source",
        default=REPO_ROOT / "datasets" / "phenobench_raw_full",
        type=Path,
    )
    parser.add_argument(
        "--dest",
        default=REPO_ROOT / "datasets" / "phenobench_raw_tiled",
        type=Path,
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Tile overlap as a FRACTION in [0, 1) (default: %(default)s)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=list(RAW_SUBDIRS),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
    )
    parser.add_argument("--exist-ok", action="store_true")

    args = parser.parse_args(argv)

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    def progress(iterable, desc=""):
        if tqdm is None:
            return iterable
        return tqdm(iterable, desc=desc, mininterval=5.0)

    print(f"source : {args.source}")
    print(f"dest   : {args.dest}")
    print(f"grid   : {args.rows}x{args.cols} overlap={args.overlap}")
    print(f"workers: {args.workers}")

    stats = materialize_tiled_dataset(
        args.source,
        args.dest,
        rows=args.rows,
        cols=args.cols,
        overlap=args.overlap,
        splits=args.splits,
        subdirs=args.subdirs,
        workers=args.workers,
        progress=progress,
        exist_ok=args.exist_ok,
    )

    print(json.dumps(stats, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
