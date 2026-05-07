from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def capture_git_metadata(
    repo_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """
    Capture git repository metadata.

    Parameters
    ----------
    repo_dir:
        Path inside the git repository.
        Defaults to current working directory.

    Returns
    -------
    dict or None
        JSON-serializable git metadata.
    """

    repo_dir = Path(repo_dir or ".")

    if not is_git_repository(repo_dir):
        return None

    return {
        "commit": get_git_commit(repo_dir),
        "branch": get_git_branch(repo_dir),
        "dirty": is_git_dirty(repo_dir),
        "remote_origin": get_git_remote_origin(repo_dir),
    }


# ============================================================
# Core helpers
# ============================================================

def is_git_repository(
    repo_dir: str | Path,
) -> bool:

    try:
        run_git_command(
            repo_dir,
            ["rev-parse", "--is-inside-work-tree"],
        )

        return True

    except Exception:
        return False


def get_git_commit(
    repo_dir: str | Path,
) -> str | None:

    try:
        return run_git_command(
            repo_dir,
            ["rev-parse", "HEAD"],
        )

    except Exception:
        return None


def get_git_branch(
    repo_dir: str | Path,
) -> str | None:

    try:
        return run_git_command(
            repo_dir,
            ["rev-parse", "--abbrev-ref", "HEAD"],
        )

    except Exception:
        return None


def is_git_dirty(
    repo_dir: str | Path,
) -> bool | None:

    try:
        status = run_git_command(
            repo_dir,
            ["status", "--porcelain"],
        )

        return bool(status.strip())

    except Exception:
        return None


def get_git_remote_origin(
    repo_dir: str | Path,
) -> str | None:

    try:
        return run_git_command(
            repo_dir,
            ["remote", "get-url", "origin"],
        )

    except Exception:
        return None


# ============================================================
# Internal
# ============================================================

def run_git_command(
    repo_dir: str | Path,
    args: list[str],
) -> str:

    repo_dir = Path(repo_dir)

    result = subprocess.check_output(
        ["git", "-C", str(repo_dir), *args],
        stderr=subprocess.DEVNULL,
    )

    return result.decode().strip()
