"""A leaderboard entry: submissions/<name>/ with recipe.md, checkpoint.pt and an optional
overlay/ of changed code (same paths as in the repo, e.g. overlay/duckjump/env/longjump_env.py)."""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from duckjump.challenge.rules import MAX_CHECKPOINT_BYTES

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS = REPO_ROOT / "submissions"
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_OVERLAY_ALLOWED = "duckjump/"
_OVERLAY_FORBIDDEN = "duckjump/challenge/"


@dataclass
class Submission:
    name: str
    path: Path
    checkpoint: Path
    meta: dict
    overlay_files: list[str]

    @property
    def recipe(self) -> Path:
        return self.path / "recipe.md"

    def problems(self) -> list[str]:
        out = []
        if not NAME_PATTERN.match(self.name):
            out.append(f"entry name {self.name!r} must match {NAME_PATTERN.pattern}")
        if not self.checkpoint.is_file():
            out.append(f"no checkpoint at {self.checkpoint}")
        elif self.checkpoint.stat().st_size > MAX_CHECKPOINT_BYTES:
            out.append(f"checkpoint is larger than {MAX_CHECKPOINT_BYTES // 2**20} MB")
        for f in self.overlay_files:
            if not f.startswith(_OVERLAY_ALLOWED) or f.startswith(_OVERLAY_FORBIDDEN):
                out.append(f"overlay may only change duckjump/ (not duckjump/challenge/): {f}")
        return out


def read_front_matter(text: str) -> dict:
    """`key: value` lines between two leading `---` lines."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return meta
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return {}


def load_submission(path: str | Path) -> Submission:
    """A submission directory, or (for local checks) a run directory or .pt file."""
    path = Path(path).resolve()
    if path.is_file():
        return Submission(path.parent.name, path.parent, path, {}, [])
    if (path / "checkpoint.pt").exists() or (path / "recipe.md").exists():
        recipe = path / "recipe.md"
        meta = read_front_matter(recipe.read_text()) if recipe.exists() else {}
        overlay = path / "overlay"
        files = sorted(p.relative_to(overlay).as_posix() for p in overlay.rglob("*") if p.is_file()) if overlay.is_dir() else []
        return Submission(path.name, path, path / "checkpoint.pt", meta, files)
    checkpoint = path / "best.pt" if (path / "best.pt").exists() else path / "latest.pt"
    return Submission(path.name, path, checkpoint, {}, [])


def all_submissions() -> list[Submission]:
    return [load_submission(p) for p in sorted(SUBMISSIONS.iterdir()) if p.is_dir()]


def build_workspace(sub: Submission, dest: Path, repo_root: Path = REPO_ROOT) -> Path:
    """Copy of the repo's code and assets with the submission's overlay applied on top."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("duckjump", "assets"):
        shutil.copytree(repo_root / d, dest / d, ignore=ignore)
    for f in sub.overlay_files:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sub.path / "overlay" / f, dest / f)
    return dest
