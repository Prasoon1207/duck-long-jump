"""Fingerprints of the frozen part of the challenge: robot, assets, arena and episode rules.

Whole files are hashed byte for byte. In longjump_env.py only the rule methods are hashed
(as normalized source, so comments and formatting don't matter), which leaves the
observation function free to change.
"""

import ast
import hashlib
import json
from pathlib import Path

FROZEN_FILES = (
    "duckjump/__init__.py",
    "duckjump/env/__init__.py",
    "duckjump/env/robot.py",
    "duckjump/env/arena.py",
)
FROZEN_TREES = ("assets",)
FROZEN_REGIONS = {
    "duckjump/env/longjump_env.py": (
        "LongJumpEnv.__init__",
        "LongJumpEnv._foot_contacts",
        "LongJumpEnv._ground_contact_xs",
        "LongJumpEnv._fallen",
        "LongJumpEnv.reset",
        "LongJumpEnv.step",
    ),
}
MANIFEST = Path(__file__).resolve().parents[2] / "leaderboard" / "frozen.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _find(tree: ast.Module, qualname: str) -> ast.AST | None:
    node = tree
    for name in qualname.split("."):
        node = next((n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name), None)
        if node is None:
            return None
    return node


def fingerprint(root: Path) -> dict[str, str]:
    root = Path(root)
    out = {}
    for rel in FROZEN_FILES:
        path = root / rel
        out[rel] = _sha256(path.read_bytes()) if path.exists() else "missing"
    for tree in FROZEN_TREES:
        for path in sorted((root / tree).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                out[path.relative_to(root).as_posix()] = _sha256(path.read_bytes())
    for rel, qualnames in FROZEN_REGIONS.items():
        path = root / rel
        module = ast.parse(path.read_text()) if path.exists() else ast.Module(body=[], type_ignores=[])
        for q in qualnames:
            node = _find(module, q)
            out[f"{rel}::{q}"] = _sha256(ast.unparse(node).encode()) if node else "missing"
    return out


def load_manifest(path: Path = MANIFEST) -> dict[str, str]:
    return json.loads(Path(path).read_text())


def write_manifest(root: Path, path: Path = MANIFEST) -> None:
    Path(path).write_text(json.dumps(fingerprint(root), indent=2) + "\n")


def check(root: Path, manifest: dict[str, str] | None = None) -> list[str]:
    """Frozen items that differ from the manifest (empty list = rules intact)."""
    expected = manifest if manifest is not None else load_manifest()
    actual = fingerprint(root)
    problems = [f"changed: {k}" for k in expected if k in actual and actual[k] != expected[k]]
    problems += [f"missing: {k}" for k in expected if k not in actual]
    problems += [f"added: {k}" for k in actual if k not in expected]
    return problems
