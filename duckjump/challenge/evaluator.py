"""Official evaluation of one entry: validate it, check the frozen rules, play the attempts."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from duckjump.challenge import frozen
from duckjump.challenge.submission import Submission, build_workspace


def _reject(base: dict, reason: str) -> dict:
    return {**base, "status": "rejected", "reason": reason}


def evaluate_submission(sub: Submission) -> dict:
    base = {
        "entry": sub.name,
        "title": sub.meta.get("title", sub.name),
        "author": sub.meta.get("author", "unknown"),
        "overlay": sub.overlay_files,
        "versions": {"python": platform.python_version(), "mujoco": version("mujoco"), "torch": version("torch")},
    }
    problems = sub.problems()
    if problems:
        return _reject(base, "; ".join(problems))
    base["checkpoint_sha256"] = hashlib.sha256(sub.checkpoint.read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        build_workspace(sub, workspace)
        changed = frozen.check(workspace)
        if changed:
            return _reject(base, "frozen rules modified: " + ", ".join(changed))

        out = Path(tmp) / "result.json"
        # The workspace copy of duckjump must shadow the installed one.
        env = {**os.environ, "PYTHONPATH": str(workspace)}
        proc = subprocess.run([sys.executable, "-m", "duckjump.challenge.rollout", str(sub.checkpoint), str(out)],
                              cwd=workspace, env=env, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            return _reject(base, "evaluation crashed: " + proc.stderr.strip()[-2000:])
        return {**base, **json.loads(out.read_text())}
