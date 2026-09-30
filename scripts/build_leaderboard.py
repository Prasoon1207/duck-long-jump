"""Evaluate every submission and build the static leaderboard site.

    uv run scripts/build_leaderboard.py --out _site
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from duckjump.challenge.evaluator import evaluate_submission
from duckjump.challenge.leaderboard import build_leaderboard, format_result, markdown_leaderboard
from duckjump.challenge.submission import REPO_ROOT, all_submissions


def git_commit() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return proc.stdout.strip() or None


def main():
    repo = os.environ.get("GITHUB_REPOSITORY", "Prasoon1207/duck-long-jump")
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO_ROOT / "_site")
    p.add_argument("--repo-url", default=f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repo}")
    p.add_argument("--branch", default="main")
    p.add_argument("--summary", type=Path, help="append the leaderboard as markdown here")
    args = p.parse_args()

    results = []
    for sub in all_submissions():
        results.append(evaluate_submission(sub))
        print(format_result(results[-1]), end="\n\n", flush=True)
    board = build_leaderboard(results, args.repo_url, args.branch, git_commit())

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "leaderboard" / "index.html", args.out / "index.html")
    (args.out / "leaderboard.json").write_text(json.dumps(board, indent=2))
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(markdown_leaderboard(board))
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(markdown_leaderboard(board))


if __name__ == "__main__":
    main()
