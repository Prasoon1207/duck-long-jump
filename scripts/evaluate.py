"""Official evaluator: mean jump over 20 fixed seeds from the full run-up, deterministic
policy, anything but a clean jump scores 0, frozen rules enforced.

    uv run scripts/evaluate.py submissions/baseline-ppo      # a submission
    uv run scripts/evaluate.py runs/my_run                   # check your own run before submitting
    uv run scripts/evaluate.py --changed-since HEAD^1        # CI: submissions touched by a PR
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from duckjump.challenge.evaluator import evaluate_submission
from duckjump.challenge.leaderboard import format_result, markdown_result
from duckjump.challenge.submission import REPO_ROOT, SUBMISSIONS, load_submission


def changed_submissions(ref: str) -> list[Path]:
    files = subprocess.run(["git", "diff", "--name-only", ref, "HEAD", "--", "submissions/"],
                           cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    names = sorted({Path(f).parts[1] for f in files if len(Path(f).parts) > 2})
    return [SUBMISSIONS / n for n in names if (SUBMISSIONS / n).is_dir()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*", help="submission dirs, run dirs or .pt checkpoints")
    p.add_argument("--changed-since", metavar="REF", help="also evaluate submissions changed since this git ref")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "eval", help="directory for <entry>.json results")
    p.add_argument("--summary", type=Path, help="append a markdown report here (e.g. $GITHUB_STEP_SUMMARY)")
    args = p.parse_args()

    paths = [Path(x) for x in args.paths] + (changed_submissions(args.changed_since) if args.changed_since else [])
    if not paths:
        print("nothing to evaluate")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for path in paths:
        result = evaluate_submission(load_submission(path))
        results.append(result)
        (args.out / f"{result['entry']}.json").write_text(json.dumps(result, indent=2))
        print(format_result(result), end="\n\n", flush=True)
        if args.summary:
            with open(args.summary, "a") as f:
                f.write(markdown_result(result) + "\n")
    print(f"results written to {args.out}")
    if any(r["status"] != "ok" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
