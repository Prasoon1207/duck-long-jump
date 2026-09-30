"""Check that the frozen rules (robot, assets, arena, episode rules) match leaderboard/frozen.json.

    uv run scripts/check_frozen.py            # exit 1 if anything frozen changed
    uv run scripts/check_frozen.py --update   # maintainers only: accept the current rules
"""

import argparse
import sys

from duckjump.challenge import frozen
from duckjump.challenge.submission import REPO_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--update", action="store_true", help="rewrite the manifest from the current tree")
    args = p.parse_args()

    if args.update:
        frozen.write_manifest(REPO_ROOT)
        print(f"wrote {frozen.MANIFEST.relative_to(REPO_ROOT)}")
        return
    problems = frozen.check(REPO_ROOT)
    for problem in problems:
        print(problem)
    n = len(frozen.load_manifest())
    print(f"frozen rules modified ({len(problems)} items)" if problems else f"frozen rules intact ({n} items)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
