"""Turn evaluation results into the leaderboard and human-readable reports."""

import time

from duckjump.challenge.rules import EVAL_SEEDS, STEP_BUDGET


def build_leaderboard(results: list[dict], repo_url: str, branch: str, commit: str | None) -> dict:
    ranked = sorted((r for r in results if r["status"] == "ok"), key=lambda r: (-r["score"], r["steps"]))
    entries = [{
        "rank": i + 1,
        "entry": r["entry"],
        "title": r["title"],
        "author": r["author"],
        "score": r["score"],
        "success_rate": r["success_rate"],
        "best": r["best"],
        "steps": r["steps"],
        "code_changes": bool(r["overlay"]),
        "recipe_url": f"{repo_url}/blob/{branch}/submissions/{r['entry']}/recipe.md",
    } for i, r in enumerate(ranked)]
    rejected = [{"entry": r["entry"], "reason": r["reason"]} for r in results if r["status"] != "ok"]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit,
        "rules": {"eval_seeds": len(EVAL_SEEDS), "step_budget": STEP_BUDGET},
        "entries": entries,
        "rejected": rejected,
    }


def format_result(r: dict) -> str:
    lines = [f"{r['entry']} ({r['title']}, by {r['author']})"]
    if r["status"] != "ok":
        return "\n".join(lines + [f"REJECTED: {r['reason']}"])
    lines += [f"  seed  {'outcome':<18}  jump"]
    lines += [f"  {e['seed']:4d}  {e['outcome']:<18}  {e['jump_distance'] * 100:5.1f} cm" for e in r["episodes"]]
    lines += [f"score {r['score'] * 100:.2f} cm | success {r['success_rate']:.0%} | "
              f"best {r['best'] * 100:.1f} cm | trained {r['steps']:,} steps"]
    return "\n".join(lines)


def markdown_result(r: dict) -> str:
    head = f"### `{r['entry']}`: {r['title']} (by {r['author']})\n\n"
    if r["status"] != "ok":
        return head + f"**Rejected:** {r['reason']}\n"
    rows = "\n".join(f"| {e['seed']} | {e['outcome']} | {e['jump_distance'] * 100:.1f} |" for e in r["episodes"])
    overlay = ", ".join(f"`{f}`" for f in r["overlay"]) or "none"
    return (head + f"**Score: {r['score'] * 100:.2f} cm** | success {r['success_rate']:.0%} | "
            f"best {r['best'] * 100:.1f} cm | trained {r['steps']:,} steps | code changes: {overlay}\n\n"
            f"<details><summary>Per-seed results</summary>\n\n| seed | outcome | jump (cm) |\n|---|---|---|\n{rows}\n\n</details>\n")


def markdown_leaderboard(board: dict) -> str:
    rows = "\n".join(f"| {e['rank']} | `{e['entry']}` | {e['author']} | {e['score'] * 100:.2f} | "
                     f"{e['success_rate']:.0%} | {e['steps']:,} |" for e in board["entries"])
    out = f"## Leaderboard\n\n| rank | entry | author | score (cm) | success | steps |\n|---|---|---|---|---|---|\n{rows}\n"
    if board["rejected"]:
        out += "\n**Rejected:**\n\n" + "\n".join(f"- `{r['entry']}`: {r['reason']}" for r in board["rejected"]) + "\n"
    return out
