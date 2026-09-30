# Submitting to the long-jump leaderboard

Train the duck to jump as far as you can, then open a pull request that adds
`submissions/<your-entry>/`. CI evaluates it and shows the score in the run summary; once
merged, the entry appears on the leaderboard (GitHub Pages).

## Rules

- **Frozen** (identical for every entry): the robot model and assets, the motors
  (`motor_strength = 2.0`), the arena, the episode rules and the scoring. These are
  fingerprinted in `leaderboard/frozen.json` and the evaluator forces the official
  `LongJumpConfig` values (see `duckjump/challenge/rules.py`).
- **Free:** the algorithm, the reward, the observations and the hyperparameters.
- **Budget:** at most 10,000,000 environment steps of training (`steps` in the checkpoint).
- **Score:** mean `jump_distance` over 20 fixed seeds from the full 40 cm run-up with the
  deterministic policy. Falls, step-overs and timeouts count as 0.

## What to put in the pull request

```
submissions/<your-entry>/        # lowercase letters, digits, ".", "_", "-"
  recipe.md                      # front matter (title, author), then how you did it
  checkpoint.pt                  # at most 20 MB
  overlay/                       # optional: your changed code, same paths as in the repo
    duckjump/env/reward.py
    duckjump/env/longjump_env.py # only the observation function may differ
```

`recipe.md` starts with:

```
---
title: My jumper
author: your-github-handle
---
```

followed by a description of the algorithm, reward and observations, and the exact
training command.

Code changes go in `overlay/`, not in the repo's own files, so that every entry can be
re-evaluated side by side. The evaluator copies `duckjump/` and `assets/`, applies your
overlay on top, checks the frozen rules and loads your policy with
`duckjump.infra.checkpoint.load_policy`. It then calls
`agent.act(obs_batch, deterministic=True)[0]` for actions. If you change the algorithm,
keep that interface (override `load_policy` in your overlay). The overlay may not touch
`duckjump/challenge/`.

## Check locally before submitting

```bash
uv run scripts/check_frozen.py
uv run scripts/evaluate.py runs/<your-run>            # or submissions/<your-entry>
```
