---
title: PPO baseline
author: prasoon1207
---

# PPO baseline

The reference entry: the repo's from-scratch PPO actor-critic with its default settings.

- **Algorithm:** PPO, clipped objective, separate 256x256 actor and critic MLPs, observation
  normalization, 8 parallel envs x 256 steps per update, 5 epochs x 8 minibatches, lr 3e-4 annealed.
- **Reward:** the default `RewardWeights` as of commit `8f9f007` (run-up speed, airtime,
  100 per metre jumped, fall penalties, small uprightness and action-rate terms). It predates the
  take-off accuracy bonus.
- **Observations:** the default 45-dim observation, unchanged. No code changes (no `overlay/`).
- **Training:** 5M steps budgeted; `checkpoint.pt` is the run's `best.pt`, saved at 4,550,656 steps.
  Runs start 5 to 40 cm before the line (the evaluation always uses the full 40 cm).

## Reproduce

```bash
git checkout 8f9f007
uv run scripts/train.py --steps 5_000_000 --name baseline
cp runs/baseline/best.pt submissions/baseline-ppo/checkpoint.pt
```
