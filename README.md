# Duck long jump

Train the [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini) to long jump in MuJoCo with PPO (actor-critic), written from scratch in PyTorch.

The duck starts on a runway, runs up to a take-off line, and jumps into a sand pit. The run-up ends at the first ground contact at or past the line. The duck then has 0.3 s to land, and the score is the distance from the line to the nearest mark made during that window, the same rule as the real event. A foot still on the runway scores 0, and a trailing foot that comes down behind the leading one shortens the jump. Falling ends the episode.

## Quick start

```bash
uv sync                                   # installs MuJoCo, CPU PyTorch, etc.
uv run scripts/view_arena.py              # look at the arena (untrained duck standing)
uv run scripts/train.py --steps 5_000_000 --name demo   # ~40 min on an 8-core laptop
uv run scripts/demo.py runs/demo          # watch it live in the MuJoCo viewer
uv run scripts/demo.py runs/demo --video --camera side   # or write an MP4
uv run scripts/plot.py runs/demo          # training curves -> runs/demo/progress.png
```

Every run writes to `runs/<name>/`: `config.json`, `progress.csv`, `latest.pt`, `best.pt`, and `jump.mp4` once training finishes.

## Layout

```
duckjump/
  algo/                 <- the learning algorithm: edit freely
    config.py           PPO hyperparameters
    networks.py         actor (Gaussian policy) + critic (value function)
    buffer.py           rollout storage + GAE
    ppo.py              clipped-surrogate PPO update
    normalizer.py       running observation normalization
  env/                  <- the task
    reward.py           reward weights and reward function: edit freely
    longjump_env.py     Gymnasium env: rules, observations, termination
    arena.py            runway / board / pit / cameras
    robot.py            Open Duck Mini v2 loading, joints, sensors
  infra/                <- plumbing
    trainer.py          parallel envs -> PPO -> logging / checkpoints
    checkpoint.py       save / load agent + configs
    playback.py         evaluation, video recording, live viewer
scripts/                train.py, demo.py, view_arena.py, plot.py
assets/open_duck_mini_v2/   robot MJCF + meshes (from Open_Duck_Playground)
```

`algo/` knows nothing about robots, and `env/` knows nothing about PPO. `infra/trainer.py` only uses this interface on the agent:

| call | meaning |
| --- | --- |
| `agent.act(obs)` | sample actions → `(action, log_prob, value, normalized_obs)` |
| `agent.store(...)` | append one step to the rollout buffer |
| `agent.bootstrap_truncated(...)` | add the value of time-limit-truncated states to their reward |
| `agent.update(last_obs)` | compute advantages and run the PPO epochs, return stats |

If you swap in another algorithm that implements the same methods, the rest of the code keeps working.

## Task details

- **Actions (10):** joint-angle offsets from the standing pose for both legs (hip yaw/roll/pitch, knee, ankle), scaled by `action_scale` rad. The head holds still.
- **Observations (45):** gravity in the body frame, gyro, body velocity, leg joint positions/velocities, the previous action, foot contacts, distance to the line, body height, and phase (run-up or landed).
- **Control rate:** 50 Hz, with physics at 500 Hz.
- **Start distance:** each episode starts 5–40 cm before the line (`start_distance_range`). The short starts make the jump easy to discover. Evaluation and the demo always use the full 40 cm run-up.
- **Motor strength:** `motor_strength = 2.0` doubles the real STS3215 servo stiffness and torque. At 1.0, the real servos manage only about 20 ms of air time, so there's hardly a jump to learn. Set `--motor-strength 1.0` for a sim-to-real-honest duck.

## Things to try

- Reward shaping in `env/reward.py`: an upright/heading term, heavier landing-stability penalty, removing the run-up velocity reward.
- `algo/config.py`: entropy bonus, `init_log_std`, network sizes, GAE λ, clip range.
- A curriculum on `start_distance_range` or `arena.runup_length`.
