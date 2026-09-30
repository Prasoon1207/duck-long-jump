"""Train the duck to long jump.

    uv run scripts/train.py --steps 5_000_000 --name my_run
"""

import argparse
import time
from pathlib import Path

from duckjump.algo.config import PPOConfig
from duckjump.env.longjump_env import LongJumpConfig
from duckjump.infra.playback import evaluate, format_result, record_video
from duckjump.infra.checkpoint import load_checkpoint
from duckjump.infra.trainer import train

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--name", default=time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--motor-strength", type=float, default=2.0)
    p.add_argument("--runup", type=float, default=0.4, help="start to take-off line (m)")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    env_cfg = LongJumpConfig(motor_strength=args.motor_strength)
    env_cfg.arena.runup_length = args.runup
    ppo_cfg = PPOConfig(total_steps=args.steps, num_envs=args.num_envs, seed=args.seed)
    run_dir = ROOT / "runs" / args.name
    print(f"run dir: {run_dir}\nbatch {ppo_cfg.batch_size} samples/update, "
          f"{ppo_cfg.total_steps // ppo_cfg.batch_size} updates")

    train(env_cfg, ppo_cfg, run_dir)

    agent, env_cfg, _ = load_checkpoint(run_dir)
    print("\nevaluation of the best checkpoint (deterministic policy):")
    for i, r in enumerate(evaluate(agent, env_cfg, episodes=10)):
        print("  " + format_result(i, r))
    if not args.no_video:
        video = run_dir / "jump.mp4"
        record_video(agent, env_cfg, video)
        print(f"video: {video}")


if __name__ == "__main__":
    main()
