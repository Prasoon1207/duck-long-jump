"""Show a trained jumper.

    uv run scripts/demo.py runs/demo              # live MuJoCo viewer
    uv run scripts/demo.py runs/demo --video      # write runs/demo/jump_iso.mp4 instead
"""

import argparse
from pathlib import Path

from duckjump.infra.checkpoint import load_checkpoint
from duckjump.infra.playback import evaluate, format_result, record_video, watch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run directory or .pt checkpoint")
    p.add_argument("--video", action="store_true", help="record an MP4 instead of opening the viewer")
    p.add_argument("--camera", default="iso", choices=["iso", "follow", "side"])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--slowdown", type=float, default=1.0, help="2.0 = half speed in the viewer")
    args = p.parse_args()

    agent, env_cfg, ckpt = load_checkpoint(args.run)
    print(f"loaded policy trained for {ckpt['steps']:,} steps")

    if args.video:
        run = Path(args.run)
        out = (run if run.is_dir() else run.parent) / f"jump_{args.camera}.mp4"
        results = record_video(agent, env_cfg, out, episodes=args.episodes, camera=args.camera)
        for i, r in enumerate(results):
            print(format_result(i, r))
        print(f"video: {out}")
    else:
        watch(agent, env_cfg, episodes=args.episodes, slowdown=args.slowdown, camera=args.camera)


if __name__ == "__main__":
    main()
