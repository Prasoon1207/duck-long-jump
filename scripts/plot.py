"""Plot training curves from runs/<name>/progress.csv into runs/<name>/progress.png."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("ep_return", "episode return"),
    ("jump_mean", "mean jump (m)"),
    ("frac_jump", "fraction of real jumps"),
    ("flight_time", "flight time (s)"),
    ("action_std", "policy std"),
    ("explained_variance", "critic explained variance"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    args = p.parse_args()
    run = Path(args.run)
    rows = list(csv.DictReader(open(run / "progress.csv")))
    steps = [int(r["steps"]) / 1e6 for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (key, title) in zip(axes.flat, PANELS):
        ax.plot(steps, [float(r[key]) for r in rows])
        ax.set_title(title)
        ax.set_xlabel("env steps (M)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = run / "progress.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
