"""
Read per-seed episode-return CSVs from runs/<algo>_<env>_<seed>/episodes.csv
and plot mean +/- std of return across seeds, one figure per (algo, env).

Each seed's returns are smoothed with a moving average, then the mean across\nseeds is drawn with a shaded +/- 1 std band.

Expected CSV format (one row per episode), header included:
    episode,return,length

Usage:
    python utils/plotting.py
    python utils/plotting.py --runs-dir runs --out-dir plots --smooth 100
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save to file without needing a display
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# One fixed color per algorithm, so it stays the same across every plot.
ALGO_COLORS = {
    "reinforce": "#d62728",  # red
    "ppo": "#2a78d6",        # blue
}
DEFAULT_COLOR = "#555555"

RUN_DIR_RE = re.compile(r"^(?P<algo>[^_]+)_(?P<env>.+)_(?P<seed>\d+)$")


def find_runs(runs_dir: Path):
    """Group run directories by (algo, env). Returns {(algo, env): [csv_path, ...]}."""
    groups = defaultdict(list)
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_DIR_RE.match(d.name)
        if not m:
            continue
        csv_path = d / "episodes.csv"
        if csv_path.exists():
            groups[(m.group("algo"), m.group("env"))].append(csv_path)
    return groups


def load_run(csv_path: Path):
    """Return (returns, steps): each episode's return, and the cumulative env steps when it ended."""
    episodes = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            episodes.append((int(row["episode"]), float(row["return"]), int(row["length"])))
    episodes.sort(key=lambda e: e[0])
    returns = np.array([r for _, r, _ in episodes])
    steps = np.cumsum([l for _, _, l in episodes])
    return returns, steps


def smooth(x, window):
    """Trailing moving average: each point is the mean of the last `window` episodes.
    Early points average over what exists so far, so the curve still starts at the beginning."""
    csum = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - window, 0)
    return (csum[idx] - csum[lo]) / (idx - lo)


def plot_group(algo: str, env: str, csv_paths, out_dir: Path, window: int, n_points: int = 1000):
    runs = [load_run(p) for p in csv_paths]
    if any(len(r) == 0 for r, _ in runs):
        print(f"skipping {algo}/{env}: a run has no episodes")
        return

    # Seeds finish episodes at different step counts, so smooth each seed over its own
    # episodes, then interpolate every seed onto one shared grid of environment steps.
    # The grid stops where the shortest run stops, so every point averages all seeds.
    first_step = max(s[0] for _, s in runs)
    last_step = min(s[-1] for _, s in runs)
    x = np.linspace(first_step, last_step, n_points)
    curves = np.stack([np.interp(x, s, smooth(r, window)) for r, s in runs])  # (n_seeds, n_points)

    mean = curves.mean(axis=0)
    std = curves.std(axis=0)

    # A return can't go above the best one any run achieved (500 on CartPole),
    # so keep the shaded band from suggesting otherwise.
    ceiling = max(r.max() for r, _ in runs)
    floor = min(r.min() for r, _ in runs)

    color = ALGO_COLORS.get(algo, DEFAULT_COLOR)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(x, np.clip(mean - std, floor, ceiling), np.clip(mean + std, floor, ceiling),
                    color=color, alpha=0.2, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=2)

    ax.set_title(f"{algo.upper()} {env}")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Average Return")
    ax.set_xlim(0, last_step)
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.grid(True, color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo}_{env}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def format_steps(value, _pos):
    """Tick labels like 250k and 1.5M instead of 250000 and 1500000."""
    if value >= 1e6:
        return f"{value / 1e6:g}M"
    if value >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots"))
    parser.add_argument("--smooth", type=int, default=50,
                        help="moving-average window in episodes (1 = no smoothing)")
    args = parser.parse_args()

    if not args.runs_dir.exists():
        print(f"no such directory: {args.runs_dir}")
        return

    groups = find_runs(args.runs_dir)
    if not groups:
        print(f"no run directories matching <algo>_<env>_<seed> found under {args.runs_dir}")
        return

    for (algo, env), csv_paths in groups.items():
        plot_group(algo, env, csv_paths, args.out_dir, args.smooth)


if __name__ == "__main__":
    main()