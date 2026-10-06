"""
Read per-seed episode-return CSVs from runs/<algo>_<env>_<seed>/episodes.csv
and plot mean +/- std of return across seeds, one figure per (algo, env).

Expected CSV format (one row per episode), header included:
    episode,return,length

Usage:
    python utils/plotting.py
    python utils/plotting.py --runs-dir runs --out-dir plots
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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


def load_returns(csv_path: Path):
    """Return a 1D array of episode returns, in episode order."""
    episodes = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            episodes.append((int(row["episode"]), float(row["return"])))
    episodes.sort(key=lambda e: e[0])
    return np.array([r for _, r in episodes])


def plot_group(algo: str, env: str, csv_paths, out_dir: Path):
    runs = [load_returns(p) for p in csv_paths]
    min_len = min(len(r) for r in runs)
    if min_len == 0:
        print(f"skipping {algo}/{env}: a run has no episodes")
        return
    runs = np.stack([r[:min_len] for r in runs])  # (n_seeds, n_episodes)

    mean = runs.mean(axis=0)
    std = runs.std(axis=0)
    x = np.arange(min_len)

    fig, ax = plt.subplots()
    ax.plot(x, mean, label="mean return")
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, label="+/- 1 std")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title(f"{algo} on {env} (n={len(csv_paths)} seeds)")
    ax.legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo}_{env}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    if not args.runs_dir.exists():
        print(f"no such directory: {args.runs_dir}")
        return

    groups = find_runs(args.runs_dir)
    if not groups:
        print(f"no run directories matching <algo>_<env>_<seed> found under {args.runs_dir}")
        return

    for (algo, env), csv_paths in groups.items():
        plot_group(algo, env, csv_paths, args.out_dir)


if __name__ == "__main__":
    main()
