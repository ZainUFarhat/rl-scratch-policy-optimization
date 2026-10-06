# My From Scratch RL Policy Optimization Algorithms Implementations

## Overview

## How to run

Setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train (writes logs to `runs/<algo>_<env>_<seed>/episodes.csv`):
```bash
python reinforce.py --env CartPole-v1 --seed 0
python ppo.py --env LunarLander-v3 --seed 0
```

Run all 5 seeds:
```bash
./run_seeds.sh reinforce.py --env CartPole-v1
```

Plot mean/std return across seeds (reads `runs/`, writes `plots/`):
```bash
python utils/plotting.py
```

Record a GIF of a trained policy:
```bash
python utils/record_video.py --env CartPole-v1 --policy runs/reinforce_CartPole-v1_0/policy.pt --out rollout.gif
```

### Logging convention

Each run writes to `runs/<algo>_<env>_<seed>/episodes.csv`, one row per episode:

```
episode,return,length
0,23.0,23
1,19.0,19
...
```

`<algo>` and `<env>` must not contain underscores (e.g. `reinforce`, `CartPole-v1`) since `utils/plotting.py` parses the directory name on `_`.

## Results

## Hyperparameters

## Bugs I hit and what I learned
