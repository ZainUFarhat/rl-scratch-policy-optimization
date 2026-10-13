"""
Load a saved policy (torch.save(policy.state_dict(), path)) and record a GIF
of one episode.

The algorithm, env and seed are read from the run folder name, and the GIF
is saved as renders/<algo>_<env>_seed<seed>.gif.

Usage (from the repo root):
    python utils/record_video.py --policy runs/reinforce_CartPole-v1_0/policy.pt
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import imageio
import torch

# reuse the exact Policy class you trained, so the architecture always matches
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reinforce import Policy  # noqa: E402


def get_action(policy: Policy, obs):
    """Greedy rollout: pick the most likely action instead of sampling."""
    logits = policy(torch.as_tensor(obs, dtype=torch.float32))
    return int(torch.argmax(logits).item())


def record(env_id: str, policy_path: str, out_path: str, hidden_dim: int, max_steps: int = 1000):
    env = gym.make(env_id, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = Policy(obs_dim, hidden_dim, act_dim)
    policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
    policy.eval()

    obs, _ = env.reset(seed=0)
    frames = [env.render()]

    with torch.no_grad():
        for _ in range(max_steps):
            action = get_action(policy, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break

    env.close()
    # keep every 2nd frame so a 500-step CartPole GIF stays a reasonable size
    imageio.mimsave(out_path, frames[::2], duration=1000 / 30, loop=0)
    print(f"wrote {out_path} ({len(frames)} env steps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True,
                        help="path to saved state_dict, e.g. runs/reinforce_CartPole-v1_0/policy.pt")
    parser.add_argument("--out-dir", default="renders")
    parser.add_argument("--hidden-dim", type=int, default=64, help="must match training")
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    # The run folder is named <algo>_<env>_<seed>, so read algo, env and seed from it.
    run_name = Path(args.policy).parent.name
    algo, rest = run_name.split("_", 1)
    env_id, seed = rest.rsplit("_", 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo}_{env_id}_seed{seed}.gif"

    record(env_id, args.policy, str(out_path), args.hidden_dim, args.max_steps)


if __name__ == "__main__":
    main()