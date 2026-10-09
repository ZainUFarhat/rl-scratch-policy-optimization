import sys
import argparse
from pathlib import Path

import torch

import imageio

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reinforce import Policy

def get_action(policy: Policy, obs):
    logits = policy(torch.as_tensor(obs, dtype = torch.float32))
    return int(torch.argmax(logits).item())

def record(env_id: str, policy_path: str, out_path: str, max_steps: int = 1000):
    env = gym.make(env_id, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n  # change if your action space isn't Discrete

    policy = Policy(obs_dim, 64, act_dim)
    policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
    policy.eval()

    obs, _ = env.reset()
    frames = [env.render()]

    with torch.no_grad():
        for _ in range(max_steps):
            action = get_action(policy, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break

    env.close()
    imageio.mimsave(out_path, frames[::2], duration=1000 / 30, loop=0)
    print(f"wrote {out_path} ({len(frames)} env steps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="e.g. CartPole-v1, LunarLander-v3")
    parser.add_argument("--policy", required=True, help="path to saved state_dict (.pt)")
    parser.add_argument("--out", default="rollout.gif")
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    record(args.env, args.policy, args.out, args.max_steps)


if __name__ == "__main__":
    main()
