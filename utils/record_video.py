"""
Load a saved policy (torch.save(policy.state_dict(), path)) and record a GIF
of one episode.

Before this will run, fill in PolicyNetwork below with your actual
architecture (same shape as whatever you trained) and set how you pick an
action from its output (see get_action).

Usage:
    python utils/record_video.py --env CartPole-v1 --policy runs/reinforce_CartPole-v1_0/policy.pt --out rollout.gif
"""
import argparse

import gymnasium as gym
import imageio
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    PLACEHOLDER - replace this with your actual network definition.
    It must match the architecture you trained, so that load_state_dict
    below succeeds.
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        raise NotImplementedError(
            "Define your policy network here to match what you trained, "
            "then remove this line."
        )

    def forward(self, obs):
        raise NotImplementedError


def get_action(policy: PolicyNetwork, obs):
    """
    PLACEHOLDER - replace with however you turn policy(obs) into an action.
    e.g. argmax over logits for a deterministic rollout, or sample from the
    action distribution your algorithm uses.
    """
    raise NotImplementedError(
        "Define how to turn the policy's output into an env action, "
        "then remove this line."
    )


def record(env_id: str, policy_path: str, out_path: str, max_steps: int = 1000):
    env = gym.make(env_id, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n  # change if your action space isn't Discrete

    policy = PolicyNetwork(obs_dim, act_dim)
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
    imageio.mimsave(out_path, frames, fps=30)
    print(f"wrote {out_path} ({len(frames)} frames)")


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
