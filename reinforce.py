import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

import gymnasium as gym

import csv
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

class Policy(nn.Module):

    def __init__(self, num_obs, hidden_dim, num_act):
        super(Policy, self).__init__()
        self.input_layer = nn.Linear(num_obs, hidden_dim)
        self.tanh = nn.Tanh()
        self.output_layer = nn.Linear(hidden_dim, num_act)

    def forward(self, obs):
        h = self.input_layer(obs)
        h = self.tanh(h)
        logits = self.output_layer(h)
        return logits

    def get_policy(self, obs):
        logits = self(obs)
        return Categorical(logits = logits)

    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    def compute_loss(self, obs, act, returns):
        logp = self.get_policy(obs).log_prob(act)
        return -(logp * returns).mean()

def train(epochs, agent, optimizer, env, csv_writer=None, seed=None):
    episode_idx = 0
    for epoch in range(epochs):
        batch_obs, batch_act, batch_weights, batch_returns, batch_lengths, eps_rew = [], [], [], [], [], []
        while len(batch_obs) < 5000:
            obs, _ = env.reset(seed=seed) if episode_idx == 0 else env.reset()
            while True:
                batch_obs.append(obs.copy())
                act = agent.get_action(torch.as_tensor(obs, dtype = torch.float32))
                obs_next, rew, done, truncated, _ = env.step(act)
                batch_act.append(act)
                eps_rew.append(rew)
                obs = obs_next
                if done or truncated:
                    eps_return, eps_len = sum(eps_rew), len(eps_rew)
                    batch_returns.append(eps_return)
                    batch_lengths.append(eps_len)
                    batch_weights += [eps_return] * eps_len
                    if csv_writer is not None:
                        csv_writer.writerow([episode_idx, eps_return, eps_len])
                    episode_idx += 1
                    eps_rew = []
                    break
        optimizer.zero_grad()
        batch_loss = agent.compute_loss(obs = torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
                                        act = torch.as_tensor(batch_act, dtype=torch.long),
                                        returns = torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()
        logging.info(
            "Epoch: %4d | Loss: %.4f | Return: %.2f ± %.2f | Ep Length: %.1f ± %.1f",
            epoch,
            batch_loss.item(),
            np.mean(batch_returns),
            np.std(batch_returns),
            np.mean(batch_lengths),
            np.std(batch_lengths),
        )
    return "Training Complete."
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env)
    env.action_space.seed(args.seed)
    num_obs, num_act = env.observation_space.shape[0], env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Policy(num_obs, hidden_dim=args.hidden_dim, num_act=num_act).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    run_dir = Path("runs") / f"reinforce_{args.env}_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "episodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "length"])
        print(train(args.epochs, agent, optimizer, env, csv_writer=writer, seed=args.seed))

    torch.save(agent.state_dict(), run_dir / "policy.pt")
    print(f"saved {run_dir / 'policy.pt'}")