import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import ActorCritic

class A2CAgent:
    def __init__(self, input_dim, n_actions=5, device='cpu',
                 lr=3e-4, value_coef=0.5, entropy_coef=0.01, gamma=0.99, max_grad_norm=0.5):
        self.device = device
        self.model = ActorCritic(input_dim, n_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.n_actions = n_actions

    def act(self, states):
        """
        Given a batch of states (torch tensor), return action distribution logits, action sample, value.
        """
        logits, value = self.model(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, value, dist

    def compute_returns_and_advantages(self, rewards, dones, values, next_value):
        """
        rewards: [T] numpy
        dones: [T] numpy booleans
        values: [T] numpy estimated V(s)
        next_value: scalar bootstrap value for V(s_{T})
        returns = G_t
        advantage = returns - values
        Using TD(0)/Monte Carlo mix per-episode (here we use full-episode returns).
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        running = next_value
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running * (1.0 - float(dones[t]))
            returns[t] = running
        advantages = returns - values
        return returns, advantages

    def update_from_trajectory(self, traj):
        """
        traj: dict with 'states' (T,F), 'actions' (T,), 'rewards' (T,), 'dones' (T,)
        """
        states = torch.tensor(traj['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(traj['actions'], dtype=torch.long, device=self.device)
        rewards = traj['rewards']
        dones = traj['dones']

        logits, values = self.model(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # compute returns (Monte Carlo with gamma)
        with torch.no_grad():
            values_np = values.detach().cpu().numpy()
            next_v = 0.0  # episodes are terminal in dataset; set 0 or bootstrap value if available
            returns, advantages = self.compute_returns_and_advantages(rewards, dones, values_np, next_v)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # losses
        value_loss = nn.functional.mse_loss(values, returns_t)
        policy_loss = - (log_probs * advantages_t).mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {'loss': loss.item(), 'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'entropy': entropy.item()}

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    def load(self, path, map_location=None):
        self.model.load_state_dict(torch.load(path, map_location=map_location or self.device))
