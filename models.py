import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(256,128), dropout=0.1):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ActorHead(nn.Module):
    def __init__(self, in_dim, n_actions):
        super().__init__()
        self.pi = nn.Linear(in_dim, n_actions)
    def forward(self, x):
        logits = self.pi(x)
        return logits  # raw logits; softmax applied by loss/sample step

class CriticHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.v = nn.Linear(in_dim, 1)
    def forward(self, x):
        return self.v(x).squeeze(-1)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions=5, hidden_sizes=(256,128)):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_sizes=hidden_sizes)
        enc_out = hidden_sizes[-1]
        self.actor = ActorHead(enc_out, n_actions)
        self.critic = CriticHead(enc_out)
    def forward(self, x):
        features = self.encoder(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

if __name__ == "__main__":
    # quick sanity check
    m = ActorCritic(379, n_actions=5)
    x = torch.randn(3,379)
    l, v = m(x)
    print(l.shape, v.shape)
