"""
policies.py - the two policy "architectures" we compare. THIS is our project's contribution.

Both take observations of shape [B, N, obs_dim] and return
    logits [B, N, n_actions]   (preference for each of the 5 moves)
    values [B, N]              (critic: how good is this situation, used only in training)

1) SharedPolicy
       ONE network used by every agent ("parameter sharing").
       Running it with N+1 agents is trivial: the new agent just uses the same network.

2) IndependentPolicies
       A SEPARATE network per agent (agent i has its own "brain").
       When we go to N+1 agents there is no trained brain for the new agent, so we
       need a rule. Our rule: agent i uses network (i mod N), i.e. the extra agent
       borrows a copy of agent 0's brain. This is a design choice - say so in the report.
"""
import torch
import torch.nn as nn


def mlp(inp, out, hidden=64):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out))


class SharedPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.actor = mlp(obs_dim, n_actions, hidden)
        self.critic = mlp(obs_dim, 1, hidden)

    def forward(self, obs):                       # obs: [B, N, obs_dim]
        return self.actor(obs), self.critic(obs).squeeze(-1)


class IndependentPolicies(nn.Module):
    def __init__(self, n_trained, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.n_trained = n_trained
        self.actors = nn.ModuleList([mlp(obs_dim, n_actions, hidden) for _ in range(n_trained)])
        self.critics = nn.ModuleList([mlp(obs_dim, 1, hidden) for _ in range(n_trained)])

    def forward(self, obs):
        n_agents = obs.shape[1]
        logits, values = [], []
        for i in range(n_agents):
            j = i % self.n_trained                 # which trained brain agent i uses
            logits.append(self.actors[j](obs[:, i]))
            values.append(self.critics[j](obs[:, i]).squeeze(-1))
        return torch.stack(logits, dim=1), torch.stack(values, dim=1)


class AttentionPolicy(nn.Module):
    """
    3) Attention policy: one shared network that looks at ALL other agents.

    The observation is [own x, own y, goal rel x, goal rel y, then (dx, dy) for every
    other agent]. The network
        - embeds "me" and each neighbour separately,
        - lets "me" ask a question (query) and each neighbour answer (key/value),
        - takes a weighted average of the neighbour values (attention weights, softmax),
        - decides the action from [my embedding, neighbour summary].
    Because it is a weighted average over a SET of neighbours, it does not care how many
    neighbours there are or in which order they come: N+1 agents just means one more
    neighbour in the set. This is why it is a natural candidate for population transfer.
    """

    def __init__(self, n_actions, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.self_enc = nn.Sequential(nn.Linear(4, hidden), nn.Tanh())
        self.neigh_enc = nn.Sequential(nn.Linear(3, hidden), nn.Tanh())   # (dx, dy, distance)
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.actor = mlp(2 * hidden, n_actions, hidden)
        self.critic = mlp(2 * hidden, 1, hidden)

    def forward(self, obs):                                    # obs: [B, N, 4 + 2*(N-1)]
        rel = obs[..., 4:].reshape(*obs.shape[:-1], -1, 2)     # [B, N, M, 2] neighbours of each agent
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        h_neigh = self.neigh_enc(torch.cat([rel, dist], dim=-1))   # [B, N, M, H]
        h_self = self.self_enc(obs[..., :4])                        # [B, N, H]
        scores = (self.q(h_self)[..., None, :] * self.k(h_neigh)).sum(-1) / self.hidden ** 0.5
        weights = scores.softmax(dim=-1)[..., None]                 # [B, N, M, 1]
        context = (weights * self.v(h_neigh)).sum(dim=-2)           # [B, N, H]
        z = torch.cat([h_self, context], dim=-1)
        return self.actor(z), self.critic(z).squeeze(-1)


# Does this policy type need to see ALL other agents (True) or just the k nearest (False)?
NEEDS_FULL_NEIGHBORS = {"shared": False, "independent": False, "attention": True}


def build_policy(kind, n_agents, obs_dim, n_actions, hidden=64):
    if kind == "shared":
        return SharedPolicy(obs_dim, n_actions, hidden)
    if kind == "independent":
        return IndependentPolicies(n_agents, obs_dim, n_actions, hidden)
    if kind == "attention":
        return AttentionPolicy(n_actions, hidden)
    raise ValueError(f"unknown policy kind: {kind}")


def save_policy(policy, kind, n_agents, obs_dim, n_actions, path, hidden=64):
    torch.save({"kind": kind, "n_agents": n_agents, "obs_dim": obs_dim,
                "n_actions": n_actions, "hidden": hidden,
                "state": policy.state_dict()}, path)


def load_policy(path):
    ckpt = torch.load(path, map_location="cpu")
    policy = build_policy(ckpt["kind"], ckpt["n_agents"], ckpt["obs_dim"],
                          ckpt["n_actions"], ckpt["hidden"])
    policy.load_state_dict(ckpt["state"])
    policy.eval()
    return policy, ckpt
