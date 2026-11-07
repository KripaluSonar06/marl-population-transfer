"""
env.py - a tiny cooperative navigation world used to study population transfer.

WORLD
    N agents live in the unit square [0, 1] x [0, 1]. Each agent has its own goal.
    Every step an agent picks one of 5 moves (stay / up / down / left / right) and
    moves 0.05 in that direction. An episode lasts 50 steps.

CONSTRAINT (the "safety" part of our project)
    Two agents are in COLLISION when their centres are closer than `collision_dist`
    (default 0.1). The constraint is: never collide.

WHAT AN AGENT SEES (observation, always the same length!)
    [ own x, own y,                     <- where am I
      goal_x - x, goal_y - y,           <- where is my goal, relative to me
      relative position of nearest neighbour #1,
      relative position of nearest neighbour #2, ... (k of them) ]
    Because we only look at the k nearest neighbours, the observation length does
    not depend on N. THAT is what lets a network trained with N agents also be run
    with N+1 agents.

REWARD (per agent, per step)
    + 10 * (how much closer to my goal I got)
    - 10.0 if I am in collision with someone (must be large, otherwise agents just
          ignore each other: with penalty 1.0 or 3.0 they learned almost no avoidance)
    + 0.2 if I am sitting on my goal

Everything is batched: `batch_size` independent worlds run at once as torch tensors
(shape [B, N, ...]), which keeps training fast on a plain CPU.
"""
import torch

# action index -> (dx, dy) direction
ACTIONS = torch.tensor([[0.0, 0.0],    # 0: stay
                        [0.0, 1.0],    # 1: up
                        [0.0, -1.0],   # 2: down
                        [-1.0, 0.0],   # 3: left
                        [1.0, 0.0]])   # 4: right
N_ACTIONS = ACTIONS.shape[0]


class NavEnv:
    def __init__(self, n_agents, batch_size, k_neighbors=2, max_steps=50,
                 step_size=0.05, collision_dist=0.1, goal_dist=0.05,
                 collision_penalty=10.0, full_neighbors=False, seed=0):
        if full_neighbors:               # see ALL other agents (used by the attention policy)
            k_neighbors = n_agents - 1
        assert n_agents >= k_neighbors + 1, "need at least k_neighbors + 1 agents"
        self.N = n_agents
        self.B = batch_size
        self.k = k_neighbors
        self.max_steps = max_steps
        self.step_size = step_size
        self.collision_dist = collision_dist
        self.goal_dist = goal_dist
        self.collision_penalty = collision_penalty
        self.gen = torch.Generator().manual_seed(seed)   # reproducible randomness
        self.eye = torch.eye(n_agents, dtype=torch.bool)  # marks "agent i vs itself"

    @property
    def obs_dim(self):
        return 4 + 2 * self.k

    # ------------------------------------------------------------------ helpers
    def _pair_dist(self, pos):
        """Distance between every pair of agents, shape [B, N, N]. Diagonal = inf."""
        d = torch.cdist(pos, pos)
        return d.masked_fill(self.eye, float("inf"))

    def _sample_separated(self, min_sep):
        """Random points for every agent, re-sampled until no two are closer than min_sep."""
        pts = torch.rand(self.B, self.N, 2, generator=self.gen)
        for _ in range(100):
            bad = self._pair_dist(pts).amin(dim=(1, 2)) < min_sep   # [B] which worlds are bad
            if not bad.any():
                break
            pts[bad] = torch.rand(int(bad.sum()), self.N, 2, generator=self.gen)
        return pts

    def _obs(self):
        rel_goal = self.goal - self.pos                                   # [B,N,2]
        diff = self.pos[:, None, :, :] - self.pos[:, :, None, :]          # diff[b,i,j] = pos_j - pos_i
        dist = torch.linalg.norm(diff, dim=-1).masked_fill(self.eye, float("inf"))
        idx = dist.topk(self.k, dim=-1, largest=False).indices            # k nearest neighbours of each agent
        neigh = torch.gather(diff, 2, idx[..., None].expand(-1, -1, -1, 2))  # [B,N,k,2]
        return torch.cat([self.pos, rel_goal, neigh.flatten(2)], dim=-1)  # [B,N,4+2k]

    # --------------------------------------------------------------- main API
    def reset(self):
        self.pos = self._sample_separated(0.15)    # start positions, not touching
        self.goal = self._sample_separated(0.15)   # goals, not on top of each other
        self.t = 0
        self.prev_d = torch.linalg.norm(self.goal - self.pos, dim=-1)
        # bookkeeping for the metrics
        self.min_sep = self._pair_dist(self.pos).amin(dim=(1, 2))   # closest any two agents ever got
        self.violation_steps = torch.zeros(self.B)                   # steps with >=1 collision
        self.ep_return = torch.zeros(self.B, self.N)
        return self._obs()

    def step(self, actions):
        """actions: LongTensor [B, N]. Returns obs, reward [B,N], done (bool)."""
        move = ACTIONS[actions] * self.step_size
        self.pos = (self.pos + move).clamp(0.0, 1.0)
        self.t += 1

        d_goal = torch.linalg.norm(self.goal - self.pos, dim=-1)     # [B,N]
        progress = self.prev_d - d_goal
        self.prev_d = d_goal
        at_goal = d_goal < self.goal_dist

        pair = self._pair_dist(self.pos)
        in_collision = (pair < self.collision_dist).any(dim=-1)      # [B,N] is agent i colliding?

        reward = (10.0 * progress - self.collision_penalty * in_collision.float()
                  + 0.2 * at_goal.float())

        self.min_sep = torch.minimum(self.min_sep, pair.amin(dim=(1, 2)))
        self.violation_steps += in_collision.any(dim=-1).float()
        self.ep_return += reward
        return self._obs(), reward, self.t >= self.max_steps

    def metrics(self):
        """Episode-level metrics, averaged over the batch. Call after a full episode."""
        d_goal = torch.linalg.norm(self.goal - self.pos, dim=-1)
        return {
            # closest any two agents ever got (higher = safer; collision below collision_dist)
            "min_separation": self.min_sep.mean().item(),
            # fraction of time steps in which at least one pair was in collision
            "violation_rate": (self.violation_steps / max(self.t, 1)).mean().item(),
            # fraction of episodes that had at least one collision
            "episodes_with_collision": (self.violation_steps > 0).float().mean().item(),
            # fraction of agents sitting on their goal at the end
            "success_rate": (d_goal < self.goal_dist).float().mean().item(),
            "mean_final_dist": d_goal.mean().item(),
            "mean_return": self.ep_return.mean().item(),
        }
