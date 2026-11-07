"""
shield.py - a runtime SAFETY SHIELD that sits between the policy and the world.

Idea in plain words
    The learned policy proposes a move. Before the agent executes it, the shield asks:
    "if I take this move, how close will I be to my nearest neighbours?"
    - Moves that would end up closer than (collision_dist + margin * step_size) are VETOED.
    - Among the remaining safe moves, the agent takes the one the policy likes best.
    - If NO move is safe (boxed in), the agent takes the move that keeps the biggest
      distance to its neighbours (we count how often this "fallback" is needed).

It only uses what the agent already observes (relative positions of its neighbours),
so it is decentralized. `margin` is a safety buffer: with margin = 1 the shield even
allows for a neighbour stepping towards us by one full step at the same moment.

Why this matters for the project
    The policy is learned, so nothing GUARANTEES it respects the constraint after
    transfer. The shield is a simple way to ENFORCE the constraint at execution time.
    We then measure what it costs (success rate) and how often it can't find a safe
    move (fallback rate).
"""
import torch

from env import ACTIONS


def shield_action(logits, obs, step_size, collision_dist, margin=1.0):
    """
    logits: [B, N, 5] policy preferences, obs: [B, N, obs_dim].
    Returns (action [B, N], used_fallback [B, N] bool).
    """
    B, N, _ = obs.shape
    rel = obs[..., 4:].reshape(B, N, -1, 2)                       # neighbours relative to me
    move = ACTIONS * step_size                                     # [5, 2] displacement of each action
    after = rel[:, :, None, :, :] - move[None, None, :, None, :]   # [B, N, 5, k, 2] neighbour rel. pos AFTER my move
    clearance = torch.linalg.norm(after, dim=-1).amin(dim=-1)      # [B, N, 5] distance to closest neighbour
    safe = clearance >= collision_dist + margin * step_size        # [B, N, 5]

    best_safe = logits.masked_fill(~safe, float("-inf")).argmax(-1)   # policy's favourite among safe moves
    least_bad = clearance.argmax(-1)                                   # used only if nothing is safe
    any_safe = safe.any(dim=-1)
    return torch.where(any_safe, best_safe, least_bad), ~any_safe
