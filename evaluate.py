"""
evaluate.py - THE transfer experiment.

For every trained policy we run it, WITHOUT any retraining, on several team sizes:
    - N agents          (the size it was trained on -> sanity check)
    - N+1, N+2, N+3     (unseen sizes -> the transfer test)
and measure the constraint-related metrics from env.metrics().

Extra rows in the results:
    greedy        a hand-coded controller that walks straight to its goal and ignores
                  others. NOT learned; shows what "no avoidance" looks like.
    <kind>+shield the same policy with the runtime safety shield switched on (shield.py).
    <kind>@ref    a policy of this type trained DIRECTLY on the larger team size
                  (--ref-dir). The gap between this and the transferred policy is the
                  price of transferring instead of retraining.

Usage:
    python evaluate.py --run-dir runs --seeds 0 1 2 --sizes 3 4 5 6 --ref-dir runs_n4
"""
import argparse
import json
import os

import numpy as np
import torch

from env import NavEnv
from policies import load_policy, NEEDS_FULL_NEIGHBORS
from shield import shield_action
from plots import plot_transfer, plot_shield, plot_training

EVAL_SEED = 12345   # same random worlds for every policy -> fair, paired comparison
KINDS = ["shared", "independent", "attention"]


def evaluate_policy(policy, full_neighbors, n_agents, episodes=500, seed=EVAL_SEED, shield_margin=None):
    env = NavEnv(n_agents, episodes, full_neighbors=full_neighbors, seed=seed)
    obs = env.reset()
    fallbacks = []
    for _ in range(env.max_steps):
        with torch.no_grad():
            logits, _ = policy(obs)
        if shield_margin is None:
            action = logits.argmax(-1)                      # plain greedy choice from the policy
        else:
            action, used_fallback = shield_action(logits, obs, env.step_size, env.collision_dist, shield_margin)
            fallbacks.append(used_fallback.float().mean().item())
        obs, _, _ = env.step(action)
    m = env.metrics()
    m["fallback_rate"] = float(np.mean(fallbacks)) if fallbacks else 0.0   # steps where no safe move existed
    return m


def greedy_action(obs, goal_dist=0.05):
    """Move along the axis where the goal is farthest away; stay once on the goal."""
    rel = obs[..., 2:4]                                  # goal - position
    up_down = torch.where(rel[..., 1] > 0, 1, 2)         # 1 = up, 2 = down
    left_right = torch.where(rel[..., 0] > 0, 4, 3)      # 4 = right, 3 = left
    act = torch.where(rel[..., 0].abs() > rel[..., 1].abs(), left_right, up_down)
    return torch.where(torch.linalg.norm(rel, dim=-1) < goal_dist, torch.zeros_like(act), act)


def evaluate_greedy(n_agents, episodes=500, seed=EVAL_SEED):
    env = NavEnv(n_agents, episodes, seed=seed)
    obs = env.reset()
    for _ in range(env.max_steps):
        obs, _, _ = env.step(greedy_action(obs))
    m = env.metrics()
    m["fallback_rate"] = 0.0
    return m


def add(results, name, n, metrics):
    for k, v in metrics.items():
        results.setdefault(name, {}).setdefault(n, {}).setdefault(k, []).append(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs")
    p.add_argument("--n-train", type=int, default=3)
    p.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="team sizes to test (default: N and N+1)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--shield-margin", type=float, default=1.0,
                   help="safety buffer of the shield in step sizes; negative = skip shield runs")
    p.add_argument("--ref-dir", default=None, help="folder with policies trained directly on a larger team")
    a = p.parse_args()
    sizes = a.sizes or [a.n_train, a.n_train + 1]

    # results[name][size][metric] = list of values (one per training seed)
    results = {}
    for kind in KINDS:
        for seed in a.seeds:
            path = os.path.join(a.run_dir, f"{kind}_seed{seed}.pt")
            if not os.path.exists(path):
                print("missing", path)
                continue
            policy, _ = load_policy(path)
            full = NEEDS_FULL_NEIGHBORS[kind]
            for n in sizes:
                add(results, kind, n, evaluate_policy(policy, full, n, a.episodes))
                if a.shield_margin >= 0:
                    add(results, kind + "+shield", n,
                        evaluate_policy(policy, full, n, a.episodes, shield_margin=a.shield_margin))
    for n in sizes:
        add(results, "greedy", n, evaluate_greedy(n, a.episodes))

    if a.ref_dir:
        for kind in KINDS:
            for seed in a.seeds:
                path = os.path.join(a.ref_dir, f"{kind}_seed{seed}.pt")
                if os.path.exists(path):
                    policy, ckpt = load_policy(path)
                    n = ckpt["n_agents"]
                    add(results, kind + "@ref", n, evaluate_policy(policy, NEEDS_FULL_NEIGHBORS[kind], n, a.episodes))

    with open(os.path.join(a.run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)

    # ---- print a readable table (mean +- std over training seeds)
    cols = ["min_separation", "violation_rate", "episodes_with_collision", "success_rate", "fallback_rate"]
    print(f"\n{'policy':<18}{'agents':>7}  " + "".join(f"{c:>25}" for c in cols))
    for name in results:
        for n in sorted(results[name]):
            row = results[name][n]
            cells = [f"{np.mean(row[c]):.3f} +- {np.std(row[c]):.3f}" for c in cols]
            print(f"{name:<18}{n:>7}  " + "".join(f"{c:>25}" for c in cells))

    plot_transfer(results, sizes, os.path.join(a.run_dir, "transfer_metrics.png"))
    plot_shield(results, sizes, os.path.join(a.run_dir, "shield_effect.png"))
    plot_training(a.run_dir, a.seeds, os.path.join(a.run_dir, "training_curves.png"))
    print("\nsaved results.json and the figures in", a.run_dir)


if __name__ == "__main__":
    main()
