# Population transfer (N -> N+1 agents): shared vs independent vs attention policies

Individual contribution to the project *Enforcing Constraints for Policy Transfer Across Agent Populations and Spatial Scales*.

**Question.** If a team of agents is trained to move to goals without colliding, does the learned
policy still respect the collision constraint when new agents join (N -> N+1, N+2, N+3) with no
retraining? And can the constraint be enforced at run time if the policy alone is not enough?

## The three policy types

| | Shared | Independent | Attention |
|---|---|---|---|
| Networks | one, used by every agent | one per agent | one, used by every agent |
| What an agent sees | own state + 2 nearest neighbours | own state + 2 nearest neighbours | own state + ALL other agents (as a set) |
| New agent at test time | uses the same network | borrows network `i mod N` (our design choice) | uses the same network; just one more neighbour in the set |

Plus two extras: a hand-coded **greedy** controller (walks straight to its goal, ignores others) as a
"no avoidance" reference, and a **safety shield** (`shield.py`) that vetoes moves that would bring an agent
closer than `collision_dist + 1 step` to a neighbour.

## Files

| File | What it does |
|---|---|
| `env.py` | Batched 2D navigation world, observation, reward, metrics |
| `policies.py` | `SharedPolicy`, `IndependentPolicies`, `AttentionPolicy` |
| `train.py` | PPO training (`--policy shared/independent/attention`, `--n-agents`) |
| `shield.py` | Runtime safety shield |
| `evaluate.py` | Transfer test on several team sizes, with/without shield, plus reference runs; prints the table and makes the figures |
| `plots.py` | Figures |
| `run_all.sh` | Whole experiment |

## Run

```bash
pip install torch numpy matplotlib
python train.py --policy shared --seed 0          # ~1 min on a laptop CPU (attention ~2-3 min)
python evaluate.py --run-dir runs --seeds 0 --sizes 3 4 5 6
./run_all.sh                                      # everything (3 seeds, reference runs on 4 agents)
```

## Setup

- 3 agents in the unit square, 5 moves, 50 steps per episode. Collision = centres closer than 0.1.
- Reward: `10 * progress_to_goal - 10 * (in collision) + 0.2 * (on goal)`.
  (With a penalty of 1 or 3 the agents learned almost no avoidance; 10 works.)
- Training: PPO, 400 iterations, 64 parallel worlds, 3 seeds per policy type, local critics (IPPO style).
- Evaluation: 500 fresh episodes per setting, same random worlds for every policy, greedy (argmax) actions.
- Reference ("@ref"): each policy type trained directly on 4 agents, tested on 4 agents.

## Metrics

- **min_separation**: closest any two agents got in an episode (higher is safer; below 0.1 is a collision).
- **violation_rate**: fraction of time steps with at least one collision.
- **episodes_with_collision**: fraction of episodes with at least one collision.
- **success_rate**: fraction of agents on their goal at the end.
- **fallback_rate**: fraction of steps where the shield found no safe move (0 in all our runs).

## Main results (mean over 3 training seeds, trained on 3 agents)

| Episodes with a collision | 3 agents | 4 | 5 | 6 |
|---|---|---|---|---|
| Greedy (no avoidance) | 40% | 67% | 82% | 93% |
| Independent | 21% | 39% | 58% | 72% |
| Shared | 16% | 33% | 50% | 66% |
| Attention | 1% | 3% | 9% | 15% |
| Any policy + shield | 0% | 0% | 0% | 0% |

Success rate with the shield (the price of safety): shared 92% -> 79%, independent 88% -> 75%,
attention 96% -> 87% (from 3 to 6 agents). Without the shield success stays at 94-99%.

Trained directly on 4 agents (episodes with a collision at 4 agents): shared 15%, independent 26%,
attention 8% (high variance across seeds). So for shared and independent policies, transferring from 3
costs real safety compared to retraining; the attention policy transferred from 3 is about as safe as
the attention policy trained on 4.

## Limitations (say these in the report and Q&A)

- More agents in the same arena means a denser world, so some safety loss is expected even for a perfect
  policy. The greedy line shows how much comes from density alone.
- **Confound:** the shared and independent policies only see the 2 nearest neighbours, so with 4+ agents they
  cannot see everyone. The attention policy sees all agents. Its advantage at larger N is therefore partly
  architecture and partly information. At N=3 both see everyone, and attention is still safer there.
- The independent-policy rule for extra agents is one choice among several.
- The shield checks that after my move every observed neighbour is at least `collision_dist + 1 step` away.
  If every agent does this and starts at least that far apart, a collision cannot happen in one step, because
  the neighbour can close at most one step. This is a sufficient-condition *idea*, not a full proof: it assumes a
  safe move exists (the fallback was never needed here), that all dangerous neighbours are observed, and ignores
  wall effects.
- Small world, discrete dynamics, 3 seeds. These are empirical results, not guarantees.
