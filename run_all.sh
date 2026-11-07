#!/usr/bin/env bash
# Full experiment.
#  1) train all three policy types on 3 agents (3 seeds each) into runs/
#  2) train all three types directly on 4 agents (reference) into runs_n4/
#  3) test on 3,4,5,6 agents, with and without the safety shield
# Takes roughly 15-20 minutes on a laptop CPU. Use OMP_NUM_THREADS=1 when running streams in parallel.
set -e
ITERS=${ITERS:-400}
SEEDS=${SEEDS:-"0 1 2"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

( for s in $SEEDS; do for kind in shared independent attention; do
    python train.py --policy $kind --seed $s --iters $ITERS --out runs
  done; done ) &
( for s in $SEEDS; do for kind in shared independent attention; do
    python train.py --policy $kind --n-agents 4 --seed $s --iters $ITERS --out runs_n4
  done; done ) &
wait
python evaluate.py --run-dir runs --seeds $SEEDS --sizes 3 4 5 6 --ref-dir runs_n4
