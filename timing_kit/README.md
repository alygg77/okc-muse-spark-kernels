# Paired cold-L2 CUDA-graph timing kit (H200)

We used this host-side harness for every local speed claim before a file went to the OKC judge.
It times a **base**, a **candidate** and an **A/A** arm (a byte copy of the base) the way the
judge does: CUDA-graph replays from a cold L2.

It never goes into a submission. It only needs Python and PyTorch with CUDA.

## Files

- `paired.py`: the A/B/A harness (`paired(arms, ...)`), version 2. What it does:
  - It builds 8 captures per arm. Each capture's state is allocated after a seeded random
    0-32 MiB throwaway allocation, so memory layout varies.
  - It runs 48 ABBA rounds, with begin/end events recorded inside the graph replay.
  - It takes each 95% CI over the per-capture means (Student t, df = captures - 1), because
    rounds that share a capture share its memory layout.
  - It runs at least 100 checked replays per arm.
  - It reports `counted_gain` = true only when the gain's CI low is above 0 AND the gain is
    larger than the largest |A/A| difference.
- `pool_runs.py`: pools several result JSONs. Big states (5-10 GiB) fit only `captures=4`, so we
  run 2 or more processes per memory pad and pool them to 8 captures.
- `stalls.py`: a small parser for profiler stall-reason CSV output.

## Use

```python
from timing_kit.paired import Arm, paired

arms = [Arm("base", prepare_a, run_a, check_a, refresh, sources=("base.py",)),
        Arm("candidate", prepare_b, run_b, check_b, refresh, sources=("cand.py",)),
        Arm("aa", prepare_a2, run_a2, check_a2, refresh, sources=("base_copy.py",))]
result = paired(arms, seed=1)          # captures=8, rounds=48 by default
```

`prepare` returns a private state. `run(state)` must be CUDA-graph safe. `check(state, out)` raises
on a wrong result. `refresh(state, seed)` writes the same fresh inputs into every arm. Each source
file's MD5 hash goes into the result, so every number binds to the exact files it measured.

## Why the A/A arm matters

On the H200, two byte-identical files differed by up to ~3 us per shape from memory layout
alone (largest |A/A| per shape: 0.7-3.1 us on muse_moe_w4a16). A gain smaller than the A/A spread
is noise. On the judge, four runs of one unchanged muse_bf16_proj file on one day scored 1.526x,
1.541x, 1.544x and 1.550x, so one judge run cannot show a gain under about 2%.
