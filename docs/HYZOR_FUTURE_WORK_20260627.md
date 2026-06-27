# HyZor Future Work Notes

Date: 2026-06-27

Scope: future work for pure HybridZ.  These items are not counted in the
2026-06-27 frozen result table.  They must preserve the current rules:
no ReLU triangle relaxation, no CROWN rescue, no input split, no sampling-based
promotion, no ORT/LP witness promotion, and no commercial solver dependency in
the default artifact.

## Most Promising Directions

1. Tighter sound S-curve operators for `dist_shift_2023`.
   This remains the clearest non-binary bottleneck.  Focus on stronger
   compressed sigmoid/tanh graph/domain cuts and degeneracy pruning that are
   still sound HybridZ constraints.

2. Open-source exact-MILP portfolio for binary-wall benches.
   Target `acasxu_2023`, `linearizenn_2024`, `relusplitter`, and `cora_2024`.
   Keep the portfolio benchmark-wide, not sample-specific.  Candidate work:
   HiGHS/SCIP scheduling, exact phase fixing, warm starts from exact HZ state,
   sparse row scaling, and MIP option portfolios with memory-governed workers.

3. Sparse/matrix-free exact HZ propagation.
   Needed for large structured models such as `cgan_2023`, `tllverifybench_2023`,
   and CIFAR-like CNNs.  The target representation is still the exact HZ
   6-tuple, with row-only materialization for final margins and exact sparse
   ReLU constraints.

4. Block and Schur presolve for large sparse exact systems.
   Use equality structure from exact ReLU and affine layers to eliminate
   singletons and connected components before MILP export, without weakening the
   represented set.

5. Resource-aware benchmark profiles.
   Productize CPU/GPU parallelism as benchmark-level profiles with explicit
   memory floors and worker caps, so long-tail benches use available hardware
   without risking OOM or VS Code disconnects.

## Current Frozen Baseline

The frozen 2026-06-27 pure-HybridZ baseline is:

`1780 / 2213 = 980 CERT + 800 ADV`, `P0=0`, `ERROR=0`.

Authoritative CSVs are under:

`/data1/Kane/ICSE/act_hybridz_soundfix_20260625`

- `FINAL_HYBRIDZ_RESULTS_20260627_FINAL.csv`
- `FINAL_CROSS_TOOL_RANKING_20260627_FINAL.csv`
- `_CROSS_TOOL_SUMMARY_20260627_FINAL.csv`
- `_FINAL_20260627_MANIFEST.sha256`
