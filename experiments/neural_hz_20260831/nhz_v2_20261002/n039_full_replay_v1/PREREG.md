# N039 preregistration: first single-path replay of the formal 2413-row universe

Written before any row of this replay is run (2026-10-02/03, Australia/Sydney).

## Candidate (frozen by FREEZE_SHA256SUMS)

Path v3 (`nhz_path_v3.py`): A aligned propagation with 300-iteration GPU LP
tightening and exact-LP polishing of small LPs (K*R <= 6e6, <= 1500 unstable);
B rigorous epigraph terminal LP per unsafe disjunct; C LP box-maximiser
witnesses; D last-ReLU-layer MILP (40 percent of the remaining budget, or the
whole budget when it already covers all ReLU layers); E all-ReLU-layers MILP
(remaining budget); HiGHS 1.14 single thread, objective cutoff, margin 1e-4.
Engine: `nhz_sound_v9.py` n009.1 (rigorous; float32 storage, float64
arithmetic, exact storage rounding; ReLU exact phases; Sigmoid/Tanh lambda
level; rigorous bilinear MatMul and Softmax; ConvTranspose, Upsample, Pad,
shape folding). One configuration for every row; three workers partition the
rows by family only.

## Universe, budgets, baseline

2,413 rows = the 2,213-row 12-family overlay
(`/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv`,
budget `csv_timeout`, baseline `raw_verdict`) + the 200-row strict ViT CSV
(`/data1/Kane/HyZor/vit_hz_legacy1_100s_20260826/consolidated_strict_100s.csv`,
budget `timeout_sec`, baseline `strict_status`). Verified count: CERT 1063,
ADV 807, TIMEOUT 274, UNKNOWN 269 (matches BASELINE_LOCK.md).

## Outcome rules

CERT if every unsafe disjunct of every input box is excluded; ADV if an ONNX
Runtime replay of a witness satisfies the original property (S1 semantics, as
the baseline); TIMEOUT if the row's official budget is exhausted (results
found after the budget count as TIMEOUT); UNKNOWN; ERROR on exceptions
(unsupported operators included).

Formal promotion (GOAL_CHARTER) requires: every one of the 1,870 solved rows
retained with the same verdict, no family solved count lower, invalid ADV = 0
(N022 audit), no CERT/ADV conflict, and at least one new solved row. Known in
advance from retention samples: dist_shift loses about 8 CERTs that need
sigmoid nonconvexity; cGAN small_transformer rows use unsupported MaxPool and
AveragePool. The replay is therefore EXPECTED TO FAIL promotion; its purpose
is the first measured single-path vector over the whole formal universe.

Mandatory post-run audits: `audit_n017_cert_sampling.py` on every CERT,
`audit_n022_witness_replay.py` (S1/S2) on every ADV.
