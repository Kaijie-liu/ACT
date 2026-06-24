# HybridZ Stop/Continue Decision 2026-06-24

Scope: strict pure exact Hybrid Zonotope only.  No input split, sampling,
LP-witness promotion, CROWN/backward tightening, Gurobi-counted proof, or
per-instance tuned rescue path.

## Evidence Checked

Current soundfix artifact:

`/data1/Kane/ICSE/act_hybridz_soundfix_20260625/`

Machine-readable summary:

`/data1/Kane/ICSE/act_hybridz_soundfix_20260625/hybridz_suite_summary_soundfix_20260625.csv`

Historical frozen artifact:

`/data1/Kane/ICSE/act_hybridz_clean_20260624_cora25/`

Current audit state:

- 12 benchmarks covered.
- 2213 expected rows and 2213 artifact rows.
- `P0=0`, `ERROR=0`.
- `CERT=977`, `ADV=786`, `V+A=1763`.
- The old metaroom `100/100` row is superseded by a soundfix:
  `94 CERT / 1 ADV / 5 TIMEOUT = 95/100`.
- Manifest check passes for `_MANIFEST.sha256`.

## Current Cross-Tool Position

| Benchmark | Current V+A | Position | Decision |
| --- | ---: | --- | --- |
| metaroom_2023 | 95/100 | #1 | soundfix freeze |
| sat_relu | 100/100 | #1 tie | freeze |
| malbeware | 150/150 | #1 | freeze |
| cersyve | 11/12 | #1 | freeze headline, keep iid11 as research |
| dist_shift_2023 | 70/72 | #1 | freeze headline, research S-curve tail |
| cora_2024 | 25/180 | #1 | freeze strict headline |
| safenlp_2024 | 1079/1080 | #2, gap 1 | freeze unless a benchmark-wide fix appears |
| cgan_2023 | 13/21 | #2, gap 6 | research sparse exact-HZ/presolve |
| tllverifybench_2023 | 17/32 | #3, gap 13 | research sparse exact-HZ/presolve |
| relusplitter | 43/220 | #3, gap 70 | do not count old 102 candidate |
| linearizenn_2024 | 40/60 | below top tier | binary-MIP research only |
| acasxu_2023 | 120/186 | below top tier | dense exact-MIP research only |

## Remaining Frontier

The unresolved frontier is 450 rows.  Only 14 rows are high-priority structural
targets:

- `cgan_2023`: 7 representation-drop rows.
- `dist_shift_2023`: 2 S-curve operator-tail rows.

The large remaining mass is not a cheap rerun target:

- `acasxu_2023`: dense exact-MIP wall.
- `linearizenn_2024`: binary-MIP wall.
- `relusplitter`: official-wall timeout plus sparse-tail MIP wall; the old 102
  candidate is not currently reproducible.
- `tllverifybench_2023`: large sparse MIP wall.
- `cora_2024`: mostly official-wall timeout under strict current rules.

## Tail Probe Update

Additional strict pure-HZ diagnostics were run for the two smallest tails:

`audit_results/hz_tail_probe_20260624/`

- `safenlp_2024` iid454 is solvable as CERT with compressed exact ReLU plus
  valid ReLU cuts, but the run takes about 34.5s wall and therefore misses the
  official 20s wall. The same configuration with a 20s MILP budget remains
  UNKNOWN. HiGHS thread/parallel/presolve/heuristic options with 1/2/4/8 solver
  threads also remain UNKNOWN under the 20s engine wall.
- `cersyve` iid11 remains UNKNOWN after a 120s compressed exact-ReLU plus valid
  cuts run.

These are useful diagnostics but not new reportable numbers. The safenlp case
is a near-wall solver scheduling target; the cersyve tail is not a cheap
one-off.

## Decision

Freeze the current artifact as the reportable result.

Continue HybridZ work only as a structural research branch, not as more
scoreboard reruns.  The next valid improvements must be benchmark-wide and
operator/solver-level:

1. Sparse exact-HZ propagation and block/Schur presolve for `cgan` and
   `tllverify`.
2. Sound, toy-oracle-validated S-curve tightening for the last `dist_shift`
   rows.
3. Exact MILP presolve/probing/compressed ReLU improvements for binary-heavy
   datasets.

Do not replace the frozen headline until a new one-shot frozen run reproduces
the improvement with `P0=0`, complete iid coverage, and the same strict guard.
