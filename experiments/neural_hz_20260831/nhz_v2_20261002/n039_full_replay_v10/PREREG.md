# N039 v10 preregistration: single-path 2413-row replay, path v6.4 + engine n017

v9 (`../n039_full_replay_v9/`, path v6.3) was stopped by me after about 45 minutes: CERSYVE 7 (baseline
CERT via a SCIP branch) is proved by SCIP in 60-83 s but not by four HiGHS seeds in 100 s
(LOG N074, N075). Terminal v7.7: the portfolio is HiGHS seeds 0-2 plus one SCIP member (forked
child process, killed when another member decides or at the deadline) on every plan with at
most 5e5 nonzeros; one uniform rule. No v9 row is used. (This file was rewritten two minutes
after launch because a shell substitution had failed; content as intended at launch.)
Earlier: v8 (`../n039_full_replay_v8/`, path v6.2) was stopped by me after about 15 minutes: the retention
sample showed ACAS Xu ADV rows 60, 74 and 88 lost although the input-box centre is a
counterexample on each (LOG N072). Path v6.3 checks the element centre (latent w = 0) as the
first witness candidate. Family order changed so that families no replay has reached run
first (wA: cersyve, cgan_2023, metaroom_2023, dist_shift_2023, safenlp_2024, sat_relu,
malbeware; wB: acasxu_2023, relusplitter, vit_2023, linearizenn_2024, tllverifybench_2023,
cora_2024). No v8 row is used. Earlier: v7 (`../n039_full_replay_v7/`, path v6.1) was stopped by me after about 40 minutes: path v6.1 would
have read a NaN LP bound as an exclusion (comparison `v >= -1e-4` is false for NaN), seen
with an experimental engine (LOG N070). Path v6.2 fails closed on non-finite states, rows and
bounds; terminal v7.6 never excludes with non-finite plan data. No v7 row is used. Earlier:
Supersedes v6 (`../n039_full_replay_v6/`, path v5.5 + engine n011.2), which I
stopped after about 1,430 rows (1,319 of 1,320 baseline solves kept; the one loss,
malbeware 103, was a CUDA out-of-memory ERROR, not a verdict). v6 never reached
the smooth or attention families. For those, LOG N065 showed that every
MILP plan of paths up to v5.5 dropped the smooth and softmax rows. v6 files are
kept and never merged.

Candidate (one configuration for every row; workers only split families):
- engine n017 (`nhz_sound_v17.py`) = n011.2 + n014 (softmax bounds from score
  differences computed as q . (k_b - k_a)) + n016 (recorded smooth units);
- path v6.4 (`nhz_path_v6.py`) = path v5.5 with terminal v8.1 (on v7.7)
  (`nhz_terminal_v8.py`): smooth units written into every MILP plan with explicit
  pre-activation columns and rigorous line rows; a final stage F (all ReLU layers
  gamma + two phase segments, split at the inflection point, on every smooth unit
  whose output range exceeds 1e-4), run only if stage E did not exclude; the
  violation-target early stop only on exact plans (all ReLU layers gamma, no
  smooth or softmax units); everything else as path v5.5 (terminal v7.5 rules,
  inward witness rounding, bound snapping, S2 then S1);
- runner `run_n039v7_full_replay.py`: v4 runner plus one retry of a row after a
  CUDA out-of-memory error inside the row's own deadline (resource failure,
  never a verdict).

Accounting unchanged: retention of every baseline CERT/ADV per family; any
loss, conflict or invalid witness fails promotion; ADVs counted after
`audit_n039_witness.py`, CERTs after the sampling audit; witness-source
admissibility and the integrality rule under hard limit 2 are reported for the
user's decision.

Expected before the run (probes, not the ledger): ViT retention incomplete
(LOG N057, N064), dist_shift 57 of 63 from smooth rows alone plus stage F rows
(row 0 CERT at 268 s in the probe), SafeNLP near-budget CERTs load-sensitive.
