# N039 v14 preregistration: single-path 2413-row replay, path v7.0 (domain-only verdicts) + engine n020.1, 4-way concurrency

v13 (`../n039_full_replay_v13/`, same code) was stopped by me after 20 minutes (384 rows):
with five formal workers plus the E0 replay it ran more concurrently than the baseline's
four-way setting (hard limit 6), and SafeNLP 272 and 290 (CERT, retained by v12 with three
workers) timed out at 20.1-20.2 s. v14 runs the same frozen code with four formal workers and
no other replay in parallel; the E0 replay N100 runs after it. No v13 row is used.

User rule (2026-10-03): gains may not come from any attack or sampling helper; they must come
from Neural-HZ itself (LOG N099). N039 v12 (path v6.7) used LP-dual box-corner witnesses, the
input-box centre and a centre-seeded MIP start, so it was stopped and is diagnostic only. No v12
row is used.

Candidate (one configuration for every row; workers only partition families):
- engine n020.1 (`nhz_sound_v20.py`) = n019 (n017 + AveragePool/MaxPool) + LP tightening and
  exact-LP polishing stop after 40 percent of the row budget (sound: looser bounds only);
- path v7.0 (`nhz_path_v7.py`): A propagate; B rigorous LP over the lambda concretisation (CERT
  if every disjunct is excluded); D last ReLU layer gamma; E all ReLU layers gamma; F E + two
  phase segments on smooth units (only if E did not exclude). ADV only from incumbents of these
  MILP plans, validated by ONNX Runtime on the original network and VNNLIB property, after
  deterministic float32 rounding of the incumbent's input (inward step, snapping coordinates
  within 1e-6 of a bound). No LP-dual corners, no fixed sample points, no seeded MIP starts,
  no attack, no sampling. Violation target only on exact plans; rerun of a stage stopped by a
  non-real incumbent; fail-closed handling of non-finite data;
- terminal v8.3 / v7.9 (Balas rows, smooth rows, segments, Proposition 1' sign rule, HiGHS seeds
  0-2 + SCIP member, valid dual bounds only, tiny coefficients folded); the MIP-start option of
  v8.3/v7.9 is not used by path v7.0;
- runner `run_n039v13_full_replay.py` (OOM retry inside the row deadline with a larger memory
  fraction when the device has room).
- workers (4, matching the baseline's four-way concurrency): wA safenlp_2024, sat_relu,
  malbeware, cersyve, metaroom_2023, cgan_2023; wB acasxu_2023, relusplitter; wC
  linearizenn_2024, dist_shift_2023; wD vit_2023, cora_2024, tllverifybench_2023. Memory
  fraction 0.2 each.

Accounting: retention of every baseline CERT/ADV per family; any loss, conflict or invalid
witness fails promotion; ADVs counted after `audit_n039_witness.py` (S1 and S2 reported),
CERTs after `audit_n039_cert_sampling.py` (audit only). Open user decisions: integrality
skipping by Proposition 1' under hard limit 2; S1 versus S2 witness semantics; the
deterministic rounding of MILP incumbents.

Expected before the run (probes): ADVs that only the removed helpers found are lost (for example
ACAS Xu 71 and 87, Cora 62 and 134); dist_shift 39, 42, 51 and 71; linearizenn rows needing
500-840 s; about half of the ViT IBP CERTs; near-budget SafeNLP rows.
