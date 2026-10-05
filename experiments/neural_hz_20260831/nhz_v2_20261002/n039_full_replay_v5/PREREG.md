# N039 v5 preregistration: single-path 2413-row replay, path v5.2 + engine n011.2

v4 (`../n039_full_replay_v4/`, path v5.1) was stopped by me minutes after launch because
sat_relu ADV rows 36, 40, 72, 78, 88 needed one more witness clean-up (path v5.2: MILP/LP
coordinates within 1e-6 of a box bound are snapped onto it before the checks; LOG N057).
v4 files are kept and never merged. Earlier:
Supersedes v3 (`../n039_full_replay_v3/`, path v4.3 + engine n009.2), which I
stopped after it had shown the SafeNLP budget losses (34 by row 1080) that
path v5 addresses (LOG N044-N053). v3's files are kept and never merged.

Candidate (one configuration for every row; workers only split families):
- engine n011.2 (`nhz_sound_v11b.py`): n009.2 plus the softmax rule of LOG N054
  (per output coordinate, the interval form when the tangent remainder alone
  is at least the interval radius; one rule, no constants);
- path v5.2 (`nhz_path_v5.py`): stages A-E as path v4; terminal v7.2
  (`nhz_terminal_v7.py`): disaggregated binary rows with the rigorous storage-gap
  tolerance, integrality skipped for units selected by Proposition 1' (last
  layer) or, on multi-layer plans with at most 2e7 unit-coordinate cells,
  Proposition 2; a stage stopped at the violation target by a spurious
  incumbent is re-run without the target; witnesses rounded to float32 toward
  the interior and checked strictly (S2) first, the baseline check (S1) second;
  every candidate is tried as decoded and with coordinates within 1e-6 of a box
  bound snapped onto that bound.
- budgets = the composite baseline timeouts; 4 HiGHS seeds per MILP; margins,
  universe and runner otherwise as v1-v3 (`run_n039v4_full_replay.py`).

Accounting (unchanged): retention of every baseline CERT/ADV per family; any
loss, conflict or invalid witness fails promotion. ADVs are counted only after
`audit_n039_witness.py` (S1 and S2 reported separately); CERTs only after the
sampling audit. The witness-source admissibility (LP box maximiser) and the
integrality-skipping rule under hard limit 2 remain decisions for the user and
are reported per row.

Known expected failures before the run (from probes, not part of the ledger):
ViT PGD rows (LOG N053/N054), dist_shift rows 0, 8, 12, 39, 48, 51, 58, 71
(LOG N027-N031), SafeNLP 844 and other near-budget SafeNLP CERTs.
