# N015 v2 preregistration: E0 single-path replay (supersedes the interrupted v1)

v1 (`../n015_e0_replay_v1/`) was stopped by me after 14 CIFAR rows. Reason:
the independent witness audit N022 v1/v2 reported MetaRoom witnesses outside
the input box; I wrongly suspected a pipeline defect. Investigation showed the
audit used a stricter input semantics (S2: the float32 point itself inside the
box) than the frozen baseline's ADV gate (S1: a real point inside the box,
evaluated by ONNX Runtime on its float32 rounding; `hz_full_worker._is_cex`).
v1's decision logic already implements S1. v1's partial file is retained
unchanged and is not merged with v2.

v2 differs from v1 only in bookkeeping: `run_n015v2_e0_sound_pipeline.py`
also saves the float64 witness point. All decision rules, iteration counts,
budgets (100 s per row), MILP plan (gamma = last ReLU layer, lambda = all
layers, HiGHS cutoff, margin 1e-4), universe (all 400 E0 rows) and ledger
rules are exactly those of `../n015_e0_replay_v1/PREREG.md`.

ADV acceptance: S1 (baseline semantics), audited independently by
`audit_n022_witness_replay.py` (v3), which also reports S2 for transparency.
CERT soundness check: `audit_n017_cert_sampling.py` (random ORT samples) on
every CERT; any violation invalidates the run.
