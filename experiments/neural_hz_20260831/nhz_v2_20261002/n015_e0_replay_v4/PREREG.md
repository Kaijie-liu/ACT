# N015 v4 preregistration: E0 single-path replay over all 400 rows

Supersedes the incomplete v1-v3 runs (retained unchanged, never merged):
v1 stopped after 14 rows (mistaken witness-defect diagnosis, LOG N022), v2
after 3 rows (memory), v3 after 4 rows because a single engine is required for
both families and the float64 engine cannot hold TinyImageNet's dense
generators.

Engine: `nhz_sound_mp.py` n006.2, the sound projection-aligned HZ engine with
float32 generator storage, float64 arithmetic and the exact storage rounding
added to the rounding radius (test `test_n025_mp_engine.py`: CIFAR rows 28/29
rigorous worst bounds -0.0529/-0.0383 versus -0.0539/-0.0398 in float64;
output radius about 3e-5; TinyImageNet rows peak about 10 GB, 12-13 s).

Decision logic, budgets (100 s per row), iteration counts, MILP plan
(gamma = last ReLU layer with phases, lambda = all layers, HiGHS cutoff,
margin 1e-4, single thread), witness sources (LP box maximiser, MILP
incumbent), ADV semantics S1 (as the 1870 baseline) and all ledger rules are
those of `../n015_e0_replay_v1/PREREG.md` and `../n015_e0_replay_v2/PREREG.md`.
Universe: all 400 E0 rows, CIFAR100 then TinyImageNet, one configuration.

Post-run audits (mandatory before reporting): `audit_n017_cert_sampling.py`
on every CERT; `audit_n022_witness_replay.py` (S1 and S2) on every ADV.
