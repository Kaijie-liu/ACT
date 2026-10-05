# N092 preregistration: E0 single-path replay with the N039 v12 candidate (path v6.7 + engine n019)

N084 (`../n084_e0_replay_v10/`) was stopped together with N039 v11; its rows are not used
(LOG N094). Code: `run_n092_e0_path_v66.py` (N068 runner with engine n019 and the OOM fraction
retry), `nhz_path_v6.py` path v6.7, terminal v8.3/v7.9. All 400 E0 rows, CIFAR100 then
TinyImageNet, 100 s per row, nice 19 next to the formal replay. Ledger rules as N015/N061.
E0 promotion is expected to fail on the attack-found anchors (83, 114 and others).
