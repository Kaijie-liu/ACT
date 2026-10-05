# N068 preregistration: E0 single-path replay with the N039 v7 candidate

Supersedes N061 (`../n061_e0_replay_v5/`, path v5.5 + engine n011.2), which I
stopped because the formal candidate moved to path v6.1 + engine n017. The E0
replay must run the same code as the formal replay. N061 files are kept and
never merged.

Code: `run_n068_e0_path_v6.py` (N061 runner + one OOM retry inside the row
deadline), `nhz_path_v6.py` path v6.1, `nhz_sound_v17.py` n017, terminal v8.1.
All 400 E0 rows, CIFAR100 then TinyImageNet, 100 s per row, nice 19 next to the
formal replay. Ledger rules as N015/N061. Expected failure of E0 promotion:
anchors 83 and 114 (attack-found witnesses).
