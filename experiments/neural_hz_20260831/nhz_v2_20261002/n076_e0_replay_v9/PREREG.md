# N076 preregistration: E0 single-path replay with the N039 v10 candidate (path v6.4 + engine n017)

N072 (`../n072_e0_replay_v8/`) was stopped together with N039 v9 (SCIP member, LOG N075); its rows
are not used. Earlier: N070 (`../n070_e0_replay_v7/`) was stopped together with N039 v8 (centre witness, LOG N072); its rows
are not used. Earlier: N068 (`../n068_e0_replay_v6/`) was stopped together with N039 v7 (fail-closed fix, LOG N070); its
rows are not used. Earlier: supersedes N061 (`../n061_e0_replay_v5/`, path v5.5 + engine n011.2), which I
stopped because the formal candidate moved to path v6.1 + engine n017. The E0
replay must run the same code as the formal replay. N061 files are kept and
never merged.

Code: `run_n068_e0_path_v6.py` (N061 runner + one OOM retry inside the row
deadline), `nhz_path_v6.py` path v6.4, `nhz_sound_v17.py` n017, terminal v8.1.
All 400 E0 rows, CIFAR100 then TinyImageNet, 100 s per row, nice 19 next to the
formal replay. Ledger rules as N015/N061. Expected failure of E0 promotion:
anchors 83 and 114 (attack-found witnesses).
