# N109 preregistration: E0 single-path replay with the N039 v14 candidate (path v7.0 + engine n020.1), relaunch

N102 (`../n102_e0_replay_v13/`, same code) exited externally after 20 rows at 21:21 on 2026-10-03
(no error in its log); its rows are not used and N109 reruns all 400 rows from the start.
Earlier: N100 (`../n100_e0_replay_v12/`, same code) was stopped by me together with N039 v13 after a few
rows, because running it next to five formal workers exceeded the baseline's four-way
concurrency (LOG N101). Its rows are not used. N102 runs the identical frozen code of N039 v14
(`run_n100_e0_path_v7.py`, `nhz_path_v7.py` path v7.0, `nhz_sound_v20.py` n020.1, terminal
v8.3/v7.9) over all 400 E0 rows, CIFAR100 then TinyImageNet, 100 s per row, and starts only
after all four N039 v14 workers have exited, so nothing else runs beside it.

Ledger rules as N015/N061: all 61 anchors must stay ADV (they are attack-found witnesses, so
retention is expected to fail); new solves only on the 339 UNKNOWN rows; CERTs count after the
sampling audit, ADVs only from the domain's own MILP incumbents after the witness audit. No
attack or sampling helper is used anywhere in the path (user rule, LOG N099).
