# N100 preregistration: E0 single-path replay with the N039 v13 candidate (path v7.0 + engine n020.1)

N092 (`../n092_e0_replay_v11/`) used path v6.7 (LP-dual corners, centre point, centre MIP start)
and was stopped (LOG N099); its rows are not used. Code: `run_n100_e0_path_v7.py`, path v7.0,
engine n020.1. All 400 E0 rows, CIFAR100 then TinyImageNet, 100 s per row, nice 19. Ledger
rules as N015/N061: all 61 anchors must stay ADV (they are attack-found, so retention is
expected to fail); new solves only on the 339 UNKNOWN rows, CERTs after the sampling audit, ADVs
only from domain MILP incumbents after the witness audit.
