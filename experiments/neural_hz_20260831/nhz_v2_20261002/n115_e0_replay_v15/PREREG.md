# N115 preregistration: E0 single-path replay, path v7.2 + engine n020.1 (all 400 rows)

N109 (path v7.0) was stopped after 170 rows because of the false CERT on row 164 (LOG N111/N112).
N115 reruns all 400 E0 rows with path v7.2 (`nhz_path_v7_2.py`: domain-only verdicts, terminal
v7.11/v8.5 with the numerical safety width, empty-plan fix), engine n020.1, 100 s per row, alone
on the machine (`run_n115_e0_path_v72.py`). Ledger rules as N015/N061: all 61 attack-found anchors
must stay ADV (expected to fail); new solves only on the 339 UNKNOWN rows; CERTs count after the
sampling audit, ADVs only from the domain's own MILP incumbents after the witness audit.
