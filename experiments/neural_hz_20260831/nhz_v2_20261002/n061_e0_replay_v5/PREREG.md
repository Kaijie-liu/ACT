# N061 preregistration: E0 single-path replay with the formal candidate (path v5.5 + engine n011.2)

E0 v4 (`../n015_e0_replay_v4/`, path v1 + engine n006.2) was stopped by me
after 127 CIFAR rows: it had already failed E0 retention (row 83,
VALIDATED_ADV -> TIMEOUT) and runs an older path than the formal candidate.
Its 127 rows are kept and never merged (15 CERT on UNKNOWN rows, 11 of 12
anchors kept by row 127).

N061 runs the code of N039 v6 (`nhz_path_v5.py` path v5.5,
`nhz_sound_v11b.py` n011.2, terminal v7.5) over all 400 E0 rows, CIFAR100 then
TinyImageNet, 100 s per row (`run_n061_e0_path_v5.py`), at low CPU priority
(nice 19) next to the formal replay. Ledger rules as N015: all 61 anchors
must stay ADV; new solves are counted only on the 339 UNKNOWN rows, CERTs after
the sampling audit and ADVs after the S1/S2 witness audit. Expected: anchors
83 and 114 are not found (their witnesses came from an attack sidecar; LOG
N033 and the note on E0 anchors), so E0 promotion is expected to fail. The run
measures capability on the same path as the formal replay.
