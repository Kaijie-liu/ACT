# N015 v3 preregistration: E0 CIFAR100 replay (resource fix of v2)

v2 (`../n015_e0_replay_v2/`) was stopped by me after 3 rows, before any
CIFAR100-large row, because a side measurement (`probe_n024_sign_plan.py`)
showed a float64 CIFAR100-large row needs about 25 GB of GPU memory and the
v2 runner (memory fraction 0.3, previous row's state still referenced while the
next row propagates) would have produced spurious ERROR rows. v2's partial
file is retained and not merged.

v3 = v2 decision logic unchanged (same frozen modules; see v1 PREREG for the
rules) + explicit release of each row's state + memory fraction 0.5 + peak
GPU memory recorded per row. Universe of this run: the 200 CIFAR100 E0 rows.
TinyImageNet (float64 dense generators of about 17.6 GB per tensor) needs a
memory-lean engine variant and will be preregistered separately; until then the
E0 replay is incomplete (CIFAR half only) and cannot promote the E0 ledger.
