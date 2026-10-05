# N039 v2 preregistration: single-path replay of the formal 2413-row universe, path v4.2

v1 (`../n039_full_replay_v1/`, path v3) was stopped by me after 786 rows
because the smoke test N042 showed that path v4 retains SafeNLP/Cersyve ADV
rows that path v3 loses to continued MILP optimisation (v1 partial summary:
safenlp 711/755 solved rows kept, 30 ADV and 14 CERT lost to the 20 s budget;
no conflict). v1's partial files are retained and not merged with v2.

Candidate: identical to v1 (engine n009.1, path v3 stages A-E, budgets,
universe, outcome rules, audits; see `../n039_full_replay_v1/PREREG.md`)
except the terminal MILP (`nhz_terminal_v4.py`, terminal-v4.2): a fixed seed
portfolio (HiGHS random_seed 0, 1, 2, 3) solved concurrently in threads, early
stop at the first solution with violation >= 1e-6 (`objective_target`), and a
MIP-interrupt callback stopping the other seeds once one is decisive; the
upper bound is the minimum of the valid dual bounds. The seed set is the same
for every instance (like the baseline's parallel portfolio branches). Two
workers partition rows by family. Expected to FAIL promotion (dist_shift
sigmoid rows, budget-bound rows); purpose: the first single-path vector.
