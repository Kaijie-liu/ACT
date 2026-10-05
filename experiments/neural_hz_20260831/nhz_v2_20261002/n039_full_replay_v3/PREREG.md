# N039 v3 preregistration: single-path 2413-row replay, path v4.3 + engine n009.2

v2 (`../n039_full_replay_v2/`) was stopped by me after 534 SafeNLP rows
(520 kept, 5 ADV and 8 CERT lost to the 20 s budget, no conflict) because its
second worker spent more than 13 minutes inside the first Cora row: engine
n009.1's exact-LP polishing had no budget awareness and polished every unstable
neuron (about 3,500 dense LPs on a 7x250 Cora network). v2's files are kept
and not merged.

v3 = v2 except: engine n009.2 caps polishing (at most 100 unstable neurons per
layer, at most 600 LPs per propagation, and only within the first 25 percent
of the row's budget; all three constants fixed here) and path v4.3 passes the
row deadline to the engine. Everything else (stages, seed portfolio, early
stop, margins, budgets, universe, S1 witnesses, audits, promotion rule) is
as preregistered for v1/v2. Smoke after the fix (not part of this run):
cora 0 ADV 1.7 s (baseline TIMEOUT), cora 2 and 5 CERT about 1-2 s,
tllverify 1 CERT 26 s.
