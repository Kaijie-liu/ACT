# N039 v12 preregistration: single-path 2413-row replay, path v6.7 + engine n019 (final candidate of this session)

v11 (`../n039_full_replay_v11/`, path v6.5) was stopped by me after about 1,900 rows. It was
used as a diagnostic (LOG N087-N090). Worker A had finished: 7 families, 1,509 of 1,524
baseline solves kept. Losses: dist_shift 39, 42, 51, 71; safenlp 704; malbeware 103, 108,
115 and one more (CUDA OOM). ACAS Xu was partial: 4 ADVs lost (55, 57, 69, 70).
linearizenn was partial: 13, 14 lost. Gains: cgan 2 ADV + 2 CERT; metaroom 4 ADV
(S1-only). No v11 row is used.

Candidate (one configuration for every row; workers only partition families):
- engine n019 (`nhz_sound_v19.py`) = n017 + AveragePool/MaxPool of n010;
- path v6.7 (`nhz_path_v6.py`): as v6.5 plus
  * a MIP start from the element centre (true ReLU phases and active smooth segments at
    latent w = 0, integer columns only) in every MILP stage (terminal v8.3 / v7.9, LOG N091);
  * stage W before D on ReLU-only networks with several phase layers: the exact all-layers
    plan with the centre start and the violation target for 5 percent of the remaining time
    (LOG N094);
- terminal v8.3 on v7.9 (all v7.8 soundness fixes kept: valid dual bounds only, tiny
  coefficients folded, segment piece bounds, fail-closed non-finite handling);
- runner `run_n039v12_full_replay.py`: v7 runner with engine n019; an OOM retry inside the row
  deadline raises the per-process GPU memory fraction to at most 0.5 when the device has room
  and restores 0.2 afterwards (resource handling only);
- workers: wA cersyve, cgan_2023, metaroom_2023, dist_shift_2023, safenlp_2024, sat_relu,
  malbeware; wB acasxu_2023, relusplitter, vit_2023; wC linearizenn_2024,
  tllverifybench_2023, cora_2024. Memory fraction 0.2 each.

Accounting unchanged: retention of every baseline CERT/ADV per family; any loss, conflict or
invalid witness fails promotion; ADVs counted after `audit_n039_witness.py` (S1 and S2
reported), CERTs after `audit_n039_cert_sampling.py`. Open user decisions: witness sources
(element centre, LP box maximiser, MILP incumbents) under the no-attack rule; integrality
skipping by Proposition 1' under hard limit 2; S1 versus S2 witness semantics.

Expected before the run (probes, not the ledger): dist_shift 39, 42, 51, 71; linearizenn
rows needing the baseline's 500-840 s portfolio; ViT rows (relaxation gap, LOG N063-N077);
TLL rows that ran out of memory at fraction 0.1; near-budget SafeNLP rows.
