# R2: saved-CSR storage compatibility only

R1 (implementationdcfb61971) is retained under its original raw directory.
The saved-model proposal refused before construction: `unsupported model
shape/storage`, because the base constraint `A` CSR is unsorted (`value_matrix`
is sorted in this artifact). Later grouping also
raised `unsupported query row` for unsorted query CSR. `assignment.json`
records no accepted point; no `summary.json` exists. Zero native calls and no
checkpoint/forward/new full request. This is NOT a mathematical infeasibility
or relaxation finding. Do not overwrite that partial run or call it PASS.

R2 changes only CSR recognition: locate the three new coefficients by explicit
column equality instead of sorted-row binary search, and allow raw unsorted
output/query CSR storage. **Never sort,coalesce,change any coefficient or change
summation order in the underlying matrices.** Duplicate new-factor entries are
unsupported and rejected. Full final feasibility and identity checks remain.
Exact query grouping compares stored bytes; differing but algebraically
equivalent CSR serializations may conservatively remain separate groups.

Same protected-R1 saved matrix,same all-zero free seed,same3s proposal/check
cap,zero optimizer calls. No alternate seed,time expansion,numerical tolerance
change or acceptance improvement. New output directory:
`/data1/Kane/MOE/baseline_runs/metamoe_current_assignment_20260923_r2`.
First run failure remains separate; R2 is a compatibility repair, not a rerun
of the complete MoE request and not an offline upgrade of old UNKNOWNs.

Controls add unsorted model/value/query rows and duplicate-factor refusal to
R1:22 assignment +7 grouping tests in ACT and pinned intake environments.
Commit/push before one R2 saved-only run; separate saved-point checking and
record reconciliation follow. Production integration and full-budget request
effects are still outside this stage. R1 protocol remains historical.
