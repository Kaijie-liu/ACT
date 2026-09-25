# Full upstream read-only integration: complete, mixed total cost

**4/4 frozen calls completed; integration validated, no consistent end-to-end
speed advantage established. Keep optional/default OFF.** Both source-check
phases were a little faster, but the whole small call was faster and the whole
medium call slower. No repeat, new size, changed budget or real request was
added to improve this table. No native output bounds or new SAFE were generated.

## Freeze, launch and independent review

- Implementation:`f916ac82f`; freeze:`4b369cae5`; actual clean launch:
  `f3b38c0d649b03c5fd4e94ce84fc765229680917`.
- Config:`configs/backend_controls/readonly_upstream_r1.json`, SHA256
  `325f544fb9cfe3212c4c014d4dcc02a230cf09697df5d6ea8fa229037658d169`.
- Controls:264PASS (24new+240prior),619source bindings preserved.
- Audit:`readonly_upstream_audit_20260925_r1.json`; fresh python-S replay:
  `readonly_upstream_replay_20260925_r1.json`. Both PASS,0issues; original
  uncached checker rechecked all4saved constructions;16cost mutations rejected.
- Audit/replay cost separately0.358786/0.377139s; neither hidden in nor
  subtracted from execution cost.
- Raw archive:`data/moe/results/readonly_upstream_20260925_r1`:
  70bound files,4,106,037bytes. Compact reports and identities committed, not
  raw matrices. Batch wall before final summary1.313770s.

Resource admission now passed:load1about4.93 on20logical CPUs,available memory
about103GiB. Earlier prelaunch refusals remain documented in the readiness
note; they were not launched scientific requests and were not spliced into
this run. All4calls were launched exactly once. No background experiment
remains running; saved results are independent of the user's laptop/session.

## The requested complete upstream path ran

Each arm generated its own declared graph/input, enclosed the input, propagated
router and needed experts, proposed and checked route exclusions, constructed
shared joins/guards/property projections/weighted LPs, serialized using the
unchanged R1 writer, ran every source check and received the candidate under
the same300s deadline (2sreserve,8GiB sampled owned-group+parent RSS,2threads).
No saved source or matrix substituted for propagation or construction.

The only arm difference was exact parsed-source read-only reuse in the four
pair check functions. Between independently generated arms, BOTH source and
construction bytes were identical, as were original checker verdicts.

| Recipe | Original obligations | Checked route exclusions | Retained output constructions |
|---|---:|---:|---:|
| Small:E4/C3/w4/d1,seed724 |12|0|12|
| Medium:E8/C10/w8/d2,seed724 |252|225|27|

Small retains6pairs; medium retains3. Both propagate4needed experts and1router.
The count is construction coverage, **not positive lower bounds or complete
MoE certification**. Output lower-bound certificates remain0 in every call.

## Full cost, seconds, once per arm (NOT medians)

Order:small direct/readonly,medium readonly/direct, as frozen.

| Phase | Small direct | Small readonly | Medium direct | Medium readonly |
|---|---:|---:|---:|---:|
| Generate source |0.000393|0.000349|0.001340|0.001786|
| Source identity |0.000099|0.000095|0.000156|0.000160|
| Input/router prefix |0.001394|0.001426|0.006422|0.006512|
| Route proposal |0.000468|0.000409|0.001447|0.001425|
| Expert/guard/property/LP construction |0.030369|0.027535|0.079060|0.080258|
| Serialization/publication |0.027942|0.027735|0.063259|0.064853|
| Complete source checking |0.041738|0.039764|0.127897|0.121716|
| Bounded receiver |0.054717|0.052735|0.052742|0.060226|
| Remaining charged wall |0.083592|0.071557|0.074480|0.101108|
| **Whole including terminal** |**0.240712**|**0.221606**|**0.406803**|**0.438045**|

Remaining charged wall is derived as whole minus the disjoint listed phases,
generation and receiver. It includes imports, process start/wait/cleanup,
additional binding/hashing/journals/report/ledger/terminal work and scheduling;
it is not omitted cost or a newly identified single causal bottleneck. Rounded
table cells need not sum exactly; original cost records close independently.

Router propagation within prefix:small0.000849/0.000854,medium
0.004167/0.004185s (direct/readonly). Expert propagation within construction:
small0.003302/0.002937,medium0.016562/0.016402s. Weighted LP construction:
small0.006191/0.005453,medium0.013058/0.013068s. These are NESTED components,
not additional rows to add to the total. Per-component counts and times are
retained in the audit, including join,guard,projection and each check.

Parser lookups/parses/hits remain60/60/0 ->60/24/36 (small) and30/30/0 ->
30/15/15 (medium). Parser-interface costs0.025836 ->0.023007 and
0.069528 ->0.060800s. Full content binding, readonly sealing, all predicates
and stats publication remain charged; no accepted proof fact is cached.

## What follows from the data

Source-check savings observed:0.001974s and0.006181s. Whole-call differences
(readonly minus direct):**-0.019106s and+0.031242s**. In medium, the six profile
phases together decrease about0.003316s while total wall increases. The saved
costs locate additional elapsed time outside those phases; a single observation
does not distinguish startup, scheduling, I/O or other causes conclusively.
Likewise small's total saving is larger than its check saving: do not attribute
the entire difference to read-only parsing.

This answers the integration question: upstream work is paid in full, all
required constructions still agree, and local parser/check savings do not
automatically yield consistent full-pipeline savings. The four single calls
are descriptive integration observations, not a speed distribution or a
statistical performance claim. Previous saved-check medians are not pooled.

## Stop boundary

Archive this stage and keep readonly opt-in/default OFF. Do not enlarge the
microbenchmark, repeat the medium call, tune capacity/order, change serialization
or reduce identity checking to obtain a nicer total. No automatic real-request
freeze or sealed98/4088/4096/4098/4099 rerun. This study does not close historical
23source gaps, produce a new output certificate or establish external-tool/
ISSTA superiority. A future research step must justify its effect on complete
MoE output obligations rather than accumulate parser timing variants.
