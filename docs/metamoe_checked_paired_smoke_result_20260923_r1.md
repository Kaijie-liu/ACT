# Combined checked ACT / author sufficient adapter: smoke passed

Implementation `a9f151804`, freeze `603129df0`; config SHA256
`b958c9aceb1e82592630e12dc7f452a6e95e6eabe73c72b43be3a13ef84c2ad9`.
New execution only; old single-input controls and pre-BN-fix attempts retained.
[Protocol](metamoe_checked_paired_protocol_20260923_r1.md),
[config](../configs/recent_moe/metamoe_checked_paired_smoke_r1.json),
[independent archive](metamoe_checked_paired_smoke_archive_20260923_r1.json).

| Old input | ACT combined | Author-backend sufficient adapter |
|---|---|---|
| CIFAR10/0 | UNSAFE_REPLAYED, 2.725965 s | UNSAFE_REPLAYED, 2.766541 s |
| MNIST/0 | HZ_POLICY_ACCEPTED, 9.445801 s | AUTHOR_BACKEND_NUMERICAL_SUFFICIENT_FILTER, 7.826143 s |

Both accepted counterexamples were separately replayed on the original full
checkpoint. For MNIST, ACT retains all 19 output obligations and complete
route/nonzero obligations; author includes 19 global margins, route dominance
and selected-score sign. Independent saved-only audit PASS, 0 issues; smoke
gate true. A second process disabled native optimization, support, and the
production exact-sign helper and recomputed the archive identically except
audit clock. This does not independently reprove either backend's positive
network bounds or all-domain upstream conversion.

The positive sets coincide on this control. ACT is about 1.62 seconds slower
on MNIST in this fresh pair. There is **no ACT speed/coverage victory** here;
previous 8.76-second timing is not substituted. One old positive input and
one old clean-error input are interface controls, not a population estimate
or clean-correct cohort. This is not a new route-changing certificate.

All three optimizations are explicit/default-off. The nonzero call took
0.013988 seconds and zero native support optimizations, same stored-HZ range
`[3.345338179462453,4.30685851511294]`. Native infeasibility, network-to-HZ,
input/guard lowering and the numerical policy remain trusted. No relaxation,
property ordering, threshold, local or outer time allocation changed.

Full charged requests total **22.764450 s**; batch through final summary
**23.828347 s** includes postflight. Separate replay and audit clocks appear
in the bound original replay and archive. 186 raw files / 22,571,763 bytes
retained at `/data1/Kane/MOE/baseline_runs/metamoe_checked_paired_20260923_r1_smoke`.
No raw arrays/checkpoint committed. 132 ACT and 97 pinned-environment controls
passed before freeze. The synthetic ERROR/TIMEOUT controls are not real failed
requests in this four-request denominator.

Decision: the execution gate is open for a separately frozen small comparison,
not automatic execution or proof/accuracy claim upgrades. Stop optimizing
these old inputs. Next freeze 5 clean-correct inputs per original dataset in
raw order (10 inputs/20 calls), exclude prior verification inputs, unchanged
checkpoint/normalized box/budgets/methods. Record rule and sources before
forward-only selection; no router complexity or bounds used in selection.
