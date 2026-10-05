# Next — exact channel-factorized dense support and packed ownership

This dense-support algorithm and static scalar preflight are prospective, NOT
implemented or qualified. The separate C127 systematic-kernel v2 primitive has
passed ordinary complete-four-source qualification and all3584 tests, with whole
work243478590; see `CHECKPOINT_C127_SYSTEMATIC_KERNEL_20260927.md`. C127 v1 remains
frozen failed. That ordinary qualification does not qualify this proposed graph
algorithm or grant real-source, native, solver or score admission.

## Concrete hook and one uniform structural rule

Create a NEW support/ownership engine overriding `_conv_counts` beneath
`c24_dense_ownership_v1.DenseOwnerEngine`. The current implementation inherits
`c6_support_affine_plan_v2.SupportEngine._conv_counts`, which visits active input
channels times output channels for each spatial stencil incidence. No existing
channel-factorized implementation was found in those engines.
`exact_linear_op.ImplicitConv2DOp._count_expanded_entries` only counts geometric
expanded slots; it is not a masked-support or packed-ownership implementation.

For a freshly checked COMPLETE finite, all-nonzero kernel, its structural
incidence tensor is all ones within each group. The new program is therefore:

1. Forward: sum the original input values over input channels ONCE per
   batch/group/spatial position; apply the unchanged spatial stencil to this
   scalar map; broadcast over output channels; apply the COMPLETE output row
   mask, including its distinct batch/channel entries.
2. Reverse: apply that row mask to the COMPLETE original packed output labels;
   sum labels over output channels ONCE per batch/group/spatial position;
   transpose-scatter through the unchanged stencil; broadcast over input
   channels. Reduce the actual packed labels, not just Boolean activity/counts.

This is equality of the existing structural integer counts and incidences,
not a claim that numerical convolution cancellations disappear. Kernel signs
and magnitudes do not enter structural support; NONZERO does not mean positive.
Batch, groups, stride, padding, dilation, spatial clipping and row masks remain
literal operator inputs, not specialized identities.

Keep the existing all-zero input/label shortcut: the exact result is zero and
no factorized arithmetic is needed. For a non-dense kernel, retain the exact
old V2 traversal with its FULL tariff, plus payment for the fresh density check.
CSR operators retain their unchanged implementation. These are one uniform
structural dispatch rule, never branches on node ID, dataset, labels, margins,
solver state or previous results. Do not select the faster path separately for
individual saved nodes; the table below deliberately includes a regression.

## Saved evidence and prospective complete operation tariff

Source: `results/c117_affine_block_census_20260922_v1/complete_census.json`.
SHA256: `b67c962b181259fe9f82e368138d01610116abb17c684aec544fa231cec42075`.
Only this saved JSON, code reads and scalar arithmetic were used. No target NPZ,
model load, HZ restoration or numerical experiment was performed.

The census contains eleven Conv operators with complete nonzero kernels. Seven
have nonempty source support. Let NI/NO be complete flattened input/output
widths, W the COMPLETE kernel entry count, B batch count, G group count, and

```
E = B*G * sum_kh(number of valid output heights for kh)
          * sum_kw(number of valid output widths for kw)
Q = 1024 + 4*W + 8*(NI+NO) + 16*E
```

Q is a proposed NEW per-fresh-call upper tariff, not a discount for old loops.
It must be registered and implemented with payment BEFORE operations: complete
kernel density inspection; full channel reduction, masks, broadcast and result
retention; every geometric gather/scatter and its temporary workspace. Validate
the implementation against this decomposition; increase/rederive the bound if
it needs additional work. There is no density-certificate/cache credit.

For these seven active nodes the current C24 graph also pays the unchanged
wrapper `2*NO+4*NI`: node-width traversal, reverse label construction, and parent
needed/ownership update. Charge TWO fresh Q calls, forward and packed reverse.
The final count pass retains the existing exact forward-result cache behavior;
it is not a new transform cache or authority to reuse a stale source receipt.

| Node | NI | NO | W | E | One-call Q | Old node support | Prospective node upper | Difference saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 25088 | 25088 | 147456 | 1600 | 1017856 | 14334336 | 2186240 | 12148096 |
| 16 | 25088 | 25088 | 147456 | 1600 | 1017856 | 27181184 | 2186240 | 24994944 |
| 19 | 25088 | 25088 | 147456 | 1600 | 1017856 | 30144128 | 2186240 | 27957888 |
| 22 | 46656 | 25088 | 8192 | 196 | 610880 | 2471040 | 1458560 | 1012480 |
| 25 | 25088 | 6272 | 147456 | 400 | 848128 | 545152 | 1809152 | -1264000 |
| 30 | 6272 | 6272 | 147456 | 361 | 696976 | 6168704 | 1431584 | 4737120 |
| 32 | 25088 | 6272 | 16384 | 49 | 318224 | 1718528 | 749344 | 969184 |

Empty-support nodes1/4/7 retain6272 each; node13 retains25088. Keep all other
graph work unchanged. The resulting CONDITIONAL graph-support upper is
`86963144 - 82563072 + 12007360 = 16407432`, a prospective saving70555712.
Node25's complete channel-dependent output mask is retained despite its cost
regression. These are structural-work bounds, not wall-clock measurements or a
full fresh-source bound. Existing source authentication/hash scans and all
other generation/proof/ledger work remain in their own paid scopes.

### Conditional composition, not a complete admission bound

For C128/K128, the ordinary-qualified C127 v2 tariff gives the still-SCALAR
preparation estimate `32768 + 3364*128*128 = 55148544`. Combining it with the
unchanged C120 constructor preparation gives100499456. Under the deliberately
optimistic assumptions in `C127_SOURCE_BIRTH_PREFLIGHT_20260927.md`, use the
largest saved single-tile direct count D=382720, remove its entire8D original
whole-encoding allowance, and omit ALL coupled work. The subtotal is then
`176407180 - 8*382720 + 100499456 = 273844876`. Replacing the one original
graph-support debit by the prospective new bound ONCE gives
`273844876 - 70555712 = 203289164` whole units.

This arithmetic excludes coupled construction/ownership/quotient work, every
new non-kernel emitter/power/native/inverse operation, fresh source proof,
physical comparison, retained ledgers and evidence publication. It neither
authorizes the optimistic direct-row credit nor grants whole-budget admission.
C123 binding was never in the fixed source base and is not subtracted again.
Both same-geometry/source premises and the new graph tariff still need proof.

The corresponding pre-support optimistic branch subtotal is220770252, but the
whole-graph saving70555712 MUST NOT be subtracted from it. C97 `lift` accumulates
`original_encoding + node_support + max(parent_costs)` and then takes the
largest branch; branches do not necessarily contain all seven changed nodes.
A prospective branch bound needs that complete nodewise recurrence, including
the node25 regression and any change of maximizing path. No new branch bound
or combined real-source admission is claimed here.

## Exactness, ownership and retained-state obligations

Before reductions/scatters, establish an ordinary signed-int64 bound. Original
support masks may be integer-valued, not only Boolean: preserve the old allowed
range and prove fan-in/fan-out times the maximum input is within INT64_MAX using
bounded scalar arithmetic. All contributions are nonnegative, so this bound
also covers every partial accumulation. Do not narrow to Boolean inputs merely
to make a support-engine test pass.

For packed ownership, retain C24's canonical dense needed-row labels and UID
range. A positive stride and dilation give at most one kernel-offset incidence
for any fixed input/output pair. Thus a given output UID contributes at most
once to an input. Count<=2^20, UID sum<2^40 and packed total<2^61 remain valid;
batch/group separation and row-mask-before-reduction are essential. Preserve
the original count-plus-UID-sum representation, retagging, exact incidence and
independent owner audit. Packed words are not arbitrary set-membership proofs.

Return the same complete owned read-only count/owner vectors. Preserve current
source mutation validation, cache-key semantics and complete cache-memory caps;
fresh full-kernel checking is not replaceable by a mutable flag. Account for
group-reduction temporaries, stencil accumulators, row-mask payloads and full
returned/cached arrays in the reachable ledger and both transient limits.

The first isolated qualification should compare ALL returned counts/owners and
complete graph support/needed masks, node counts and UID allocation with the
old engine, plus independent explicit incidence references on bounded ordinary
cases. Include signed dense coefficients, zero/non-dense kernels, all-zero
masks, arbitrary channel-dependent output masks, integer supports, multiple
batches/groups, stride, padding and dilation. The original source equations,
binary factors, frame, aliases, native coefficients and inverse must remain
unchanged; faster graph counts do not prove their later admission automatically.

## What remains deferred

Do not resume the unqualified source-birth emitter merely because this scalar
projection is promising. Its original-power, slot/UID, radix, alias-decision,
full source/owner/inverse and physical-cost obligations remain in
`C126_SOURCE_BIRTH_HANDOFF_20260927.md`. First implement and completely qualify
this graph primitive, then derive a fresh whole/branch/resource bound for any
combined source path. Unknown costs are not zero; unchanged real-target reruns
are not authorized by this handoff.

Use new isolated versioned files, default off. Preserve all inherited ordinary
sources/tests and verification limits. Formal1870/2413 and separate E0
CIFAR25/Tiny36 (61/400) are unchanged; every old solve and all13 family
results remain protected. No production/default/commit/push change, and no
historical artifact or `/data1/Kane/HyZor` write.
