# Implemented pair scoped capacity and independent Work refusal

**The implemented pair-scoped consumer now has a registered-shape owned-buffer
upper below 128 MiB, but cannot finish checking present candidates for the full
endpoint inventory under the unchanged 8M Work limit.** This latter refusal holds even if the source were
already admitted: checking only the mandatory input-factor portions needs at
least 17,031,168 Work. No real execution is needed to establish that obstruction.

This assessment binds `d2c7a6f68184c7773ba3f2496db03398a9156ff4`, the
[implemented consumer](hz_pair_scoped_consumer_controls_20261014.md) and the
[registered topology](hz_model_scaled_resources_20261013.md). Its
[script](../scripts/assess_pair_scoped_capacity.py) and
[machine report](hz_pair_scoped_capacity_20261015.json) use source text, fixed
shape metadata and integer arithmetic only. Production code, quotas and
scientific results are unchanged.

## Owned buffers now fit this component envelope

Use the old conservative global counts `N=19496`, `F=32048`, `H=136232`, and the
checked pair-closure uppers `P=7592` nodes and `Q=9400` factors. Each working
slot is 2065 bytes. Root storage is `R=16N+4H`; scope maps are
`I=4(N+F+P+Q)`. Each of eight sparse descriptors has at most four
`(u32 node ID, 2064-byte scalar)` entries, so `D=8272` bytes per descriptor.

The native frame upper at eight endpoints and 2700 dual rows is `T=362944`.
It is sealed before the large working vectors exist. Objective pullbacks are
sequential; two objective factor maps remain while two dual node maps are
transferred, not copied, into the dual pullback. Its two output factor maps
coexist with those four maps until node cleanup.

| Stage | Owned-byte upper |
| --- | ---: |
| Root index and every ordered membership | 856,864 |
| Scope forward and reverse maps, additional | 274,144 |
| Eight descriptors, additional | 66,176 |
| Last descriptor sealing overlap, total | 1,205,456 |
| Frame sealing or intake overlap, total | 1,923,072 |
| Second objective pullback, total | 56,059,608 |
| Dual pullback peak, total | **110,559,088** |
| Norm evaluation after dual-node release, total | 79,204,128 |

The peak is `R + I + 8D + T + 2*2065*(P+2Q)`, leaving 23,658,640 bytes below
the existing arena ceiling. At most 266 registered Blocks coexist, below 4096.
Even substituting the full admitted 4 MiB frame ceiling gives 114,390,448 bytes.
The tighter global-factor upper 25648 lowers the native figure by only 25,600
bytes; it is unnecessary for this component fit decision.

This is **owned payload, not process RSS**. Source/native storage, arithmetic
temporaries, Python scalar/group/plan/receipt objects and external retained
copies remain outside it. The old 117,009,527-byte proposed packed layout also
included banks that this implementation leaves as Python objects; the two
figures are not comparable total-memory measurements. No full-size buffers or
captured model were constructed to obtain this table.

## Complete consumption independently exceeds Work

The source checker allocates one input factor per coordinate, including zero
radius coordinates. Those 3072 factors belong to common ancestry and hence
every pair scope. Every present endpoint traverses all global factors when
merging objective and dual residuals and again in each of its two norms.

The following conservative counts come directly from `Block.get/put`,
`ExactMap`, `ScopedMap`, `Scope.local`, `_consume_block` and `weighted_norm`:

| Work per input factor and endpoint | Dense owned reference | Pair-scoped |
| --- | ---: | ---: |
| A get, assuming only the presence byte is needed | 1 | 3 |
| A set | 1 | 4 |
| Merge: loop visit + two gets + exact add + set | 5 | 12 |
| One norm: get + loop visit + exact group accumulation | 3 | 5 |
| Merge and both norms | **11** | **22** |

Scoped gets include forward/inverse index reads; sets additionally perform
their exact check. These counts deliberately omit payload reads when present,
including reads known to occur after the residual has been written. They also
omit all private factors and final scale-group sums. Thus they are lower
bounds, not invented total-cost estimates.

Even with one gate endpoint per property, the unchanged plan requires
`28 pairs * 9 properties = 252 endpoints`. Consequently:

```
Dense owned: 11 * 3072 * 252 =  8,515,584 Work > 8,000,000
Pair-scoped: 22 * 3072 * 252 = 17,031,168 Work > 8,000,000
```

The conclusion concerns **complete present-candidate checking**. Missing
records can consume less, but cannot yield a complete positive result. All-zero
dual values do not avoid these loops. Source construction, planning, pullbacks,
dual-row validation, proposal generation, serialization and publication only
add cost; none is needed for this refusal proof.

It follows that smaller microblocks, caching Scope once per pair, or further
shrinking vector storage cannot alone admit the current complete consumer.
Both internal owned-workspace paths have this obstruction, not just the new
method. This is a resource-policy/implementation result, not evidence that
MoE properties are unsafe or that the mathematical support relaxation is weak.

## Candidate and source limits remain separate

For pair endpoint counts `e_p`, row counts `M_p`, and `T_tiles` candidate tiles,
the current complete-frame extent is exactly

```
128*T_tiles + sum_p e_p*(2152 + 2*scalar_bytes*M_p)
```

Here `scalar_bytes` is 8 or 1040. Explicit-zero roles and missing-record padding
still occupy full strides. Checking width two does not reduce published width
eight frames. The full shape's conservative 504-endpoint/84-tile envelope is:

| Candidate profile | Largest frame upper | Whole stream upper |
| --- | ---: | ---: |
| Native binary64 | 362,944 B | 22,868,160 B |
| General rational | 44,945,344 B | 2,831,559,360 B |

The fixed limits are 4,194,304 B per frame and 16,769,024 B cumulative. Upper
overflow is **not** proof that actual trained objects overflow. In particular,
the large general frame is a conditional preflight refusal, not an allocation
included in the accepted-frame arena bound above. No actual stream is measured.

Scope is rebuilt per tile (up to 84), not once per pair (28); old design scan
counts cannot be reused. Upstream mandatory membership count 126632 still
exceeds 65536, and source construction/checking first-weight visits still have
the independent 13,369,344 lower. The new root index does not replace them.
Complete two-method Work uppers, RSS and whole-worker lifetime remain unset.
No Work quota is derived from any of these lower bounds.

## Decision and bounded next direction

Eight new static arithmetic/identity tests and thirteen navigation/review tests
pass. Two read-only AI reviews independently derived the arena and Work
formulas; they are not human review or model reproduction. Recompute with:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/assess_pair_scoped_capacity.py --check-report docs/hz_pair_scoped_capacity_20261015.json
```

**Stop full-shape retries of this consumer under the old policy.** The remaining
decision is not another vector-size tweak. A separate source/evidence and
resource contract must address repeated checked-index/buffer accesses and the
full explicit-zero candidate encoding, alongside source consumption. A proposed
checked traversal or zero-role descriptor must preserve identity, every row,
both roles, arithmetic check points, strict acceptance and cumulative failure
semantics; none is implemented or authorized by this static assessment.

First establish complete comparable uppers and a versioned policy/representation
scope. Any quota adjustment or real/GPU execution remains separately approved.
No additional small-source runs, sealed input retries or model-capacity probes
are warranted by the present result. The 300-second budget and all G1–G6
scientific status remain unchanged; complete cost is still **NOT CLOSED**.
