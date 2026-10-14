# Actual direct join cost and admission decision

The direct source, independent checker and portable interfaces are connected at
`8b22bcd9bd249c9e2199e847415239446a31e17f`; the
[108-control result](hz_direct_join_controls_20261015.md) remains valid in its
fixed-stub scope. **The registered eight-expert topology is not admitted.**
This source-bound assessment identifies costs of the implementation, not of the
earlier compact-store design. It changes no runtime, quota, mathematical check
or 300-second budget, and executes no source, model, optimizer or GPU.

The [machine record](hz_direct_join_cost_20261015.json) and
[read-only calculator](../scripts/assess_direct_join_cost.py) bind the current
implementation and frozen topology. Complete request Work and RSS upper bounds
remain open. The source-core upper bound below is not a complete resource policy.

## What the new connection actually executes

[execute](../scoped_source/direct_execution.py) copies an already captured
declaration, constructs a direct forest, independently checks it, plans all
obligations, consumes fixed candidates, reseals the source, publishes the package,
then independently checks that package online. The second check constructs a
fresh forest and Driver. Both use the original Work object and deadline.
The subsequent copied-package control is offline and cannot rescue this request.

The registered shape has 3,072 inputs, router widths 3,072/128/8, eight experts
with widths 3,072/256/128/10, and 252 properties over all 28 conservative pairs.
There are 252–504 endpoints. This pair roster covers tie-legal obligations; it
does not establish reachability or a route-flip witness.

## Membership remains a certain structural refusal

Pair views alone retain `28*3072 = 86,016` input memberships. Including all other
mandatory views gives at least **126,632**, versus the unchanged **65,536** cap.
Unique root values do not remove ordered membership obligations.

The [encoder](../scoped_source/direct_source_format.py) performs this check after
the builder has constructed the graph and the encoder has collected all uses.
The receiver rejects an excessive count from the fixed header before decoding
sections. Thus the implementation is fail-closed but the producer does not yet
have an early topology preflight. This known shape must be rejected statically;
there is no reason to allocate it just to encounter the late check.

Local entries, references, active factors and scalar representability impose
separate gates. Their conservative upper counts exceed some current ceilings,
but that alone does not establish the actual trained-object sparsity or the
first rejection it would encounter. No weights were loaded here.

## Source accounting includes the new encoder and intake

Let `H` be all memberships, `N` nodes/unique roots, `L` local entries, `R`
reference entries and `F` factors. Direct encoding adds exactly `2H+2N` row
visits and `L+R+F` coefficient visits on a successful path. Direct intake itself
adds `3N+H` row visits and `L+R+F` coefficient visits; independent source
initialization and the input/affine/ReLU/guard equations still run afterwards.
Relative to the old checker core this adds `3N` rows and `L+R` coefficients,
not a saving of repeated-root checks.

Using the [frozen structural envelope](hz_full_resource_ledger_20261014.md),
the conditional **source-core** bounds are:

| Stage | Row visits | Coefficient visits | Exact arithmetic visits |
| --- | ---: | ---: | ---: |
| Construct and encode | 366,200 | 14,000,128 | 27,745,712 |
| One independent direct source check | 249,628 | 21,010,128 | 27,742,424 |
| Construct plus both online checks | 865,456 | 56,020,384 | 83,230,560 |

The last row totals **140,116,400**. It excludes copying, terminal reseals,
planning, candidates, endpoint consumption, archive inventories and reception.
These are structural upper bounds conditional on completing the core, not
consumption observed under the eight-million cap. They cannot set a new quota.

A separate lower bound proves the current complete schedule cannot fit that
cap: constructing and twice checking the first affine layer alone visits
`3*6,684,672 = 20,054,016` weight positions, including zero weights. Two complete
pair-scoped candidate checks add at least `2*17,031,168 = 34,062,336` visits from
the mandatory input-factor merge and two norms. Their sum is **54,116,352**,
before the omitted work. This assumes all required evidence is present; missing
evidence can cost less but cannot yield a complete positive result. It is not an
algorithm-independent complexity bound or an observed failed-run total.

## Physical bytes differ from the earlier tensor alias design

The actual binary file has the exact extent

```text
S = 36 + h + 32*N + 12*(L+R) + 4*H + 8*F
```

Here `h` is canonical metadata bytes. This format stores first-affine coefficients
explicitly; it does not alias the captured tensor bytes. The registered wide
shape envelope gives **85,545,220 + h bytes**. Under the current section caps,
any accepted file instead has an upper bound of **24,379,428 bytes**. The former
substitution violates some caps and is not an accepted file size; neither number
is Python memory or full archive size.

The online archive contains a copy of the captured source, forest, candidates,
code, manifests, result and pending terminal. The raw captured tensor payload
is 55,715,520 bytes before metadata. A **dense-nonzero first-layer subcase** needs
at least `12*6,684,672 = 80,216,064` bytes for explicit first-layer local entries.
Just those two terms total **135,931,584 bytes**, exceeding the 134,217,728-byte
archive cap even before roots, references, headers or candidates. This is a
conditional counterexample to a general capacity guarantee, not a claim that
the actual checkpoint is dense or reached archive failure. That subcase also
violates the local-entry cap earlier.

The previously derived native-candidate stream upper remains 22,868,160 bytes,
above its separate 16,769,024 cap; it is only a dimension-conditional upper, not
an observed overflow. Explicit zero vectors remain stored. General-rational
candidates have their own larger envelope. The new source format additionally
refuses reference sums not exactly representable as binary64; it does not round
them and does not claim to accept every V3 source. The proposed 96.8 MB tensor-
alias store cannot be used as the size of this implementation.

## Wall cost includes more reads than archive size measures

On the successful schedule the source tensors are fully read at least ten times:

| Source tensor read | Multiplicity |
| --- | ---: |
| Original source to owned package copy | 1 |
| Builder and two independent source checks | 3 |
| Two terminal source reseals | 2 |
| Portable initial and final inventories | 2 |
| Archive audit and reception archive audit | 2 |

For raw tensor bytes `T`, forest bytes `S` and total candidate bytes `C`, these
paths alone read at least `10*T + 5*S + 5*C` file bytes. The tensor term is
557,155,200 bytes for the registered shape. Metadata, code and repeated center
reads are additional. These are logical file-read counts, not measured disk
traffic: the OS may cache data.

Most hash/I/O checks call `tick()` without incrementing Work visits. The phase
timer includes them, but `archive.logical_bytes` measures newly written files,
not read/hash volume or RSS. Current instrumentation does not aggregate all
read/hash bytes. Removing checks is not an authorized way to reduce this cost.
The original supplied source and separately copied offline package are outside
this online archive's byte counter and must be included in a whole-stage disk
lifetime account when that scope is proposed.

## Object ownership still leaves a full RSS and cleanup gap

| Phase | Relevant simultaneous objects |
| --- | --- |
| Construction final encoding | Builder graph and fractions, metadata, full bytearray and its immutable bytes copy |
| First source check | Original forest bytes, decoded nodes/roots/views and checker temporaries |
| Online portable source check | Original bytes and first checked graph/plan/receipts/last candidate, second raw bytes and second parsed graph |
| Portable candidate checking | First graph and retained records, second graph, one current checking arena and intake/arithmetic temporaries |

`Driver.finish()` releases the first owned arena before replay, so adding two
110.56 MB arenas would be wrong. But it does not release the forest, Session
plan, receipts or the caller's original `raw`; `owner.live_bytes == 0` therefore
does not mean those objects are gone. The earlier 114,390,448-byte bound is for
one owned arena with an admitted frame, not total resident memory.

There is also a timing boundary: `execute()` records elapsed time before its
Python frame is finally destroyed. Final release of retained local objects is
not explicitly measured in its cleanup phase. A caller or outer supervisor
must measure through return and final ownership release before claiming full
end-to-end cleanup cost. The fixed-stub interface is still cooperative; it
does not interrupt blocking callbacks or file operations. No new hard-budget
or whole-process memory guarantee is established here.

## Comparison boundary and next decision

The direct execution and portable package currently select pair-scoped Driver
only. Dense Driver and old Session have same-candidate controls, not a new full
matched native execution arm. Their old native/Work ledgers cannot be silently
combined with this source format and online-replay schedule to claim a fair
end-to-end comparison.

**Stop full-topology execution at this gate.** Another small control, a smaller
arena or an earlier release of the first forest would not remove the membership,
complete Work and explicit-coefficient storage barriers. Do not derive a quota
from the 54.1M lower bound or the 140.1M source-only upper bound.

The next resource decision must jointly specify the exact source scalar domain,
membership representation/accounting, native candidate schedule, matched full
reference, both online checks, and an ownership/reception contract through the
hard supervisor. Preserve all mathematical checks and separately identify any
representation or resource-policy change. This assessment does not authorize
those implementations, a quota increase or real/GPU execution. G1–G6 and all
historical scientific claims remain open and unchanged.

## Reproduction and review

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/assess_direct_join_cost.py --check-report docs/hz_direct_join_cost_20261015.json
```

The calculator parses source layouts and performs integer arithmetic only.
Two read-only reviews independently checked the source counters, extents and
object lifetimes. The report recomputation and 13 navigation/historical-binding
regressions pass; all four changed documents' local links resolve. They are
static checks, not human technical review,
performance measurements or full network proof. No new control-count claim is
used to justify admission.
