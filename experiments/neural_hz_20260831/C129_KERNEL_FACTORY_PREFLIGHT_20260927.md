# Prospective source-owned kernel factory — concrete algorithm, budget NO-GO

Disposition: source-text and authenticated saved-scalar preflight only. No
implementation, numerical array load, test, model execution or target retry.
The proposed factory below changes actual arithmetic and ownership, but its
conservative complete planning tariff does NOT fit the real source budget.
Do not implement a real source-birth wrapper on the strength of this document.
This is not a theorem that every possible exact HZ kernel algorithm is blocked.

The real geometry here is **C=128, K=128, KC=16384**, not the C128 experiment's
ordinary C=16, K=32 fixture. The unchanged source domain is all complete 3x3
zero/normal finite binary32 kernels, or their exact binary64 lifts, within the
ordinary raw alignment span <=33. Every source coefficient and ALL36 outputs
of every K,C pair remain in scope, including residual/unselected channels.
Neither a saved tile identity nor target status is a dispatch condition.

## Audited sources

All paths below are relative to `experiments/neural_hz_20260831`; hashes were
checked without importing the modules or loading numerical artifacts.

| Source | SHA256 |
| --- | --- |
| `c120_word_f4_v1.py` | `51f2b5a5b5666ba98956a097cbd1bb79d8055eb7479b4ea074bf77e8df78ca06` |
| `c126_support_word_mixed_v1.py` | `2678dcadc9103f9ae5503d94f17fed7600d35295c3b876a483a9319360629e7c` |
| `c127_systematic_kernel_proof_v2.py` | `e4184d522ec0ed333e9864094fcaaed149fb26c98f11cec11ecaa08b636f783d` |
| `c127_systematic_native_oracle_v2.py` | `252a785f93ef6174f9c6caba9513d06587194d1c45225ee75ee536fab0fdcf6a` |
| `C128_SOURCE_BUDGET_PREFLIGHT_20260927.md` | `2933c7f186907f5c2a012513accb7d3ae288be6a450b87459bcf49d9f9d0e986` |
| `C128_COUPLED_WORK_HANDOFF_20260927.md` | `83ddf7d6f57cb7e4829768837c8f26eb06b240b386c7a922d86e3533fb0201e6` |
| `results/c117_affine_block_census_20260922_v1/complete_census.json` | `b67c962b181259fe9f82e368138d01610116abb17c684aec544fa231cec42075` |
| `results/c122_channel_route_20260927_v1/complete_channel_census.json` | `0382e972ea5d5015ebb0314c43d12cac46255e610dd988b033e40f27e5cf2acb` |

The saved JSON is a source of counts and graph topology only. It does not bind
fresh coefficients, powers, aliases, masks, row survivors or runtime storage.
C128v1/v2 remain FAILED. C129 custody work is separate and receives no assumed
success or free storage allowance from this preflight.

## One fresh transaction, not a caller-supplied preparation receipt

A possible new interface would encompass source observation, construction,
independent checking and actual row consumers in ONE invocation. It would:

1. Prepay headers, all source copying/decoding, arithmetic, evidence allocation
   and source comparisons before their operations. Own a complete exact-byte
   source snapshot; bind shape, dtype, operator geometry and current source.
2. Bit-decode the producer's source words and compile the complete transform
   through the new shared nine-operation axis below. There is no inverse-only
   acceptance test and no call to old C120 with a discounted tariff.
3. Independently decode the actual source by the C127-style frexp/ldexp
   identities, without importing the producer decoder, transform or constants.
   Check EVERY producer canonical/raw/aligned word and common exponent against
   those independently obtained values, then check all transformed components.
4. Pass only those checked words to selected M and residual direct formation
   and their independent actual-row oracle. Retain every complete numeric
   evidence array and live original; bind every consumer to this transaction.
   Compare the current full source with its snapshot again at completion.

No external transformed array, mutable receipt, source-ID cache, historical
hash or boolean `proved` flag authorizes a later use. Merely marking an owned
ndarray read-only is not an immutable-lifetime proof: callers can reset its
writeability. The actual implementation would need closed call-local custody,
no untrusted array exposure/interleaving during use, and full retained-root
accounting afterward. Each new invocation pays its complete fee. Whether any
multi-tile lifetime can be proved is a separate question, not a saving here.

This representation could remove C126's separate full original-weight decode
`512+64*9KC`, because its actual residual loop would consume the proved original
canonical words. It would also avoid producing/retaining a second transformed
36-tensor merely for equality. C127's current native oracle ALREADY consumes
canonical original words for residual terms: there is no per-residual float
decode there to remove a second time. Existing row/source/physical tariffs
are not discounted just because they receive the new words.

## Concrete producer and independent ALL36 checker

For source triple `(a,b,c)`, compute:

```
negative_sum = -a-c             # 2 signed operations
twice_b = 2*b                  # 1
shared = a+4*c                 # 2
y1 = negative_sum-b            # 1
y2 = negative_sum+b            # 1
y3 = shared+twice_b            # 1
y4 = shared-twice_b            # 1
return (a,y1,y2,y3,y4,c)
```

Two separable axes require 3+6 triples: **81**, not C120's 99, signed operations.
The shared `-a-c` and `a+4*c` are real common-subexpression changes. There is no
claim that changing a label or reusing the old Python loop deserves a lower fee.

The independent checker first checks all nine anchor cells at row/column
indices {0,1,5} against complete source-derived forms: four corners, four
negative three-source forms and one nine-source sum, totaling20 signed
operations. All9 original source coefficients participate, not just a checksum.
For each checked anchor triple `(a,first,c)`, compute:

```
middle = -a-first-c             # 3 signed operations; then check its bound
twice_middle = 2*middle         # 1
expected2 = first+twice_middle  # 1
shared = a+4*c                 # 2
expected3 = shared+twice_middle # 1
expected4 = shared-twice_middle # 1
```

Compare those three values to the actual candidate entries immediately, rather
than constructing a second full tensor. Check three anchor rows first (9 new
cells), then all six columns (18 new cells). The triangular order gives exactly
9 authenticated anchors +27 constrained cells = ALL36. Producer and checker
must have independent constants/code and independent source decoding. A fixed
paid nine-source-basis theorem must check all324 literal tensor coefficients
and the checker's triangular coverage on every invocation. Ordinary qualification
also retains the complete independent Fraction reference and all existing tests;
their fees are not absorbed into this prospective deployment arithmetic.

### Exact domain and overflow obligations

Let A=2^57. Validate the full raw exponent span <=33 BEFORE shifts and prove
each aligned source has magnitude <A. Producer first-axis forms have bounds
given by row L1 norms `(1,3,3,7,7,1)`; every second-axis intermediate is <49A,
strictly less than2^63. For checker triple scale s<=7, a,c are <sA and first
is <3sA; recovering middle has partial magnitude <5sA<=35A. Check middle<sA
BEFORE doubling it. All later partials are <7sA<=49A. Check each candidate
component's full row/column envelope before using it as an anchor. Never use
the naive unchecked `3*a+2*first+6*c` expression.

Normal source raw powers remain in[-149,104]; canonical source numerator and
denominator bounds, and transformed signed-word/exponent bounds, imply reduced
numerator/denominator sizes below512 bits. These kernel facts do NOT prove a
native row: every actual coefficient, positive original pivot, odd M defining
denominator, L1 redundant box, full-row gauge and exact binary64 window
[2^-20,2^40] still require the independent complete row checks. Parent aliases,
coalescence/cancellation and original semantic powers remain unchanged.

## Conservative explicit planning tariff — insufficient, not prepaid

Use a fixed65536 header for complete basis/coverage theorem, metadata, shape
guards and small dispatch state. The following is a proposed source-level
operation envelope, not a fee applied to any old function or a certified new
implementation. All retained old operations keep their existing component
rates; new 81-operation programs use the existing composite12-unit rate.

| Per-KC component | Planning units | Full work covered |
| --- | ---: | --- |
| Complete source custody | 288 | 32*9: snapshot, initial/final exact source observation |
| Producer full word decode/canonical/domain | 576 | 64*9; no omitted original coefficient |
| Producer full raw alignment | 288 | 32*9 including min/span, shifts and envelope |
| New producer axis program | 972 | 12*81 including gathers/temporary/stores |
| Producer complete output envelopes | 288 | 8*36 |
| Independent frexp/ldexp source decode | 576 | Retain C127's64*9 decode/domain tariff |
| Independent full alignment | 288 | Retain32*9, no source-receipt shortcut |
| Source-word equality | 368 | 8*(5*9+1): canonical pair, raw pair, aligned, common exponent |
| Independent nine source anchors | 160 | C127's8*20 composite rate |
| New independent dependent-cell program | 972 | 12*81 including per-triple prebounds/temporaries |
| ALL36 actual candidate comparisons | 288 | 8*36, anchor and dependent cells, full component envelopes |
| Complete source/transform census | 360 | 8*(9+36), full zero/nonzero and count scans |
| Total per KC | **5424** | No Fraction materialization occurs in this NEW interface |

Thus `F(K,C)=65536+5424KC`; at128x128, **88,932,352** before actual consumer
binding and graph route census. The existing two preparations cost100,499,456;
the proposed arithmetic nominally removes11,567,104, not the whole old kernel
bill. Existing C120/C127 calls would still pay their FULL fees including unused
materialization reserves; no caller may use this formula as their refund.

Full implementation would have to justify every tariff's allocation, gather,
guard, temporary and store coverage before freezing. If a bound is insufficient,
increase it; this NO-GO does not rely on approval of a reduced rate. Kernel
proof alone does not pay source graph generation, maps, row emission, physical
owners, ledger, serialization or qualification/reference runs.

Keep the exact eight-array numeric evidence scope: complete source snapshot,
canonical mantissa/exponent, raw mantissa/exponent, aligned words, all36
numerators and common exponent. This is91KC retained numeric entries. At128x128:
1,490,944 entries and10,092,544 bytes for a binary32 source, or10,682,368 bytes
for its binary64 source. These are kernel-payload sizes, NOT whole-state or
peak-memory claims. Independent decoding/alignment temporaries and the producer's
18-word intermediate, Python metadata, snapshots and every live original also
belong to the complete measured ledger. No object/Fraction payload may be hidden.

## Actual use and the complete route census cannot be free

One additional conservative consumer-binding plan is `8U`, where
`U=3*36KS+2*(B+ER)`: two constructor coverage lookups plus the independent
oracle's coverage lookup for every selected channel/component, and both actual
M/direct consumers and independent M/direct reconstructions. B is the actual
selected M connection count; the saved all-transformed-nonzero B is only an
UPPER. ER counts every live residual occurrence. This is extra custody/index
checking, not a refund of the existing complete coverage/row fees. A final
implementation must include any additional coefficient-read site in U.

The unchanged C122 route costs280576 per128x128 tile. ALL52 saved structural
tiles cost14,589,952, before fresh mask extraction/custody. The saved population
is16 tiles each at nodes10/16/19 and4 at node30. Adding4489216 at nodes10/16/19
and1122304 at node30 to ALL36 saved max-parent recurrences gives:

```
whole route addition  = 14589952
branch base before    =  95936620
branch base after     = 100425836
branch route addition =   4489216
```

The maximizing path remains18->19->20->24->32->33->34->35. This is a complete
scalar recurrence, not subtracting a whole-graph sum from one branch. For
reproduction, use `report.nodes[].parents` and `graph.node_counts` in C117,
the seven support replacements from the authenticated C128 preflight, and
`P_i=12*(continuous+binary+center+aux)_i+support_i+census_i+max(P_parent)`;
add the unchanged8407200 predicate term once. Non-route costs and all other
node regressions remain included. A changed future route population or custody
placement needs a fresh complete recurrence; neither is assumed free.

## Real-budget comparison

Keep the prior complete coupled allowance46,305,837. Grant the same optimistic
12D old encoding and22O lifecycle credits; preserve O=2048 and the conditional
injective surviving-parent/nonzero-original-kernel premises. The current
constructor+independent output-only fee lower remains528O+176ER, excluding all
V/M rows and the other missing obligations. The saved M connection UPPER is
`B=m_nnz_upper-kept_m`; it is not a proof that transformed coefficients survive.

| Tile/origin (evidence only) | S | ER | B upper | Extra use plan8U | All-kernel/other-cost ceiling BEFORE route census | Ceiling AFTER full branch route census |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|1/(0,4)|17|151936|78080|5560320|34055527|29566311|
|5/(4,4)|26|125824|104192|6555648|39168871|34679655|
|6/(4,8)|24|95360|82816|5505024|43796327|39307111|
|9/(8,4)|28|145408|118912|7325696|35715943|31226727|
|10/(8,8)|23|99072|76544|5353472|43055463|38566247|

Even granting ZERO consumer-binding charge,88,932,352 exceeds each29.6–39.3M
ceiling. Those ceilings must cover the ENTIRE kernel plus V/M, source birth,
owner/inverse/physical/ledger/evidence and all other new work. They are not
standalone kernel allowances. With only the fixed65536 header and no other new
cost, a per-KC rate would already have to be at most1800/2112/2395/1901/2349 for
the five rows, versus this plan's5424. Actual headroom is strictly smaller.

Combining the proposed kernel, its conservative use upper and full branch
census with the previous source/output-only budget gives264926361/260808345/
255130265/265031321/255719577 respectively, still BEFORE V/M and other new work.
These mix conservative budget components with a conditional output count lower;
they are rejected planning allocations, not measured actual runtimes or universal
work lower bounds. Original coordinates/kernels are still unbound, not assumed
true for admission. Do not jointly select the five tiles or dispatch by their IDs.

## Decision and next algorithmic boundary

Do NOT launch this factory as the next real-network source-birth attempt. It
has a specific genuine arithmetic/lifetime design, but modest shared integer
subexpressions plus a new interface do not remove enough complete work.
Nor can one replace independent original-source decoding with trust in the
producer's words just to cross the budget line.

A materially different next target is a block-native proof representation that
joins source-word use with complete residual/native row comparison: compile
source coefficient significands once, propagate exact semantic exponents through
the live-support plan, and independently compare EVERY actual native coefficient
and whole-row gauge in bulk. This must really remove repeated per-coefficient
decode/normalization/dictionary machinery; vectorizing an unchanged loop while
lowering its fee is not that change. Alias/coalescence, odd-D box proof, every
original row/pivot/RHS/binary term and exact composed inverse remain mandatory.
It needs a NEW operation/custody/storage bound jointly with kernel preparation,
not an assumption that a 29–39M ceiling remains after those row costs are paid.

An exact radix-packed relation checker was considered only algebraically: a
bounded injective digit embedding could constrain all outputs, unlike a checksum,
but it introduces large-integer packing/products/comparisons and has no complete
cheaper limb-work or custody bound here. It is NOT a second authorized path,
implemented candidate or asserted budget solution. No numerical experiment is
justified until one genuinely different complete bound closes the deficit.

Formal1870/2413, all13 families/every old solve and separate CIFAR25/Tiny36=61/400
are unchanged. All whole256M/branch200M, shared auxiliary/entry/emission,
64M-entry/BOTH1GiB and single-path verification prohibitions remain. Unknown
costs are unknown, not zero. No production/default/archive/git mutation occurred.
