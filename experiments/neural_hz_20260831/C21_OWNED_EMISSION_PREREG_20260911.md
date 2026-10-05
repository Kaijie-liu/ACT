# C21 fresh owned generation with exact dyadic product certification

NEW default-off version in the SAME S0 Tiny143 ADD75-to-ReLU78 affine suffix.
C18 is CLOSED and is not retried. C19 complete geometry and C20 complete product
censuses support the new implementation but create no verdict or gate pass.
Formal1870/2413, E061/400 (CIFAR25+Tiny36), gain0 at this stage. No new native
ReLU, unit splice, live publication, solver, CIFAR/shadow/family/full replay.

## Uniform exact product theorem and implementation

Given original emitted nonzero binary64 coefficient a with |a| in[2^-20,2^40]
and alias ratio r with |r| in[2^-60,1], their exact real product has magnitude
in[2^-80,2^40]. It is normal and cannot overflow. If EITHER operand is a signed
power of two, multiplying shifts the other operand's exponent and preserves
its significand exactly. No general significant-bit reconstruction is needed.
For the complement, compare the rounded binary64 product with the exact
Fraction(a)*Fraction(r). The final lower coefficient-window guard remains;
the upper guard follows from |r|<=1 and the exactly representable upper endpoint
2^40 (round-to-nearest cannot cross it from a real product below it).

Strict equal-shape one-dimensional float64 array interface, not a promise from
an external odd/bits cache. Check both operands' absolute min/max against the
original windows on EVERY call. NaN propagates through min/max and fails the
positive comparisons; infinity/zero/subnormal/out-of-window also fail. No
unchecked cache flag, hz.exact boolean or source theorem skips these checks.

Compute right-dyadic flags. If the ENTIRE current batch is right-dyadic, all
products are exact; otherwise compute left flags and perform exact rational
checks only where neither operand is dyadic. This is one product certifier
selected solely by operand structure, not iid/model/status/LP state or a rescue
after failure. Mixed batches remain fully supported, despite zero such rows
in the complete C20 target census. Numeric multiplication and accepted product
bits match the original function, as do every eligibility/frontier/alias tag,
binary/RHS relation, global slot, reconstruction and ownership event.

## Work and memory frozen before target

Keep C18's WHOLE256M, largest-branch200M, entries64M, construction1GiB,
radix16384/entries131072/work16M, UID2^20/radix2^40, tests60s, worker240s,
AS16GiB, CPU1/GPU0. Original affine/metadata/sort/ownership/retirement/range
prices stay unchanged. In particular32*MAIN stays despite removal of now-unused
odd/bits cache allocation and extraction. No old tariff is retrospectively
lowered while running the old algorithm.

Precharge before each operation:

-2 per hit before gathering owned coefficient and alias ratio vectors;
-12*n+4 per nonempty batch:2 abs maps,4 min/max reductions,4 scalar bound
  comparisons, right frexp/equality/all (3*n), multiplication(n), output
  abs/lower comparison(2*n).12*n totals the vector passes; +4 scalar checks;
-if not all-right-dyadic,4*n BEFORE left frexp/equality(2*n), OR(n) and flag
  walk(n); no separate flatnonzero or nonempty-copy scan;
-64 per both-general product BEFORE exact rational conversion/product/equality.
  Under these windows the significands/exponents are bounded; this is the
  explicit logical general-arithmetic tariff, not a machine-cycle claim.

Empty vector returns empty arrays without arithmetic. No capacity exception
can trigger another algorithm or return a partial vector/candidate. Vector
buffers, stored original hit products and Fraction temporaries all remain in
the unchanged measured/preallocated construction envelope. The C18 extra96
bytes per possible one-row column remains a conservative transient reserve;
the removed metadata buffers receive no negative work or memory credit.

Using complete C20 counts BEFORE target:216572 hits in6396 all-right-dyadic
rows,106364 hits in36860 other rows,5632 both-general products. Product cost is

    14*216572 + 18*106364 + 4*43256 + 64*5632 = 5480032.

Against old10333952 this saves4853920 logical work; C19's complete C18 work
model260713354 becomes255859434, leaving140566. This is an a priori model,
not permission to increase work if the actual run differs or to call remaining
physical/equivalence gates passed. The old failed C18 result is unchanged.

## Fresh target and ALL inherited independent gates

Require893 tests (78 C21+815 inherited). Then ONE fresh original ADD75 snapshot
d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed build through
the exact registered affine suffix/global frame. Complete C9 pre-HZ/maps and
C10 post-HZ are independent proof/event oracles only, never generator inputs.
Same loaders, retained original snapshot, retained complete declared oracles,
same native/reference ordering and same conservative HWM/tracer semantics.

Require completed construction, full original-affine/quotient proof, every
actual MAIN ownership word including radix/retired aliases, complete post-row
ownership append replay and actual incidence agreement, and independent
discovery of ALL268 C15 unit pairs. C14 table is consulted only afterward.
Metadata replay is not a new native NN phase or a unit-splice implementation.

Strict BOTH bytes and entries decrease for the candidate component including
its owner array against original C9 HZ/maps without that NEW owner array.
Also require the SAME complete offline union as C18: original snapshot/expr,
complete C9 oracle/maps, complete C10 post-HZ, phase-owner copy, unit columns
and all candidate roots, strictly below the SAME frozen expanded reference in
BOTH bytes/entries. No oracle release, omitted diagnostic roots or altered
comparator can rescue a failure. Work improvement does not guarantee this
independent gate. Then ordinary native passModel/getLp must preserve every
coefficient/bound/integrality; no optimization/presolve/base solve bypass.

## Freeze, retention and score boundary

Exclusive results/c21_owned_emission_20260911_v1/. Freeze all inherited/new
sources, docs, input hashes, branch/commit/dirty provenance before tests/worker.
Automatically preserve tests, events, completed proof files when reached,
result/exit/hashes and owned_hz.pickle ONLY if every gate passes. Any violation
closes the version, without retry, repricing, production/default/HyZor/archival
edits, commit or push. No new score is recorded from this infrastructure step.
All original full live/native/terminal and single-source full2413/400 replay
requirements remain open; no complete overall-goal claim is authorized.
