# C56: one exact row-gauged radix lift per common affine consumer

Previous goal turn: PROGRESS. C55 proved binary64-literal exact realization,
but scalar-coordinate radix16 failed the complete chain/Conv storage gates.
It remains closed. C54 and C55 archives, sources and production remain unchanged.

The SAME ordinary general-scalar/output-dead definition structure is retained.
Choose ONE uniform direct-row carrier design, not a menu of alternative lifts.
For each exact C54 consumer row, group long continuous coefficients by exact
absolute mantissa/exponent. A group has a canonical signed root vector and a
common magnitude c. Identical coefficient/signed-vector groups share a lift
across rows. Short binary64 coefficients, binary terms and RHS remain unchanged;
an unsupported long binary/RHS scalar rejects rather than selecting a rescue.

For N distinct signed roots define k=ceil(log2(N)), carrier t=sum(sign_i*x_i)/2^k.
All source continuous factors are in[-1,1], so |t|<=1. For abs(c)=m*2^e and
L=bit_length(m), take low-to-high integer digits of width at most min(53,60-k).
For each actual digit width w use q=max(0,w+k-20) and emit the single equality

  2^q*y_new - 2^(q-w)*y_old
             - sum(sign_i*d*2^(q-w-k)*x_i) = 0.

The first row omits y_old=0. Integer digits have at most53 bits, hence are exact
binary64; all nonzero literal coefficients remain in[2^-20,2^40]. Every prefix
is an unsigned fractional scale of t, so its new[-1,1] box is redundant. The
original group is replaced by2^(L+e+k)*y_final only if that literal also passes
the ORIGINAL coefficient window. No precision/window increase or rounding.

The actual auxiliary equalities ARE their complete reconstruction maps: their
birth-order pivot identifies each new coordinate and their coefficients retain
the full signed original-root relation. Do not also store a redundant lift map.
An independent Fraction audit derives EVERY actual auxiliary expression from
the emitted rows, proves every new box redundant, and compares ALL substituted
original predicates/output/RHS against the source-bound exact C54 state. Keep
original inverse/scalar/UID/frame data, new equality UIDs and all phase binaries.
Transient grouping/digit maps are not retained; construction costs are measured.

Use the unchanged width128 C54 chain/shared_add/conv_relu measurement cohorts
and unchanged SAME-source C52 binary64 reference, including complete original
source on BOTH sides. All four strict physical gates remain: total predicate
nnz, complete numeric bytes, complete numeric entries, and numeric+reported
Python shallow accounting. Preserve all failure records; do not select only a
passing cohort. Tests additionally cover ordinary signed carriers and reuse,
without changing the three registered measurement workloads.

Retire all five unshared exact CSR/RHS/old-EQ-UID arrays before accounting.
All original retained scalar/inverse/global/removed UID roots remain charged.
CPU1/GPU0,AS16GiB,64M entries,whole256M/nested200M,60s focused tests,240s worker,
both1GiB measured transient caps,16384 radix auxiliaries,131072 added numeric
entries and16M lift work stay. Pre-count the actual group-expanded added CSR/
RHS/UID entries before allocation; a per-coordinate shortcut is insufficient.
Full source/reference/C54/lift construction, actual-row proof,48 full original
inverse/output vectors, retirement, ownership, protocol5 export, C41 verified
decode, fresh C54 regeneration and restored proofs/physical checks stay inside
the measured transaction. Temporal whole_work is excluded only from the exact
semantic binding, not from actual work charging or execution provenance.

Before one run freeze all sources and provenance in
results/c56_gauged_carrier_20260912_v1. Keep auto-saved test/event/result/exit
records. This stage proves mathematical coefficient realization only, NOT
current native solver-layout integration, real source generation or a target run.
No iid/family/LP-state selector, attack/PGD/BaB/split/backward/dual rescue,
binary pivot, convex replacement, default/source-history modification or score
promotion. Formal1870/2413 and E0 CIFAR25/Tiny36 remain unchanged. The full
source/frame/native/witness, real target/shadow/family/replay gates still apply.
