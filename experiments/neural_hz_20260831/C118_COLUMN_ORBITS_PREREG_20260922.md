# C118 — complete exact coefficient-column orbits

Status: preregistered, default-off, isolated numerical diagnostic. Formal gain 0.
The goal/baseline/all 13 family requirements remain unchanged. No production,
defaults, historical archives, solver parameters, commits or pushes may change.

## New mathematical question

C117 excluded repeated WHOLE source-bound programs. C118 instead tests whether
DISTINCT input columns of the same explicit CSR operator have exactly identical
complete output coefficient vectors. A new exact sum factor can route those
columns without identifying their inputs. Original continuous and binary
factors, predicates, correlations and original-input reconstruction remain.

For a group of q identical nonempty columns, each with d nonzeros, the raw
coefficient cost is q*d. A new sum equation and routed coefficients cost q+1+d,
including the new pivot. Strict reduction requires (q-1)*d-q-1 > 0. Old output
pivots cancel in the comparison. With actual parent values 2^e_j*x_j and
x_j in [-1,1], choose s=ceil(log2(sum_j 2^e_j)) using exact arithmetic;
2^s*z=sum_j 2^e_j*x_j has a unique extension with redundant [-1,1] z-box.
The routed coefficient is w*2^s. Independence of x_j is NOT required.

This is a conditional algebraic opportunity, not a ready native HZ rewrite.
Actual parent exponents, C65 mappings/coalescence, binary and owner identity,
all native dyadic representability/windows, full inverse, nnz/physical bills
and source-generation costs must still pass before any source candidate.
Do not restore already-removed old factors or book raw savings as final nnz.

## Complete frozen scope

Use C117's exclusively saved complete operator arrays, authenticated by:

- census JSON SHA256 b67c962b181259fe9f82e368138d01610116abb17c684aec544fa231cec42075;
- NPZ SHA256 6096be4cb1fe691067df9543c3664f494241b0cbbdb6d46438f7dbea640ccd18;
- failed exit SHA256 381e5448b7b092e47a558ddaee824323953130c13342511712caad0a4c225da7.

Select ALL nodes whose stored operator kind is CSR, by structure only: the
frozen inventory has 13 operators (nodes 2,5,8,11,14,17,20,23,26,27,31,33,35).
Inspect every column, every row incidence and every stored coefficient. Include
empty and singleton classes in coverage. Equality uses complete canonical row
indices and exact float64 bytes, never digest-only matching, tolerance, inferred
pooling geometry, LP information or an instance/public-verdict menu. Implicit
convolutions are outside this explicit-CSR question and are not expanded.
Stored explicit zeros reject the diagnostic; no input is silently canonicalized.

Source-code review supplies no evidence that the dense head is pooled. Shape
200x6272 is not evidence of 128 repeated 49-column groups. A zero-hit complete
result closes this proposal on this population; do not relax equality after it.

No archived HZ is restored. This archived operator theorem/census does not
repair C117's failed ledger or missing post-expression-preservation check. It
does not supply a fresh original-source/LIVE binding or a terminal verdict.

## Paid execution and stop conditions

Before numerical work, inherit ALL C117 tests (144 files / 3341 nodes) plus all
new C118 tests. Exact collection and execution inventories must match, with no
old test edits/skips, within the same total 60 s. Freeze all sources, inherited
results and production provenance. New exclusive result directory only.

Use one 256M diagnostic pool for all 13 operators and their complete held-state
ledger. Fees, fixed before the numerical run:

- Before archive loading: 16*(2*nnz+n_rows+1) per complete CSR payload.
- CSR validation: 16*(2*nnz+n_rows+1).
- CSC conversion: 16*(2*nnz+n_columns+1).
- Exact full column keys: 16*(2*nnz+n_columns)+256*n_columns.
- Complete group aggregation: 16*n_columns.
- Existing C62 complete numeric/metadata header fees remain unchanged.

JSON-only preflight predicts 48,075,984 load units +192,189,600 census units =
240,265,584, leaving 15,734,416 for complete held-root accounting. This is a NEW
mathematical question and its own diagnostic budget, not a reset/rerun of the
C117 CSE or ledger. None of this is waived into source-generation admission.
Source-file and numeric-payload hash traffic is counted separately in bytes,
not covered by integer tokens; it is not a count of all CPU or I/O work.

Keep all original loaded arrays, all complete CSC payloads, class/size/degree/
membership evidence and reports until the single measured body finishes. The
temporary CSR/CSC wrapper objects naturally expire after each census; their
complete numeric owners remain held as ordinary arrays. This is not deletion
of evidence or repair of C117's simultaneous CSR/dense role ledger. Record
complete current numeric entries, bytes and Python metadata, not only a winner.
Save full per-operator results and arrays before late ledger rejection.

Unchanged limits: worker wall 240 s, CPU threads 1, GPU off, AS 16 GiB, numeric
entries 64M, BOTH RSS growth <=1 GiB AND trace peak+metadata <=1 GiB. Fatal-only
faulthandler. No periodic Fraction sampling. No solver, BASE bypass, witness
injection, PGD, BaB, split, backward/dual rescue or convex substitution.

Candidate admission remains whole/branch 256M/200M and shared 16384 auxiliaries,
131072 extra entries,16M extra work, with no per-group reset or repricing. A
later changed source must independently pay these and preserve every baseline
solve before formal gain/default promotion. This diagnostic cannot certify it.
