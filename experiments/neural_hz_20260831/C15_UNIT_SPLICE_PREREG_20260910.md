# C15: actual unit signed row splicing, isolated component

Frozen before the first C15 target execution. Same S0 Tiny143 final predicate
checkpoint and original C10 MAIN maps; C14 completed all102571 definitions and
proved268 individual unit-multiplier pairs. This version must construct the
simultaneous rewrite, not count those individual proofs as an HZ reduction.
No iid, margin, label, optimizer status or C14 table is an algorithm input.

## Exact identity and compact witness certificate

For a registered MAIN definition a*x_j+P=h and its sole other predicate
occurrence -s*a*x_j+Q=t (or <=t), s in {-1,+1}, replace both by
s*P+Q=t+s*h and erase the defining equality. The pivot is a positive power of
two in the unchanged [2^-20,2^40] window. P is binary-free with all continuous
columns <j; the consumer's first continuous column is j, all remaining ones
are >j. Consumer binaries remain intact. The selected variable is output-dead.
The exact box guard is sum(abs(P))+abs(h)<=a. The RHS sum must be exactly
representable as float64 (Fraction checked); signs/copies introduce no rounding.

All unit pairs are selected by structural incidence and coefficient equality.
Reject the proposed component as a whole for unsupported ordered/binary shape,
shared consumers, selected-definition dependencies, any failed norm/RHS guard,
or any cap. No greedy prefix or subset rescue. Non-unit rows are untouched.
Sorted disjoint supports give exactly two fewer coefficient nnz per pair.
Raw continuous indices/frame, every binary factor, original output/input maps,
and every unselected predicate are unchanged. No binary pivot or convexization.

The erased P buffer is NOT retained. The new consumer's prefix below j equals
s*P; reconstruct x_j=(h-s*new_prefix*x)/a. The box proof makes this extension
legal for every retained box point. Noninterference makes extensions independent.
The two directions preserve the full nonconvex feasible output set; the old
input prefix is unchanged. This theorem is not a solver/witness repair path.

Certificate: int32 column and uint64 descriptor per pair; descriptor low31bits
consumer row, bit31 inequality kind, next8bits exponent+20 (6bits), negative
sign (1bit), nonzero offset present (1bit); all upper24bits zero. A third
float64 array stores only nonzero h in column order. No retained old rows/maps.
All arrays/scalar metadata and HZ payload have a sealed closed ownership schema.
Independent Fraction audit decodes the format independently, checks every
changed coefficient/RHS, all unaffected rows and the input/binary/frame map.

## Fixed component acceptance and resource ledger

The actual component AND its certificate must strictly decrease ALL of:
coefficient nnz, numeric resident entries, unique numeric owner bytes, and
controlled bytes = numeric owner bytes + reachable Python shallow payload.
Python payload includes the closed object graph, dictionaries, scalar metadata,
ndarray/view/CSR headers and bytearray capacity; no hidden old-definition array
is allowed. Allocator rounding, interpreter/class globals and Torch C++ internals
are exclusions explicitly bounded separately by unchanged process/tracer limits.
This component metric is NOT the whole-runtime representation comparator.
Both complete original source checkpoint dictionaries remain live in the worker.

Same256M work,64M entries,1GiB measured construction,tests60s,worker240s,
address space16GiB,CPU1/GPU0. Full preflight before norm/RHS/emission:
8*input coefficient nnz+32*logical MAIN structural charge (before validation);
16*selected parent terms for exact norm;32*selected pairs scalar/metadata;
12*original coefficient nnz for compact emission, constructor/seals/ledger;
4*affected final row coefficient terms;4*(old equality+inequality rows).
Charges describe this standalone implementation only. No generic multiplication
path is executed; sign copy and ordered concatenation are the new primitives.
No cap, coefficient threshold or measurement formula changes after a failure.

Development33tests cover signed EQ/INEQ transformations, mathematical fixtures,
independent audit, exact input reconstruction, binary retention, lower pruning,
serialization, mutation/hidden metadata, incomplete maps, invalid arithmetic,
zero hits, small non-amortizing state and before-emission resource rejection.
Inherit all611 frozen tests, expect644 total. Native target inspection may use
ordinary lowering and passModel/getLp ONLY; no optimizer or presolve call.

## Single target and stopping boundary

Exclusive results/c15_unit_row_splice_20260910_v1/; seal sources/config/provenance
before tests/target. Original final file SHA192f3a5a95637933bbbed8b57b10fd20f7d237b4c6f2066573e5dfa14e6b1971;
original live/maps SHA1c242085191040cdea7db9975aba5a8c2134e4b2776213cfee2c99a7e3ec1c9d.
After construction only, compare selected columns with the complete sealed C14
table: all268, not a chosen prefix. Predicted certificate3768bytes/605entries,
536 fewer coefficients; predictions are not results or editable acceptance caps.
Require independent exact proof and unchanged native matrix/bounds/integrality.
Save component, proof, native fidelity, result/events, failures/exits and hashes
automatically and exclusively, including negative/timeout evidence. Never rerun
this version or edit any prior manifest, production source or HyZor archive.

Success authorizes ONLY a later, separately preregistered generation-time
integration design. A 253M-class postpass cannot be appended to the existing
254750077-work C10 build. No new live network or terminal run, CIFAR expansion,
shadow/family/full replay, default switch, speed promotion or score change here.
Failure closes C15 v1 as recorded; do not alter gates to rescue it. Formal score
1870/2413 and separate E061/400 remain unchanged; gain0 regardless of outcome.
