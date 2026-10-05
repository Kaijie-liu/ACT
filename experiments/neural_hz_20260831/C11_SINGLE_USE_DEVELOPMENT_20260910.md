# C11 pre-target development record

Before freezing or invoking the actual S0 target: removed one inert placeholder
from the structural loop; tightened exact-domain/frame, complete1D map/radix
and existing predicate coefficient-window guards. No target outcome was read
to make these edits. New28 focused tests pass in1.56s: independent Fraction
substitution identities (including offsets, binaries and inequality consumers),
box sufficiency, exact cancellation and nnz deltas, each rejected arithmetic
guard, existing tagged coordinates, liveness/degrees, maps and fixed caps.
The all-candidate arithmetic preflight is tested to reject before individual
evaluation when even one unit over budget. No representation is constructed.

Before target launch, clarified that the ordinary unchanged-input lowering
diagnostic is independent of arithmetic census acceptance. A rejected census
remains rejected and nonzero-exit even if lowering succeeds; no retry, prefix
evaluation, extra work budget or solver is allowed. Both complete deserialized
source checkpoints stay reachable throughout diagnostics, and the unchanged
1GiB construction measurement is applied separately to census and lowering.

The supervisor inherits all490 frozen tests/sources from C10 terminal, then
adds these28 tests. All new outputs are exclusive; tests60s/worker240s,
address space16GiB, CPU1/GPU0. Production/HYZor/frozen archives are untouched.
