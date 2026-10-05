# C44: qualify explicit compact IDs against retaining consumers (synthetic only)

Full objective remains Neural-HZ above 1870/2413 with all old cases and 13
families preserved; E0 stays separate at CIFAR25 + Tiny36 = 61/400. No solver,
actual C34 source decode, source admission, target/shadow/family expansion,
production edit, numeric-ledger extension or score promotion is authorized.

Motivation: C43 completed integer-literal statistics, but it did not establish
which fields own these integers or a safe pooling transformation. Fresh source
list.copy shares existing integer objects, unlike reconstructing values from
arithmetic runs. The exact downstream ConSet.add_op uses tuple(var_ids).

## Candidate and fixed diagnostic

Default-off c44_var_ids_v1 stores immutable bounded signed64 ID runs, preserving
ordered integer values, slice/reverse/repeat/concat behavior and outer copy
identity. It is NOT a list subclass; scalar Python-object identity and mutation
are explicitly outside its proposed closed-consumer contract. No runtime
receipt is issued. All nontrivial descriptor work requires a paid scope.

Compare fresh Python lists against this descriptor in an actual typed
INPUT / INPUT_SPEC / RELU / ASSERT metadata Net, node-output shallow copies,
and 1, 2, 4 independently retaining ConSet.add_op consumers. Widths are fixed
at 4096 and 65536, in that order; run all six cases, baseline then candidate.
These are scalar-allocation fanout guards, not benchmark selections, solver
paths, or claimed production fanout frequencies. They use no model or labels.

Both sides must have identical complete ordered-ID SHA256 images and all outer
sequence alias relations. Repeated scalar-object identity is separately
reported. Call existing Net/Layer validators, global-ID stride, and ConSet
methods unchanged; no mocked successful validation. Binary-operand list/tuple
guard and C5 full-LIVE rejection of the unknown type remain unchanged tests.

Measure every object reachable from the constructed fixture: Net/Layer fields
and __dict__, node-output containers, ConSet signature keys, Con fields,
sequence entries, integer objects, strings and run metadata. Exact-type closed
walker deduplicates by object identity and sums sys.getsizeof. Reject numeric
or unknown opaque roots. This is a complete *synthetic Python ID fixture*
metric, NOT the full original live HZ or C5 numeric metric. Allocator/C-workspace
omissions are separately covered by measured peak gates, not counted as zero.

General retaining-consumer payment passes only if ordered values / outer aliases
agree AND all six candidate fixture sizes are strictly smaller. Any regression
closes this version as an unqualified general payment; do not select only a
favorable consumer count after observing results. Even a pass is not runtime
admission: unsupported binary params, collector type, whole-source identity,
all persistent numeric roots and source/native payment still need proof.

## Resources, work and immutable recording

CPU1 / GPU0, AS16GiB, entries64M, tests60s, worker240s. One whole diagnostic
pool256M across ALL arms/cases; source branch cap200M unchanged and no source
branch runs. Per construction+evidence+closure retain the existing measured
RSS-growth and trace-peak+metadata <=1GiB gate, including diagnostic walker
storage. No result may be used to exempt the failed C41 full-load gate.

Prices in source: fixture metadata1024; baseline allocation/copies12*width;
unmodified graph/stride consumers128*width; each Con consumer16*width;
semantic image4/ID+32/sequence; outer alias matrix square; closed object walk
32/reference; descriptor reads/construction independently paid. This is a
diagnostic tariff, NOT free credit that can be appended to C40's381747 or
C34's943763 remaining work. Every arm shares the single unchanged pool.

Supervisor freezes inherited C43 dependencies/tests, all new C44 source/tests,
this card and directly exercised production consumers before the scale run.
Exclusive output results/c44_id_consumers_20260911_v1 holds events, per-case
measurements, complete result and exit even on failure. Prior archives and
production hashes must stay unchanged. A failed candidate does not terminate
the overarching goal or authorize a different structure.

Development already observed: 15 sequence tests passed; first consumer import
used the wrong transformer class name (collection error); then four fixture
schema failures (missing INPUT dtype). Fixed only isolated fixtures/imports,
not validation. Complete focused result before freezing: 26 passed in1.37s.
