# C22 complete run-index proof PASS; no graph/HZ retirement

Goal ACTIVE, this turn PROGRESS.928 tests passed7.33s; worker exit0,
supervisor32.0324817718938s, worker22.64140592701733s. Source/provenance driftfalse;
the entire successful C10 checkpoint dictionary/graph/source/HZ/maps remained
reachable and unchanged. No C21 rerun, generator, node-field retirement, new
phase, native ingestion, solver, terminal verdict or wider replay. Formal
1870/2413 and separate E061/400 unchanged, gain0. No jobs remain running.

## Complete exact index evidence

The index covers ALL142559 surviving MAIN EQ rows using11038 maximal monotone
UID/physical-row runs. Each60-bit record packs20-bit UID start,20-bit physical
row start and20-bit(length-1) into one uint64. Actual retained index11038
numeric entries,88304bytes; no second UID-order permutation or dense inverse
is part of the index. Existing old-prefix/radix maps remain required.

Every one of542344 reserved MAIN UIDs (range3450..545793) was queried, including
399785 unused/erased coordinates; every143737 physical EQ row was queried in
the reverse direction. Old-prefix/radix rows must return None from this MAIN-
only index.2300 old INEQ UIDs were independently derived by the retained original
mapping procedure, not silently folded into MAIN runs. All686081 probes match
the independent complete graph-derived labels. Structural validation and
content digest matched before and after the entire query transaction.

This is exact INDEX compression. The original graph arrays were NOT discarded
and the index is not a standalone source proof receipt. As a unit test explicitly
shows, a structurally valid but re-encoded corrupt run can change a mapping;
only full source binding plus independent mapping checks authenticate it.

Source portable identity4ca034b0ce88eae8bdf6b252e60124bd6b20dccc4dafe53d9f3940f8543c7423,
source HZ41a3bb791a7887da1088d69678952668a08433b1efeb5cdf9fc182f9afe2adb3.
Numeric run-word digest0a6b8dbb72f779e700d491d4322cf03014de97b669af8deb0b53b97435b9e18e.
The C10 source is a successful archived diagnostic input, not a generated C21
candidate or proof that a new original-expression run will have passed gates.

## Actual work and memory

Standalone build1469998 work: stream1149896, pack176608, array conversion11038,
complete validation132456. Complete diagnostic113609094: source UID metadata
29809376, standalone build1469998, all probes82329720. Same256M/200M pool passed.
Measured construction+probe stage19.848088768310845s; entry RSS843341824,
conservative lifetime-HWM growth87478272, traced peak8022322 plus9056 metadata;
unchanged1GiB cap passed. Absolute process peak911780KiB includes full archive.
Loading/authentication2.1997243966907263s separately recorded; not credited as
free generator work.

1469998 is much larger than C21's140566 remaining work, so this code CANNOT
be appended to the current generator. Fusion must reuse actual emitted labels/
existing filtering traversal, remove real work or find an independently proved
payment; merely lowering this standalone tariff is forbidden. C19's complete
all-row sort evidence offers at most115456 work versus C21's current combined
order-check/normalization cost, not enough for this standalone pass either.

The exact11038-entry index changes the prospective physical arithmetic. Retiring
all four C21 node fields but adding this index alone would leave52483246 entries,
still54446 ABOVE the fixed52428800 reference. So index compression alone is NOT
a full physical pass. The duplicated phase owner state is a separate measured
candidate for an exact sparse-event representation; see the next design note.

## Retained artifacts

All exclusive results/c22_uid_runs_20260911_v1/ artifacts:

- preregistered.json6898436227ff9e0c55982997d0cdbaf689a423c60294622f0469f23c21dd5226
- uid_runs.npz982af5aeaecd2d2d634af196b319eff0777cecd0d2dc379e67239008ecf702d5
- index_audit.jsonead4e3f79214f19713ea9887adde30a361c6668b7bc90b7a28c556888f27b7b7
- result.jsonb88d8556365684a0a02405943f3a2bbe70a682721795c8e089bde56bbf2a57c8
- exit.json3abf8d58cfa26032c4d197137363f756e7e14e4b8400c5ffa6b89e6cecfd44d8
- tests.logd26cf1cffbc14e816ba6b9a83f67ce9c7c329e7a4ba8c73e5cbd4d60571ae92b
- worker.log5e0f0e97dfe4e976b2a81e8c71c2c5f33a3ac7ea583489174478658e9c8b7963
