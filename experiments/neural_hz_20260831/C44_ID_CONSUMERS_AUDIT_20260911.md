# C44 complete diagnostic: naive compact-ID payment rejected

Formal1870/2413 unchanged; E0 CIFAR25 / Tiny36 unchanged. Full goal ACTIVE.
No actual benchmark load, new HZ, solver, witness or native/source admission.
No production edits. All old52 checkpoint manifests,1997 unique files,
28303572215 bytes checked in9.659820361062884s: zero mismatch/conflict.

Frozen run results/c44_id_consumers_20260911_v1:1810 tests passed in49.17s;
641 source hashes; supervisor exit0,worker exit0. Supervisor60.87661779578775s,
worker9.711655463092029s,whole work197610212/256000000. Source branch unused.
Source/provenance drift false. All12 measured arm gates passed; maximum RSS
growth136056832B; maximum trace peak+metadata106565597B, both below1GiB.
The closed Python walker has temporary identity sets and recursive closure
bookkeeping, all included in measured arm peak; allocator bytes are not the
same metric as unique fixture shallow size. This is not a full LIVE audit.

| Width | Retaining consumers | Baseline Python bytes | Compact Python bytes | Delta |
|---:|---:|---:|---:|---:|
|4096|1|466536|304230|-162306|
|4096|2|532415|599021|66606|
|4096|4|664317|1189611|525294|
|65536|1|7347000|4726782|-2620218|
|65536|2|8396063|9445837|1049774|
|65536|4|10494141|18883979|8389838|

All six complete ordered ID images and outer alias matrices agree. At width
65536 baseline unique integers stay131077 across all consumer counts; compact
counts are131084,262156,524300. The cause is observed scalar duplication in
the real unchanged ConSet.add_op tuple construction. Fresh list.copy keeps
scalar sharing. This does NOT prove production uses the synthetic fanout;
it refutes a fanout-independent saving based solely on the compact sequence.

The existing residual binary list/tuple guard rejects VarIds, and C5 full-LIVE
collector rejects the unregistered type. Both rejection tests pass; neither
guard was weakened. Explicit materialization is a semantic control, not free
payment. A small descriptor or tiny pickle is not a source-construction win.

General retaining-consumer payment gate FAILED in four of six cases. C44v1 is
closed as an unqualified payment, not integrated, and no favorable single-
consumer subset is promoted. The primitive remains isolated evidence only.
Do not retry the same target/version or append it to C40/C34's remaining work.

Result SHA256 ebd020cb17a987ad4773609dd95c4788c1a9d82411af263429eaa9ee999c19d7.
Exit SHA256 a59b28e360d12adc1d955674173e9724a783a3b41cfc99218e8c263b90596aff.
Events SHA25662fd3acb5844be7a89c45a4657e41ae4f4b393f444589427b8934b0d3a38ba3a.
