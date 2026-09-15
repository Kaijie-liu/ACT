# Portable input98 proof — transport protocol V1

This is a read-only derivative of the completed pre-F0 R2 proof, not a new
model query or changed mathematical obligation. It retains all nine output
properties, both router order supports, sixteen difference supports, eight
rational McCormick LPs and one scoped interval proof. The source HZ, scope
snapshot and request/route assertions are pinned in the external statement
hash. Network/input→HZ, expert/source binding, guards and route exclusions
remain trusted. No production gate changes.

`scripts.build_portable_conv_proof` packages only the mathematical dependency
closure. Long JSON arrays are stored once by content hash in a compressed ZIP;
logical files are reconstructed byte-for-byte under the original JSON encoding
and checked against their original SHA-256. Sparse matrices remain sparse.
Checks run on reconstructed data, not on a claimed summary. Solver proposal
functions and historical directory validators are excluded from the vendored
checker; selected mathematical functions are copied verbatim by AST source
spans, with source hashes recorded. Bundle and statement hashes must be supplied
out-of-band; self-described hashes are not an authenticity mechanism.

Acceptance: copy outside the repository, run the act-py312 interpreter with
`-I -S` from outside the checkout, without checkpoint/data/history/solver reads.
A Python audit hook restricts reads to the relocated bundle and interpreter
stdlib, prohibits solver/model imports and subprocess/network calls. This is
an observable dependency restriction, not a sandbox for adversarial Python code.
Result must equal the archived independent result. Negative controls must
reject a removed obligation, changed source binding, changed classification
property and damaged content, including semantic controls with recomputed
transport hashes but the original external statement/checker identity.

Record source-directory size separately from the proof dependency closure,
deduplicated uncompressed bytes, final bundle bytes, packing time, checking
time and relocation/check process wall time. No subtraction from old timing
tables, no new speedup claim. Historical evidence is never overwritten.
