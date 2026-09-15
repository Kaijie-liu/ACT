# Input98 portable proof: acceptance complete

Execution `f9dc8f2dd369bd380da6e573eac7672164ceadb1`; no model or solver queries.
Raw review: `/data1/Kane/MOE/portable_conv98_20260915_v1/REVIEW.json`.
The package was copied to the `relocated` sibling of `packed`, both outside
the ACT checkout. The checker ran from that outside directory with `-I -S`,
an empty Python environment and read/import/process restrictions. It reproduced
the archived independent result exactly: **9/9 positive**, eight rational
McCormick obligations plus one interval reuse, minimum
`199593373867685/1125899906842624` (about0.1772745273844).

| Measurement | Observed value |
|---|---:|
|Selected proof dependency files|56|
|Their original serialized bytes|428,185,262|
|Deduplicated uncompressed object bytes|32,096,276|
|Final bundle including checker/metadata/license|7,181,520 bytes|
|Unique objects / large-array references|202 /750|
|Packaging, no new generation|9.256s|
|Copy|0.003s|
|Isolated checker process, including startup|30.222s|

Source-directory size428,562,773 bytes includes other records; neither that
number nor the proof-only denominator includes checkpoints. These are storage
and portability costs, not production speedups. Original proof generation
costs remain separately reported in `conv_pre_f0_r2_results.md`.

All four negative controls rejected. Deletion, source substitution and property
changes were tested with recomputed transport hashes but the original external
statement identity, so they did not merely fail a ZIP checksum. Reasons were
`missing/duplicate output property`, `trusted source binding changed`, and
`wrong output property`; damaged archive rejected on file identity. The five
unit tests separately cover extraction, JSON ambiguity and content references.

## Reproduce after copying the entire relocated directory

No server paths are stored in the proof's mathematical dependency closure.
Run from any directory, using the copied location:

```sh
python -I -S /COPY/verify.py --bundle-hash 8f35a4ba23b51bdcc829535a47880e5f6158a6fbaaf119b4f7744e0c2278606b --statement-hash 7c31f551137b33257e178c40eeea55bf4b94e3438ae00ddb3ed16e2808e56b00
```

Keep the two hashes independently of the bundle. This is an independently
checkable *conditional* request proof, not proof of network→HZ, route
exclusions, or deployed floating-point execution. No previous production
TIMEOUT is relabelled. Raw arrays and bundles remain local, not committed.
