# Checked compensated expert prefix: seven steps complete

Frozen execution `0c2a50693e495dd4e23279db879b89d5162c335c` completed once.
Independent review **PASS,0 issues**. Read the
[protocol and local soundness arguments](source_enclosure_v1.md) and
[exact review](source_enclosure_v1_review.json). Raw results:
`data/moe/results/source_enclosure_conv98_20260920_v1`.

The previous audit detected input/affine gaps. This stage constructs a **new**
checked enclosure that compensates them and continues through the first ReLU
and the shared/private-factor join. No old source, accepted result, matrix,
LP dual or model is overwritten. The old local audit's findings remain true
for its old states.

## Exact source steps

All seven required steps passed. The independently checked router source still
covers the sole pair[1,2], with five strict exclusion proofs. The prefix checks
the following, using the original source parameter and input bytes:

| Step | Result |
|---|---|
| Input | All3,072 represented coordinates covered; inward coordinates29→0 |
| Pair guard | Four actual saved rows exactly redundant and attached without deleting assignments |
| Expert1 first Conv |4,096 rows compensated with4,096 fresh continuous error factors |
| Expert1 first ReLU |2,943 active,608 inactive,545 unstable; all4,096 ranges/outputs checked |
| Expert2 first Conv |4,096 rows compensated with4,096 fresh continuous error factors |
| Expert2 first ReLU |1,145 active,2,595 inactive,356 unstable; all4,096 ranges/outputs checked |
| Joint map |3,072 input factors shared; all other continuous/binary factors private and remapped with every constraint |

The maximum affine compensation magnitudes are
`1642475027/2417851639229258349412352` (about6.79312e−16) and
`303082151/302231454903657293676544` (about1.00281e−15). Unlike the previous
error-only audit, the new output states **contain explicit checked generators**
of those per-row magnitudes. Exact recomputation accounts for the corrected
input state; old nominal coefficient arrays are merely proposals.

The final prefix HZ has8,192 output coordinates,13,066 continuous factors,
901 binary factors,901 equalities and1,806 inequalities. Its new state identity
is `976a3727bf24e8375ca34aed2c0d2edb874560712d1c76380b70f29312adba4a`.
This larger representation exposes a real future scaling cost; this control
does not establish that per-row compensation is an efficient full-network
default or will fit the same budget after all remaining layers.

## What the checked prefix proves

For any real input in the pinned represented box, the input check supplies a
valid coordinate assignment. The redundant guards preserve it. Each affine
step supplies fresh bounded error-factor values representing the exact real
affine output; each ReLU step supplies a checked sign/continuous extension.
Both experts retain the same input assignment, and disjoint private factors
allow both extensions to coexist in the joined state. Thus the joined HZ
contains the **common-input relation of the two declared first-Conv/ReLU
prefixes**, not just two independently sampled output boxes.

This removes numerical input/affine/ReLU/map assumptions **for this new prefix**,
conditional on the declared graph correspondence and checker execution. It
does not retrofit the historical whole-expert HZ or its proof. No assumption
is removed globally from historical complete-output certificates.

The endpoint is explicitly
`TWO_EXPERTS_AFTER_FIRST_CONV_RELU_NOT_CLASSIFICATION`.
**Zero classification output properties were checked, zero old LP certificates
used, and no complete strict network certificate or new route-changing SAFE
was produced.** Later Conv/pool/linear/ReLU layers, weighted output aggregation
and any reused membership/common-fact sources still need a new checked chain.
The declared source graph is not a proof of deployed floating-point execution.

## Portable check and controls

Bundle manifest:
`01e7a3b62ae49c1ed89eb6ddc3aaaf23ce8f1ca202c062be5483b04b896a2579`.

Copy the new `relocated/` directory, then:

```sh
python -I -S /new/location/verify_prefix.py --manifest-hash 01e7a3b62ae49c1ed89eb6ddc3aaaf23ce8f1ca202c062be5483b04b896a2579
```

Checking imports neither ACT nor the producer, model or solver. It reads only
the moved package and standard library. The separate real relocation exactly
reproduced all state identities and results. Four real semantic mutations
with recomputed transport hashes were rejected: missing prefix step, zeroed
required error bound, an inward/false ReLU range, and a wrong private-factor
map. The synthetic moved control additionally rejects wrong source/parameter
bindings and works after its test-owned source directory has been removed.

32 relevant tests passed:11 new tests,8 upstream local regressions,2 lifecycle
regressions,5 portable regressions and6 original-router checks. The exact
witness-extension tests validate constructed assignments, not floating probes
used as a substitute for a universal inclusion argument. Main-table rebuild
still passes and no main-table counts changed.

## Cost and endpoint accounting

| Stage | Seconds |
|---|---:|
| New whole build, including stored-source reads and serialization |3.0345 |
| Build child including startup |3.0918 |
| Independent checker child |4.9806 |
| Total through terminal publication |**8.0934** |
| Separate moved checker internal time |4.7776 |
| Separate review, including four semantic rejections |13.1824 |

The package is36,690,861bytes. The300s outer budget was respected. Nested
input/affine/ReLU/join and state-serialization costs are preserved in the review
and are not added twice. No checkpoint load, forward, native solve or full
expert/network propagation occurred; old input/nominal source generation costs
are excluded. This is not a production speedup or full-request timing result.

## Next boundary

The next source work should extend the checked chain through the remaining
declared expert layers, retaining original parameter and source identities and
checking model-to-prefix-to-output correspondence. It must account for the
added factor/matrix size before launching a full real proof. Only newly generated
output obligations on those new matrices can support a complete certificate;
the old positive9-property proof cannot be grafted onto this prefix. No new
cohort, training, backend search or larger frozen budget is scheduled here.
