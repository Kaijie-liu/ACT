# D015 v1 preregistration — source-only pilot, no score promotion

2026-09-28; branch `redu-hz`, HEAD
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. Read DESIGN.md for the mathematical
question, full source semantics, fixed scope, resource accounting and limits.
This document and every new source/test are frozen BEFORE the first import,
pytest collection, original-model decoding or numeric evaluation. Source-only
reading/review and test-format corrections before that freeze are not test runs.

## Fixed sources and populations

Use both original universe manifests under `manifests/`:

- `cifar100_2024_universe_v1.json`, SHA256
  `fa30dafe17cdcafeb08b56da66189795e1623b5556ced7a8247903fef507d948`;
- `tinyimagenet_2024_universe_v1.json`, SHA256
  `a8a0dc7504af2c6b89d099fd5c74f27aa5ae98458c5c151cbb6d0da2ef5c1f59`.

Include all three distinct model files, choosing one spec per model by the
lexicographically first ORIGINAL spec path. Root is the benchmark collection
root PLUS manifest family PLUS relative path. Record exact chosen paths/hashes
in preregistered.json before import. Selection does not read labels/verdicts.
For each admitted direct next-ReLU Conv branch use all output channels and
four corner/center coordinates as defined in DESIGN.md, no post-result choice.
Keep all residual side-consumer records. Unsupported source/grammar fails
the complete attempt; no successful subset can be called the complete pilot.

## Tests, comparator and immutable attempt

Inherit every D003 test:160 files,3684 exact node IDs and the complete975-entry
source identity registry, plus all3 prior original byte inputs. Add exactly
18 kernel tests,12 source-packet tests and11 worker/source-bound tests:
**3725 tests in163 files**. No parameterized expansion, skips, xfails, dropped
old tests or requalification by a smaller suite. Collection+execution share
ONE unchanged60s wall clock. Only a complete pass can start the census.

New tests check exact interval arithmetic, sqrt/dyadic enclosures, source
sharing, D014 controls/zero phases, strict VNNLIB bounds, raw FLOAT/DOUBLE
and BN/source extraction, padding/stride, residual side records, failures and
a complete synthetic source-to-shielding control. These are bounded correctness
tests, not the user-requested full2413/400 replay or an established-new-domain
test. Independent algebra/paper proof remains D014 and DESIGN.md.

Freeze and authenticate every old identity without refreshing its expected
hash; additionally freeze new files and the ONNX/protobuf/upb/ml_dtypes/numpy
decoder .py/.so files, optional numpy.libs and typing_extensions.py. D003's
registry did NOT previously cover ONNX/protobuf decoding. Bind the inherited
Python executable, all13 production-provenance files and the D014 six-entry
seal (`cc3b9715df411d7c94168180d62e7a0ed310d81b1852d8735f535e63de6bc7e4`).
No old model/HZ checkpoint is deserialized. Fresh original ONNX decode only.

One exclusive directory: `results/d015_source_shielding_20260928_v1`.
The runner creates it once, writes preregistered identity/input/test lists,
collection inventory, complete pytest/JUnit output, worker evidence and
terminal exit.json. Every failure retains its bytes; no rerun or edit of
frozen files under this name. Before/after hash and production checks remain.
The existing9 tracked edits are preserved; no production/default/history
change, commit or push is part of D015.

## Worker and outcomes

The worker needs explicit `--enabled`. Use CPU1/GPU0, AS16GiB, worker240s,
whole256M/nested200M work,64M retained numeric entries and BOTH1GiB transient
limits with full evidence held. Charges and their scope are fixed in DESIGN.md.
The runner independently checks diagnostic counters and memory fields; a
positive-hit count is NOT required for a completed diagnostic.

Report separately: complete component tests, faithful admitted source scope,
valid local certificate hits, resource/cost measurements, and unknown native/
real-network utility. Report proved-stable target rows separately from rows
whose outer box crosses zero. The latter count does NOT prove both signs
actually reachable. A certificate
failure or zero hit does not prove the structure absent from other inputs,
positions, models, layers or stronger sound bound methods.

The diagnostic contains zero solver calls, zero full model forward calls,
zero property verdicts and formal gain0. Inherited tests retain their existing
small solver tests, so the ENTIRE test run is not labelled solver-free.
No source certificate alone validates an ADV or updates1870/E061. This pilot
does not establish native floating-path equivalence, native storage savings,
speedup, four-concurrent nonregression or abstract-domain novelty. All original
math/real-structure/shadow/family/full2413+400 promotion gates remain.
