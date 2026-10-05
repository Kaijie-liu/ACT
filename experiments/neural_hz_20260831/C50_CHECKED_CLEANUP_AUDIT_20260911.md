# C50 audit: compiler evidence; qualification closed at source-origin audit

C50 v1 is CLOSED, not qualified. The full Neural-HZ goal remains ACTIVE.
This turn produced test-executor evidence and a diagnosed audit defect, not
a Neural-HZ representation breakthrough, actual target run or solved case.

## Implemented before freezing

The direct plain-assert hypothesis was rejected: the installed pytest
rewriter has different chained-comparison and successful walrus evaluation
traces. Neither plain mode nor Python optimization was used for qualification.

The isolated default-off cleanup compiler preserves the ordinary assertion
rewriter, its conditions, diagnostic branches, exceptions and evaluation
multiplicities. It only replaces cleanup stores on proved-bound private
function locals by deletion, preserving release order; possibly unassigned
short-circuit slots retain None stores. No HZ or old test source was changed.
See the frozen compiler contract for the scope and conservative control-flow
rules; the observed canaries are evidence, not a theorem about arbitrary
reflective Python programs.

Pre-freeze focused result:16 tests passed in0.11s. The17 original-rewriter
comparison cases cover outcome/message/evaluation traces, repeated short
circuits and temporary-owner release. The actual qualification also executed
and saved all17 canaries successfully before collection.

All75 original test sources compiled:1502 assertions,1498 changed cleanup
groups,6324 proved-bound deletions,1143 retained None targets. Recursive
bytecode size1482820->1471320 bytes. A fixed seven-pair100000-iteration probe
gave identical checksums and median0.00748717226088047->0.007328060455620289s,
ratio1.021712676392858. This is a small compiler microbenchmark only.
It does NOT prove a60s full-suite pass or any Neural-HZ runtime gain.

## One frozen qualification, exact negative result

Branch redu-hz, commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac; unchanged
production candidate15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75.
The new executor froze721 source/evidence paths and76 selected test files,
retaining all2104 original ordered cases plus16 new cases, total2120.
The original ordered-ID digest remains
143258636d9b6840970829d30810ae55649671a30f0e40a43a63cd7302797c0e.

Supervisor wall5.8730615600943565s, test subprocess exit3. This was an internal
collection audit error, NOT a60s timeout and NOT a failed HZ test.
The runner records wall5.242642871104181s; pytest reports no tests ran in4.99s.
Exactly64 module audits completed; the next selected file was
test_c6_support_affine_plan_v2.py. The exception was
ValueError: unknown wrapped selected test/helper function.

ZERO full-suite test calls and ZERO setup/call/teardown phase reports ran.
The17 pre-collection canaries and16 earlier focused checks must not be
reported as2120 successful tests. test_result.json has completed=false.
Its all_original_ordered_cases_retained=true records collection identity
only; it is not execution, coverage completion or a passed qualification.

The supervisor correctly withheld the actual generator. The output directory
has only preregistered.json, tests.log, test_events.jsonl, test_result.json,
and exit.json. There is no worker.log, actual generator event/result, new
Closed object, new source proof, native state, solver result or witness.
The C49 generator's predicted243824504 whole/158799938 branch work remains
unmeasured. C49's earlier60s timeout remains immutable.

## Read-only diagnosis, no qualification retry

A subsequent collection-only diagnostic collected2120 items and executed
zero tests. It identified31 imported items collected from
test_c6_support_affine_plan_v2.py but defined in
test_c6_support_affine_plan_v1.py. The v2 file explicitly imports the old
tests with import * and an autouse fixture binds their plan/SupportEngine
to the v2 implementation. This is existing intentional reuse, not new code
injection. Definition source SHA256
1bebecbec36f63ea58afaf086074ab03901856c2eb1c5495341c0e40478c66fd
already matches the frozen source manifest.

C50's new audit incorrectly requires every selected function's co_filename
to match the collecting module. It has no protocol for these source-bound
import aliases. Its rejection is fail-closed but incomplete. The focused
synthetic module tests did not cover imported test definitions.

No source-origin check was removed or bypassed. The diagnostic did not run
the failed full test suite, change any test outcome, optimize an unregistered
module, or launch the real worker. Fixing this audit in a later version alone
would NOT prove the outstanding full-suite timing requirement.

## Integrity and remaining gates

All58 prior checkpoint manifests,2097 unique files/28305103514B were hashed
before (8.421165600419044s) and after (8.732454938814044s); zero mismatches or
conflicts. All721 frozen run source paths were rechecked after diagnosis:
zero source drift and unchanged branch/commit/production provenance.
No C50 processes remain live. Existing histories, production/defaults and
/data1/Kane/HyZor were not modified.

No cap or window changed: CPU1/GPU0, AS16GiB,64M entries, one60s test process,
one240s actual-worker limit, whole256M/nested200M and original radix limits,
both1GiB transient checks. Since the real worker never ran, none of its
load, generation, audit, owner-retirement or save gates is newly satisfied.

Formal remains1870/2413 (1063 CERT+807 validated ADV), with no new full
13-family retention replay. Separate E0 remains61/400 = CIFAR25+Tiny36.
No attack/PGD/BaB/split/backward/dual/LP-status rescue, binary pivot,
iid menu, Zono/CZ/box replacement, or target/family expansion.

Next work is constrained by C50_VERIFICATION_HANDOFF_20260911.md.
This audit is an immutable trial-log addendum; the previously hashed
TRIAL_LOG.md and failed C49/C50 sources are not rewritten.

## Terminal anchors

- preregistered.json:7aeb3f7b91f50eede18108ab112f44f14a679d0be584fdb074ad9d9b2120a3c0.
- tests.log:e98fb85478b3871727946e01e84540c9710b79e584b3faeb57ae2b26fa3ec411.
- test_events.jsonl:f5fca935a57892a1996f21083475a5ae15441e5f601cace5ea1ed09ef966738c.
- test_result.json:11e086295ded704cd55e805aca7a73ea8f27492aa68fc13c23141a80ae270b02.
- exit.json:5ba5fdecfd495746cc45710d26ceb8283001741105cec26baa62f12ba45e996d.
