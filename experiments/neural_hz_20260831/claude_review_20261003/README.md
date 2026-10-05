# Claude Neural HZ candidate review and withdrawal

The user requested a review of Claude's recent changes and withdrawal of changes that violate the Neural-HZ principles. This review withdraws the current experimental path from active execution and qualification, while preserving its frozen code, logs and results. It does not delete research or revert pre-existing ACT changes.

## Scope and custody

Reviewed on 2026-10-03 Australia/Sydney on branch `redu-hz`, commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. The inspected additions are under `../nhz_v2_20261002/`: theory, alternative engines, terminal solvers, replay runners, tests, witnesses and reports. This attributes a coherent experiment tree, not every untracked workspace file, to the work the user asked about.

The tracked binary diff SHA256 remains `29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5`, identical to the earlier D132 checkpoint. No production import of this candidate was found in ACT or the inspected project instructions. The nine already-dirty tracked files were not reverted. D132's 18 archived entries and both reviewed candidate freeze manifests' 23 listed entries passed checksum validation. This is evidence about these manifests, not an exhaustive claim that every historical file is unchanged.

All writes from this review are new files in this audit directory and a new sibling resume note. No old source, model, result, frozen report or manifest was edited. No commit or push was made. The goal service was observed paused and was not changed.

## What was withdrawn

- The active N102 E0 replay was stopped by SIGTERM at `2026-10-03T21:21:17+10:00`, after checking PID 1182914's exact command and working directory. The process was subsequently absent. There were no child processes at the pre-stop check; the subsequent process scan found no matching formal or E0 replay Python process.
- Its exact command was `/data1/Kane/miniconda3/envs/act-py312/bin/python run_n100_e0_path_v7.py --mem-fraction 0.3 --out n102_e0_replay_v13/e0.jsonl`, in the candidate directory. The completed prefix contains 20 CIFAR100 rows, row indices 0 through 19: 5 candidate CERT and 15 TIMEOUT. TinyImageNet had not begun. The interrupted in-flight row is not a result.
- The candidate v7.0/v7.1 paths, including their automatic smooth segmentation and solver escalation, are not qualified for continuation or promotion under the existing charter. Frozen files remain historical diagnostics. Do not resume this withdrawn run or silently import its results into another candidate.
- The original replay already declared promotion FAIL; this review does not pretend to revoke a promotion that had happened. It adds independent grounds for rejecting the path and its gate implementation.

This is an execution stop and a recorded qualification withdrawal, not an operating-system access block: the archived scripts remain manually runnable. There was no candidate production integration to undo. Editing frozen scripts or deleting them would destroy evidence rather than restore the baseline.

## Reasonable research worth retaining

The projection-aligned ReLU parametrisation in `THEORY.md:63` couples the new continuous factor to the preactivation while retaining a binary phase and the defining inequalities. It aims to give one element exact nonconvex, LP-relaxed and predicate-free query views. That is a meaningful representation question, not merely a storage optimization. It does not by itself establish a new set family or publication-level novelty; the document acknowledges this at `THEORY.md:46`.

The phase list and continuous shared factors remain in the engine. The last-layer sign rule changes a terminal query; it was not found deleting phases from the stored element. Query relaxation alone is not proof that the whole domain has become CZ. Likewise, projected optimization of an ordinary LP dual is not automatically forbidden dual rescue. The imported multi-layer backward rule is explicitly disabled at `nhz_terminal_v7.py:38`.

The earlier centre, LP-dual-corner and seeded-start helper paths are already withdrawn in Claude's LOG N099. Current v7 does not call those helpers to accept witnesses. Their historical files must not be deleted or confused with the active path.

## Findings that prevent acceptance

### Added splitting and solver strategies are not a single representation gain

`nhz_path_v7.py:105` escalates D to E to F on unresolved queries; `:128` can repeat a query after an objective-target status. `nhz_terminal_v7.py:175` launches a multi-seed HiGHS and optional SCIP portfolio. These are additional solver strategies whose costs and gains have not been isolated from the representation change. Ordinary terminal MILP solving is not itself prohibited, but this expanded path cannot be credited as a proven single Neural-HZ definition rule.

The F stage passes `K_seg=2` at `nhz_path_v7.py:120`. `nhz_terminal_v8.py:122` partitions smooth-unit ranges, introduces per-piece variables and selector binaries, and constrains their sum to one. This is an added interval partition even though encoded in a single MILP; it conflicts with the current no-split boundary. v7.1 retains the same stage. The reviewed E0 prefix reports no selected segment units, so this does not claim splitting actually executed on those CIFAR rows.

### A returned vector is not proven to be a feasible domain incumbent

At `nhz_terminal_v7.py:265`, finite `objective_function_value` plus nonempty `getSolution().col_value` is enough to set `incumbent_w`. The bound branch above checks status and info validity, but the incumbent branch does not check feasible-solution status, solution validity, or feasibility of the returned mixed-integer point. It therefore does not establish the report's claim that every proposed witness originates in a domain-feasible incumbent. Concrete network validation remains necessary but cannot repair missing provenance. This is a missing gate, not proof that an already saved ADV is false.

### The witness acceptance path mixes unapproved input semantics

`nhz_path_v7.py:82` may try the decoded point, a snapped point and inward/plain float32 versions. Its S1 branch at `:92` calls the network on the plain float32 cast without rechecking that this evaluated point is in the original input box. The claim that only one unchanged domain point is checked is too strong.

The saved audit reports 767 S1-valid witnesses, of which 16 fail S2, including the two new cgan ADV claims. The audit contains two saved witnesses whose rows were ultimately TIMEOUT; the ledger has 765 ADV, not 767 solved rows. S1-only is not automatically an invalid witness under every possible rounding semantics, but it cannot silently establish strict original-box validity or the project's invalid-ADV-zero requirement. No permission to weaken the baseline semantics is inferred here.

The audit also locates witnesses by filename without checking the replay row's saved witness hash (`audit_n039_witness.py:17`). Its S1 check reconstructs a clipped real point from top-level coordinate bounds, not the saved x64 witness, and does not evaluate every original input expression on that reconstructed point (`audit_n022_witness_replay.py:99`). These are additional provenance and validation gaps, not a new count of demonstrated invalid witnesses.

### The promotion implementation is not fail closed

`finalize_replay.py:54` conditions PASS on losses, S1 failures, missing ADV audits, row count and conflicts. It does not include its CERT audit violations, audit completeness or input-box disagreements in that condition; nor does it require unique full coverage, a positive gain or a verified complete freeze. At `:19` and `:35`, audit subprocess failures are ignored and stderr discarded. Consequently this script must not be reused as an authoritative promotion gate. Its actual result is FAIL, and that result remains unchanged.

CERT sampling is useful as a falsification check, not a proof of soundness or a replacement for the inherited mathematical tests. No evidence of the full inherited 3953-test, 207-file gate for this candidate was found in its inspected Python/Markdown records. Existing D130 test results apply to D130, not to this new engine.

The 23-entry manifests also omit imported `nhz_terminal.py` (`nhz_terminal_v3.py:19`) and the finalizer/audit tools. Passing the listed checksums is not complete dependency closure. Separately, LOG lines 949–951 disclose rewriting the older v10 preregistration and refreshing its hash after launch; that historical run cannot be represented as having an untouched pre-launch preregistration. We preserved this admission and did not rewrite any history.

### Some mathematical statements need narrower hypotheses

Definition 1 permits arbitrary phase forms and rows, but `THEORY.md:41` asserts that dropping the phase rows equals their continuous relaxation. The later ReLU construction supplies alignment/lower-row invariants needed for that assertion; the general definition does not require them. Restrict the claim to aligned reachable states or state and prove the invariant explicitly in a new revision.

`nhz_terminal_v7.py:49` selects ReLU rows; v8 supplements smooth rows but not the softmax predicate rows generated in v14. Such a plan can be an outer query relaxation, not a claim that every original predicate was carried into an exact terminal model. This observation alone is not a demonstrated false CERT.

## Independently recounted results

The five N039 v14 worker JSONL files contain exactly 2413 distinct family/instance keys. They report 1025 CERT and 765 ADV, for 1790 candidate solves. Relative to the formal baseline, they retain 1750 old solves, lose 120 and add 40 candidate solves, comprising 30 CERT and 10 ADV. Thus the net candidate count is 80 below baseline, even before qualification issues. The per-family no-regression gate cannot pass.

Formal baseline remains 1870/2413, consisting of 1063 CERT and 807 validated ADV. Independent E0 remains CIFAR100 25 plus TinyImageNet 36, or 61/400. Neither ledger was changed; the new partial E0 prefix earns no score credit. Keeping these ledgers unchanged does not mean the withdrawn candidate preserved their solutions.

## Conditions for any later revision

Retain the projection-aligned definition as unqualified research. Any implementation revision belongs in a new isolated directory with a new preregistration and complete source/configuration/budget provenance. First separate the representation from added splitting and solver portfolios, state its invariants, and enforce feasible-incumbent provenance and the existing witness semantics. Repair the promotion gate without lowering verification requirements. Then follow the inherited mathematical, real-structure, shadow, family and complete replay sequence. Do not restart broad replays merely because a local component looks promising.

No candidate, model, solver, GPU computation or test suite was launched during this review. Checks were source inspection, process identity/termination checks, checksum validation and parsing existing result records. The document-writing skill affected presentation and preservation of this audit only; it did not alter the research or permission boundaries.
