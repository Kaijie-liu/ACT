# Automatic joint forward support results

The default-off D158 operator passed its only frozen run: 4032 tests from 212 files, preserving all 4020 inherited tests. The meaningful new result is an automatic whole-box forward bound, not the test count: the actual query computes -3/55 for the positive control's next preactivation, and the actual next ReLU state has output bounds (0,0).

This fills a concrete D157 operator gap without adding an external verifier. It is not a new domain definition or a real-network result. Formal new solves remain zero.

## What changed mathematically and operationally

The [uniform rule](THEORY.md) combines the stored mass energy relation, branch caps and mixed-consumer dominance. It retains signed phase coefficients and combines shared source terms before taking a box bound. One sweep substitutes the owned banks' affine expressions; it never traverses the network backward, reads a solver state, searches for certificate weights or invokes a new optimization routine. The old norm certificate is computed alongside the new certificate on every query, and their scalar bounds are intersected. Failure in either calculation does not trigger fallback.

The [implementation](joint_support.py) preserves D157's native domain, all original binary phases, source identities, predicates, banks and decoder. It adds no domain coordinates or constraints. It incurs additional arithmetic and temporary certificate costs, which the inherited Work meter charges; zero additional coordinates is not zero runtime or a physical-memory qualification.

The key [test](test_joint_support.py) invokes child.bounds directly, constructs the following ReLU with that bound, and checks its whole-state output bounds (0,0). It supplies neither a precomputed upper bound nor the three proof weights used in D157. The incompatible original active label is rejected through guards and remains an original bit. The nonuniform-bias control and mixed-bias phase-affine control also pass, including their negative phase coefficients.

Other new tests cover negative mass coefficients, norm cancellation across duplicate consumers, shared skips, two-bank substitution, physical non-unit source boxes, zero labels, identity and arithmetic limits, and unchanged domain-coordinate counts. These are small mathematical controls, not a coverage census of trained networks.

## Evidence deliberately retained against overclaiming

D157's three-gate false same-input abstract output remains a legal native member. The new query bounds it soundly; its next positive false amplitude is still retained. Thus D158 improves how a relation is consumed, not every relation's precision. The tests also keep the difference between native membership and finite outer rows.

Two paper results in THEORY.md change the definition research decision. First, exact recovery through a fixed linear source quotient requires the relevant masked consumer rows to factor through that quotient. Independent or sufficiently rich ordered phases can force full coordinate rank; this is a scoped application of established invariant-subspace ideas, not a general impossibility result for Neural-HZ. Second, a convex fixed-phase enlargement containing a complete affine graph cannot be exact at one relative-interior source while lossy elsewhere in that same domain.

The investigated source-directed kernel contraction does repair the three-gate origin in its native nonconvex semantics. However, two strict same-phase interior members have a convex combination equal to the original fake point. Even the complete fixed-phase convex hull loses that repair; ordinary continuous LP auxiliaries cannot recover it. The contraction is therefore archived as a rejected execution candidate, not implemented as a second path. The proofs are paper derivations and independent review, not additional unregistered numerical runs.

## Frozen run and provenance

Run directory: [d158_joint_forward_support_20261004_v1](../../results/d158_joint_forward_support_20261004_v1). All six registered files and freeze.json were fixed before the first import, AST parse, collection or numerical execution. There was one run and no rehearsal, filtered test or retry.

- 4032 passed, 13 inherited warnings, no failures, errors or skips; exact ordered nodeid population authenticated before execution.
- Pytest elapsed time: 47.78 seconds. Combined supervised startup, collection and tests: 49.092110965400934 seconds, within the unchanged 60-second limit. Whole supervisor: 63.22842537611723 seconds, including evidence work outside that test deadline.
- Same CPU 0 and interpreter, one numerical thread, CUDA hidden, address-space cap 16 GiB.
- Supervisor RSS high-water growth: 0 bytes; traced peak: 19455194 bytes; tracer metadata: 6767952 bytes; reserve: 65536 bytes. Registered supervisor observations remained within 1 GiB. No aggregate child-process memory or candidate physical-memory claim follows.
- Source/input drift empty and production provenance unchanged. The new manifest authenticates 7327 source identities and 14 original inputs.
- All source, actual-model, original-phase binding, production, GPU and complete-physical qualification flags remain false. The [exit receipt](../../results/d158_joint_forward_support_20261004_v1/exit.json) contains artifact hashes and exact statuses.

The inherited fixed LP mathematical controls remain in the complete population; the new support operator itself contains no solver call. No candidate model worker, source census, GPU initialization, family shadow or benchmark replay was registered. The process finished successfully and no component job remains running.

## Next research decision and unchanged accounting

Retain this operator as support for the definition-first Neural-HZ line. Do not spend another round merely reproducing the same Householder certificate or expanding the harness. The remaining substantive questions are mixed-sign precision, recursive budget growth, complete consumer/source binding on ordinary real structures, and the full cost of using the domain. A real same-structure experiment needs its own frozen scope and complete binding before any model execution; partial blocks cannot be reported as whole-network gains.

GPU arithmetic, smooth activations and Transformer capability are still unimplemented requirements of the full goal. The source-directed rejection does not authorize new nonconvex solvers, phase splitting or changing the user's forbidden methods. Existing mathematical, real-source, shadow and full-replay gates remain intact.

Formal score stays 1870/2413 = 1063 CERT + 807 validated ADV. Independent E0 stays CIFAR100 25 plus TinyImageNet 36 = 61/400, never added to 1870. These baselines were not newly replayed in this run. No formal gain, default enablement or production integration occurred.

2026-10-04 Australia/Sydney; branch redu-hz; HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. Tracked diff SHA256 remained 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. All writes are new isolated experiment/evidence files. Historical models, production changes and frozen archives were not modified. The goal remains active and incomplete.
