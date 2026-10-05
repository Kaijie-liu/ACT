# Resume after shared nonlinear decoder research

The active goal is unchanged: definition-first nonconvex Neural-HZ for ordinary CNN/PWA, smooth/Transformer, full GPU, all 13 families, independent CIFAR/Tiny and new families. Formal baseline remains 1870/2413 = 1063 CERT + 807 validated ADV. Independent E0 remains CIFAR100 25 + TinyImageNet 36 = 61/400. Both gains are zero; an unchanged ledger does not prove candidate preservation.

## Latest mathematical work

[D131 theory](definition_first_20260928/d131_nonlinear_common_decoder_20261002/THEORY.md) proves that y=A^T ReLU(Ax+b) uniquely determines the entire activation vector. A single strictly convex decoder supplies all mixed consumers; nullity(A^T)=1 admits a shared scalar clip closed form. A three-nonparallel-gate control separates this from constant affine decoding. Original x, all bits and zero labels, EQ/LE, shared identities and every consumer remain protected. Binding y=A^T diag(beta)(Ax+b) is necessary; retaining the name of x alone does not bind the decoder to it.

This is paper evidence, not a qualified implementation or a new-domain/novelty claim. General decoding hides an m-variable convex problem; linearizing its KKT or materializing the corank-one clip can restore the old amplitude width. Potential/Fenchel/complementarity precedents are documented in [primary sources](definition_first_20260928/d131_nonlinear_common_decoder_20261002/SOURCES.md). Do not start a numerical wrapper merely because the toy decoder is closed form. Next proof work must supply a useful, fully paid forward/query rule for ordinary multi-consumer structures, or revise the hypothesis. Do not return to sorting/cache/test-maintenance work as the main research objective.

[Research decision](definition_first_20260928/d131_nonlinear_common_decoder_20261002/RESEARCH_DECISION.md) and [status](definition_first_20260928/d131_nonlinear_common_decoder_20261002/STATUS.json) separate the mathematical result, negative controls, prior art, costs and missing qualifications. All derivations were independently reviewed on paper, not machine-checked. No new numerical candidate, model, solver or GPU was run this turn.

## Latest numerical execution

D130 unique session 22072 is terminal, supervisor exit 1. [Postrun archive](definition_first_20260928/d130_import_isolation_20261002/RESULTS.md) and [status](definition_first_20260928/d130_import_isolation_20261002/STATUS.json) now accompany the original frozen evidence. Full 3953 tests/207 files passed within 48.93076469562948 seconds, zero failures/errors/skips. Source execution stopped on its internal 235 second evidence-preserving timeout, not a work-cap event. Completed 38/192 rows and 233/1152 roots; the last five roots do not form a complete 39th row. IBP did not start. Source/native/model/GPU/full-physical/shadow/full-replay qualifications remain false.

The saved 38 original-PGD-model rows all improve preactivation lower and upper bounds against the same-information independent score/value rectangle, not against complete old HZ. ReLU lower improvements 11, upper improvements 13, newly stable phases 0. These are real-arithmetic local coefficient-derived bounds, not full-network floating Softmax certification or new benchmark solves.

D128 session 33461 and D129 session 54232 are also terminal. Their protobuf/container and test-import failures remain immutable. No session should be restarted. The D130 source run artifacts and frozen files were rehashed this turn. Latest complete CNN source remains D120: 1600 local readouts, zero next-ReLU improvements. D128 CNN paper controls remain covered by stronger D018/D052 comparisons.

## Execution boundaries and provenance

All writes are new isolated files under experiments/neural_hz_20260831. Historical HyZor data, models, production code, prior frozen sources/results and scores remain read-only. No commit or push. No new candidate import/AST/compile/collection/execution before a new preregistration and freeze. Keep the full inherited test population and existing resource/physical/numerical gates; no instance/solver-state menus, attacks, BaB, input/phase splits or backward/dual rescue. Formal promotion still requires the same candidate/configuration/budget/path full replays and every old solve preserved.

2026-10-02 Australia/Sydney; redu-hz; HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac; tracked binary diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. Previous user turn was status-only/no progress; this turn adds new mathematical evidence and completes postrun documentation. Goal remains active and incomplete. Archive SHA256 manifests are post-readback custody, not preexecution numerical freezes.
