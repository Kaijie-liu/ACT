# §1 — Introduction

Neural network verification asks whether a trained network satisfies a behavioral specification — for image classifiers, whether an adversarial perturbation can flip the predicted class; for safety-critical controllers, whether a state in the operational envelope can drive the system out of safe bounds. Sound verifiers compute an over-approximation of the network's reachable output set and check it against the unsafe specification.

The dominant approach in recent VNN-COMP iterations combines **forward bound propagation** through linear layers with **branch-and-bound** search and **gradient-based attacks** on unstable ReLUs. While effective on the leaderboard, this combination has substantial implementation complexity and depends on heuristic choices that complicate formal soundness arguments — particularly when gradient attacks are used as a falsification channel parallel to the sound proof channel.

This paper takes an orthogonal stance. We **restrict ourselves to a strict forward-only, principle-compliant verifier**: no CROWN backward bounds, no autograd-based gradient attacks, no MILP via Gurobi, no fallback heuristics, no branch-and-bound. The verification engine consists exclusively of (a) sound forward propagation through hybrid-zonotope (HZ) abstract operators and (b) LP-feasibility checks against the abstract output set. Witnesses, when produced, are validated by **strict ORT replay** against the original ONNX network at zero tolerance.

The contributions are:

1. **HZ as the first mixed-integer abstract domain in the Cousot-Cousot sense (§3)**. We define a partial order `⊑_HZ`, a bounded join `⊔_HZ` (Theorem 3.1) that avoids binary explosion, a widening operator `∇_HZ` with termination proof (Theorem 3.5), and a systematic enumeration of sound abstract transformers for all standard NN operations. To our knowledge no prior work formalizes HZ as an abstract domain — Bird's seminal dissertation (Bird 2022) develops HZ as a set representation but stops short of the Cousot formalism.

2. **A new sound multi-neuron ReLU operator (§4)**. The joint K=2 envelope captures sound upper bounds on PAIRS of unstable ReLU neurons, using inner LPs over the pre-ReLU HZ to compute the joint upper envelope in 8 octant directions (default) or in spec-aware directions (last-ReLU mode). Theorem 3.6 gives the soundness proof. The operator integrates cleanly into the abstract-domain framework as a composable transformer.

3. **A multi-corner LP witness extractor for the verifier's Phase 4 (§5)**. Standard HZ-based verification accepts only the FIRST LP-feasible witness; if it fails ORT replay, the verdict is UNKNOWN. The multi-corner extractor enumerates additional LP corners and re-replays, sound by Theorem 5.1.

4. **Empirical evaluation across the VNN-COMP 2025 benchmark suite (§6)**. We demonstrate ~1000 sound verdicts across 13 small-to-medium benchmarks, including 47.8% decidability on acasxu_2023, 76.7% on linearizenn_2024, 97% on metaroom_2023 (the latter via the documented N=1 override). These are competitive with the strongest forward-only verifiers under strict soundness.

5. **A load-bearing negative result: the conv 0-verdict structural ceiling (§6.3)**. On the seven conv-heavy VNN-COMP benchmarks where the baseline produces 0 V+A (cifar100, dist_shift, soundnessbench, tinyimagenet, traffic_signs, vggnet16, yolo), we test THREE independent principle-compliant precision-side levers — multi-corner LP sidecar, joint K=2 octant, joint K=2 spec-aware. All three produce 0 lift across 47-54 sampled instances; the spec-aware variant introduces +6 OOM regressions on conv-heavy networks. We argue this constitutes empirical evidence that forward-only HZ + LP-relaxation has a structural precision ceiling on conv 0-verdict benchmarks, and we diagnose the ceiling as representation-bound (specifically: Girard reduction + project_eq_elim drop the cross-layer shared-ξ correlations that conv layers create; no post-hoc cut on the output HZ can recover this information).

The negative result is not a failure but a precisely-located limit. We propose (§8) that closing this limit requires representational change to HZ — preserving shared-ξ across conv layers without OOM — rather than further engineering on the existing representation.

## 1.1 Design principles

The five hard principles our verifier obeys:

**P1**. No CROWN-style backward bound propagation. The verifier propagates only **forward**.

**P2**. No autograd / no gradient-based attack. Witnesses come from LP-feasibility on the abstract output set + ORT-replay validation, never from PGD, FGSM, CW, AutoAttack, or related attacks.

**P3**. No Gurobi or MILP solver. The verifier uses scipy's `linprog` (HiGHS) for LP and never calls a MIP solver.

**P4**. No fallback to a different verifier on UNKNOWN. The verifier's output is `(verdict, witness?)`; UNKNOWN is honestly reported, never silently replaced by another tool's output.

**P5**. No branch-and-bound search. The verifier never splits the input box; the abstract operators must be tight enough on the full input region.

Plus a 2026-05-28 addendum after empirical investigation:

**P6** (no random-sample-then-check). Falsification candidates must come from a STRUCTURED procedure (LP-feasibility on the abstract output set), not from random or corner sampling on the input followed by ORT replay. This excludes the OrtSampleFalsifier approach which would otherwise produce a few "boundary" FAL witnesses.

The principle set is strict by design. We measure what sound forward-only HZ verification can achieve under these principles, and what it provably cannot.

## 1.2 Paper structure

§2 reviews background on NN verification, abstract interpretation, and HZ as a set representation. §3 introduces HZ as an abstract domain. §4 develops the joint K=2 ReLU abstract operator. §5 describes the multi-corner LP witness extractor. §6 reports the empirical evaluation including the 3-experiment negative result. §7 compares with related work. §8 concludes and lists open problems. Appendices A, B, C contain the formal proofs of Theorems 3.1, 3.6, 3.5 respectively.
