# §7 — Related Work

## 7.1 Abstract domains for NN verification

The CROWN family (Zhang et al. 2018, α-CROWN Wang 2021, β-CROWN Wang 2021) uses per-neuron linear lower/upper bounds propagated backwards through the network. CROWN is the basis of most leading verifiers (α,β-CROWN, MN-BaB) and is sound. However, CROWN uses **backward propagation** through ReLU bounds, which is excluded by our principle set.

DeepPoly (Singh et al. 2019) and Star sets (Tran et al. 2019) similarly rely on backward-aware abstractions. DeepZ (Singh et al. 2018) is fully forward; its triangle ReLU relaxation is per-neuron and is a special case of the HZ triangle ReLU operator under our framework (§3.3.7).

Polyhedral abstractions (Cousot & Halbwachs 1978) give the tightest convex over-approximation but scale exponentially in dimension. ELINA (Singh et al. 2017) provides production-grade polyhedral abstract domain libraries; none currently use HZ as an underlying representation.

The present paper extends this lineage by adding HZ — a mixed-integer set representation — as the **first formal abstract domain with binary generators**, supporting cross-neuron correlation that pure convex domains cannot capture.

## 7.2 Hybrid zonotope literature

Bird (2022) defines HZ and proves the closed-form set operations (linear, sum, intersection, union) we use in §3.3. Bird focuses on hybrid systems (state-space reachability for piecewise-linear control); the NN verification application is developed in subsequent work.

Ortiz et al. (2023, hz1) prove the exact ReLU encoding (+4 gens, +1 binary, +3 cons per unstable neuron). This is the foundation of the per-neuron ReLU abstract transformer we build on (§3.3.7).

Zhang et al. (2022, 2023, 2024) extend HZ to neural feedback systems (closed-loop reachability with plant + controller), backward reachability sets (BRS), and nonlinear activations (SOS, OVERT). The BRS work uses backward propagation through HZ, which our principle set excludes; the SOS/OVERT activation extensions are orthogonal to and compatible with our framework.

To our knowledge, no prior work casts HZ as an abstract domain in the Cousot sense. The closest related is Bird's discussion of "containment hierarchy" (Bird §3.4) which observes the set inclusion relation between HZ representations but does not develop it into a Cousot-style lattice with abstract operators.

## 7.3 Multi-neuron precision techniques

Singh et al. (PRIMA, 2019) develop k-neuron convex hulls (k=1, 2, 3) as post-hoc cuts added to a baseline relaxation. PRIMA's k=2 is conceptually similar to our joint K=2 envelope (§4) but differs in two respects:
1. PRIMA's cuts are added at the LP level **outside** the abstract domain; ours are added as additional inequality rows **inside** the HZ representation, so they survive composition with downstream operations.
2. PRIMA targets standard small-dense MLP benchmarks; we test on the conv 0-verdict frontier where the empirical lift is structurally zero (§6.3).

Anderson et al. (2020) derive the IDEAL hull formulation for ReLU MIP, equivalent to per-neuron exact HZ encoding under appropriate parameterization. Their cuts are also post-hoc relative to the MIP solver.

Müller et al. (PARC, 2022) explore per-layer partition refinement for abstract interpretation; orthogonal to our work.

## 7.4 Forward-only verification

The strict "forward-only" principle our work operates under is uncommon in modern verification. Most leading verifiers (α,β-CROWN, MN-BaB, NNV, Marabou) use some combination of backward CROWN, BaB search, MILP solving, or gradient-based attacks. The principle constraint allows us to focus on **what sound forward-only operators can achieve** — and identify the structural ceiling (§6.3) clearly.

The Star-set verifier (NNV, Tran et al. 2020) is partially forward but uses LP solvers extensively. Our HZ abstract domain framework is a strict superset in expressiveness (HZ generalizes Star sets via binary generators).

## 7.5 Soundness-first verification

The soundness regression gate (§6.1, ACT regression pack) is in the tradition of Bak et al. (NNENUM 2020, NeuralSAT 2022) which prioritize formal soundness over benchmark speed. Our 8-instance regression pack is smaller-scope but covers distinct fix areas (conv path, dense ReLU, MaxPool, Sigmoid) and runs in ≈5 minutes — fast enough to be a pre-commit gate.

## 7.6 Comparison with VNN-COMP 2025 leaderboard

VNN-COMP 2025 evaluated 11 verifiers across 22 benchmarks. The leaderboard (publicly available) shows the verdict counts. We do not claim leaderboard-competitive results on conv 0-verdict (where we obtain 0 V+A); we claim formal soundness across all 22 benchmarks, abstract-domain framework completeness, and structural empirical evidence for the forward-only precision ceiling.

Concrete leaderboard comparisons on small-dense benchmarks (acasxu, linearizenn, tllverifybench) are provided in §6.2 and show that HZ matches or exceeds the next-strongest forward verifier on these classes while remaining sound under the stricter principle set.
