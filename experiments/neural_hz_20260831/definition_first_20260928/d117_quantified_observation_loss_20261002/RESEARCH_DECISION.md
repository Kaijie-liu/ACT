# Observation loss research decision

This turn adds a quantitative design result, not an executable candidate. The [ambiguity theorem](THEORY.md) relates exactly what a finite mean interface forgets to its best uniform consumer approximation. Independent reviews checked both inequalities, the distinction between a global worst-case summary and a particular summary, and the original-bit ReLU example. The theorem supports explicit bounded loss; it does not impose a new requirement of exact closure at arbitrary depth.

## What the archive review prevented repeating

The qualitative finite-feature obstruction, including a signed-measure proof and retained phase marginals, already appears in [D051](../d051_interface_sufficiency_20260930/PROOFS.md). Merely rediscovering it would not be progress. D004 concerns exact continuous interface compression; D100 concerns loss from separately boxing basis observations. Neither inspected record contains the factor-two minimax statement used here, but that is an archive distinction, not an external novelty claim.

The earlier D031 and D080 reviews already discussed junction-tree gluing of complete separator marginals. Matching only means does not meet those hypotheses. The [Mao, Zhang and Vechev paper, sections 4 and 5](https://arxiv.org/html/2410.06816v4), also already cited in D073/D080, separates fixed-range convex relaxation limitations from an existence construction carrying original inputs across layers. Its transformation still relies on strong convex-hull computations. Neither that result nor its input-partitioning alternative supplies a new admissible cheap operator here. No split or BaB is adopted.

Shared ReLU remainder coordinates were already reduced to known relations in D074. Positive common-denominator reparameterization and its residual limitation were already reviewed in D104; the nonzero-center comparison in D105 remains negative for the unlocalized radial row. No earlier negative finding was overwritten or revived as a positive result.

## A narrow positive spline boundary examined but not selected

One paper-only alternative has a precise scope. Suppose all retained value functions depend on the same original scalar z in an interval and share a piecewise-linear dictionary:

```text
f_j(z)=a_j*z+c_j+sum_{tau in T} mu[j,tau]*ReLU(z-tau).
```

For a mixed affine successor h=b*z+d+sum_j w_j*f_j, merge the shared curvature coefficients nu[tau]=sum_j w_j*mu[j,tau]. If all merged coefficients are nonnegative, h is convex. ReLU(h) is then convex, and adds at most two new breakpoints: the boundary of the interval where h<=0 has at most two endpoints. Existing breakpoints may disappear. Across G admitted gates, the union of breakpoints is therefore at most the initial count plus 2G. This permits some negative neural weights; the condition is on the merged function, not on each weight in isolation.

For example, on x in [-1,1], set

```text
u=ReLU(x+1/4), v=ReLU(-x+1/4)
r=ReLU(2*u+v-x/2-3/4)
s=ReLU(2*r-v+x/5+1/10).
```

r has curvature atoms at -1/3, 0 and 1/4 with coefficients 3/2, 1/2 and 1. The preactivation of s has merged curvature coefficients 3, 1 and 1 at those same locations, despite its negative v weight; its two new zeros are -23/36 and 3/44. These are paper identities. All original gates and bits would still have to be retained by any domain using them.

Convex composition is classical; [Amos, Xu and Kolter, ICML 2017, Proposition 1](https://proceedings.mlr.press/v70/amos17b/amos17b.pdf) gives a parameter-sign sufficient condition for input-convex networks. [Plonka, Riebe and Kolomoitsev](https://arxiv.org/abs/2207.14609) study one-dimensional ReLU spline representations. We claim neither a new general spline theory nor a neural verification algorithm from this observation.

This is not selected as the next executable path: common one-dimensional dependence has not been established on the target networks, generic mixed consumers need not preserve convexity, and a linear breakpoint count does not imply linear total storage. Retaining every gate's coefficients can still take quadratic space; roots, source authentication, exact phase guards, all consumers and terminal reconstruction remain payable. The old D014 general spline-growth warning is not contradicted. No root enumeration, source partition or numerical spline program was executed.

## Decision and next work

Do not implement a generic moment-closure, scalarized observation basis or convex-spline wrapper as a purported new domain. Retain this turn's quantitative result as a tool for evaluating a subsequent finite, source-preserving, possibly inexact interface. A useful subsequent proposal must identify a particular shared relation and a constructive certificate on ordinary model structure, compare against the same-information HZ/known relational reference, and account for propagation and terminal costs. This document registers no replacement runtime candidate or model experiment.

This does not close the Neural-HZ objective or narrow it to approximation theory. Domain definition innovation, nonconvex composition, useful GPU execution, trained-model gains and complete no-regression replays all remain required and unfinished. Known results are not counted as new formal solves.

## Custody and execution state

Date 2026-10-02 Australia/Sydney; branch redu-hz; commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac; tracked binary diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. Configuration was paper-only derivation, read-only archive review, primary-source reading and independent mathematical review. No candidate code, AST/import/compile/collection, tests, model, solver or GPU was run. There is no new freeze or RUN.

The previous goal turn was progress: D116 completed its fixed numerical comparison. This turn is progress from the quantitative proof and checked positive/negative scope; it is not another executed component pass. Latest executed evidence remains D116, 3857 tests in 191 files, with zero improved directions. Its 22-item seal was rechecked successfully.

All new files are isolated under experiments/neural_hz_20260831. Historical/production files and /data1/Kane/HyZor were not changed. No commit/push, external Page or remote backup was performed. The documentation skill separates proved statements, classical precedents, rejected hypotheses and missing qualification; no external rendered Page preview is claimed.

Formal baseline remains 1870/2413 (1063 CERT and 807 validated ADV); independent E0 remains CIFAR100 25 plus TinyImageNet 36, or 61/400. Both new gains are zero. The full Goal stays active and incomplete.
