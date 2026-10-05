# Shared decoder source binding and the remaining query problem

Goal active, incomplete. D144 was progress: its frozen old-evidence screen changed
the priority of the fixed common-anchor grouping. D145 returns to definition
mathematics and adds a concrete binding theorem, not another census or helper.

[Full derivation and limits](definition_first_20260928/d145_stable_anchor_decoder_20261004/THEORY.md)
define D(y)=argmin_{q>=0,A^Tq=y} ||q||^2/2-b^Tq. If original strictly active
rows B span rowspace(A), the external decoded equations q_B=A_Bx+b_B>0 imply
q=ReLU(Ax+b). A direct exchange-direction proof replaces D131's explicit
phase-product source binding in this semantic representation. Original bits,
zero labels, all source predicates and live residual inputs remain protected.

Critical distinction: do not put anchor equations or arbitrary old P into the
decoder's argmin feasible set. That prevents the variations proving source
binding. A four-gate/two-source control with two ordinary crossing gates shows
the resulting conditional QP accepts a false amplitude. Old complete HZ already
excludes it; this is correctness evidence, not stronger verification capability.

Correct internal elimination is available: choose B as row basis, A_U=T*A_B,
A_B^Tv=y, q_B=v-T^Tu. The reduced objective has H=I+T*T^T on u=q_U, with
u>=0 and q_B>=0. It has |U| variables, not necessarily m. At the external
strict-positive anchors, its complementarity reduces exactly to the original U
gates. Do not call this a cost win or implement an optimizer wrapper. In the
control the old path already has only two nonlinear amplitudes.

The same exchange proof works for a specified proper closed strictly convex
potential family with a finite attained decoder minimum and interior smooth
anchors. It gives g_i in subgradient F_i(q_i); activation inversion must exist
on the actual range. Sigmoid/tanh anchor equations remain nonlinear. This is
paper scope, not smooth/Transformer coverage or a new Fenchel principle.

Prior active-row spanning and frame reconstruction were checked in Puthawala
et al. JMLR2022 and Haider et al. ICML2023; Fenchel activation graphs in Gu et al.
AISTATS2020. Do not declare external novelty. The novel-to-project delta is the
decoder source-binding contract and its exact proof/incorrect-restriction control.

A sparse old nonlinear frontier is not a recursive width guarantee: a single
q=ReLU(x+1/4), stable v_i=2+y_i-y0/4+c_iq, then ReLU(v_i-17/8) can realize all
successor phase patterns through independent live y_i. Do not infer cheap
closure from D144's 89/4800 crossing count.

Next substantive issue remains a useful, uniform joint query/re-abstraction for
shared nonlinear amplitude and mixed successor ReLU/live skip, without restoring
the whole original graph or relying on helper optimization. No candidate, tests,
model, solver, GPU, replay or background job was run this turn. Latest component
execution remains D136 3965/208; no test/resource requirement changed.

Formal 1870/2413 and external CIFAR100 25 + TinyImageNet 36 =61/400 unchanged;
both gains zero. Branch redu-hz, HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac,
tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5.
New isolated paper records only; old archives and production edits preserved.
Documentation skill used for local provenance/limitations, not an external Page.
