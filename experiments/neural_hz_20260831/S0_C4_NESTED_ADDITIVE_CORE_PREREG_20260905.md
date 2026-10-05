# S0-C4: complete additive-subgraph Conv contraction, V1

2026-09-05, `redu-hz`, base `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
This follows the measured C3 zero-hit and the completed negative TLL thread
diagnostic. It neither reopens nor changes C1/C2/C3. Formal 1870/2413, E0
61/400, baseline preservation, representation and promotion gates are intact.

## Measured blocker and one mathematical rule

The fixed Tiny143 ReLU36 predecessor contains an entire nested ADD subgraph,
not a single ADD. Its four complete nearest-ReLU paths contain two Convs each;
all share the last Conv. The proposed representation contracts this whole
additive core, retaining every operand occurrence and every shared HZ source.

For a finite acyclic additive graph of branch affine maps, write its exact
source expansion before the shared outer Conv as

    r = sum_p D_p C_p h_source(p) + d.
    y = D_out [B r + b] + e
      = sum_p D_out (B D_p C_p) h_source(p)
        + D_out (B d + b) + e.

Here p indexes operand PATH OCCURRENCES, not unique nodes. Every preactivation
source uses the SAME continuous/binary frame and predicates, even when sources
are different intermediate HZs. Duplicate paths retain multiplicity; branches
are not given independent latents. Bias d is computed by the affine DAG with
zero source values, including every Conv and explicit BIAS exactly once per
graph evaluation (reuse at ADD still has its true multiplicity). Output bias
is never distributed once per source. This is an exact real set identity over
the unchanged nonconvex factor domain; arbitrary float reassociation is not a
bitwise or outward-rounding proof.

V1 accepts only complete paths containing exactly two Conv2D occurrences, with
one common outer Conv; the first multiplicative event is the inner Conv.
Every ADD is strictly between those Convs. All intervening/output SCALE maps
must be finite and exactly channel-stationary; BIAS transitions remain explicit.
All reachable parents are expanded back to INPUT/INPUT_SPEC/RELU, without a
caller-invented affine cut. Cycles, incomplete arity, unsupported operators,
post-outer ADD, another Conv, or a nonstationary scale reject the whole core.
Maximum path occurrences is 4096. No instance identity, label, margin, solver
state or terminal outcome is available to the rule.

## Immediate graph/resource preflight, before runtime work

Implement and run a necessary-condition screen on the same pinned, independently
certified corrected Tiny143 graph. It must enumerate all four source occurrences
without treating nested ADD as identity. A graph hit is not runtime lineage or
HZ simplification evidence. No solver or abstract propagation runs at this stage.

Reuse the existing descriptor's channel-group-intersection arithmetic to count
per-branch coefficient entries, coefficient-byte lower bound and contraction
products WITHOUT allocating coefficients or expanding spatial CSR. In this V1,
all occurrence descriptors are charged separately: no numerical cache discount.
Limits stay at 2,000,000 coefficient entries, 200,000,000 contraction products
and 64 MiB descriptor resident bytes; total contraction limit 256,000,000.
Any violation of these support-independent NECESSARY caps closes V1 immediately.
Passing them is not a resource acceptance: actual support-dependent emission,
whole reachable-state strict reduction, retained caches/other consumers, bias,
metadata, predicates, transients/RSS, and the existing work/concurrency gates
remain mandatory before integration. Kernel bytes are not the whole ledger.

## Exactness/shadows and advancement

Before executable integration, test nested/multiply-consumed ADDs, shared
continuous and binary factors, equality/inequality predicates, nonzero biases,
grouped/padded/strided/dilated Conv and zero-hit barriers. An independent exact
coefficient oracle on dyadic examples and concrete witness reconstruction must
agree; otherwise close V1. Resource estimates alone never authorize execution.

Target remains Tiny143 ReLU36 only. Same-structure small residual HZ shadows
precede any target abstract run. ReLU63/71, CIFAR100166/153 and formal Conv
families remain behind the existing advancement gates; this preflight does not
authorize jumping to them. The corrected BN loader remains default-off and its
13-family retention prerequisite remains open. No original archive is written.

New artifacts: `run_s0_c4_nested_add_preflight_v1.py`, its tests, and exclusive
`evidence/s0_c4_nested_add_preflight_20260905_v1.json`. Bind source, graph,
model/spec, configuration, branch/commit and this preregistration in the record.
On a necessary preflight failure, record and stop this version. On a pass,
implement the exact same-frame full-core materializer and prove REAL physical
reduction before any benchmark/default/score promotion. Formal gain stays zero.
