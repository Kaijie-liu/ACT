# Checked binary64 enclosure: finite kernel result

The separately frozen [design](hz_binary64_enclosure_design_20261001.md) is now
implemented as an opt-in kernel. It closes **given exact owned HZ → actual
binary64 SparseHZono outer enclosure**, not network-to-HZ or a complete output
proof. Production defaults, old equality contracts and numerical gates are
unchanged. All six [research goal gates](ALGORITHM_RESEARCH_GOAL.md) remain OPEN.

## What changed mathematically

The producer retains exact rational reference coefficients and original
continuous/binary identities and owners. For each rounded output row it adds a
fresh continuous error factor whose radius covers the exact coefficient error.
For each rounded equality it adds a separate bounded slack. Inequality right
sides are weakened outward from the **original exact** right side plus the
coefficient error, including binary columns. The checker independently verifies
these inequalities, identities and zero-support layouts using rational arithmetic.

For every feasible original factor assignment, its same old coordinates extend
to a feasible target assignment with identical outputs. This is inclusion, not
set equality: the new target may be weaker. New continuous columns shift binary
flat offsets, so separate maps are checked. Expert-private error factors cannot
be identified merely because their coefficients match. Weakened guard rows do
not establish exact reachability or justify excluding other legal routes.

This is an enabling representation repair for the specific nonrepresentability
found by the [real-intake diagnosis](hz_real_intake_20261001.md), not a claim that
generic rounding compensation is a new MoE abstract-domain theory. Source
integration and useful large-model bounds remain separate obligations.

Code:

- [Untrusted producer and actual HZ instantiation](../scoped_source/hz_binary64.py).
- [Independent rational inclusion and assignment checker](../scoped_source/check_hz_binary64.py).
- [Seventeen finite control groups](../scoped_source/test_hz_binary64.py).
- [Execution and saved-evidence audit](../scripts/run_hz_binary64.py).

## Frozen scope and observed outcome

Execution is bound to design SHA-256
`62d58dc9f18a5a498679c92f905b99910badc7f2e6e89c8aa7f2da49520b4ab5`
and freeze HEAD `128c69c4b456ff05b99413b6bc016634d34cde9c`. Per-execution source
snapshots bind the new, then-uncommitted implementation. The accepted archive is
`/data1/Kane/MOE/baseline_runs/hz_binary64_enclosure_20261001_r3`.

| Supplied reference pattern | Added continuous factors | Outcome |
|---|---:|---|
| Exactly representable identity | 0 | Unchanged target; binary preserved |
| Affine residual `2^-104 + 2^-200` | 1 | Stored radius strictly rounds outward |
| ReLU endpoint expression `2+2^-54` | 1 | Output compensation checked |
| Equality RHS `2^-56−1/2` | 1 | Separate equality slack checked |
| Positive/negative guard RHS | 0 | Outward RHS checked |
| Half-minimum-subnormal coefficients | 3 | Underflow compensation checked |
| Shared input/two private experts | 2 | Ownership and binary offset checked |
| Empty factors/exact zero rows | 0 | Actual zero-width HZ instantiated |
| Overflow/no finite outward endpoint | — | Both supplied cases rejected |

All **17/17 groups** pass. Independent saved-evidence audit checks all eight
normal reference/target objects, actual HZ snapshots, 13 fixed exact assignment
embeddings, 24 specifically bound rejected mutations and two overflow refusal
sources. Assignment samples supplement, rather than replace, the inclusion proof
checks. Overflow is refusal by the registered construction, not a general theorem
that no alternative finite representation exists.

The 24 mutation records bind their entire concrete input and expected error to
fixed per-key hashes. Cross-key reuse of the same rejected record is tested and
refused. Actual errors are replayed by the checker, not accepted from the log.
Mutation during execution and expiry during final identity calculation also
reject. The mathematical checker does not invoke the producer or an optimizer.

The final saved-evidence audit passed with `python -B -S`, without importing
NumPy, SciPy or ACT for checking. It reads the repository's standard-library
checker modules and saved snapshots; this is **not** a new portable-package or
clean-environment reproduction claim. [Compact final audit](hz_binary64_enclosure_20261001_r3.json).

The 111 relevant source/HZ/support/device/propagation/intake/documentation
regressions pass. No dependency was installed. Existing Gurobi-license warning
does not indicate a solver call: there were **zero native solves, zero CUDA
calls and zero real requests** in this stage.

## Preserved attempts and cost boundary

R1 and R2 remain unchanged beside R3, with their compact reports
([R1](hz_binary64_enclosure_20261001_r1.json), [R2](hz_binary64_enclosure_20261001_r2.json)).
R1 predates final-deadline and concrete negative-record hardening. R2 fixes those
checks but its archive audit allowed rejected records to be substituted between
keys. R3 binds each specific mutation and tests substitution rejection. Earlier
PASS reports describe their then-current checks, not the stronger final audit.

All eight normal references, mathematical proofs, snapshots, exact embeddings
and checker outputs are identical across R1/R2/R3. No compensation rule or
mathematical acceptance threshold was tuned after outcomes. This was control and
archive hardening, not repeated search for stronger bounds.

R3 suite clock is about **0.8233 s**. Its first actual instantiation includes cold
numeric-library imports (about 0.7727 s); normal-case creation, instantiation,
serialization and checking costs are recorded separately. These are small local
controls with cooperative deadlines, **not** supervised request performance,
GPU acceleration or evidence that real models fit the 300 s budget. Audit time
is separate; no end-to-end speed claim is made. The three raw directories total
807,985 logical file bytes, including implementation snapshots and failed-control
history; nothing was deleted.
Stage handoff storage check measured 223,833,700,208 logical bytes under
`/data1/Kane/MOE` (a workspace snapshot, not this experiment's size). No raw
checkpoint, data or external repository is included in this commit.

## Recheck and next step

Using the existing environment:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m scripts.run_hz_binary64 audit /data1/Kane/MOE/baseline_runs/hz_binary64_enclosure_20261001_r3
```

Next separately connect the existing exact affine/ReLU/guard reference checks to
this enclosure kernel, preserving shared/private identity and generating fresh
evidence for every new matrix. Start within the existing finite source scope;
do not replace the old exact-reference contract with approximate equality.
This stage alone does not admit real models, larger caps, a physical GPU retry
or a new supervisor wrapper. Shared/private block capacity is a different factor.
Sealed inputs and all external-comparison results remain unchanged.
