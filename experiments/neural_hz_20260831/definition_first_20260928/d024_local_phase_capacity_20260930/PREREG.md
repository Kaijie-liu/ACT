# Rational phase capacity reference qualification

This is one explicit opt-in CPU rational reference experiment for the transfer in THEORY.md. It is not a full Neural HZ implementation, a GPU fallback, a census, a solver experiment or a scored verification run. Defaults remain disabled. New code is not imported by production.

## Frozen execution and population

Use a new exclusive results directory experiments/neural_hz_20260831/results/d024_local_phase_capacity_20260930_v1. Its first execution consumes this version regardless of success. Freeze this document, THEORY.md, INTERFACE_LIMITS.md, capacity.py, test_capacity.py, runner and source hashes before collection. No exploratory candidate execution precedes that freeze. Static code review and syntax inspection are permitted.

Authenticate the D020 trace experiment's frozen provenance and all inherited identities. Preserve all 3739 tests in 166 files and add exactly seven plain top-level nonparameterized tests in test_capacity.py, for 3746 tests in 167 files. Freeze their AST names and exact combined node IDs before collection. Collection and execution share the unchanged single 60-second budget; no failure, error or skip is allowed. There is no smaller component-only test gate. No retry or reduced population follows a failure.

Use /data1/Kane/miniconda3/bin/python with assertions enabled, one CPU affinity and one thread for BLAS/runtime libraries, AS16GiB, bytecode disabled and all runtime caches/TMPDIR relocated inside the new results directory. CUDA is disabled for this CPU reference. Do not import or initialize Torch/CUDA from new component code. Do not call old runner main or old result-writing functions. Source/input/provenance drift fails closed, and output, inventories, JUnit and exit status are retained automatically.

## Reference component scope and checks

The compiler accepts immutable tuples of exact fractions and a shared frame identity. Each original phase identity must be distinct. It computes its own affine box and paired difference bounds; external uncertified capacities are not accepted. Width and explicit source dimension are at most 128 for this reference qualification, fractions are at most 512 bits, and inputs and arithmetic results are checked. The restriction is not a new whole-project scope.

The whole 256M and nested 200M numerical-work budgets, 64M retained-entry limit and existing memory gates are not increased. The component charges exact arithmetic work and refuses to exceed its limits; structural size bounds and every stored rational are checked. No floating approximation, LP search, sampler, phase enumeration or witness attack is introduced.

The seven tests cover explicit opt-in and unsupported input rejection; exact scalar and common-frame difference bounds; fixed opposite-sign pairing and unmatched magnitudes; the complete rational two-layer and three-layer old-hull controls and their new rejection; preservation of zero-phase freedom through the valid inequalities; general-bias capacities and sign orientation; and work/bit/shape/identity fail-closed boundaries. The four fixed zero-point assignments are semantic checks, not a phase-enumerating runtime algorithm. Fixed rational evaluations are checks, not general soundness proofs: soundness rests on THEORY.md and the earlier derivation.

The supervisor retains host high-water growth and tracemalloc accounting with the inherited 1GiB limits and 65536-byte reserve. Test children retain the original test-gate scope; the experiment must not call aggregate CPU/GPU physical memory qualified. Keep complete_physical_qualification=false, native_HZ_admitted=false, GPU computation=false and formal_gain=0, even on a passing reference test gate. If any measured resource bound fails, the reference experiment fails and is preserved.

## Promotion boundary

A pass supports only the arithmetic compiler and its stated finite reference interface. It cannot establish real-network applicability, multi-block strictness, complete cost advantage, GPU readiness or score improvement. Original phased math-to-real-structure-to-shadow-to-family-to-full replay gates remain unchanged. Every old solve and all 13 family totals must survive before promotion; independent 400-case replay remains separate.

2026-09-30; branch redu-hz; commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. The old historical results, model files, production implementation, default configuration and earlier frozen experiments remain untouched.
