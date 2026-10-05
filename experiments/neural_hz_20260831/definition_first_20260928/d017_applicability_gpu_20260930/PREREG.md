# Shared source applicability and GPU prerequisite experiment

Date 2026-09-30. Branch redu-hz, commit
f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. New immutable result destination:
`experiments/neural_hz_20260831/results/d017_applicability_gpu_20260930_v1`.
This version is consumed by its first execution, successful or failed.

## Decision and scope

D016 established a conditional exact compression theorem, not actual CNN motif
prevalence. Its necessary source-dependence condition needs a complete, efficient
check. The companion applicability theorem derives sound negative rank
certificates. Before spending implementation effort on a full census, this
experiment tests the new goal's GPU prerequisite under the UNCHANGED host gates.

The only executed new numerical population is two synthetic integer matrix
fixtures with 64 rows, all 41,664 triples in each, independently checked on CPU.
The first is the Vandermonde matrix with rows (1,i,i*i); the second has dependent
rows (1,i,1+i), i=0,...,63. It is not a neural-network run, source prevalence
census, substitute for D015's failed population, or score-bearing experiment.
Every original model/spec selection and identity remains authenticated, but no
model is decoded and no property is solved here. A successful probe authorizes
no source or domain promotion by itself.

## Frozen qualification and execution

Authenticate D015 v2 preregistration, inventory, exit, diagnostic, helper runner,
and all inherited source/input identities against their original hashes. Preserve
its failed three-model census. Reproduce the same three selected_sources records
and all decoder identities. Reuse only authenticated side-effect-free helper
functions, not the old runner main or old result-writing functions.

Freeze these new files before any candidate import or test collection:
gpu_preflight.py, rank_probe.py, test_rank_probe.py, PREREG.md and
APPLICABILITY_THEOREM.md. Add Torch/NVIDIA/torchgen and the selected supporting
Python/native packages, driver libraries, and the absolute nvidia-smi executable
to the dependency inventory before Torch import.
Do not refresh inherited expected digests to conceal drift.

The new component population is exactly 3735 tests in 165 files: every inherited
3730 node ID in 164 files plus exactly five top-level, nonparameterized CPU tests
in test_rank_probe.py. Collection and execution share ONE 60-second budget;
no failures, errors, skips, retries or smaller population. CPU1, assertions on,
single-thread environment, no bytecode/cache writes, CUDA hidden for this gate.

Only after that complete pass and repeated identity checks may the separate
240-second worker import Torch and execute the GPU fixture. Bind
CUDA_VISIBLE_DEVICES to GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc and set
CUDA_MODULE_LOADING=LAZY. Do not silently switch to CPU if import, initialization,
telemetry, allocation or kernel execution fails.
Relocate CUDA, Torch, XDG, Triton and Inductor caches and TMPDIR into this new run directory;
do not let the experiment write cache artifacts into historical or home paths.

## Numerical and resource obligations

Use prime 32749, exact rational residues and int64 elementwise operations,
not floating-point determinants or assumed int64 CUDA matrix multiplication.
Prove the arithmetic overflow bounds. A nonzero projected modular determinant
can prove rational rank; zero is INCONCLUSIVE, not dependence. The synthetic
dependent fixture is known dependent from its construction, not inferred solely
from modular zeros. CPU closed-form references check every GPU result.

Keep AS16GiB, CPU1, worker240s, whole256M/nested200M work, 64M retained numeric
entries, 512-bit rational bounds and both existing 1GiB host-memory gates. Charge
scalar operations and container/transfer entries, not one unit per GPU launch.
Measure host memory/tracing from BEFORE Torch import, so setup is not baselined
away. Record initial/final virtual size and GPU allocated/reserved peaks.

Query this process's CUDA context memory through nvidia-smi. Require observed
RSS growth plus context sample plus 65536 reserve <=1GiB, and traced peak plus
tracer metadata plus context sample plus reserve <=1GiB. This conservatively
adds the observed device cost without weakening either old host gate. A context
snapshot is not a peak bound: complete_physical_qualification remains FALSE
even if these observations and all GPU computations pass. Initialization failure
under AS16GiB is evidence, not permission to enlarge the cap after observation.

## Evidence and stop conditions

Automatically save preregistered identities/configuration, complete collection
inventory, pytest/JUnit logs, GPU log/diagnostic, before/after drift checks, exit
status and artifact hashes in the new directory. If the supervisor times out,
retain its flushed logs and failure; do not invent a missing worker diagnostic.
Any failure consumes this version. Fixes require a new isolated version.

Report component qualification, GPU arithmetic execution, resource observations,
real-source applicability, native admission and formal scores separately.
No GPU speed claim follows from this fixture. No whole-network runtime, physical
qualification, new CERT/ADV or formal gain can be claimed by this experiment.
The 1870/2413 and independent 61/400 ledgers remain unchanged. Historical models,
source files, failed experiments, defaults, commits and pushes are untouched.
