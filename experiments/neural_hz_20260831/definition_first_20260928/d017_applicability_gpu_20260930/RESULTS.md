# Applicability proof and failed GPU initialization

The complete inherited component population plus the five new CPU tests passed:
3735 tests in 165 files, with combined collection/execution 57.95531381107867
seconds inside the unchanged 60-second gate. The separate GPU prerequisite then
failed during CUDA initialization before executing any determinant fixture.
This is neither a GPU-qualified candidate nor a real-network census.

Date 2026-09-30; branch redu-hz; commit
f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. Supervisor session 3141 finished with
exit 1 and was consumed. No D017 worker remains running. This failed version is
closed and must not be patched or rerun.

## Executed evidence

The sole run is
[d017_applicability_gpu_20260930_v1](../../results/d017_applicability_gpu_20260930_v1/exit.json).
It authenticated the unchanged original source population, 4417 newly inventoried
GPU/support dependencies, and every inherited test node. Pytest reported 3735
passed with the existing 13 warnings; no test failure, error or skip was admitted.
Collection reported 11.19 seconds and test execution 44.55 seconds; the combined
gate includes orchestration overhead and is the authoritative 57.955314 seconds.

The GPU worker then emitted, before any tensor fixture:

    cudaGetDeviceCount: Error 2: out of memory

The warning originates from torch.cuda.is_available(). The explicit no-fallback
path raised `CUDA unavailable; no fallback is registered`. Worker time was
1.8133665435016155 seconds; whole supervisor time was 71.45680274255574 seconds.
Neither of the two planned 41664-triple GPU populations ran. Do not report their
planned 83328 checks as executed successes or as a source-rank result.

Recorded host observations include initial VmSize 32047104 bytes, final VmSize
4856750080 bytes, RSS high-water growth 571654144 bytes, traced peak 65679181
bytes, and tracer metadata 29194848 bytes. These observed host components fit
their caps. They do NOT prove that an attempted, failed virtual reservation
would fit AS16GiB.

The process-context telemetry branch was never reached. The diagnostic's
`observed_memory_within_caps=true` uses a zero placeholder when that sample is
missing; it is NOT evidence of zero device/context memory or a passed combined
physical gate. `complete_physical_qualification=false` is authoritative.

Source drift and input drift are empty; production provenance drift is false.
All five pre-run file seals were rechecked after execution. The frozen D016
proofs and all earlier code/results remained unchanged. Temporary pytest outputs
were kept under the new run's TMPDIR, not mixed into historical evidence.

## What the failure does and does not establish

This attempt establishes that this registered Torch/CUDA initialization path did
not work with its frozen environment and resource limits. It does not identify
the unique cause. A subsequent read-only nvidia-smi snapshot reported 96450 MiB
free on the selected 97887 MiB device, and /proc/meminfo reported 85931080 kB
MemAvailable. Those later snapshots are not measurements at the failing call;
they do not prove a driver fault or rule out a virtual-address-space limitation.

No cap was relaxed, no CPU computation replaced the failed GPU run, and no source
models were decoded. A new, separately preregistered same-cap initialization
trace is the appropriate next diagnostic. `/usr/bin/strace` is available; tracing
failed mmap/mremap/brk reservations would distinguish evidence for the AS limit
from other CUDA initialization failures. Do not restart this version, assume
AS16GiB is the proven cause, or increase it without new authority.

## Mathematical progress independent of GPU execution

[The applicability theorem](APPLICABILITY_THEOREM.md) turns D016's literal
existing-gate secant requirement into a testable affine-rank obstruction. With
the explicitly stated dense convolution/support/scaling/free-input premises,
small cached rank certificates can cover all spatial/channel triples. In the
64-channel, pad-one 3x3 setting, the analytic sufficient populations are 374976
triple-rank tests and 32256 pair tests, rather than an unrestricted enumeration
of all spatial/channel triples. These counts are NOT observed model results.

The new CPU-tested primitive gives only one-sided exact modular certificates:
nonzero projected determinant proves rank; zero remains unknown. It correctly
keeps modulus/projection false negatives inconclusive. It is supporting
applicability machinery, not a new abstract-domain definition or a solver.

Fixed coordinates, zero weights/scales, unsupported geometry, later nonlinear
prefixes and paid new anchors remain outside any unproved shortcut. No conclusion
about D016's actual CIFAR/Tiny prevalence has been obtained. The next mathematical
decision still depends on authentic source evidence, not the control example.

## Custody and continuation

The previous D016 turn is progress through its exact/LP/composition theorem.
This turn is progress through the rank obstruction, an implemented default-off
primitive, the complete 3735-test pass, and a terminal GPU initialization failure
that changes the next executable action. It is not a verified wait or a claim
that the complete research goal is blocked. The goal remains ACTIVE.

The write-page guidance was used to separate proofs, tested CPU behavior,
unexecuted GPU intentions and unmeasured real-network claims in this repository
record. No external Page, production default, baseline table, commit or push was
changed. All new work is confined to the new D017 source and result directories.

Formal baseline remains 1870/2413 = 1063 CERT + 807 validated ADV. Independent
external E0 remains CIFAR100 25 + TinyImageNet 36 = 61/400. New formal gain is
zero. Full original-network/structural qualification, GPU kernels, physical cost,
new solves, and complete retention replays remain open.

## Artifact identities

- Exit: `e6e33752f227a81c12552cc856a9b41c1d40f7098cfa3022d3d5d8f7b5e16f8e`.
- Preregistered configuration: `e57f934ca7e8b5161667734c6bd10eeed9b929bb4b3c63e9f8c98c69bfb83cac`.
- Inventory: `ad99d5a92610fb3cd8415e7aeaf53856e0a1b2f7cf4d686493718cde3106415d`.
- GPU diagnostic: `4107fb5967647e76f361df7010ba7c51766833e53b84e4ad2ab5d12edc325980`.
- GPU log: `645566f102716eddba0070007ec2aa8b2081b0c5143973f7bdd5fe78c1b938f8`.
- JUnit evidence: `d1d2fbf1b5f3456f141b5f280cedabf515b447c2ab242c082694bfdb26260cab`.

The exit record binds collection/test logs and retained pytest artifacts as well.
PRE_RUN_SHA256SUMS binds the five immutable source/preregistration files.
