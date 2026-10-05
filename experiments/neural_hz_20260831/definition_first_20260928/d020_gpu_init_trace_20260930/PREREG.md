# Same cap CUDA initialization trace

This is one default-off diagnostic, not a Neural-HZ candidate, GPU kernel test,
network census or verification run. It follows D017's observed initialization
failure and tests the hypothesis that an attempted virtual reservation is
incompatible with the unchanged AS16GiB cap. Generic ENOMEM alone does not prove
that hypothesis. No cap is relaxed, and no CPU fallback or retry is permitted.

Branch redu-hz, commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac, date
2026-09-30. New exclusive result directory:
`experiments/neural_hz_20260831/results/d020_gpu_init_trace_20260930_v1`.
The first execution consumes this version regardless of its result.

## Frozen population and authority

Authenticate D017's source, preregistration, inventory, exit, results and every
inherited source/input/artifact identity. Preserve all 3735 tests in 165 files.
Add exactly four top-level nonparameterized parser tests, frozen by AST before
collection: the total gate is 3739 tests in 166 files. Collection plus test
execution must finish within the original single 60-second budget, with no
failure, error or skip. There is no smaller diagnostic-only gate. Test failure
means the trace is not launched. Reproduce the original selected-source list
without decoding models, and compare dependency populations before import.

Bind the new runner, parser, tests and this document before collection. Bind
`/usr/bin/strace` to SHA256
`28f957c227012de0b18d1bd7fff2d396cb693ea60ed8013be68de071e84b5001`,
as well as the unchanged interpreter, Torch/NVIDIA/driver dependencies and
production provenance. Old helper functions are read-only reuse; neither old
runner main nor any old result-writing function is invoked.

## Single initialization sequence

Only after the full gate passes and identities are rechecked, launch strace
with `--kill-on-exit -q -f -ttt -T -s 256
-e trace=mmap,mremap,brk,ioctl` around a new worker. The single `-q` suppresses
attach messages, not the process-exit footer required by the parser.
The worker sets AS16GiB before Torch import, records epoch/monotonic event times
and process memory, imports Torch, then calls `torch.cuda.is_available()` once.
Only if available, it calls `torch.cuda.init()` once. This is one initialization
sequence, not a retry. No tensors, kernels, fixtures, models or solvers run.
The GPU is fixed to GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc with LAZY module
loading. All caches and TMPDIR are confined to the new result directory.

CPU affinity remains one CPU; BLAS/runtime threads remain one. The traced
process group has one 240-second deadline and is killed as a group on timeout.
The trace and each worker output file have a stricter 16MiB RLIMIT_FSIZE. A
truncated or capped trace is incomplete, never evidence that a syscall did not
occur. Exit logs and artifact hashes are saved automatically even on failure.

## Resource and interpretation boundaries

The existing 256M whole / 200M nested numerical-work caps, 64M retained numeric
entries and 512-bit rational cap remain unchanged. This initialization-only
diagnostic performs no candidate numerical work and makes no qualification
claim for CUDA runtime internals. Parser output retains at most 1024 events,
reads at most 16MiB and rejects oversized or unrecognized partial trace syntax
for completeness. Raw evidence remains available in the bounded trace file.

Measure host high-water growth and tracemalloc from before Torch import; retain
both original 1GiB gates and the 65536-byte summary reserve. There is deliberately
no telemetry subprocess inside the traced scope: device-context memory remains
UNKNOWN, never zero. Therefore the combined physical gate is UNKNOWN and
complete_physical_qualification remains false even if initialization succeeds.
Host-only observations cannot substitute for the missing device measurement.
The strace sidecar also runs under AS16GiB, but its physical peak is not separately
measured. No sum-of-process physical qualification is claimed by this diagnostic.

A failed mmap/mremap request larger than the entire 16GiB address-space cap is
direct positive evidence that that request cannot fit this cap. It does not
identify the unique root cause of all CUDA failures or authorize a larger cap.
Small ENOMEM, brk results, ioctl failures and incomplete traces are observations
only. The parser never authorizes absence-based conclusions; even a complete
selected-syscall trace cannot rule out untraced driver/resource causes.

Record source/input/provenance drift, test inventory/JUnit, worker events,
bounded trace, parsed observations, failure and artifact digests. A missing
worker record is recorded as missing, not fabricated. Init failure can still
yield useful complete diagnostic evidence; this is reported separately from
initialization success. No GPU readiness, speedup, domain admission or new solve
follows. Baseline 1870/2413 and independent CIFAR100 25 + TinyImageNet 36 =
61/400 remain unchanged. All prior evidence and production files remain read-only.
