# CUDA initialization trace stopped by environment permission

The full component gate passed, but the syscall diagnostic produced no trace because strace's startup check was denied `PTRACE_TRACEME`. This does not establish the cause of D017's CUDA OOM. No GPU readiness, physical qualification, kernel result, speedup or new solve was obtained.

Date 2026-09-30; branch `redu-hz`; commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. This version has been executed once and is consumed. Its source, preregistration and failed evidence must not be edited or rerun. Root session 87157 finished with exit 1; tracer PID/process group 1509002 was no longer present in subsequent process inspection. No diagnostic from this run remains live.

## Executed gate and diagnostic evidence

All inherited 3735 tests and the four new pure parser tests passed: **3739 tests in 166 files**, no failure, error or skip, with 13 existing warnings. The authoritative combined collection/execution time was **56.908445063978434 seconds**, inside the unchanged 60-second gate. Pytest execution alone reported 43.65 seconds; that smaller number is not substituted for the combined time.

Only after that pass and identity checks did the supervisor launch the registered trace command. The complete worker log is:

```text
/usr/bin/strace: is_exitkill_supported: PTRACE_TRACEME: Operation not permitted
/usr/bin/strace: PTRACE_O_EXITKILL is not supported by the kernel
```

The first message establishes a tracing-permission denial. The second is strace's resulting capability report, not independent proof that this kernel fundamentally lacks EXITKILL. No disabling of EXITKILL, permission workaround, privilege change, cap change or alternative execution was attempted.

The raw trace is zero bytes and the worker record is absent. The result fields correctly say `trace_complete=false`, `worker_record_present=false`, `host_observations_within_caps=null`, and `combined_physical_gate=unknown`. `initialization_succeeded=false` means no successful initialization was established; it is not a new observed CUDA failure of the same kind as D017. No CUDA syscall or attempted allocation size was captured.

The supervisor and tracer exited 1. Total supervisor wall time was 69.20639469847083 seconds, including authentication and post-run checks outside the component test clock. No network model or property was evaluated. Inherited tests include their existing operations; this is not a claim that the entire test population makes zero solver calls.

## Integrity and qualification

Pre-run seals were verified before execution and again afterward. Source drift and input drift were empty; production provenance drift was false. Every artifact named in exit.json was independently rehashed successfully after execution. The two static-review corrections, made before the one execution, were conservative handling of malformed trace bodies and unconditional recording of worker resource failures. The full test population covered the parser cases; it did not certify the new phase-difference mathematics.

The [exit record](../../results/d020_gpu_init_trace_20260930_v1/exit.json) has SHA256:

```text
1cc060804dfb3928ba4f099134f11519b9bce1574265fd9123aa7f2305a36487
```

It binds the registration, collection inventory, JUnit/test logs, empty trace, parser report, worker log and retained test artifacts. Their absence/presence distinctions are preserved, not repaired after observation. All former source/results, including failed D017, remain read-only.

## Consequence for GPU work

This syscall-trace route cannot proceed in the current permission environment. Repeating it or removing the safety option would not be an authorized diagnosis. Continuing that route requires an appropriately authorized tracing environment; no such change was made. The underlying CUDA OOM and its possible relation to AS16GiB remain unresolved. The later read-only GPU memory snapshot is not evidence about memory at D017's failing call.

This narrow diagnostic obstacle does not invalidate or complete the mathematical Neural-HZ work. The parallel phase-difference theorem is stored separately and remains paper-only. A future GPU experiment requires a fresh isolated registration under unchanged numerical, resource and witness gates; the present failure cannot qualify a fallback or silently broaden permission.

Formal baseline remains **1870/2413**, external E0 remains **CIFAR100 25 + TinyImageNet 36 = 61/400**, and formal gain is **0**. No production/default change, score update, commit or push occurred. This record follows write-page guidance by separating executed evidence, missing measurements and unresolved cause; it is local Markdown only.
