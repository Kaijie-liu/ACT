# B2 CPU deployment: execution completed, acceptance stopped

Execution used clean `6d025afb4ef4abee4eeb10c179d02b6e65d016be` and the
[frozen deployment](genbab_cpu_deployment_20261021.md), manifest SHA256
`729741a042d102c658858e38ea8024beda8dbef8f6628ebcd270459fc22d69e3`.
[Lossless records](genbab_cpu_deployment_records_20261021.json) retain all nine
original files, including the saved-audit failure. **Stage acceptance is
`NOT_ACCEPTED_SAVED_AUDIT_FAILED`.** The worker's `PASS` means execution/import
completion only; it was not and is not an accepted verification result.

| Phase | Execution seconds | Cleanup seconds | Exit |
| --- | ---: | ---: | ---: |
| Isolated venv | 3.681531 | 0.040639 | 0 |
| CPU dependency installation | 63.835278 | 0.042180 | 0 |
| Fixed converter installation | 3.440248 | 0.035904 | 0 |
| Import-only conformance | 4.545975 | 0.041110 | 0 |

Script-entry to pre-terminal-export cost was77.363190s, not full cold-start cost.
All four owned sessions were reaped with no live session member recorded; no
complete descendant-zombie or escaped-session proof is claimed. Two phases had
one incomplete RSS sample each, retained as telemetry under the frozen policy.
Environment storage is1,534,524,711 bytes and temporary/cache storage24,663 bytes.
These are terminal sizes, not measured peak storage. Nine original records total
628,924 bytes; original plus compact archive is below128MiB. Installation cost
is a separate B2 charge; real-stage usage remains0 requests/0 model calls/0 solves.

The probe observed Python3.11.16, torch2.11.0+cpu, no CUDA build, the new venv,
and the pinned API/auto_LiRPA paths. It reused the explicitly declared base
interpreter's standard library, not its site-packages. No existing environment
or external source was modified, and no GPU or trained source was loaded.

## Frozen saved-only audit failure

The once-run auditor exited1 with `ValueError: fixed direct dependency: packaging`.
The following saved/disk-only diagnosis ran **no imports or installation**:

| Dependency | Pip install report and top-level distribution | Probe's name-only map | Setuptools-vendored distribution |
| --- | --- | --- | --- |
| packaging | 26.3 | 26.0 | 26.0 |
| wheel | 0.48.0 | 0.46.3 | 0.46.3 |

The pip artifact hashes are
`d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c` (packaging)
and `3217dcc807155e45db462d7ef2431f5ddda0d7273b700d05a67b271ceb1287ab` (wheel).
The corresponding top-level `METADATA` and package `__init__.py` agree with those
installed versions. `setuptools/__init__.py` appends its `_vendor` path; the probe's
`{Name: version for distributions()}` loses location and overwrites duplicate
names. This explains the inconsistent inventory without evidence of an install
downgrade. **The actual loaded packaging/wheel paths were not saved**, so current
disk state cannot supply the missing online identity or retroactively pass R1.

The original frozen auditor and original worker record are unchanged. A separate
archive hash/decompression check is only lossless structural retention, not a
repair of failed dependency acceptance. No native graph/control/GenBaB solve was
started after the failure. There was no reinstall, import retry or limit change.

## Stop and next decision

Retain the environment and failure. The minimal correction would preserve
distribution location and duplicate identities, independently bind top-level
installed packages and actual loaded module paths, then perform a separately
approved bounded import reception without reinstalling. This is a finite
deployment-evidence correction, not algorithm progress or an excuse to restart
the observer chain. It has **not** been executed or used to claim B2 compatibility.

B1's independent read-only historical-input inventory may continue. Real
selection and comparisons remain unfrozen; G1–G6 and the scientific source gaps
remain OPEN. Actual nonlinear branching, full-route external competitiveness,
and new complete certificates are not established by these installation results.
