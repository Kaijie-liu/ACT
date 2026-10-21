# B2 isolated CPU deployment — one attempt

This is a necessary strong-baseline dependency stage, not a MoE algorithm
experiment or a completed GenBaB comparison. It uses the current B2 authorization
in [the goal](ALGORITHM_RESEARCH_GOAL.md). ACT remains in `act-py312`; the new
environment is `/data1/Kane/MOE/envs/genbab-cpu-20261021-v1`. No trained source,
model evaluation, solve, GPU or prior control is executed by this protocol.

## Source and dependency identity

- alpha-beta-CROWN: `e5c7e17bf0488843acb77b7519f59876717a49f4`.
- auto_LiRPA: `5a098e8f9fb5786a428a024981d833d303921f2d`.
- onnx2pytorch: `0826948e0d50cd09d909b7f26b2ee9ca2d3d71b3`.

The existing clean external checkout is read-only. Its `pyproject.toml` requires
Python3.11 and torch2.11, so this freeze uses CPU torch2.11.0 and torchvision0.26.0,
not the older `environment_pyt280.yaml` recipe or the old CUDA environment.
[Direct requirements](../configs/genbab_cpu_20261021_requirements.txt) are pinned;
transitive dependencies are resolved once and retained with their downloaded
artifact hashes in pip reports. They are **not** claimed pre-locked. The converter
is installed from the fixed author commit without replacing dependencies.

The old external interpreter supplies Python3.11's executable/standard library
for `venv --copies`, not its site-packages. The probe checks the new prefix,
Python family, CPU wheel, actual API/LiRPA source paths and absence of inherited
old site-packages; it records `base_prefix` and standard-library provenance.
No system/ACT packages are installed or changed. Author imports require ONNX and
Gurobi Python packages even for this CPU API; importing Gurobi is not using its
solver or asserting a license. Actual solver compatibility remains untested.

## Finite execution and storage

[Driver](../scripts/deploy_genbab_cpu.py) and [immutable manifest](genbab_cpu_deployment_freeze_20261021.json)
reuse the existing owned-session supervisor. The four stages are fixed:

| Phase | Total allowance, including cleanup | Action |
| --- | ---: | --- |
| 0 | 60s | Create isolated venv |
| 1 | 300s | Install the frozen CPU direct requirements and dependencies |
| 2 | 300s | Install the fixed converter source without dependencies |
| 3 | 120s | Import-only source/API/CPU checks |

Each phase reserves10s **inside** its allowance for owned cleanup. Deployment
has one900s cumulative deadline, including dispatch/checking/terminal export;
this is installation, not a900s verifier request or synthetic mechanism control.
The measured installation cost is separately charged to the eventual B2 stage's
6h envelope. There are zero real-request slots at this deployment stage. Import,
loading, serialization and process startup in later requests remain chargeable.

Every managed process inherits8GiB `RLIMIT_AS`; RSS is telemetry under
`DEV_MARGIN_20261021_V1`, and missing RSS is not zero. The existing16KiB console
limit remains; quiet/no-progress installers retain full dependency provenance
in JSON reports. This does not increase a request's evidence allowance.

New environment plus temporary build/cache storage has an8GiB **phase-boundary
stop threshold**, not a hard aggregate or peak disk guarantee. Installation
files have a1GiB `RLIMIT_FSIZE` (CPU shared objects can exceed128MiB). Deployment
storage is separate from the128MiB evidence limit for receipts/reports/results.
Before execution require16GiB free disk and8GiB available RAM, inspect concurrent
jobs, and perform only one bounded admission. No other user's process is touched.
Actual final deployment and evidence bytes, including failures, are reported.

## Acceptance and stopping

There is one attempt, fresh paths only. Bind source commits, direct requirements,
driver, supervisor helpers and interpreter hashes before launch. Require each
phase's normal exit and owned cleanup; accept deployment only after the actual
CPU/API import checks. Preserve partial installations and receipts on the first
unexpected failure; no automatic dependency repair, retry, alternative version,
increased time/space limit or fallback to the old environment.

After execution, saved-only audit checks receipt identity, finite cost, direct
versions, downloaded provenance, absence of CUDA runtime packages, converter
commit and final storage. It does not rerun imports or solve. Archive a compact
lossless copy of reports/receipts plus hashes; original and archive together count
towards128MiB. Neither a successful deployment nor the preceding25 specification
controls establishes native graph compatibility, actual nonlinear branching,
full-request performance, strict SAFE or G1 completion.

Next, if deployment passes, separately freeze necessary native graph/spec/dtype
and actual-engine controls. Actual B1/B2 inputs and complete request/query roster
must be frozen after the history exclusion ledger is closed. Do not launch a
trained comparison just because imports work.
