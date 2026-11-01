# Fixed S2a closeout artifact (private, not a publication)

The archive reconstructs the two-model, first-50-input reproduction authorized
on 2026-11-01. It does not train, tune, expand a queue, or claim an independent
human replication. The empirical run used fresh venv installs on the original
Linux host. A new OS/container or other-host reproduction has not been tested.
The controller's resource probe still expects `/data1/Kane/MOE`; unpack under
that root for the supported path. No old checkpoint/data/run directory is
needed by the inference commands: the selected assets are included.

Contents: frozen ACT execution; clean pinned author sources; the actual R7
compatibility snapshot and full diff; two selected weights and input archive;
hash-locked wheels/requirements; original and reproduced subset records; paper
source, compact results, four-main-table renderer and seven-historical-table
renderer. Large dependency wheels are downloaded from their indexes under the
hash locks; the locally built onnx2pytorch wheel is included. No environment
site-packages are copied into a newly installed environment.

## Install in an unused unpacked directory

Use Python 3.12.12 and Python 3.11.16, matching the recorded base interpreters.
Do not use `--system-site-packages`. Set `REPRO` to the unpacked directory.
These commands install into that directory only; they do not alter act-py312.

```sh
python3.12 -m venv --copies "$REPRO/env-primary"
python3.11 -m venv --copies "$REPRO/env-engine"
"$REPRO/env-primary/bin/python" -m pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cu128 -r "$REPRO/primary-lock.txt"
"$REPRO/env-engine/bin/python" -m pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu --find-links "$REPRO/converter-wheel" -r "$REPRO/engine-lock.txt"
python3.12 "$REPRO/configure.py" "$REPRO"
```

The CUDA-enabled primary wheel matches the historical build; both commands
below hide GPUs and run CPU-only. The engine wheel is CPU-only. `configure.py`
only binds relocated paths and inventories; it performs no model inference.
Source checks retain the original byte identities. The original install
receipts are labeled historical provenance, not proof of the new installation.

## Ten-call smoke and remaining fixed queue

```sh
python3.12 "$REPRO/run_reproduction.py" smoke
python3.12 "$REPRO/run_reproduction.py" run
```

The smoke contains two IBP batches (50 inputs each) and eight complete calls.
The suffix contains the other 92 complete calls: 102 worker invocations total,
covering 200 model-input-mode measurements. Never rerun the prefix in the same
results directory. An UNKNOWN or TIMEOUT is a recorded outcome, not a reason
to change a configuration. Each complete request has one 300-second budget
for the whole network and all histories. Four single-thread workers run after
resource admission; the combined execution wall budget is four hours.
An 8 GiB address-space limit is per process, not a group physical-memory claim.

IBP processes 50 rather than the historical 256 inputs in its batch. Compare
all discrete flags exactly and list numerical element differences, rather than
using a tolerance to upgrade a result. Every full-verifier terminal is compared
exactly, including returned unknowns and outer timeouts. The original records
remain in `reference/`; `recorded_reproduction/` contains the completed internal
recreation. Neither is used as a solver answer during a new run.

## Tables and figures without a model or dataset

From the included analysis source root:

```sh
python3.12 -I -S -B scripts/architecture_closeout_tables.py --check
python3.12 -I -S -B scripts/render_current_review_tables.py --check
```

The first command regenerates all four main-text tables from committed S1/S2a
summaries; the second regenerates the seven historical verifier tables.
For figures, an existing environment with NumPy/Matplotlib can run:

```sh
python3.12 scripts/architecture_s2_figure.py --data paper/results/s2a_primary.json --output paper/figures/s2a_primary
python3.12 scripts/architecture_closeout_figures.py
```

Figure reconstruction adds no model calls. It is not empirical reproduction
of bounds. TMLR style files and their license/source hash are included. Use two
`pdflatex -no-shell-escape` passes on `paper/architecture_main.tex`, with output
in a new build directory. The manuscript is a preprint-style internal draft,
not a submission or an accepted paper. Authorship, licensing of any release,
independent human review, anonymization, and submission remain PI decisions.

After the fixed queue, run `env-primary/bin/python compare_records.py` from
the artifact root. It compares with the included fixed reference, not with
server-only historical paths. `SHA256SUMS` verifies the packaged files before
use. This comparison helper reports differences; it never revises either
original or reproduced acceptance.
