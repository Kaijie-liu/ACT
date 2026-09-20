# Reviewer workflows and their evidence boundaries

There are three different entry points. None silently substitutes a toy result
or a reconstructed table for an empirical complete-network proof.

| Workflow | Needs trained weights/data? | Establishes |
|---|---|---|
| Fresh source-defined request below | No | Executable tie-complete weighted proof control |
| Copy/check the input98 bundle | No, once bundle is supplied | Stored real request, conditional on upstream lowering |
| Rebuild committed outcome tables | No | Archived result arithmetic and source identity, not bound truth |

## Rebuild the main tables

From any directory, without site packages or numerical dependencies:

```sh
python -I -S /path/to/ACT/scripts/rebuild_moe_main_tables.py
python -I -S /path/to/ACT/scripts/rebuild_moe_main_tables.py --check
```

The first prints [the main tables](results/main_tables.md); the second rejects
a stale committed rendering. Four committed reviews supply the separate
confirmation, external, convolutional and evidence-mode cohorts. Denominators,
state/grade distinctions, paired count deltas and terminal inventories are
checked. No embedded server paths are followed. This reconstructs recorded
results, not their original model computations or independent SAFE proofs.

## Check a stored real complete request

Copy the entire input98 portable directory and retain the statement/bundle
hashes independently. The command and original portability tests are in
[the portable-proof result](../docs/portable_conv_proof_v1_results.md).
The checker uses `python -I -S`, forbids model/solver imports, outside reads,
subprocesses and network access, and checks all nine necessary properties.
It does not regenerate the upstream HZ. The bundle is currently a local
7.18MB artifact, **not distributed by this Git checkout**; access is needed
before this workflow is reproducible by an external reviewer. Publication of
weights, inputs or bundles remains a PI-managed decision.

This command checks the historical **conditional** positive bundle. The later
declared-source construction has different matrices and no complete positive
request; passing the former check does not validate the latter, nor can its
positive duals be reused there. The [source-contract section](sections/05_soundness_engineering.md#source-checking-and-non-transferable-certificates)
and [preserved history](appendices/source_proof_history.md) explain the boundary.
Input98 experimental follow-up is closed; these instructions are a reviewer
workflow, not an instruction to restart generation or solve missing properties.

## Generate a fresh source-defined request

This workflow needs no private checkpoint, dataset, server address or download.
The model and input are specified in source: three affine experts with margins
(-0.2,1,2), a tied router, and a two-dimensional input box. All three tie-legal
pairs must be covered. It is a correctness demonstration, **not** a replacement
for the trained-model tables or an estimate of accuracy/coverage.

Use the tested ACT Python environment. This revision's exercised dependency
versions are Python3.12.12, torch2.9.1+cu128, torchvision0.24.1+cu128,
numpy2.3.5, scipy1.16.3, PyYAML6.0.3, sympy1.14.0 and networkx3.6.1.
The demo runs on CPU and does not require a Gurobi license. These are recorded
versions, not a claim that an untested clean container installation has passed;
the repository's broader dependencies remain described in `environment.yml`.
The proof checker alone requires only Python's standard library.

From any working directory, using your checkout location and a **new** output
directory (no overwriting old attempts):

```sh
python /path/to/ACT/scripts/run_moe_proof_demo.py --output /path/to/new-demo-output
python -S /path/to/ACT/scripts/check_moe_request_lp.py /path/to/new-demo-output/proof
```

The first command creates the model/input, runs the standard scheduled verifier,
audits its evidence package, freshly constructs rational LP obligations, and
invokes the independent checker in a separate process without site packages.
`summary.json` records model/input/config identities, environment versions and
timings. The second command reconstructs the checked request conclusion from
the proof files; it neither loads the network nor calls an LP solver. Tampering
with a referenced file or omitting a required obligation must fail checking.

Expected: three legal pair/property obligations, one discharged by two positive
expert facts and two by direct rational weighted LPs, with a conditional positive
request result. The expert with negative margin cannot discharge those two
pairs alone. This demonstrates weighted fallback and complete tie coverage,
not a literal route-flip experiment on a trained model.

The remaining trusted base is network/input-to-HZ propagation, membership/pair
guard lowering and route exclusions. Exact rational checking does not establish
deployed floating-point semantics. The external-comparison checkpoint/data
distribution and a tested clean installation are still needed for the full
empirical submission artifact. No external weights or code are bundled here,
and this local preparation does not publish an artifact release.
