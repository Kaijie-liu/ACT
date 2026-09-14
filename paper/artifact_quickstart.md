# Minimal reviewer workflow (source-defined model)

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
