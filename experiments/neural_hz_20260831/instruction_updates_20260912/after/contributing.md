# Contributing to ACT

ACT (Abstract Constraint Transformer) is a PyTorch-native neural-network verification framework. Thanks for considering a contribution — this guide explains how to get set up, what kinds of changes are welcome, and what we expect in a pull request.

## What we welcome

The changes that get merged most easily:

- **Bug fixes** — especially soundness bugs (the verifier must never certify an unsafe property).
- **New layer support** — adding a `LayerKind` and its transfer functions.
- **New transfer functions / precision** — interval, hybrid-zonotope, or dual bounds for existing layers.
- **New solvers** — additional bounding/optimization backends behind the `Solver` interface.
- **Benchmark / dataset support** — new VNN-COMP categories or TorchVision dataset–model pairs.
- **Performance** — faster bounds propagation, batching, or BaB exploration (with before/after evidence).
- **Documentation** — READMEs, docstrings, examples.

For larger or architectural upstream contributions (new verification paradigm, major API change), discuss scope before submitting a large feature PR — see [Feature requests](#feature-requests).

## Before you start

- **Discuss non-trivial upstream contributions.** Use a short issue to agree on scope before submitting a large feature PR, and reference it in the PR. Already-authorized local implementation, isolated research, and testing may proceed without opening an issue or waiting for maintainer agreement. Creating an external issue or PR requires authorization; this guide does not grant it.
- **Check it doesn't already exist.** ACT has three TF families, several solvers, and many layer kinds — search the codebase before adding something new.
- **Trivial fixes** (typos, doc tweaks, obvious one-liners) don't need an issue.

## Development setup

**Requirements:** Conda (Miniconda), Python 3.12. Gurobi is optional (only for MILP/exact bounds).

```bash
# 1. Clone repository
git clone --recursive https://github.com/SVF-tools/ACT.git
cd ACT

# 2a. Primary core environment
conda env create -f environment.yml
conda activate act-py312

# 2b. (Optional) Run setup script (same act-py312 env, plus Gurobi)
bash modules/setup/setup.sh

# 3. (Optional) Gurobi license for MILP / exact HybridZ bounds
cp /path/to/gurobi.lic ./modules/gurobi/gurobi.lic
```

Verify the install:

```bash
python -m act.pipeline --help
```

## Project layout

ACT is a three-tier architecture:

- `act/front_end/` — data/model/spec loading and synthesis (`specs.py`, spec creators, `torchvision_loader/`, `vnnlib_loader/`).
- `act/back_end/` — the verification core: `core.py`, `verifier.py`, `layer_schema.py`, the `bab/` package, the transfer-function families (`interval_tf/`, `hybridz_tf/`, `dual_tf/`), and `solver/`.
- `act/pipeline/` — orchestration, Torch↔ACT conversion (`verification/`), validation, and fuzzing (`fuzzing/`).
- `act/util/` — shared device/path/config utilities.

## Running ACT

All native modules share `--device {cpu,cuda,gpu}` and `--dtype {float32,float64}`:

```bash
python -m act.back_end  --generate                          # build example nets from YAML
python -m act.back_end  --verify --network <name>           # verify a single ACT net (optionally --bab)
python -m act.pipeline  --verify vnnlib                      # end-to-end VNN-COMP verification
python -m act.pipeline  --fuzz --category acasxu_2023        # whitebox fuzzing
python -m act.front_end --list                               # list datasets/benchmarks
```

## Where to add things

| Contribution | Start here |
|---|---|
| New layer type | add to `LayerKind` in `act/back_end/layer_schema.py`, then wire transfer functions |
| Transfer function | `act/back_end/{interval_tf,hybridz_tf,dual_tf}/` (register the kind in the dispatcher + add the `tf_*` impl) |
| New solver | `act/back_end/solver/` implementing the `solver_base.py` interface |
| Benchmark / dataset | extend `BaseSpecCreator` (`act/front_end/spec_creator_base.py`) + register in `creator_registry.py` |
| Example network | add to the examples YAML in `act/back_end/examples/`, then `python -m act.back_end --generate` |

## Soundness first

ACT is a verifier — **soundness is non-negotiable**. A change is unsound if the verifier reports `CERTIFIED` when a concrete counterexample exists, or if abstract bounds fail to over-approximate concrete activations.

If a change affects transfer-function, solver, or BaB behavior, arithmetic, or numerical semantics, validate before and after. Comments, formatting, and documentation-only edits do not trigger soundness runs; run checks appropriate to those edits instead:

```bash
# Unified two-level validation (Level 1 counterexample + Level 2 per-neuron bounds)
python -m act.pipeline --verify netfactory --validate-soundness --device cpu --dtype float64
python -m act.pipeline --verify netfactory --validate-soundness --tf-modes interval hybridz dual
```

When a layer or solver cannot handle an input soundly, raise an explicit typed failure. At the verifier boundary, map expected unsupported-input, capacity, or unproved-precondition failures to a logged `UNKNOWN`. Keep unexpected implementation defects as explicit errors with diagnostics; do not silently relabel them as routine UNKNOWNs. Neither kind of failure may yield a positive verdict.

## Testing & how to verify your change

CI runs across both precisions and all tiers. The relevant workflows:

- `act-backend-float64.yml` / `act-backend-float32.yml` — backend + verifier validation
- `act-bab.yml` — branch-and-bound (branching + bounding strategies)
- `act-frontend.yml` — loaders / spec creation
- `act-pipeline-verify.yml` — end-to-end VNNLIB + TorchVision
- `act-pipeline-fuzz.yml` — fuzzing

Complete the relevant local checks before opening a PR and state **what you ran and what the result was**. An already-completed check may satisfy both change validation and the PR requirement when its source, configuration, and environment are unchanged. Repeat or broaden checks only after relevant changes, failures, or unresolved concerns. This does not waive independently required witness checks, preregistered experiment gates, or the complete 2,413-case Neural-HZ promotion replay and all 13-family retention requirements.

## Pull request expectations

- **Branch and target `main`.** ACT integrates into a single branch — there's no separate `dev` branch. Work on a short-lived topic branch (e.g. `fix/...`, `feat/...`) and open your PR against `main`.
- **Keep PRs small and focused.** One concern per PR.
- **Explain the problem and your fix** in your own words — what changed and why.
- **Show how you verified it.** For logic changes: what you ran and how a reviewer can reproduce it. For numerical/soundness changes: include the `--validate-soundness` outcome.
- **No AI-generated walls of text.** Short, specific descriptions. If you can't explain it briefly, the PR is probably too large.
- **Don't suppress problems.** No bare `except:`, no silently swallowing errors, no deleting failing checks to go green.

### PR titles — Conventional Commits

`<type>(<scope>): <description>`, where `type` ∈ `feat | fix | docs | chore | refactor | test | perf | ci` and `scope` is the area touched (`back_end`, `front_end`, `pipeline`, `bab`, `dual`, `vnnlib`, `torch2act`, …).

```
fix(dual): correct ReLU backward LB on negative multipliers
feat(bab): add N-ary input splitting
perf(interval): vectorize per-sample CE collection
docs: refresh back_end README for dual solver
```

## Coding style

Follow the standards in [`.github/copilot-instructions.md`](./.github/copilot-instructions.md). In short:

- **Type hints + docstrings** on functions/classes; PEP 8 / PEP 257.
- **`dataclass`** for simple containers; ABCs for extensible interfaces.
- **`logging`, not `print`**, for diagnostics; clear, actionable error messages.
- **Precise types** — avoid `Any` and type-ignore escape hatches.
- **Device/dtype** via `act/util/device_manager.py`; **never** hardcode `torch.device("cuda")` or a dtype.
- **Paths** via `act/util/path_config.py`; never hardcode paths.
- Remove only code made obsolete by the requested change. Preserve backward compatibility unless its removal is explicitly authorized.
- Prefer comprehensions, f-strings, and context managers where they read clearly.

## Feature requests

For net-new upstream functionality, agree on the problem, proposed approach, and fit for ACT before opening a large feature PR. Maintainer agreement remains required for that submission, not for already-authorized local implementation, isolated research, or testing. Do not create an external issue or PR without authorization.

## License

ACT is licensed under the **GNU Affero General Public License v3.0 or later (AGPLv3+)**. By contributing, you agree that your contributions are licensed under the same terms. Add the standard ACT license header to new source files.
