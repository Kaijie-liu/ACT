# SoPlex exact input/output compatibility (controls only)

Date: 2026-09-20. Starting ACT HEAD `03b1c5f6a9c06af245b5c407242bb6c578089dc1`,
clean feature branch. PI approved isolated installation after the priority change
in `/data1/Kane/MOE/Advice/dd.md`. No real four-LP launch is authorized here.

## Result and boundary

SoPlex **8.0.3**, upstream commit
`13e2ab2467e0016d02116802ac4dc7a89560dbc1`, is installed at
`/data1/Kane/MOE/envs/soplex-8.0.3/bin/soplex`. A small exact-I/O probe is next
to it. Source/build/dependency copies are outside ACT, not committed. GMP 6.3.0,
MPFR 4.2.1 and Boost 1.90.0 were copied from existing local dependencies;
no package manager was run and `act-py312` was not modified. GMP/MPFR are
statically linked; runtime still relies on the recorded system C/C++ libraries.

- **10/10 compatibility tests PASS**, with ten tiny analytic native solves.
- Nine returned points pass the unchanged original rational LP checker after
  relocation under `python -I -S`. The infeasible analytic control supplies no
  point and remains unresolved in our checker: no infeasibility proof is claimed.
- All ten rational models read back identically before/after optimization,
  including `1/7`, the exact binary64 value of `0.1`, and a signed `2^-40`
  matrix coefficient. This is control evidence, not a general proof of parsing.
- Separate saved-evidence review: **PASS, zero issues**, 111 artifact hashes,
  ten fresh isolated checks, changed-byte rejection and exhausted-check deadline
  rejection. Zero new native calls during review.
- Unchanged original-checker regression: **21/21 PASS**.

Evidence: [controls](soplex_compat_controls_attempt001.json),
[fresh review](soplex_compat_review_attempt001.json),
[installation and regression inventory](soplex_compat_installation_v1.json).
Native/check subprocess times and point/bundle sizes are retained per case;
these tiny controls are **not** a performance experiment or a complete MoE
request-cost measurement. No new network certificate or real feasible U results.

## Exactness contract

`export_lp` interprets stored finite floats as their exact binary rationals,
emits fraction tokens without decimal rounding, and retains all original
inequalities, equalities and finite bounds. Names map every original coordinate.
The native objective deliberately omits the additive constant only; the original
LP (including this exact constant) remains unchanged in the final checking
bundle. The candidate objective is recomputed rationally including that offset.
This translation preserves minimizers; it does not claim the native printed
objective is the original value.

The probe sets/readbacks READMODE=1, SYNCMODE=1, SOLVEMODE=2, CHECKMODE=2,
minimization, zero feasibility/optimality tolerances and a ten-second native
limit. It emits every coordinate, including zero, in rational form. No native
status grants acceptance; partial, duplicate or misbound coordinates reject.
No dual witness is exported in this stage, so **checked optimality is not
claimed**, even where the analytic expected optimum equals the feasible U.

An important source-level limitation: this version's CLI `--writefile` calls
`writeFile`, which serializes `_realLP`, not the rational LP. It is unsuitable
as the rational-fidelity audit. Our small probe instead reads `objRational`,
`lowerRational`, `upperRational`, `lhsRational`, `rhsRational` and
`rowVectorRational` directly. Independent Python comparisons inspect all entries.
No general MPS/export compatibility is asserted; this stage uses LP syntax only.

The final unchanged checker validates every box, equality, inequality and
objective against the original LP. Wrong solver points remain non-feasible,
and a feasible relaxed point is **not a full-model counterexample**. Neither
SoPlex's exact label nor the structural audit removes network→HZ, guard,
route-exclusion or F0-lowering assumptions.

The Python export explicitly rejects more than eight variables or sixteen
constraints. The probe is an analytic capability tool, **not** a registered
real-study adapter. The controls have process timeouts, not a new unified
300-second production supervisor; peak-memory accounting, large-output bounds,
complete partial-evidence publication and real-study audit remain unimplemented.

## Reproducible installation details

Official source: <https://github.com/scipopt/soplex/tree/v8.0.3>.
Exact options also follow the pinned source's `settings/exact.set`.
The external prefix contains `src`, `deps/include`, `deps/lib`, `build`, `bin`.
`deps` contains copied GMP/MPFR headers and static libraries plus Boost headers.
No external source or binary is added to this repository.

Configure with CMake Release, `GMP=ON`, `STATIC_GMP=ON`, `MPFR=ON`,
`STATIC_MPFR=ON`, `BOOST=ON`, `Boost_NO_BOOST_CMAKE=ON`, `PAPILO=OFF`,
`ZLIB=OFF`. GMP_DIR, MPFR_DIR and BOOST_ROOT point to this prefix's `deps`.
Build only the `soplex` target with `nice -n 10`, `--parallel 1`.

The first link failed because static MPFR appeared after GMP. Preserve
`build.log`; the repaired configuration repeats `deps/lib/libgmp.a` in
`CMAKE_CXX_STANDARD_LIBRARIES`. `configure_attempt002.log` and
`build_attempt002.log` document success. No upstream source was patched.
`--version` is not a supported CLI flag; the banner is captured using no
arguments, with its expected missing-input diagnostic, not as a solve.

Compile `soplex_compat/probe.cpp` with `g++ -std=c++14 -O1 -ffp-contract=off`,
include `build`, `src/src`, `deps/include`, then link `build/lib/libsoplex.a`
and a linker group containing copied `libmpfr.a`, `libgmpxx.a`, `libgmp.a`.
Copy the two executables into the isolated prefix's `bin`. Source commit,
generated configuration, dependency/header identities, executables, failed and
successful logs are bound in the inventory. This is a local standalone binary
installation, not an installed C++ SDK or a claim of bit-reproducible rebuilding.

## Next bounded decision

**Do not continue the amortized-basis → supervisor → four-LP loop.** Its 106
controls and 9.496% local operation saving stay sealed; latest real diagnostics
remain four LIMIT and zero checked feasible U.

The next candidate is one finite SoPlex full-original-LP study, not another
self-written arithmetic algorithm. Before execution, independently verify
large sparse export/readback, bind the four unchanged original LP identities,
and freeze wall-time/memory/output-size/checking-cost limits and terminal rules.
Keep comparison to a fixed-basis solve conceptually separate: SoPlex is allowed
to choose a basis. Python `visit()` counts are not cross-tool work units.
No new sample, training, budget rescue, production-acceptance relaxation or
automatic real launch follows from these compatibility controls. SPEX is not
installed in this stage.

A checked `U<=0` limits the supplied relaxation, not the network; a checked
`L>0` proves that LP obligation under lowering assumptions. Missing U or
`L<=0<U` stays unresolved. Failure of this finite mature-tool comparison should
not block organizing the already substantial method and positive/negative
evidence into the paper.
