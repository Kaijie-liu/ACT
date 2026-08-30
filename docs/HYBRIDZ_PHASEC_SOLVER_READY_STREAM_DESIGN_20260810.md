# HybridZ Phase-C solver-ready stream design (2026-08-10)

## Status and scope

This is a paused, design-only checkpoint.  It does not authorize production
integration, a verifier result, or a CIFAR100/TinyImageNet run.  No real or
large workload was used for this design.

The frozen invariants remain unchanged:

- propagation is forward-only HybridZ;
- every unstable ReLU has an exact phase/binary encoding;
- no triangle/convex ReLU fallback, ACT network BaB, backward pass, or dual
  tightening is allowed;
- ADD is the only RANGE family and any RANGE fallback rejects the whole build;
- every resource, arithmetic, provenance, cleanup, or solver ambiguity returns
  `UNKNOWN`.

The frozen Phase-A/B core and Operator candidate are not edited by this
design.  The disconnected Phase-C loader is also not a production candidate.

## Why the current route is stopped

The fresh 96-source-row AB/BA diagnostic measured the complete disconnected
route, including program construction, legacy-HZ reconstruction needed only
for disconnected binding, native loading, solving, exact Python replay, and
cleanup:

| Measurement | Candidate | Legacy baseline |
|---|---:|---:|
| median wall time | 17.533 ms | 2.015 ms |
| wall speedup | 0.1149x | 1.0x |
| CPU speedup | 0.1148x | 1.0x |

The important candidate stage medians were 3.596 ms for program build/seal,
2.066 ms for disconnected legacy-HZ replay/materialization, 1.135 ms for
object binding, 5.020 ms for Fraction-based transform/addRows, and 3.772 ms
for whole-program Fraction incumbent replay.  This route is therefore a hard
`NO-PROMOTION`.

There is also a Q=1 lower bound.  A 1.50x result against 2.015 ms requires the
candidate to finish in at most 1.343 ms.  The existing 3.596 ms program
build/seal alone exceeds that bound; even making every later stage free would
give at most `2.015 / 3.596 = 0.560x`.  Keeping the current core construction
cost and adding a solver sidecar cannot meet a Q=1 1.50x gate.

The 96-row fixture is deliberately binary-heavy: every row has one `A_bin`
nonzero.  It is a useful arithmetic stress test but is not a performance model
for C89.  The historical C89 K4 case had only four binary columns, and the ADD
body is expected to contain mostly `A_bin.nnz == 0` rows.  The 0.1149x result
must not be extrapolated to a C89-shaped direct path.

## The two admissible research routes

### Route A: solver-ready primary representation

An opt-in V2 representation owns solver-coordinate source rows directly.  It
does not simultaneously retain the old source program, a transformed sidecar,
and an expanded legacy HZ.  It is published by the Operator producer at the
same transaction boundary as the final factor frame.

This is the only route that can honestly attempt a Q=1 1.50x gate.  It would
require a new offline candidate and fresh lifecycle/soundness audits; it is
not an edit to the frozen Phase-A core.

### Route B: fair multi-query session

One producer-owned schedule and one HiGHS model are loaded once, then an exact
ordered query batch changes objective/cutoff data and solves Q queries.  Build,
transform, and model-load costs are amortized.  The legacy comparator must be
given exactly the same model/session reuse and query order.  Results must be
reported separately for Q in `{1, 4, 16, 99}` and may be claimed only for the
Q actually used by the verifier.  A Q=99 win cannot promote a Q=1 caller.

The first research prototype should measure both routes, but must not merge
their headlines.

## Producer-owned capability and API boundary

The intended object chain is:

```text
Operator producer transaction
  -> sealed ConstraintProgram / solver-ready schedule pair
  -> program-backed Operator build capability
  -> one-shot ordered HiGHS query-session lease
  -> non-authoritative solver candidates
  -> independent forward witness gate
```

Suggested APIs (names are provisional) are:

```python
program, schedule = sink.seal_solver_ready(
    final_frame=final_frame,
    highs_contract=HIGHSPY_CONTRACT_V1,
    input_factor_map=input_factor_map,
)

build_capability = publish_program_backed_operator_build(
    build=build,
    program=program,
    schedule=schedule,
)

session = issue_highs_query_session(
    build_capability,
    ordered_queries=queries,
    active_source_rows=ALL_SOURCE_ROWS,
)

result = consume_highs_query_session(session, resource_limits=limits)
```

`SolverReadySchedule`, the build capability, and the session lease have no
public constructors.  A process-local ABA-guarded registry retains the exact
producer, build, program, schedule, input-factor map, typed factor namespace,
query arrays, active source-row identities, and publication token.  Successful
append/seal/query publications receive monotonic epochs, but an epoch is only
a staleness check and never authority by itself.  A query lease is one-use and
converges to `CONSUMED` or `POISONED`; all occurrence and query identities are
burned on failure.

Digests and raw integer IDs remain diagnostic only.  Authenticity requires
the exact registry object graph, typed `ExternalFactorID` identities and
namespace, exact factor order, and the producer's non-serializable publication
token.  Cleanup uses bounded retries, and a cleanup exception can annotate but
never replace the primary exception object.

The optimized build must use a distinct program-backed HZ type which rejects
`.Auc`, `.Aub`, `.ub`, `solver_tuple()`, prefix/dual consumers, equation
substitution, FBBT, and any other whole-matrix path.  Supplying empty matrices
while pretending to be an ordinary `SparseHZono` is forbidden.

## Solver-ready row layout

Columns receive producer-owned solver ordinals when factors are allocated.
The ordinal map is bound to typed factor identities, not recovered from raw
IDs.  This permits later factors to append without shifting earlier binary
columns.  Query objectives and returned values use an immutable permutation
between Operator `(continuous, binary)` order and solver ordinal order.

Each source segment stores solver-native CSR (`float64` data and `int32`
indices/row starts), source-row typed identities, RANGE/LE sense, original
bounds, and only the mixed-row metadata needed below.  RANGE activation is
atomic at the source-row level: its lower and upper sides can never be
activated separately.

### Direct block: `A_bin.nnz == 0`

For a whole source block/batch with `A_bin.nnz == 0`, zero binary shift is
exact.  If every coefficient and finite bound is already inside the pinned
HiGHS thresholds, the loader passes the producer-owned `A_cont`, row starts,
indices, lower, and upper buffers directly to `addRows`.  There is no Python
Fraction, coefficient merge, `2*A_bin`, shifted-bound allocation, or batch CSR
reconstruction.  “Zero-copy” here means no Python-side coefficient or bound
copy; HiGHS necessarily owns its internal model copy.

If a direct block requires exact power-of-two row scaling, it is classified as
a prepared fallback rather than falsely counted as direct.

### Mixed sparse fallback

Only blocks containing binary nonzeros enter the mixed path.  During the
producer transaction it:

1. verifies that every `2*A_bin` is finite, exact, nonzero-preserving, and
   exactly reversible by halving;
2. computes each binary row sum with an exact dyadic integer accumulator, not
   `Fraction` or `numpy.sum`;
3. computes the exact shifted RANGE/LE endpoints and rounds lower bounds
   outward toward `-inf` and upper bounds toward `+inf`;
4. optionally chooses one positive exact power-of-two row scale; and
5. publishes an immutable solver-ready segment or fails the transaction.

The solver coordinate is still exactly `xi_b = 2*z - 1`:

```text
lower + sum(A_bin)
  <= A_cont*x + (2*A_bin)*z
  <= upper + sum(A_bin).
```

The primary V2 layout stores transformed coefficients once.  Compatibility
replay can exactly halve them into bounded transient batches; it cannot retain
a second full `A_bin` matrix.  A frozen-core shadow prototype may duplicate
data only for parity testing and must set `primary_storage=false` and
`promotion=false`.

## Exact arithmetic and HiGHS thresholds

The simplest authoritative arithmetic kernel is a native or Python-integer
dyadic superaccumulator which decomposes binary64 sign, significand, and
exponent.  It supports exact row sums, exact products for validation, exact
comparison to stored bounds, and directed integer-to-binary64 conversion.
No Python `Fraction` is allowed on the timed load or validation path.

An optional faster validator may first compute a rigorously outward interval
dot product (outward product and addition at every operation).  Rows wholly
inside or outside a bound are decided from that interval; only overlapping
rows fall back to the exact superaccumulator.  Ordinary `np.dot`, `np.sum`,
`longdouble`, BLAS error guesses, and tolerance-based row acceptance are not
rigorous substitutes.

At schedule construction, a row scale `2**k` is selected only if every
transformed coefficient round-trips bitwise and satisfies the strict HiGHS
conditions after scaling:

```text
small_matrix_value < abs(coefficient) < large_matrix_value
abs(finite shifted bound) < infinite_bound
```

The HiGHS options are pinned and read back before loading.  An option mismatch,
coefficient underflow/overflow, no feasible scale, `addRows` warning, changed
row/nnz count, finite value treated as infinity, or objective crossing
`infinite_cost` clears native state and returns `UNKNOWN`.  A zero-coefficient
row is checked exactly as a constant constraint; it is never silently dropped.

The initial schedule accepts only `ALL_SOURCE_ROWS`.  A later active subset
must be a producer-issued, immutable, strictly ordered sequence of typed
source-row identities.  It cannot reorder rows, split RANGE rows, or imply a
prefix from an integer count.  Any partial model is candidate-only: an omitted
row can never make solver status authoritative for SAFE.

## Rigorous witness path without whole-graph Fraction replay

For falsification, validating every auxiliary HZ row is unnecessary if the
candidate input independently violates the original network/property.  The
producer must therefore bind an immutable input-factor map containing, for
every flattened input coordinate, its typed continuous factor identity or
point marker, exact stored input bounds/center/radius, tensor order, shape,
and dtype.  The current raw `input_col_ids` array alone is not authority.

Only input-factor lanes are decoded.  Decoding uses exact dyadic arithmetic,
requires `xi in [-1,1]` at zero tolerance, chooses a representable model input
inside the raw input box, and rechecks the final cast exactly.  The existing
tolerance-and-clip decoder is not admissible for this exact path.

The decoded input then enters an independent, forward-only witness gate over
the raw ONNX and raw VNNLIB:

- Conv/Gemm/Add and other affine operators use exact accumulation or directed
  outward intervals;
- ReLU uses the monotone forward interval image, never a triangle relaxation;
- the raw Boolean/linear property is evaluated outward and must hold with a
  rigorous margin;
- no backward pass, dual computation, or ACT network BaB is called.

CPU ONNX Runtime plus zero-tolerance raw-VNNLIB replay can remain a useful
independent first filter, but it is not by itself a rigorous real-arithmetic
enclosure.  Until the forward enclosure supports every live operator and
proves the raw property, the candidate remains `UNKNOWN`.

This path removes the 3.772 ms whole-program Python Fraction replay from
witness authority.  Exact auxiliary-row replay remains a synthetic parity
oracle and an optional diagnostic.  If any solver status or abstract-row
claim is ever used for authority, it must instead be checked by the rigorous
interval/superaccumulator validator over all required source rows.

## Memory model

Let `R` be source rows, `N = N_c + N_b` source nonzeros, `R_m` mixed or scaled
rows, and `N_L, V` the expanded legacy nonzeros/facet rows.  Ignoring small
Python object headers, a primary int32 solver-ready schedule is approximately:

```text
M_schedule <= 12*N                  # float64 data + int32 column index
              + 4*(R + 1)           # int32 row starts
              + 16*R                # original lower/upper
              + 18*R_m              # shifted bounds + int16 scale
              + O(rows + factors + blocks).
```

For direct `A_bin == 0`, unscaled rows, shifted bounds alias original bounds,
so the `18*R_m` term is absent.  Mixed scratch is bounded by one configured
batch and is not proportional to the whole program.

The current frozen bytes-backed program is roughly `16*N + 32*R` before
Python headers, and disconnected binding additionally creates the expanded
legacy matrices, approximately `12*N_L + O(V)`, before HiGHS makes its native
copy.  The primary design must never have all of those representations live
together.

Peak memory must be measured in isolated subprocesses.  Endpoint RSS is not a
peak metric.  The required headline is incremental RSS HWM through final
validation and cleanup, with candidate HWM at most 80% of the fair legacy
baseline and no whole-program transformed scratch allocation.

## Offline correctness and performance gates

The old binary-heavy 96-row case remains a soundness/adversarial fixture only.
A new C89-ratio synthetic is mandatory.  Its structural ratios are taken from
the historical static observation:

- total source rows `57,418`, virtual facets `98,378`;
- total source nnz `6,243,172`, virtual nnz `9,267,556`;
- ADD source rows `40,960`, virtual facets `81,920`;
- ADD source nnz `3,024,384`, virtual nnz `6,048,768`;
- four binary columns.

A smaller fixture may scale these ratios exactly.  Because those totals do
not reveal the historical `A_bin` occupancy, the fixture must report direct
block/row/nnz hit rates and sweep mixed occupancy rather than inventing one
number.  At minimum it includes pure-ADD, sparse-mixed K4, and K4-dense
non-ADD variants.  No C89 projection is valid without the hit-rate fields.

Required gates are:

1. small exhaustive RANGE/LE parity in original `xi` and solver `z`
   coordinates, including cancellation, subnormal, overflow, zero rows,
   widths 1/129, and the offset-256 cursor boundary;
2. fault injection for producer publication, one-shot lease consumption,
   `addRows`, run, exact validation, and cleanup while preserving primary
   exception identity;
3. explicit forbidden-hook tests proving exact unstable-ReLU binaries and no
   triangle/BaB/backward/dual path;
4. fresh AB/BA wall and process-CPU timing from first append through final
   cleanup, with stage medians and semantic equivalence;
5. isolated subprocess RSS HWM;
6. Q=1 primary and fair Q=`{1,4,16,99}` session results reported separately.

A claimed route needs median wall and CPU speedup at least 1.50x and a paired
bootstrap lower confidence bound at least 1.25x, plus the 80% HWM gate.  If a
primary C89-ratio prototype remains below 1.25x or above 90% baseline HWM
after direct-block optimization, stop that layout instead of integrating it.
If a session wins only beyond a measured break-even Q, callers below that Q
must remain on the legacy path.  No real/large ladder starts until all offline
gates are green and independently audited.

## Minimal next prototype and next first step

The smallest disconnected file surface is two new files only:

```text
act/back_end/solver/constraint_program_highs_schedule_candidate.py
act/back_end/solver/test_constraint_program_highs_schedule_candidate.py
```

They may simulate producer-issued primary storage and compare it bitwise with
the frozen core, but must not edit `constraint_program.py`, `operator_hz.py`,
`solver_hz.py`, `verifier.py`, configuration, or production receipts.  A
native accumulator is a later third file only if the exact Python-integer
prototype passes the structural stop-loss and native speed is the remaining
measured blocker.

On resume, the first action is to implement only the C89-ratio schedule
fixture and the `A_bin.nnz == 0` direct-block load path, then run the Q=1
primary-simulation and HWM stop-loss.  Do not begin producer/verifier
integration before that result.

## Largest unresolved risks

1. A sidecar retaining the frozen core cannot meet the Q=1 timing lower bound
   and may erase the source-memory advantage.
2. The actual C89 `A_bin` occupancy is not present in the current static
   receipt; ratio synthetics must not masquerade as measured model telemetry.
3. Highspy may make additional native copies even when Python buffers are
   direct; only isolated HWM can settle this.
4. Existing whole-matrix HZ consumers are numerous.  A program-backed object
   must reject them explicitly rather than expose missing constraints.
5. A rigorous independent forward witness for all CIFAR100/TinyImageNet ONNX
   operators is a separate proof obligation and may be dominated by Conv.
6. Exact power-of-two scaling may have no feasible exponent when tiny and
   large coefficients share one row; the only sound outcome is `UNKNOWN`.
7. Multi-query amortization is valid only if the production query count and a
   fair reusable legacy comparator match the benchmarked session contract.

This checkpoint intentionally ends before implementation or promotion.
