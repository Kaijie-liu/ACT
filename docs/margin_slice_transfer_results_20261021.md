# Saved margin certificate transfer results

The frozen saved-evidence diagnostic passed once at
`0f87a541c39b71d247ac3cb7069ce9d3a1363ce2`, outer exit0.
[Derivation and protocol](margin_slice_transfer_20261021.md),
[freeze](margin_slice_transfer_freeze_20261021.json), and
[lossless records](margin_slice_transfer_records_20261021.json) bind the result.
No new optimizer call, model execution, candidate, slice, gate bound or source
was introduced. This attempt is consumed and is not a performance experiment.

## The complete bound did not improve

| Saved record | Improved cells | Improved local endpoints | Original and transferred aggregate |
| --- | ---: | ---: | ---: |
| Zero tilt | 7/8 | 14/16 | −1.295650847 |
| Fixed median tilt | 0/8 | 0/16 | −1.289611148 |
| Matched zero using median's zero certificates | 7/8 | 14/16 | −1.295650847 |

Cell0 remains the minimum in all three records, so every aggregate difference
is exactly zero. Zero/matched-zero retain
−746891861839360909/576460752303423488; median retains
−743410212416693823/576460752303423488. These are three records of one synthetic
source, not three independent models or samples. All remain nonpositive.

The diagnostic verifies48 anchor-to-slab transfer identities and48 combined
local endpoint residuals across24 cells. It confirms that scalarizing residuals
early can lose local strength, but that loss does not explain the complete
bound here. It supplies no basis for increasing cells/tilts or building another
production optimization around residual cancellation on this source.

The finite-tilt bound still exceeds the separate zero bound by about0.00604,
as previously archived. The new finding is that a sliced shared-certificate
reference reproduces it exactly with no additional optimizer calls. It reuses
the already obtained tilted certificates and pays for construction/checking;
it is not a free independently executed baseline. Fresh strong-slice support
may be tighter, but its end-to-end cost has not been measured here.

## Independent checking and diagnostic cost

The diagnostic independently projects original expert outputs and checks every
combined dual against the same P plus both exact affine slab inequalities.
Both row signs, the nonzero margin center, duplicate row retention and residual
triangle identity are checked. A separate read-only Fraction implementation
reconstructed the original augmented CSR, full property and constraints and
rechecked all48 transfers and48 endpoint residuals without importing the
diagnostic or any native solver. A separate structural review checked all12
frozen files, original source/property identities, saved cells and accounting.
These AI reviews are not independent human review or model reproduction.

The single diagnostic used1,216,636 Work visits. Original-evidence rechecking
took0.901791s; the three transfer/recheck scopes took0.018745s,0.019001s and
0.018784s. Entry through result export was0.961287s. These are offline diagnostic
costs, not a baseline speedup or costs that can repair prior online acceptance.
Interpreter pre-entry and stdout/exit tail remain outside that measurement.
The original tool exit0 and stdout export time are retained separately in the
archive's `outer` field; PASS JSON alone is not their evidence.

Original run files total30,039 bytes, and the repository archive30,606 bytes,
for60,645 bytes under128MiB. Every original byte count/hash is reconstructed
from the archive. The process used the frozen8GiB AS limit and300s hard alarm;
the alarm is uncatchable, no observer or child process was introduced. Original
data, failed runs and frozen source files were not modified or deleted.

## Continue and stop decision

Stop the residual-combination optimization branch for these saved records:
there is no complete aggregate improvement. Do not rerun with new cut points,
extra supports, alternative maximizing pairs or a tuned source.

Continue the main A2 comparison as a **checked-bound and complete-cost study**,
not a claim of an irreducibly stronger representation. The next missing
interface is an exact closed-slice support view on the original bound HZ, so
fresh endpoint supports can be compared without changing P or guarantee grade.
Reuse existing candidate and residual kernels, rather than adding another
supervisor. A fair roster must include both a priced shared-certificate
reference and fresh slices, account for all range/feasibility/support goals,
and give N1/M3 their existing conditional-range and reasonable subdivision
capabilities. Native numerical-policy paths remain separately labelled unless
their slice/range construction is independently checked.

No A2 effectiveness gate, real-development stage or external-tool execution is
completed by this diagnostic. G1–G6 remain OPEN; real-stage usage is still
0 stages/0 calls/0 execution seconds. Historical source gaps and sealed
boundaries remain unchanged.
