# C107 same complete canonical row, fused actual checks

Scope: _Encoder.encode after the unchanged original _bounded_powers has checked
each caller power array's integer type/range BEFORE broadcasting and signed
copying. Exactly one call per continuous/binary logical power input is retained.
Those bounded int64 arrays are actual operands, not caller flags. Generic emit
and _prepare retain the C104 numerical program for real radix definitions.

The new entrance checks actual continuous/binary ndarray dimensions and shapes,
every original coordinate's integer type/frame range and strict order, and RHS
finiteness. Native dtype and stride inspection precede reads. Original valid
coordinates are unchanged; the old store still copies them to owning int64
arrays. Invalid unsigned descending coordinates are rejected rather than relying
on unsigned-diff wraparound; no valid canonical original row is narrowed.

For each actual coefficient, the same finite/nonzero guard precedes frexp.
Original parent powers give the same exponent minima/maxima and largest
mantissa at the maximal exponent. Thus lower=-19-minexp and
upper=(41 if top_mantissa==.5 else40)-maxexp are identical, as is
shift=min(max(0,lower),upper). Only lower>upper returns None, preserving the
original radix/relay/mask/RHS fallback and its independent and global budgets.

Each output is exactly ldexp(original,power+shift), with the same finite and
ldexp inverse equality guard on EVERY continuous and binary coefficient. The
RHS uses the same finite ldexp(rhs,shift) and inverse comparison. Head sign and
mantissa derive from the same original first continuous coefficient and shift.
No approximate interval, floating sum or changed coefficient enters a row.

Only after this program returns may Python issue the private _Prepared payload.
The exact same one-use store enforces entry capacity, appends owning arrays and
head, and updates physical/logical counts. Actual encode_uid/pending UID,
radix-owner relocation, labels and final retirement checks remain. Matching new
strict encoder/tracker/stream types are used; no old type guard is loosened.
No C106 raw row or arbitrary serialized flag is a prepared certificate.

Existing preflight counts coordinate and numerical work; the same16 decision
and1 logical-power charges remain, with no credits or expanded budget. Moving
checks into one call removes temporary comparisons/diffs and Python dispatch;
it does NOT claim that work vanished merely because it is native. Successful
full source/report/owners/inverse fingerprints must equal the original proof.
Invalid-input exception timing is not a positive verdict or accepted row.

The native call holds arguments/GIL, handles typed strided inputs and creates
only two NumPy owning result vectors. No buffer exporter, hidden numeric cache,
borrowed output, long-lived pointer or manual NumPy metadata mutation. The
existing frozen x86_64 compiler/NumPy ABI and all headers are bound in the build
receipt; cross-platform portability is not admitted by this experiment.

Existing179 row/power/owner/source checks and18 ordinary complete encode/RHS/
Fraction/work comparisons pass locally; full inherited qualification, complete
ordinary source comparison and original-network resource/native/LIVE/terminal/
concrete witness gates are still required. No local test or proof paragraph
establishes a new model verdict. Formal gain remains0 until actual promotion.
