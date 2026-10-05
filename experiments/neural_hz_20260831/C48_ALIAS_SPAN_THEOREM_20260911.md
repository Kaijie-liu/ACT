# C48 exact no-hit routing lemma and ownership contract

Let L be the complete local-alias lookup used by the original quotient:
L[x] >= 0 iff column x is a local alias, and L[x] = -1 otherwise. Let n be
the full continuous frame and S[x] = min({y >= x : L[y] >= 0} union {n}).
Reverse minimum accumulation of an array initially n except at known hits
constructs exactly S. Bounds n <= 64000000 make its int32 representation
exact, including the sentinel n.

Consider the SAME strictly increasing continuous-column slice scanned by
the original quotient. Its own eligible alias defining pivot, if any, has
already been excluded by that unchanged rule. If its terminal column t has
L[t] = -1, removing t cannot remove a hit. For the remaining first/last
columns a,b, S[a] > b proves no member of [a,b] has a nonnegative lookup.
All considered columns lie within [a,b], so the original scan has zero hits.
Conversely S[a] <= b says only "possibly a hit": sparse gaps may give false
positives, which merely execute the original scan. There are no false misses.

The gate runs only when original row width >16; excluding at most two
endpoints leaves a nonempty interval. Short rows always take the old path.
The rule depends only on the canonical source and lookup, not row/instance
identity, learned thresholds, numerical margin or solver status.

Consequently every old hit row and every product input survives unchanged.
No product result, eligibility decision, rewrite, redundant-box proof,
frontier membership or witness is inferred from a no-hit summary. This
component does not issue a source proof. Integrating it into a changed C31
generator will require an independently verified new complete report and
source proof even if numerical output is byte-identical.

## Read transaction and limitations

The caller must already establish actual canonical ordering and exact
frame membership; NumPy/SciPy cached flags are not evidence. The complete
census explicitly checks these properties on every original row. Lookup L
is the original full table, borrowed without constructing a second table.
S is newly owned and read-only; hashes of both L and S are checked at entry
and retirement. The transaction is single-threaded and contains no source
mutation. All scans and final seals complete BEFORE any quotient rewrite
would be permitted. A failed seal cannot publish a miss-based quotient.
Read-only flags alone are not claimed to prevent mutation by another owner.
The helper weak-checks physical retirement of S; the full diagnostic keeps
all independently saved source/checkpoint owners visible until completion.

The standalone helper intentionally is NOT an arbitrary-row validator:
caller ordering and canonical-source preconditions are essential. Small
invalid inputs cannot cause a false miss because they execute the old path.
The reference census independently computes lookup hits for every source
row, including the miss rows, and rejects any counterexample before success.

The fixed work tariff charges 10*n+512 for complete construction/seals/
retirement, 4*K for known-hit scatter, one dispatch per row, and 16 per long
row query. These are declared operation-accounting envelopes, not empirical
speed measurements. The complete census separately pays its independent
oracle and source qualification; those diagnostic costs are not proposed
as additions to the production generator. The original generator still
owes all of its independently verified source obligations. Positive
component accounting alone does not prove whole-runtime or LIVE payment.
