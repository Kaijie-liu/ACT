# C108 v2: fixed named-phase deduplication

V1 is closed before any real target: the ordinary fixture returned the same
point/objective and passed input/delegate identity checks, but the exact phase
list failed because SciPy's two-line _highs_wrapper expression visits line374
at two bytecode locations. LINE DISABLE applies per location, not named phase.
The complete v1 log/exit/source remain unchanged.

V2 adds one bounded set of at most14 fixed phase names. A repeated phase never
emits a second event; every callback still counts toward the original512 limit
and returns DISABLE. This changes only diagnostic event deduplication, no test
expectation, numerical code, native object or solver call. The fixture now
prints raw facts before its phase assertion, so any failure keeps them too.
All C108 preregistration limits and exactly one real target allowance remain;
v1 launched zero targets. New exclusive v2 source/output paths only.
