# C6 V1 pre-target development record

First focused test run: 21 passed, 8 failed. All eight failures were in the
expanded-geometry reference fixture, before comparing support counts: the
fixture mutated an existing implicit operator's row mask after construction,
leaving its precomputed expanded-nnz size stale. The native reference correctly
rejected that mismatch. Fix the fixture by constructing the masked operator
through its constructor. No native implementation or assertion was weakened;
no actual target had been evaluated at this point.

Before target evaluation, refine the preregistered work certificate from a
uniform output-row multiplier to a proved per-coordinate multiplicity upper
bound. An initial selected identity has multiplicity one, not output-count.
All ceilings, targets and score semantics are unchanged. Validate the tighter
bound against independent explicit row-reachability sets on complete small
programs. Freeze this final source and preregistration before the actual run.
