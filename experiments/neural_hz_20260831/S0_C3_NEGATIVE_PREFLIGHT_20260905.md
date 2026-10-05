# C3 negative necessary-condition preflight

2026-09-05, `redu-hz`, base `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
This records a read-only early rejection check before execution. It leaves the
frozen C3 grammar, target order, all positive gates and score rules unchanged.

The private C3 adapter currently accepts exact source boundaries only at INPUT,
INPUT_SPEC and RELU. Expand every affine operand path in the repaired graph
back to the nearest such boundary. This grants every RELU a fresh exact source,
even where a phase-sliced runtime would retain a longer history. If a resulting
path already contains two ADD events, every longer permissible history also
contains those events. It cannot match the frozen single-ADD grammar. A caller
cut at an affine ADD/BIAS/CONV is not an allowed source boundary. Unknown events,
malformed graphs and path-budget exhaustion produce an error, never a partial
negative conclusion. Repeated operand occurrences must retain multiplicity.

This is a necessary-condition screen only. A passing path is never a runtime
certificate, exact-set proof or authorization to plan. A failing condition can
close the registered occurrence without numerical propagation. This is the
same kind of topology-based rejection used in the preserved C1 closure.

The runner rebuilds the pinned Tiny143 model and specification, checks the V2
source and repaired graph hashes, and uses only a private repaired graph clone.
It also compares one fixed box-center forward evaluation by predecessor edges
with an independent variable-producer reading and the converted PyTorch model.
That diagnostic checks loader mechanics; it is not an exhaustive numerical
proof, sampling-based verification, a witness search or a verification result.

The runner does not import or invoke the verifier or C3 planner. Results use a
new exclusive JSON in `evidence/`. Trial9 source files and all historical data
remain unchanged. The formal baseline is 1870/2413, E0 is 61/400, and gain is 0.
