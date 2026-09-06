# Production staged verifier v1

## Purpose and scope

`act.pipeline.moe.staged_verifier` separates the executable verifier from the
Experiment 1 measurement harness.  It accepts a model, one represented
L-infinity box, and a top-1 prediction property.  It does **not** search for a
route boundary, run the matched no-support ablation, or propagate an
unguarded expert solely to collect before/after statistics.  Those operations
remain available to experiment runners, but they no longer consume the
verification request's budget.

Version 1 deliberately supports only the formal configuration established by
Experiment 1: an eval-mode, CPU/float64, selected-softmax top-2 output MoE.
Unsupported gate families and execution contracts fail before propagation.
This is an implementation boundary, not a claim that the staged proof rule is
restricted to top-2.

## State machine

The public entry point is:

```python
verify_staged_linf(model, center, epsilon, config, ...)
```

It executes the following fixed sequence.

1. Bind the model state, represented center/lower/upper tensors, property,
   epsilon, checkpoint identity, configuration hash, and numerical policy to a
   request identifier.
2. Propagate the router and decide the exact feasible unordered top-2 route
   family.  An incomplete candidate or route-set query terminates as
   `UNKNOWN`; it is never interpreted as an empty remainder.
3. Run guarded expert-wise gate elimination for every candidate expert.  The
   production path enables guarded support but disables both experiment-only
   controls.
4. Return `SAFE_GATE_ELIMINATION` if every obligation is proved.  Return
   `UNSAFE_FULL_FORWARD` only for a candidate that changes the literal full
   model prediction on replay.
5. Invoke F0 only for the registered semantic-incompleteness reasons.  It
   reuses the same exact router, feasible route sets, represented box, and
   property.  Every feasible pair and every property row must be proved before
   returning `SAFE_WEIGHTED_RANGE`.
6. A negative relaxation objective remains `UNKNOWN`.  Only a recovered input
   that lies in the represented box and violates the complete selected-softmax
   model can return `UNSAFE_FULL_FORWARD_FALLBACK`.

Solver limits and numerical failures remain explicit `TIMEOUT` or `UNKNOWN`
outcomes.  The verifier does not silently enter the segmented F1 fallback.

## Budgets and numerical acceptance

The tracked v1 configuration is
`act/pipeline/moe/configs/staged_verifier_v1.json`.  It allocates separate
budgets to candidate feasibility, guarded support, Tier-1 branch solves, F0
router-margin support, F0 expert-difference support, and F0 property solves.
These are algorithm budgets: no time is spent on a no-support comparison or a
boundary-finding experiment.

The configuration embeds the implementation's complete HZ/HiGHS numerical
policy.  Startup rejects any mismatch, including drift between F0's positive
acceptance tolerance and the policy-wide `safe_positive_margin`.  For F0, the
evidence records the outward-corrected accepted minimum separately from the
solver's raw bound contribution; the two values are not interchangeable when
the represented output has a nonzero center.

Version 1 has per-query and per-branch solver limits rather than one outer
process-kill deadline.  Consequently, its total wall time grows with the
number of exact feasible pairs and property rows.  The evidence reports every
invoked stage and total elapsed time.  A caller that needs a service-level
deadline must isolate the request in a worker and preserve a right-censored
active-stage record, following the confirmatory runner's hardened timeout
schema.

## Evidence package

`write_evidence_package` creates a new directory and refuses to overwrite an
existing one.  A package contains:

- `request.pt`: the literal center and represented lower/upper tensors;
- `evidence.json`: identities, route coverage, branch/property results,
  solver bounds, transitions, budgets, numerical policy, and final verdict;
- `witness.pt`: present only when a concrete counterexample was recovered;
- `manifest.json`: hashes binding all artifacts to the request identifier.

The request identifier is a canonical hash of model state, checkpoint
identity, represented tensors, property, epsilon, and verifier configuration.
A `SAFE` package explicitly records complete exact route coverage and the
decision tier.  An `UNSAFE` package must carry a full-model-validated witness.

`act.pipeline.moe.audit_staged_evidence` independently recomputes file and
tensor hashes, the request identifier, represented box, route/pair coverage,
Tier-1 or F0 proof completeness, positive F0 acceptance, and witness presence.
With `--replay-unsafe`, it also reloads the bound checkpoint and replays an
unsafe witness.  Structural audit cannot independently reproduce a SAFE solve;
the row-level solver evidence and pinned numerical policy remain the replay
inputs for a future full proof rerun.

## Relationship to frozen experiments

This code does not rewrite the immutable Experiment 1 confirmatory,
Experiment 1D, or multi-seed R1 results.  Their 300-second worker included
registered scientific controls such as matched no-support solving.  Their
coverage therefore measures the frozen experiment pipeline, not the best
production scheduling of the staged method.  Historical F0 hard-deadline
rows with missing duration remain right-censored rather than being imputed as
zero.

The v1 entry point is an engineering artifact.  Its toy tests establish state
transitions, F0 invocation, fail-closed configuration checks, immutable
artifact binding, and mutation detection.  It does not by itself establish a
new coverage or runtime result.  Any comparison on previously observed models
must be labelled development-only, and a later confirmatory cohort requires a
separate pre-execution manifest.
