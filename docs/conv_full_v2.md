# Authorized full convolutional three-arm V2

PI instruction2026-09-15 explicitly authorizes freezing and executing90 calls:
the ORIGINAL30 unexecuted clean-selected inputs, selected epoch89 checkpoint,
epsilon2/255,300s complete-request cap. No reselection, extra training, support
change,25% retuning, numeric gate relaxation, or additional solver search.
`scripts/conv_full_v2_protocol.json` is the new execution authority.

All old ACT source, method configs, selection and failed R1 stay immutable.
The old request `protocol` field continues to identify E4/C10 top2 SEMANTICS;
the new `full_execution` binds the full-run source/protocol. Both ACT arms use
exactly the V2 implementation from the successful smoke, with5s reserved
INSIDE300s. Its embedded historical policy says it did not itself authorize
full90; that policy is not a live experiment status or the new launch authority.
Only this explicitly separate protocol grants that authority. CROWN receives
no V2 budget adapter and uses its unchanged ACT route frontend, plain CROWN
whole-box variable selected-softmax pairs, fixed CPU/float64 and environments.

The same30 stored inputs, frozen30-rank cyclic three-arm ordering, checkpoint,
model state and config hashes are independently reconstructed before execution.
The ACT gate is the successful V2 four-request smoke; the CROWN gate is its two
complete audited original smoke records and unchanged source/backend. The old
overall smoke remains FAIL. No new smoke calls or fresh outcomes select a gate.
Preflight re-audits both historical runs and the clean-only selection in a
separate process, then stores an identity-bound freeze review before Git freeze.

Run root: `data/moe/results/conv_three_arm_full_20260915_v2`. One worker, shared
project lock,1 thread, no GPU. Start is before request serialization/imports,
so loading, routing, common facts, support, construction, cross-environment
handoff, all solving and terminal submission count. Resource waiting is separate
and bounded24h per request with existing16GiB/5GiB/load-per-core gates.27000s
is the sum of REQUEST CAPS, not a guarantee including waits/audits or scheduler
latency. An owning process-group watchdog remains mandatory: a correct passed
native allocation does not ensure the native solver returns within it.

No resume, overwrite, replacement, budget extension or outcome-dependent stop.
ERROR fail-stops with all attempted terminals and explicit unattempted roster.
UNKNOWN and TIMEOUT continue normally. Partial/late packages remain inventoried,
never accepted after the outer cutoff. The runner records no unreturned native
result as zero. All tie-legal branches remain mandatory; expert/relaxation
violations need full dynamic-model replay to establish UNSAFE.

Final analysis inherits R1:30-input primary paired adaptive-minus-monolithic
SAFE indicators,10000 input-block bootstrap draws with seed20260915, descriptive
unadjusted2.5/97.5 percentiles.90 calls are not90 independent inputs. Incomplete
runs retain denominator30/arm and unattempted counts but no final cohort CI.
Report all states, complete UNKNOWN, internal TIMEOUT, outer TIMEOUT, ERROR,
positive/solved intersections and gains/losses, all-terminal time and paired
costs. Single/multiple/unavailable route strata are descriptive. CROWN POSITIVE
is a numerical filter, not formal SAFE or a full independent dynamic verifier;
its external overlap is explicitly cross-evidence-level, not strict domination.

The final terminal/package auditor replays dynamic UNSAFE, checks common facts,
E4 route coverage and per-arm policy, and independently runs Python -S V2 journal
checks. Automatic audit and a second interpreter re-audit must match. The
runner then writes `FULL_SUMMARY.json` locally, even for a retained failure;
it does not silently publish/commit results or queue a new experiment.
This is structural/numerical-policy conformance, not independent network-bound
reproof. Any follow-up diagnosis uses these saved records first; neither a
solver-limit stop nor a negative relaxation bound alone identifies a cause.

Launch once after tests, clean-only freeze review, commit and push:
`python -m scripts.run_conv_full_v2` in act-py312. Planned durable tmux session
`moe-conv-full-v2`, external log `data/moe/results/conv_full_v2_pipeline.log`.
No checkout edits while active. After execution inspect both audits and raw
identities, archive compact outcomes/docs, then commit/push separately.
