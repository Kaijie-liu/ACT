# Final TMLR closeout, 1 November 2026

The PI authorized this finite closeout after the S2a STOP. No S2b, tuning,
new mechanism, or expanded scientific queue is authorized by this stage.
Starting branch `feat/moe-route-verification`, clean HEAD
`c3217bd4cfc336f231c09e6076ccedf280b3743b`.

## Fixed empirical reproduction

Two selected models, D-act seed 0 and M seed 0; test indices 0–49; 2/255.
IBP is evaluated once in one 50-input batch per model (the original batch was
256); complete verification is evaluated once per model-input. Thus 102 worker
calls cover 200 model-input-mode measurements, within the authorized 200-call
ceiling. The ten-call smoke is the queue prefix, not ten additional calls.
Unknown/timeouts are valid terminal outcomes, not reasons to retune or repeat.

Fresh Python environments and wheel installations reside under
`/data1/Kane/MOE/reproduction/s2a_20261101_r1`; original environments remain
untouched. Primary Python 3.12.12 / torch 2.9.1+cu128 and engine Python 3.11.16 /
torch 2.11.0+cpu reproduce the original installed versions. Execution is CPU-only;
the primary CUDA-enabled wheel matches the original build but is not GPU use.
This is a same-host environment reconstruction by the same AI, not independent
human replication, a new OS image, or a container-isolation claim.

The immutable execution binds unchanged numerical files from original
`8087806b695ed55cb5a8b4b300181648ba91ebdb`. Only relocation, the primary subset,
and a finite aggregate queue clock are added. Original weights, represented
input domains, output obligations, acceptance policy, and engine configuration
are unchanged. Per-query 300 seconds includes loading through acceptance and
cleanup; four single-thread workers, 8 GiB per-process AS, CPU/memory admission.
Combined smoke and suffix execution has a 4-hour wall budget, with summed
request time separately reported. Setup/install and offline comparison costs
are separate. Shared evidence remains 128 MiB including the copied input
archive; environment/source/weight installation bytes are separate.

Compare every complete terminal exactly and every IBP flag/array element;
report differing indices and maximum absolute numerical differences, without
a post-hoc tolerance that changes a result. Original records remain immutable.
Timeout differences are reported even if no positive/negative contradiction
occurs. No other objects are rerun. A tool error may be repaired within the
user's authorization while preserving its failed attempt and accounting.

## Manuscript and review scope

The supplied `survey/drafts/act_final_review_20261101.md` is internal Claude
AI review, explicitly non-independent. M1–M7 guide a TMLR negative-result
working manuscript; human sign-off and submission remain PI responsibilities.
The requested causal interpretations must be tested against saved records:
certified accuracy equals clean accuracy times conditional certification, so
both factors contribute to the M–D-act difference. Learned-scale interval
widths are not causal abstraction comparisons. Top-2 clean routing alone does
not imply that each expert's robust-training gradients use one quarter of the
data. No first-in-literature claim or universal negative is asserted.

The artifact discloses all current external compatibility edits: five original
GenBaB edits, a root-input branching-score edit, and three LiRPA edits (nine
edits across eight files). Calling the current engine a five-edit snapshot
would omit provenance. Locks, immutable source identities, weight hashes,
input indices, smoke-prefix invocation and table rebuild are included.

Completion means this finite reproduction, manuscript and artifact are done;
it does not reverse S1/S2a STOP or close unachieved scientific guarantees.
After the final report the finite goal is completed and automatic work stops.
