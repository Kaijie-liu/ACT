# Current-problem feasible factors and repeated-query diagnosis

## Scope and stop condition

User requests solver-free current-request feasible assignments and continued
diagnosis, not more solver time or relaxation tuning. This stage develops a
pure proposal/checker interface and checks compatibility on the **already
stored** protected-R1 expert matrix. No checkpoint load,forward propagation,
new native query,full-request rerun,new sample or historical result relabeling.
The old base UNKNOWN and16 output UNKNOWNs stay unchanged. No default or frozen
production source is modified, and no result is imported into a live request.

One deterministic free-factor seed (all zeros); no search or restart. Proposal
plus full feasibility checking share3s, the prior base allocation. Matrix load,
source audit,analysis and diagnostic serialization have separate recorded costs;
this is NOT an end-to-end production speedup experiment. Native production
integration,outer supervisor and full cost accounting must be frozen separately
before claiming saved time/certificates in a real request. Stop after the one
saved-model control regardless of success; no alternate seed to rescue failure.

## Why construction can avoid a base MILP

In the registered compressed ReLU graph each unstable neuron appends two
continuous factors and one binary. After the existing lowering to0/1 binaries,
the new coefficients in equality j are `a<0,b<0,2a`. Earlier factors have been
assigned. Write R for the equality RHS minus earlier-column contributions;
then `t=R-a-b` is the represented preactivation. The proposal sets:

- t>=0: new factors `(1,(R-a)/b,0)`;
- t<0: new factors `((R-b-2a)/a,1,1)`.

These satisfy the local equality algebraically in real arithmetic and encode
the appropriate ReLU branch. Stored floating arithmetic may have residuals;
there is NO clipping,changed tolerance or reliance on this derivation as the
acceptance test. All constraints,variable bounds and integrality are checked
after construction using the original feasibility policy. In particular a
zero free-factor seed need NOT satisfy a route guard; that case is rejected.
The proposal recognizer refuses unsupported layouts or forward dependencies.
Failure means no proposal, not infeasibility.

The checker independently hashes all current constraints,bounds,integrality
and output mapping, compares the request/input/evaluation identities and checks
the complete assignment. It never calls the construction routine,never trusts
its claimed row coverage,and never returns SAFE/UNSAFE/infeasible. It returns
only a policy-checked BASE feasible point or no accepted point. Request nonce
and full model identity prevent historical/cross-expert reuse. This is still
floating feasibility evidence conditional on lowering,not an exact rational
certificate or deployed-network input witness.

## Remaining-obligation diagnosis

Re-read every saved expanded query and bind it to the sealed model/config.
Group only EXACT stored sparse matrix/threshold identities,checking bytes as
well as hashes. Preserve all19 property identities and their old terminal
states. A group is not a solved obligation: UNKNOWN is never promoted to proof.
Report per-row setup-to-entry windows,completed native time if present and
deadline-censored records separately. No optimality/relaxation conclusion may
come from an interrupted call. Duplicate handling is a separate future schedule
change,not silently combined with the base-assignment control.

## Controls and execution

20 constructor/checker tests pass in ACT and pinned intake environments:
active/inactive/tie,multiple dependent ReLUs,affine/constant cases,guard refusal,
native tiny-base differential,request/input/nonce binding,changed matrix/output
map,missing variables,binary/equality corruption,nonfinite input,deadlines and
base-vs-property distinction.6 exact-query identity controls reject one-ULP or
threshold changes,missing/reordered rows and invalid arrays.

First control invocation exposed two fixture mistakes: assigning a field of
the frozen `_HZMILP` dataclass and mutating an earlier affine coefficient rather
than the intended canonical new-binary coefficient. Corrected fixtures use
`dataclasses.replace` and the exact new-binary index; no acceptance rule changed.

Commit/push tested implementation first, then run once:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python scripts/diagnose_metamoe_current_assignment.py
```

New output root is `/data1/Kane/MOE/baseline_runs/metamoe_current_assignment_20260923_r1`.
Source protected-R1 config and archive are hash-bound; native calls are disabled
inside the diagnostic. Separately recheck the saved proposal and full matrix,
archive identities/costs,and document whether production integration is justified.
Never replace the previous UNKNOWN with a new offline base-feasibility result.
