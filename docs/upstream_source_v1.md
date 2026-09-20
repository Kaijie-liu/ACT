# Local expert / guard source audit V1

Scope: follow the remaining upstream trust in the frozen convolutional input98
proof. This is a source-lowering audit, NOT another attempt to increase its
margin, change its verdict, or reopen a cohort/backend search.

The old checkpoint, input bytes, sole pair `[1,2]`, conditioned router HZ and
terminal joint expert HZ are pinned. There are no new solver calls, network
forwards, full-model propagations, output obligations or acceptance changes.
The old single-pair9-obligation proof remains conditional, not a high-accuracy
or cross-family route-changing strict certificate.

## Read-only inspection that motivates this check

The old stored HZ does not contain a per-layer source trace, per-ReLU range
certificates or the historical private-factor mapping. `_sparse_relu_bounds`
uses interval/generator bounds and optionally numerical support. A final
`exact=True` flag is not proof that this entire real-arithmetic conversion
was outward-rounded or independently checked.

Exploratory exact inspection, before this protocol, found:

- the four actual pair inequalities have positive box slack and occur
  unchanged in the first four joint-HZ inequality rows;
- reconstructing the current input-box formula gives29 inward coordinates,
  maximum1/72057594037927936. The historical input HZ was not saved, so this
  initial observation is about the current formula, not recovered history.

These observations are disclosed, not presented as untouched confirmation.
The bounded follow-up checks an actual freshly constructed input HZ and first
Conv2d states from both covered experts, using the same pinned source bytes.

## What is independently checked

`upstream_source/checker.py` uses only exact standard-library `Fraction`
arithmetic. `verify.py` runs after relocation with `python -I -S`, hashes every
file and binds expert parameter tensors to the full original model inventory.

1. For an unconstrained diagonal input HZ, check exact containment of the
   represented input bounds. Nonzero inward gaps remain failures of this
   containment test; `exact=True` cannot override them.
2. Read **actual** old pair guard coefficients and prove each is redundant
   over its whole `[-1,1]` factor box. Check its exact prefix in the old joint
   HZ, including RHS and no private continuous/binary contribution. Also
   record exact coefficient/RHS differences from the intended score subtraction;
   no numeric tolerance is used to claim equality. Redundancy does not require
   the two coefficient forms to be identical.
3. Independently assemble the declared CHW Conv2d linear operator from original
   parameter bytes (padding, stride, dilation, groups), not using the production
   matrix-construction function in the checker. Compare both actual fresh
   first-conv states with that operator and the captured input HZ.

For source `z=c+G_c xi+G_b beta` and stored affine result `z_hat`, at the same
factor assignment, the checker proves the rowwise bound

`|Wz+b-z_hat| <= |delta_c| + sum|delta_Gc| + sum|delta_Gb|`.

Zero proves this local affine step exact, conditional on the supplied source.
A nonzero error bound is **not** an accepted target enclosure: an explicit
compensation construction, or another checked containment argument, would be
needed. No compensation is inserted into the production/frozen graph here.
Even a nonzero same-factor error does not alone prove failure of set containment
under every possible alternative factor assignment.

## Limits and missing trace contract

The newly captured first-conv states are not the old hidden layer states.
The complete original expert propagation, nonlinear range proofs, fact reuse,
membership big-M guards and frame merging remain unchecked. Actual pair-row
redundancy must not be generalized to membership guards or all constraints.
No global upstream trust string is removed in this stage. The isolated checker
trusts its mathematical code; transport hashes are not a machine-checked
implementation proof. Declared graph/native-program correspondence remains
separate, as does enclosure of an intended exact epsilon ball.

A future complete trace needs: input containment; each layer's input/output
source identities; original operator parameters; independent sign/range
evidence for every ReLU; exact affine/constraint error compensation; shared
versus private factor maps; and the membership-bound sources supporting reuse.
Missing entries cannot be inferred from the final HZ or finite concrete probes.

## Execution, controls and costs

One new root: `data/moe/results/upstream_source_conv98_20260920_v1`.
One 300-second outer budget, build cap60s, isolated check cap180s, owned-child
termination at remaining deadline, partial records retained, no retries.
Independent postterminal review is separately timed and compares the entire
exact output, including per-row gaps. Source capture, local transfer,
serialization, isolated check and publication costs are all recorded; old
output propagation/proposal costs are excluded. Not a production timing study.

Eight new controls cover exact/inexact affine arithmetic, missing small box
radius, changed constraints, continuous/binary factor bindings, grouped/padded/
dilated convolution, ties and guard inventory, CSR rejection, checkpoint-bound
capture and moved isolated checking. Three hash-rebound semantic mutations
remove an expert, substitute an expert parameter name, or change the input
frame. Two unchanged lifecycle regressions cover cutoff, exception and partial
evidence. Initial control run failed on rejecting an internal `Fraction`;
admission was corrected before any real capture, without float tolerances.

After preparation commit/push: `python -m upstream_source.run`, then
`python -m upstream_source.review <new root>` in act-py312. Archive either
outcome. This stage does not schedule another arithmetic or full-model search.
