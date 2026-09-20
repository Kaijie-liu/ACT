# Bounded saved-evidence diagnosis after property-directed selection

Scope fixed before this analysis: only the two reviewed packages from
`property_ranges_conv98_20260920_v1`, input98, all eight common checked
properties1,2,3,4,5,6,8,9. Property7 remains in the nine-obligation endpoint
ledger, but neither its saved point nor a new solve is used for paired
diagnosis. The frozen failed/missing result is unchanged.

No numerical optimizer, new bound/range, source propagation, model forward,
checkpoint loading, parameter search or acceptance change is allowed. Use
stdlib/exact rational arithmetic on the stored graph coefficients, appended
factor mappings, LP points and checked-bound records only. Pin the archived
review and both package manifests; verify every referenced file hash before
analysis. Reject changed identities, missing properties and aliased factors.

## Exact accounting, not proof or causal attribution

Each arm uses its OWN source and factor frame. For its unverified saved point:

* `J = u + w` is the exact evaluation of that arm's stored LP objective.
* `P = lambda*d - w` is the free-gate product discrepancy.
* `A` is the weighted discrepancy between final LP output coordinates and
  the exact affine classifier applied to stored last-ReLU outputs.
* `R` sums signed contributions
  `weight_i * (W_i[y,j]-W_i[k,j]) * (h_ij-max(a_ij,0))`.
* `T = J + P - A - R` replaces both the free product and last ReLU at the
  same stored factors. Preserve all last-ReLU rows, including stable rows.

Also report `J+P` (product-only replacement), `J-A-R` (last-ReLU and final
affine consistency replacement), gate/difference ranges, and local sign,
range and triangle discrepancies. A positive `T` is not a network output,
a feasible repaired point or a new lower bound. Earlier source relations and
the equality of the free gate to the actual softmax remain unchecked here.

For the already checked lower bound `L`, retain its recorded dual constant
`D` and residual-box correction `C`, and verify `L=D+C`. This correction
includes the legitimate finite-box minimization of a stationarity residual;
it is NOT entirely numerical rounding loss. Record residual L1 and nonzero
coordinates without relabelling them as a proof of solver inaccuracy.

Let `e=J-L`. This is an **unverified point-minus-bound difference**, not an
optimality gap. Across the two independently generated points, check exactly:

```
Delta L = Delta T - Delta P + Delta A + Delta R - Delta e
```

Partition `R` into the four prefix-selected rows, the four property-selected
rows, and all other rows. These are signed terms at different LP points; their
paired changes are descriptive accounting, not interventions holding every
other variable fixed. Never transfer a point or old dual to the other matrix.

## Continue/stop decision

Deliver a finite table of all eight pairs, residual and local-discrepancy
accounting, and the complete nine-property exclusion ledger. Do not add a
new row-ranking search or propose querying more hidden rows.

Continue toward a separate control ONLY if the records identify a concrete,
testable representation change and a scope that does not merely repeat the
already known product/ReLU coupling. State what evidence supports that choice
and what remains unseparated. If this analysis still supplies only coupled,
unverified-point symptoms with no well-isolated next intervention, STOP the
input98 follow-up rather than expanding samples, queries, time or tunings.
Even a proposed control is not authorization to run it in this stage.

The new analysis and its independent arithmetic/identity controls must call
no solver. A fresh saved-record recomputation may check reproducibility, not
change outcomes. Old proof packages, protocols, production gates and main
tables remain frozen. Input98 is single-route and67.06% clean accuracy; this
diagnostic cannot fulfill high-accuracy/cross-family route-changing goals.
