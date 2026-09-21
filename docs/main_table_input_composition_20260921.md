# Composed input containment: append-only review addendum

This new saved-only analysis supplements, and does not overwrite, the
[parent audit](main_table_source_applicability_20260921.md).
`main_table_input_composition_20260921.json` binds the parent SHA-256 and all
100 representative request tensor files. It rechecks their tensor identities
and exactly interprets the frozen binary64 midpoint/radius formula.
Torch only decodes tensors; no checkpoint, dataset, model, ACT propagation,
guard lowering or solver is loaded/executed. The parent audited 739 package
copies; this addendum rereads 100 representatives, not all 739 copies.

Let X be the clipped requested box around the saved binary center and H the
exact denotation of the reconstructed binary64 input formula. Neither denotes
recovered historical intermediate HZ traces. Two failed component edges do
not imply a failed composed edge; this analysis tests X subset H directly.
It also tests H subset X, rather than assuming inward gaps imply strict subset.

| Radius interpreted as a real number | Inputs with X not contained in H | Inward coordinates | Maximum inward gap | Inputs with H not contained in X | Outward coordinates |
|---|---:|---:|---|---:|---:|
| Rational 2/255 |100/100|257,850|1/35888059530608640|100/100|50,512|
| Saved binary64 epsilon |100/100|197,704|1/36028797018963968|100/100|50,512|

Maximum outward excess is respectively 127/4593671619917905920 and
1/36028797018963968. All 23 gain model–input pairs (21 distinct inputs) fail
requested containment. The two inputs without inward BOX→H reconstruction
error in the old audit also fail this direct requested→H test.
Each input has 3,072 coordinates, 307,200 total. Inward and outward coordinate
sets can overlap: they are not required to partition that total.

**Conclusion:** the requested and reconstructed boxes are not nested. This is
an exact source-containment obstruction, not a counterexample to output
safety, not a new lower bound and not a proof that another legal route exists.
Recorded positive output margins do not establish coverage of omitted regions
or their potential routing obligations. Historical 179/156/141 HZ-policy
acceptance counts, gains, losses and statistical intervals are untouched.

The parent schema's `coordinate_count_min/max` means *inward coordinate count
per input*, not tensor dimension. Its original keys are preserved; this new
schema names inward/outward counts explicitly. No positive radius was dropped
in the parent cohort; separate pruning risks are not attributed to it.

## Reproduce without solving

Using the existing ACT environment and access to saved request tensors:

```sh
python scripts/audit_moe_input_composition.py --check
python -I -S scripts/test_moe_input_composition.py
```

Without `--check`, generation is exclusive and refuses an existing output.
The tests cover equal/degenerate intervals, nonnested boxes, failed-component
edges with a successful composition, radius semantics, positive-radius
pruning versus genuine zero, invalid tensors/identities, and JSON roundtrip.
This is author-side arithmetic reproduction, not a third-party network proof.
