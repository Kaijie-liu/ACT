# Expert / guard upstream audit: completed, strict chain remains open

Execution `0f8af0fa04ad5ad0ad992571fc46d1f029ba6aed`, one frozen local audit,
no retries. Independent review: **PASS,0 issues**. Read
[the compact exact results](upstream_source_v1_review.json) and
[scope/protocol](upstream_source_v1.md). Raw:
`data/moe/results/upstream_source_conv98_20260920_v1`.

This audit found a concrete reason **not** to remove the input/expert-lowering
assumption. It did not attempt another complete output proof or change any old
SAFE/UNKNOWN/UNSAFE result. In particular, it is not a high-accuracy or
cross-family route-changing certificate.

## Checked results

| Local obligation | Exact check | Interpretation |
|---|---|---|
| Newly reconstructed input HZ vs pinned3072-coordinate represented box | 29 inward coordinates; maximum `1/72057594037927936` ≈1.38778e−17 | This actual current input conversion does not contain the full represented box |
| Four saved pair[1,2] guard rows | All redundant on `[-1,1]`; minimum slack ≈0.034151814 | These actual rows cannot remove any factor assignment |
| Guard rows in old joint expert HZ | Exact prefix identity, including RHS and zero private-column contributions | Proven preservation of these four rows, not proof of the rest of the joint HZ |
| Expert1 first Conv2d, all4096 output rows | Nonzero same-factor error; maximum bound ≈6.79312e−16 | No exact-step acceptance or error compensation |
| Expert2 first Conv2d, all4096 output rows | Nonzero same-factor error; maximum bound ≈1.00281e−15 | No exact-step acceptance or error compensation |

Each Conv transfer differs in110,128 stored center/generator coefficients from
the independent rational construction. Per-row residual vectors and all29
input gaps are preserved in the full logs, not replaced by rounded summaries.
Both locally reconstructed output HZs carry the production `exact=True` flag;
the checker does not use that flag as real-arithmetic authority.

The exact first-layer maximum error bounds are respectively
`1642475027/2417851639229258349412352` and
`303082151/302231454903657293676544`. They bound the discrepancy at the **same
factor assignment** for the supplied input HZ. They are not estimates of the
complete network error; nonzero discrepancy alone also does not exclude every
alternative set-containment argument. In contrast, the diagonal input-box
containment failures are exact endpoint failures for the reconstructed state.

The guard coefficient differences from exact score subtraction are also
nonzero, and one RHS differs by1/9007199254740992. No equality was accepted
within a tolerance. Instead, exact positive redundancy slacks independently
justify that these particular stored rows are harmless on the entire factor
box. This reasoning does **not** extend to arbitrary guards near a route tie.

## What remains unproved

The old proof saved final joint and router HZs but no full historical layer
trace. The new input and Conv states are newly constructed local controls from
the same model/input bytes, not recovered historical intermediates.1,243 joint
equalities and2,486 additional inequalities are not independently derived in
this audit. No per-ReLU source/range/encoding chain, shared/private factor map
or membership-interval source for the reused output property has been checked.

Accordingly, no global upstream assumption is removed. The previous9-obligation
conditional output proof and source-router extension remain immutable. Those
9 LP obligations were not re-solved or rechecked in this local stage; this
stage independently checks its own source conditions, not a new complete MoE
certificate. Model safety/unsafety and the historical HZ-policy table are not
changed by these local findings.

Small local errors cannot simply be subtracted from the old0.1772745 final
margin: an error-propagation argument through all remaining layers, activations,
guards and property construction would be required.

## Portability, rejection and cost

Manifest identity:
`bdf39f969664a5e1b2b49eb16ad7da1872bc36aa6fc1351a601c14510079c6d9`.

Copy the new `relocated/` directory, then run:

```sh
python -I -S /new/location/verify_upstream.py --manifest-hash bdf39f969664a5e1b2b49eb16ad7da1872bc36aa6fc1351a601c14510079c6d9
```

No checkpoint, data, history, ACT imports or solver is needed during checking.
The second copy reproduced the **full exact** results. Missing expert, wrong
expert-parameter binding and altered input frame were rejected even after
transport hashes were recomputed. Eight new tests plus two lifecycle and five
portable regressions passed; main-table reconstruction remained unchanged.

| Cost | Seconds |
|---|---:|
| Source capture and two local Conv transfers, nested in build | 1.4623 |
| Copy/serialization, nested in build | 0.0736 |
| Complete build child, including imports | 2.0483 |
| Complete isolated checker child | 1.6669 |
| Total through terminal publication | **3.7283** |
| Separate relocated checker internal time | 1.5478 |
| Separate review including three mutations | 3.3472 |

The local package occupies15,967,372bytes. The300s budget was respected; no
native solver, network forward or complete-model propagation was called.
Nested costs are not added twice. These are source-audit costs, not verifier
speedup or fresh complete-proof generation costs.

## Next justified source work

The blocker is now explicit: a checked source chain needs an input enclosure
that contains the represented box, and local affine error compensation that
survives composition. Develop those as separate optional source-proof controls,
then require independently justified ReLU ranges/constraints and frame maps.
Do not edit frozen coefficients, silently enlarge the old HZ, reuse old LP
certificates against changed matrices, or waive these gaps using a fixed epsilon.
No further precision search, new sample, native LP or main-table rerun is
scheduled by this audit. The next full claim remains contingent on **all**
upstream and output obligations, not two checked first layers.
