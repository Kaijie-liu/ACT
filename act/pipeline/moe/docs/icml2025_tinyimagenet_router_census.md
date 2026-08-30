# Official TinyImageNet router census

## Scope

This census measures only the hard-router applicability geometry of the
released TinyImageNet pipeline at commit
`30ef94d77b5451595b82e739aa8938e1f4c4521f`. It performs no training, uses no
validation labels, and makes no output-robustness or expert-certification
claim.

The primary model is the released `MOE_ViT`: four pretrained
`vit_small_patch16_224` experts are constructed before the router, and the
router is `Flatten -> Linear(150528,4) -> argmax`. For each preregistered seed
0--19, the runner constructs the complete official model before extracting the
router. This preserves the official random-number consumption order.

## Data and domains

The source is the Stanford Tiny ImageNet archive. Its 10,000 validation JPEGs
are sorted by filename; labels are never loaded. The archive size, SHA-256,
validation-annotation SHA-256, and ordered filename digest are recorded.

The five registered radii are `0.5/255`, `1/255`, `2/255`, `4/255`, and
`8/255`. Two input domains are reported separately:

1. `official_post_resize_224` is the primary 150,528-dimensional unit-pixel
   space after the released 224x224 bilinear resize.
2. `official_composed_raw_64` is a secondary analysis of the *same router* in
   the original 12,288-dimensional unit-pixel space. The deterministic
   bilinear resize is folded into the router by its exact real-arithmetic
   adjoint. It is not a separately initialized model.

The released transform converts decoded pixels to float16 before resizing.
The formal analysis uses the real-arithmetic bilinear operator and separately
replays the literal float16 preprocessing. Consequently, the secondary domain
is labelled `real-arithmetic preprocessing composition`; it is never described
as a bitwise model rewrite. Route mismatches and maximum score drift against
the literal transform are retained as audit fields.

## Exact fixed-radius decision

For a clean route `i`, competitor `j`, point `x`, and clipped L-infinity radius
`epsilon`, the maximum reachable score reduction is computed exactly as

```text
sum_d |W_i,d - W_j,d| * min(capacity_d(x), epsilon),
```

where capacity selects distance to the lower or upper input face according to
the coefficient sign. A competitor is reachable when its minimum guarded gap
is non-positive. The implementation evaluates all five radii together and
uses explicit outward gap brackets, producing disjoint `stable`, `reachable`,
and `undecided` states. Ties are never classified stable under
`ANY_LEGAL_TOPK` semantics.

This fixed-radius method avoids constructing a `10000 x 150528` capacity-sort
tensor. Images are decoded and evaluated in chunks, while the K=20 router
weights remain resident on the selected device.

## Reproducibility and failure policy

- All repositories, environments, caches, data, and outputs remain under
  `/data1/Kane/MOE`.
- The official clone must be clean and at the pinned commit; both relevant
  source files are hash checked.
- Outputs are create-only. A failed directory is retained and never silently
  overwritten.
- The independent audit reparses the raw NPZ, closes every partition identity,
  checks all hashes, and recomputes a deterministic sample against the affine
  support formula.
- CIFAR-10 and TinyImageNet are reported as separate datasets and architecture
  families. Only the official 224 domain enters the main cross-dataset figure;
  the composed raw-64 analysis is secondary.

The frozen configuration is
`act/pipeline/moe/configs/icml2025_tinyimagenet_router_census.json`.
