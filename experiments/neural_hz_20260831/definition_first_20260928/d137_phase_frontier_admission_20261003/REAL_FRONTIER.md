# Complete residual consumers and static source capacity

This is a read-only audit of saved graph evidence, not a new model run. It refines the existing [consumer boundary](../d125_signed_phase_component_20261002/NEXT_REAL_STRUCTURE.md). A valid Neural-HZ definition must support the real shared consumers; it cannot obtain a reduction by omitting the residual port or the rest of a convolution's spatial outputs.

## Evidence and model identity

The D120 saved [large](../../results/d120_mixed_consumer_source_20261002_v1/complete_0.json), [medium](../../results/d120_mixed_consumer_source_20261002_v1/complete_1.json) and [Tiny](../../results/d120_mixed_consumer_source_20261002_v1/complete_2.json) records give first_shape, original_source_relu, branches and side_consumers. The [D025 large graph](../../results/d025_interval_capacity_20260930_v1/complete_0.json) is under packet.graph.nodes. The [D015 medium graph](../../results/d015_source_shielding_20260928_v2/partial_source_evidence.json) is under element 1, packet.graph.nodes. These JSON files are single-line artifacts; field paths, not invented line ranges, identify the evidence.

Saved model SHA256 values match across those records:

```text
CIFAR large  5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16
CIFAR medium aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4
Tiny medium  234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776
```

Artifact hashes are in ANCHOR_SHA256SUMS. The historical D025 and D015 source attempts did not qualify overall; authenticated graph facts do not transfer source qualification to this checkpoint. No new original model bytes were loaded here.

## Surviving consumer topology

For CIFAR large, Q=127 is the output of Relu_2 with shape [1,64,32,32]. One branch reaches Relu_5 through Conv_3 and BN_4. Q also enters Add_8 as an identity skip, producing P=133. P feeds Conv_9, BN_10, Relu_11 (output 136), and also remains live as the skip into Add_14. The immediate joint boundary is therefore at least (136,133), not just 136.

For CIFAR medium, Q=121 has shape [1,64,15,15]. Its shortcut passes through the 1x1 stride-2 Conv_8 and BN_9, then joins at Add_10, producing P=129. P feeds Conv_11, BN_12, Relu_13 (output 132), and also enters Add_16. Its immediate boundary is at least (132,129).

For Tiny, the saved first source shape is [1,64,27,27], and the first main/shortcut outputs have shape [1,128,14,14]. The first shortcut is recorded, but this audit does not have authenticated complete Tiny downstream graph evidence. The CIFAR tail must not be silently substituted for it.

Following every graph path by which raw Q can remain affine gives these longer surviving Add chains:

```text
CIFAR large:
133 -> 139 -> 147 -> 153 -> 161 -> 167 -> 175 -> 181
    -> Flatten_57 -> Gemm_58 -> Relu_59
First nonlinear cut along all such paths:
Relu indices 5, 11, 17, 25, 31, 39, 45, 53, 59.

CIFAR medium:
129 -> 135 -> 141 -> 147 -> 155 -> 161 -> 167 -> 173
    -> Flatten_55 -> Gemm_56 -> Relu_57
First nonlinear cut along all such paths:
Relu indices 5, 13, 19, 25, 31, 39, 45, 51, 57.
```

This is graph-level potential dependence, not a claim that every downstream scalar coefficient is nonzero. The saved evidence does not supply all numerical composed kernels and shapes needed to price the complete frontier. A sound local rule may retain the unmatched ports; it cannot claim they were eliminated.

## Static capacity of the first complete Conv

These are exact shape-derived counts, not measured nnz, runtime or physical memory. First-Conv kernels are [64,64,3,3] for large and [128,64,3,3] for medium/Tiny, with group=1, dilation=1, pad=1. Strides are 1, 2, 2 respectively.

| Quantity | CIFAR large | CIFAR medium | Tiny medium |
| --- | --- | --- | --- |
| Parent coordinates | 65,536 | 14,400 | 46,656 |
| Conv output coordinates | 65,536 | 8,192 | 25,088 |
| Stored kernel coefficients | 36,864 | 73,728 | 73,728 |
| Canonical stencil slots | 37,748,736 | 4,718,592 | 14,450,688 |
| Valid non-padding slots | 36,192,256 | 3,964,928 | 13,107,200 |
| Dense output-by-parent entries | 4,294,967,296 | 117,964,800 | 1,170,505,728 |
| Bytes for that hypothetical dense FP64 matrix | 34,359,738,368 | 943,718,400 | 9,364,045,824 |
| Separate coordinate lower/upper directions | 131,072 | 16,384 | 50,176 |

Canonical slots are outputs times input channels times nine. Valid one-dimensional stencil totals are 94, 22 and 40; their squares times input/output channels give the valid-slot row. Dense entries are output coordinates times parent coordinates; bytes multiply that by eight. Direction counts are twice the output coordinates, not a lower bound for all possible joint algorithms.

The large identity shortcut has 65,536 scalar reads. Medium/Tiny 1x1 shortcuts have 8,192 kernel coefficients and 524,288/1,605,632 stencil slots. These are additional consumers, not substitutes for the main Conv cost.

D120 executed only five spatial windows per source, or 320/640/640 rows. Its archived results were zero improved ReLU bounds for all three; Tiny had one improved upper preactivation bound. Those historical local outcomes are not a new gain and are not full spatial qualification.

## Meaning for a tensor fiber definition

A possible operator-level expression is 0<=e_b<=c_b+sum_{a<b} L_ba(e_a), with nonnegative linear bound operators, and reverse support recurrence K_b=positive_part(w_b+sum_{c>b} L_cb^* K_c). Storing kernels and exact layout/stride metadata can avoid the hypothetical dense matrix. That observation alone is implementation support, not a new nonconvex calculus or a demonstrated total-cost advantage.

Positive parts do not commute with composition: (AB)_+ generally differs from A_+ B_+. The latter may be a valid majorant but requires its own precision accounting; it is not automatically an exact compact encoding of the former. All predicates, original bits, shared queries, source constraints, terminal conversions, evidence, decoder, and host/device coexistence remain charged.

The next research hypothesis must address the source-phase-amplitude relation over these consumers, not just port the old convex recurrence to GPU. No new source/GPU execution is admitted by this document. The old initialization failure and unchanged resource gates remain recorded in the D136 resume checkpoint.
