# C93: packed source footprints and exact histogram cardinalities

Previous C91/C92 turn was PROGRESS. Full goal ACTIVE; formal1870/all13 and
separate E0 CIFAR25/Tiny36 unchanged. C92 exact additive post-fold implementation
remains CLOSED at282186397>256M. No unchanged target/test/census is restarted.

Implement the SAME C92 sufficient dense-circuit upper bound using packed
boolean source footprints and observed-mask histograms, not full coordinates
and exponents for all163 tiles. Original source masks/slot/local-root/row-length
maps remain authoritative. Do not feed old selected tile IDs, cost tables,
family, model, iid, label, margin or verdict to the rule.

Ordinary bounded row geometry: input/output spatial widths and heights <=64,
same stride1/dilation1/group1/3x3 operator as C92. Unsupported geometry does not
acquire a new representation. Pack each actual boolean image row into uint64
with explicit little-bit/byte order, then extract every padded4x4 input and2x2
output footprint. Every actual mask bit is read; padding contributes zero.
Only masks that actually occur are counted; no65536-entry or model-specific
precomputed table. Original source scalar aliases/offsets/binaries retain the
same C92 exclusion, via exact spatial OR and direct-nnz sums over ALL channels.

For transform position t, let A be the number of nonempty input forms, B the
number with degree>1, and S their total degree. From output-mask histograms
let U be total uses, V the number of output channels with>1 use and W their
total uses. Dense-transform M is retained iff A>1 and uses>1. V-form consumption
is the number of active output channels when A>1, otherwise U. When that total
exceeds1, kept V count=B, V nnz=S-A+2B and M term width=A; otherwise width=S.
M nnz is V*(width+1) when A>1. Output terms are width*U+(1-width)*W when A>1,
otherwise width*U, plus original output pivots. These integer formulas must
equal C92's complete matrix-degree upper bill exactly, not approximately.
Numerical admission/dense/window/box/source/inverse obligations are unchanged.

Freeze logical prices: one row packing costs3*boolean_pixels+16*image_rows;
complete tile extraction costs128+24*C+16*K per tile. Histogram/cardinality
costs512+12*(C+K)+64*(observed_input_classes+observed_output_classes) per tile.
Source discovery costs64 per graph node and one read per eligible output-mask
cell. Source scalar/direct routing pays128+4*parent_width+6*output_width+
16*active_parent+32*active_output per eligible nonempty operator, covering the
complete mask index scans, slot/root gathers, RHS/binary/row-length checks,
allocation and spatial reductions. Same C92 canonical order and same C88
coordinate gathering only for selected tiles. Complete source proof/input
fingerprints/native comparison/export retain their existing diagnostic prices.
The unchanged C85 transform/C88 zero classification/numerical construction
remain paid; do not mix a second new numerical optimization into this version.

Tests independently expand packed ordinary zero/partial/full/padded/odd masks
and compare every integer bill field with C92. Full inherited2144 plus new
tests must collect/execute within60s before the actual source diagnostic.
Retain COMPLETE C9/C69/C89/C90/native dependencies plus complete C92 result JSON
as an AFTER-THE-FACT oracle, never as planner input. Compare all163 decisions,
every conditional bill, the whole canonical plan and every selected native
literal to the unchanged C92/C90 proof. Transform only selected operators and
construct only selected numerical packets. No reduced-input claim about older
whole numerical-custody or LIVE gates.

Report new measured planner/emission work and a necessary additive/fused bill.
No new generation credit without actual removed source operations. If even
optimistic direct-row removal cannot pay the full necessary bound, close that
exact realization before original-network execution and derive the next
same-structure change from actual counts, not a changed tariff or larger cap.
CPU1/GPU0,AS16GiB,256Mwhole/200Mbranch,64Mentries,both1GiB,240s worker and collective
16384/131072/16M stay. No source-HZ/native/LIVE/solver/witness/default/score
promotion is implied. No production/history/commit/push mutations. All new
outputs exclusive under results/c93_packed_mask_20260913_v1.
