# C106 exact original defining-row lowering argument

Scope is the original graph's op/sum definitions before the unchanged C104
encoder. Source definitions and original EQ/INEQ/binary operands are unchanged.
No lookup of an input identity, result, LP state or witness occurs in construction.

For CSR, the same original indptr extent is traversed; exactly those original
indices with needed[index] and data!=0 survive, in unchanged index order. Float32
data widens exactly to binary64, as in the original box/encoder conversion. For
Conv, the same b/output-channel/output-spatial/group and stride/padding/dilation
equations select input coordinates. Original output masks, parent masks and
zero kernel entries are respected. With positive dilation and finite spatial
bounds, channel-major then kernel-row/column-major visitation is strictly
increasing global NCHW input order. This is exactly the original argsort order;
no repeated index or differently summed coefficient is introduced.

For sum, the original sorted Counter(parents) is constructed once per node.
Each actual parent support[row] selects its original global slot, exponent and
positive integer multiplicity. The count is checked <=2^53 before binary64
conversion, so no rounding occurs. Only descriptors are retained for the node;
arrays are read live under the GIL, without a cached finite/power/shift assertion.

For all supported rows, let e_i=frexp(abs(a_i)).exponent+p_i. First pass counts
retained terms and M=max(e_i). Second pass accumulates exactly
T=sum_i(1<<max(e_i-(M-26),0)). Each shift is0..26, and the bounded row population
keeps T below2^52; hence signed64 accumulation equals C8's exact integer sum.
The same bit_length(T-1) and clamp give unit=max(0,M-26+bit_length(T-1)), with
the same finite1023upper gate. No floating reduction substitutes for this formula.

The outputs are exactly original parent slots followed by the new slot,
negative original coefficients followed by+1, and original powers followed by
unit. Negation is exact; no fused floating multiply/add is used. The original
C104 encode_uid still validates canonical coordinates, finite coefficients,
original bounded powers, feasible native window and every inverse ldexp;
unchanged radix behavior handles an unfit direct window. No private prepared
payload is issued by this new constructor, and no check is skipped in its
consumer. Binary coefficients and original source constant use the old code.

Consequently every accepted row entering the same encoder, owner ledger, exact
quotient and circuit stream is identical. The complete source report uses the
same conservative charges, not an extra budget or a timing-derived discount.
The old mathematical/source identity can be reused ONLY when the complete actual
fresh state fingerprint matches it. Unit tests and an ordinary fixture are not
proof of the real input/native/LIVE/terminal/score gates; those remain mandatory.

The implementation holds input Python references and the GIL, checks actual
ndarray type/shape/index extents before reading, uses memcpy for strided scalar
loads, and owns three NumPy output buffers. There is no buffer-protocol exporter,
heap-side numeric cache, manually freed NumPy metadata, or long-lived raw pointer.
Allocation failures unwind partial outputs. Both measured memory boundaries
must still pass; this argument is not a resource measurement or model verdict.
