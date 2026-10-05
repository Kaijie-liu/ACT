# D015 — First real-source partial-shielding certificate pilot

2026-09-28; `redu-hz`, HEAD `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Default-off isolated diagnostic, not a verifier, production integration or
claimed new abstract domain. Builds on D014's proved partial-negative rule.
Formal1870/2413 and separateE061/400 remain unchanged irrespective of this
pilot's outcome. Full same-path2413/400 replay gates are not replaced.

## Question and population

Can faithful original first-bank Conv/ReLU sources establish the sufficient
D014 shielding condition for candidate unstable next-ReLU consumers in a
fixed, ordinary three-model pilot? A zero result is useful evidence about
THIS certificate/population, not a proof that the mathematical rule never
applies. Hits are source certificates, not solved properties or proof of
end-to-end savings. Neither success nor failure establishes research novelty.

The two frozen universe manifests contain THREE distinct model files: CIFAR100
medium, CIFAR100 large and TinyImageNet medium. Include all three. For each,
choose its lexicographically first ORIGINAL spec path, independently of public
labels, history, margins or any trial result. For every direct first-ReLU
Conv->optional affine/BN->ReLU branch, inspect ALL output channels at the four
output corners and integer center, deduplicating coincident positions. This
is a first source pilot, NOT all400 inputs, every spatial position or deeper
layer prevalence. The full user goal is not narrowed to this population.

All immediate residual/projection consumers are recorded in the source packet
and remain unchanged. A Conv that stops at a merge with a different nonlinear
branch is not mistaken for a direct next-ReLU path. Unknown local grammar
rejects the entire source attempt; it is not silently dropped from coverage.
No old deep HZ construction or C131 operator admission is needed for this
different first-bank source question.

## Source and arithmetic semantics

Parse authenticated original ONNX bytes by tensor producer/consumer identity,
without torch, onnxsim, graph simplification, opset conversion or shape
inference. The scoped grammar is static batch1 RGB NCHW, optional channel
affine/Identity input preprocessing, Conv, optional channel affine/BN/Identity,
and ReLU. Original constants may use embedded FLOAT/DOUBLE raw/repeated fields
or Constant/Identity aliases. External or overridable tensors and unsupported
local operators reject. Needed source scalars are exact rational values of
their stored floating payloads, not decimal approximations.

Parse the ORIGINAL VNNLIB decimal/rational endpoint tokens exactly. Require
every flat X_i declaration and both finite ordered endpoints, with no duplicate
endpoint, non-box input assertion or missing-bound substitution. Original
output clauses are retained as raw bytes but never queried. Flat indexing is
explicit C-order NCHW. Inputs need not lie in [0,1]; these original properties
may already have normalized coordinates.

The mathematical target is the real-valued affine/ReLU network with those
stored constants. BN uses the original inference expression
gamma*(x-mean)/sqrt(variance+epsilon)+beta. The original raw parameters, epsilon,
opsets and order remain in the packet. Square roots are bracketed with an
integer-square-root proof on the 2^-64 grid; division endpoints are rounded
outward to the same grid. Exact rational interval arithmetic propagates these
enclosures, including negative scales. No rounded BN fold is called exact.
See the primary [BN operator definition](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html)
and [Conv geometry](https://onnx.ai/onnx/operators/onnx__Conv.html).

This does NOT certify equality to finite-precision ONNXRuntime/torch execution
or ACT's rounded folded graph. Any later use in that path needs its own semantic
binding and concrete witness/replay checks. Original source coordinates and
all raw side consumers remain available; there is no score-producing shortcut.

## Implemented sufficient rule and proof

Each first-bank preactivation g_i has a sparse interval-affine enclosure over
the SAME original input IDs. For a fixed reference tau_i in {0,1},

    ReLU(g_i)=tau_i*g_i+e_i, e_i=ReLU((1-2*tau_i)*g_i).

Choose tau_i=1 iff the certified center enclosure has lower endpoint>=0,
otherwise0. This identity is valid for either reference. If the center sign
is uncertain, e_i(center) need NOT vanish; no center/network-baseline shortcut
is used. Record every original gate identity, including stable and zero gates.
A cap zero suppresses only unnecessary certificate work, never an original bit.

For an original direct consumer f=d+sum_i w_i ReLU(g_i), retain the factored
baseline s=d+sum_i w_i*tau_i*g_i, so f=s+sum_i w_i e_i. Post-Conv affine/BN
coefficients may have outward intervals. Only strictly proved coefficient
signs participate; a nonzero ambiguous-sign interval rejects the row's
certificate, not the original relation. Known exact zeros are explicit.

Sbar is computed AFTER composing the baseline's sparse affine terms on shared
original input IDs. The combined form is retained with each row as evidence;
independently boxing each incoming g_i would lose valid source cancellation.
Uncertain coefficient intervals may still make this bound conservative, so a
failed test does not establish mathematical impossibility. Pair tests likewise
combine sparse source forms on shared IDs before bounding their sum, so
neighboring patches are not spuriously identified or cloned for cancellation.

For each positive/negative pair needed by a row with Sbar<=0, certify

    upper((1-2*tau_i)*g_i+(1-2*tau_j)*g_j)<=0.

This proves e_i*e_j=0, including non-strict zero cases. It does NOT imply a
count restriction on original phase bits. Pair bounds that fail remain
possible overlaps. For every negative j calculate the outward bound

    Sbar + sum_{positive i without certified conflict to j} upper(w_i)*cap_i.

If <=0, D014 proves that contribution may be removed from the gate's LOWER
row while the original upper row, original gate bit and other consumers
remain. All qualifying terms can be removed simultaneously. The diagnostic
only records that certificate; it emits no native row and changes no model.
A row with Sbar>0 is rejected by this fixed sufficient rule before pair work.
This is bound-based forward algebra, not LP/dual rescue, phase/input splitting,
backward propagation, attack or success-dependent dispatch.

The proof follows from interval inclusion, exact ReLU reference decomposition
and D014's n>0=>R<=0 lemma. Tests can falsify the implementation, not replace
the proof. Native row/byte savings, LP-bound impact and concrete-network
benefit remain unmeasured. Proved-stable target rows are counted separately
from those whose outer box crosses zero; the latter are only possibly
unstable, NOT proof that both actual signs occur. The synthetic regression
includes explicit two-sign witnesses; original-model sign reachability is
not measured by this pilot and cannot be inferred from its candidate counts.

## Complete cost and failure accounting

All work is in the new diagnostic process. The shared WorkBudget enforces
256M declared scalar/container work before operations; a per-model ceiling
of current_start+200M bounds each nested source. Parser reservation per source
is4096+model_bytes+8*spec_bytes. Receptive-field metadata prepays16+8*fanin for
each first form and32+8*fanin for each output window. Window row bookkeeping
prepays64+20*actual_patch_size*output_channels. Kernel arithmetic/form walks
separately charge bounded-size operations, including baseline composition.
Before each eligible row's pair generation, reserve32+24*|P|*|N|+8*|N| units
for generation, hashing/deduplication and the later full certificate loop,
including conflicting pairs that skip arithmetic. Before sorting q unique
pairs reserve q*(bit_length(q)+16). Raw BN variance/positivity helpers charge4.
Rational stored endpoints have numerator/denominator<=512 bits; transient
integer arithmetic may be wider. Work units are not CPU instructions or a
bit-complexity/runtime claim. Authentication byte hashing is separately timed.

Before complete evidence encoding and root accounting, reserve40M additional
work units plus65536 summary units. Require measured
8*(retained_numeric_occurrences+unique_objects)+encoded_evidence_bytes<=40M.
Failure of that bound rejects qualification; no second evidence budget exists.
Retain all3 original model/spec bytes, extracted coefficient packets, original
boxes, sparse forms, composed baselines, every tested pair bound, every candidate/rejection and
the full JSON encoding together during accounting. Released protobuf parser
objects are not retained roots but their allocations are in the whole trace.
Opaque model bytes are fully counted as storage, not invented scalar tensors.
No NumPy/HZ native-storage or old live-caller qualification is claimed.

Worker240s, CPU1/GPU0, AS16GiB, numeric entries64M, BOTH RSS high-water growth
and trace peak+metadata<=1GiB, including the65536-byte final-summary reserve.
The scalar-work and retained roots include source, certificates and evidence;
the full end-to-end cost of a future native rewrite remains an open obligation.
Every failure is terminal for the frozen attempt; partial evidence is not a
successful census, a new solve or permission to relax the original gates.
