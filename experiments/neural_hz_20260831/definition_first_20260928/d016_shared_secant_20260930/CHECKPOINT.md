# Shared source secant research checkpoint

On 2026-09-30 the definition-first goal advanced from the cross-domain literature
review to a source-relative secant generator candidate with an exact multi-output
compression theorem, LP containment, and a restricted residual composition
proof. This is a paper milestone; no implementation, numerical qualification,
benchmark gain or established novel-domain claim is made.

## Current evidence and decisions

[Definition and proofs](D016_DEFINITION_AND_PROOFS.md) record both the useful
four-row secant construction and why the earlier product-only construction
was insufficient. Two independent mathematical reviews checked exactness and
preservation of every original phase, including legal independent zero choices.
One review also recomputed the one-layer and two-layer row counts and strict
LP separation. The residual composition and its sparse upper-row cancellation
were reviewed.

The one-layer paper control reduces continuous coordinates 13 to 6 and predicate
rows 44 to 32, retaining all 11 original bits. A two-layer residual extension
uses 6 rather than 21 continuous coordinates and 48 rather than 76 rows, with
all 19 original bits retained. These are explicitly bounded arithmetic counts,
not physical-memory or speed measurements. Original HZ with the same known
lifting can reproduce the formulation; the complete contribution needs further
novelty and real-structure evidence.

The previous goal turn is classified as progress: it created the archived
cross-domain review and changed the mathematical next action. This turn adds
proofs and a strict-relaxation counterpoint, rather than restating a plan.
The overall goal remains ACTIVE and far from completion.

## GPU readiness and limits

Read-only `nvidia-smi` inspection found GPU 0, NVIDIA RTX PRO 6000 Blackwell
Max-Q Workstation Edition, UUID `GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc`,
97,887 MiB total, driver 580.126.09. Root's later reading showed 37 MiB used;
the earlier reviewer reading showed 1,240 MiB. Utilization is transient.
Package metadata indicated PyTorch 2.8.0+cu128 and Triton 3.4.0; no CUDA
initialization or kernel was tested, so runtime readiness is not qualified.

The current live user goal explicitly adds GPU acceleration. This new authority
does not rewrite D015: its frozen runner sets CUDA_VISIBLE_DEVICES to empty and
its results remain CPU-only. The next numerical version must separately
preregister GPU use and account for device roots and work under unchanged
physical/resource obligations; it cannot count one matrix multiply as one
scalar operation or assume CUDA fits the existing address-space cap.

The candidate exposes batched coefficient algebra: alpha uses signed products
and reductions, while residual closure updates Ba and BA. These are potential
GPU kernels, not measured speedups. Any implementation must pay packing,
conversion, transfers, setup, synchronization, exact certification, terminal
lowering, witness reconstruction, and host/device retained memory. FP64 alone
is not a proof: either prove outward conversion and reduction bounds, or use
GPU proposals followed by independent exact checking of accepted certificates.
Neither route is implemented or selected as a runtime menu here.

## Original source readiness

The saved [D015 v2 partial evidence](../../results/d015_source_shielding_20260928_v2/partial_source_evidence.json)
was authenticated without modifying it.
Its SHA256 is
`bbf0d17e48439edc11b352ad4d38ea8fc3ad0e9e0bf99108802b51fd7faac456`.

The CIFAR-large packet and box are complete, with 39,104 decoded scalar
occurrences and a completed five-window local result. CIFAR-medium has a packet
and box with 84,928 decoded scalar occurrences but NO completed result object.
Tiny's packet is absent. Both CIFAR input boxes contain 3,072 coordinates.
These packet counts are read-only inspection, not new candidate measurements.

The packets support future authenticated saved-source algebra for the two
CIFAR models; they cannot complete the original three-model pilot without new
Tiny extraction. They belong to an OVERALL UNQUALIFIED run. A successful packet
parse, a valid hash or a complete local result cannot stand in for the failed
whole-population and retained-storage qualifications. No shared-secant motif
prevalence has yet been measured on any of these sources.

If numerical work is chosen later, retain the inherited 3730 tests / 164 files
and append the required delta tests; combined collection/execution remains
60 seconds. The existing worker 240-second, CPU1, AS16GiB, 256M whole / 200M
nested work, 64M retained numeric entries, 512-bit rational and dual 1GiB host
memory requirements remain. GPU authorization does not waive any of them.
Use a new immutable preregistered version, never rerun or patch D015 v1/v2.

## Custody and formal status

Branch `redu-hz`, commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
New files are confined to this new D016 directory. No historical source,
archive, score, production file, default, commit or push was changed.
The old navigation index was not overwritten; this checkpoint is the new
continuation entry. The standalone documentation uses the write-page skill's
claim-checking guidance while honoring the required repository destination.

Authenticated references inspected this turn:

- Cross-domain review: `06d70c10a6730b93e55637d2554f9460ffcef9ab70b927f1d782fb1c68f3dccc`.
- D015 saved audit: `07b4a133fae5410e0f4b56be6b1c68ffa9a159e135c938cd08dc2f12c1ca2465`.
- Partial source evidence: hash above.

No numerical worker was launched. A separately owned pytest process was observed
at initial inspection and left untouched; it is not a D016 run and supplies no
evidence for this candidate. Its status is not assumed from an old process listing.

Formal baseline remains 1870/2413 = 1063 CERT + 807 validated ADV. Independent
external E0 remains CIFAR100 25 + TinyImageNet 36 = 61/400. New formal gain is
zero, and no invalid-ADV check is claimed for an experiment that was not run.
Full real-network structural prevalence, GPU execution, physical qualification,
new solves and all retention replays remain open.
