# Separate pinned external frontend compatibility R1

This is a bounded capability probe, NOT an external performance benchmark,
not a reopening of AdvMoE/CROWN search, and not a new formal SAFE experiment.
No model training, data download, installation or external repository edits.

Pin local alpha-beta-CROWN e5c7e17bf0488843acb77b7519f59876717a49f4 and
auto_LiRPA 5a098e8f9fb5786a428a024981d833d303921f2d. Use the already installed
`/data1/Kane/MOE/envs/alpha-beta-crown/bin/python`, but put these pinned source
trees first on sys.path and RECORD actual import paths (the installed wheel
must not silently substitute a different implementation).

Three predetermined,120second-per-process,CPU/float32 cases:

1. A two-input affine E=3 weighted top2 toy with dynamic torch.topk and gather.
   Scores=(x0,x1,-x0); expert scalar values=(1+x0,1-x1,1+x1).
   Box[-.1,.1]^2 includes route changes/ties. Test conversion then plain CROWN.
   Literal PyTorch tie selection is NOT by itself the project's ANY_LEGAL_TOPK
   contract; even successful conversion would need that semantic gap closed.
2. Static pair{0,1} from the same functions, variable softmax weights and both
   experts retained. This is a WHOLE-BOX static pair obligation, not an
   equivalent full dynamic MoE or a guard-conditioned input domain. Test one
   plain CROWN call, not optimization/search. Positive values are numerical
   filters, not independently outward-rounded certificates.
3. Full tool Python API input parsing: a box as positive control, then the
   conjunction of that box with x0+x1<=.1. Record acceptance/rejection or import
   failure separately; do not remove the relational constraint to claim success.

For all cases record versions, source hashes, exact phase/error, bounds and
finite grid forward agreement if applicable. Errors and timeouts are valid
compatibility observations, not model UNSAFE or a general tool incapability.
No changed cases after outcomes. F0 outer-relaxation solving and full BaB are
explicitly NOT tested. Source anchors for TopK/GatherElements, Softmax and API
input polytope parsing accompany runtime records. External records are separate
from request-LP and relation ablation outputs and never pooled with them.
