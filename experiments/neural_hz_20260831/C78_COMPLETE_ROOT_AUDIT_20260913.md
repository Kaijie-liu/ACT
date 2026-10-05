# C78: full traversal passes resources; original storage sharing is not restored

All84files/1875 tests pass:21 new complete-C5-oracle checks plus1854 inherited.
Full collection/execution27.424069406s, pytest21.32s, no skip/failure/error.
Compiler source/binary and257 dependencies are frozen. Two pre-freeze local
runs were20pass/1fixture-setup failure and21pass, both1.07s; the unsupported
SimpleNamespace weak-reference fixture was corrected to its actual torch model.
No collector assertion was weakened.

The complete actual C77 checkpoint is decoded and traversed without omitted
fields. The new C loop emits exactly the original integer path/type/value
tokens and records exact pointer bits, with no alignment or value-identity
assumption. All original numeric owner/schema checks remain. Final full traversal
visits6746342 integer occurrences/1300611 distinct integer objects, emits
1177976461 exact token bytes and uses5382144B identity pages/6132 prefixes.
There are8522 other seen objects,662 numeric roots,1309133 total objects and
93506276 Python shallow bytes. Root collection completes at25.934326733s and
the unchanged owner ledger at26.279328106s; C77 previously timed out60s inside
the same complete traversal. No F-speed or benchmark gain is claimed.

Worker ends26.457461933s, work97642765. RSS entry582569984B/HWM1622597632B,
growth1040027648B; traced706517654+76974912=783492566B. Both1GiB metrics pass
through the actual failure, and the whole checkpoint stays held. The original
256M/60s/64M and stricter original-entry-envelope gates were not changed.

Failure is MemoryError('complete restored LIVE entries'): complete restored
numeric500661000B/47235662entries exceeds the frozen42949161-entry envelope
(original42948961 plus readonly200), despite remaining below64M. Sources/native
binding and inverse are not executed after this rejection. The candidate is
closed, not complete-restored, promoted, or allowed to solve.

The complete old/new numeric ledgers locate the entire excess:

- Original LIVE:522 torch storage groups,81811064B/10226383entries.
- Restored:663 torch storage groups,116103072B/14512884entries.
- Both sides have663 Tensor objects. All original394 NumPy arrays retain the
  same384556328B/32722578entries; the new readonly backing adds1600B/200entries.
- Thus141 extra torch storage groups account for34292008B/4286501 entries.
  This is not missing HZ data or a reason to enlarge the envelope.

Read-only follow-up inspected the installed torch storage reducers/legacy
serializer and two C76-indexed original byte payloads. torch storage.__reduce__
uses separate legacy torch.save bodies. That format explicitly records
storage_key=str(storage._cdata); its loader normally scopes the storage map to
each separate body. The two10035445B bodies at original offsets282745475 and
518211900 carry the SAME origin key510393008 and identical full SHA
dd39c17ceb02b5fb38e363c43a338b50b2e490751a06fd751f9bad74c7ec96ed.
They are literal DoubleStorage/cpu/1254400-entry bodies, not guessed aliases
from equal tensor values. Header inspection executed no constructors. See
C78_TORCH_STORAGE_EVIDENCE_20260913.json for complete compact ledgers and headers.

Next evidence-based rule: in ONE authenticated complete checkpoint, reuse a
decoded tensor STORAGE only when its original encoded storage identity, full
header, and every serialized byte agree. Preserve distinct Tensor objects,
shape/stride/offset, and distinct equal-value storage identities. Scope and
retire the temporary map with that one decoder. No numeric/mutable value-based
deduplication, dropped field or larger budget. Full original owner envelope,
source/native/inverse still must pass; keep this failure immutable.

Supervisor24334 terminal exit1 after55.237695253s, source/provenance drift false.
Result SHA4dda558de45eb3db5a7b3c4f53e9308545af50cd45aca9f78762561e4b2dfbee.
Formal1870/all13 and E0 CIFAR25/Tiny36 unchanged. No production/default/history/
network/solver/commit/push change. Goal remains ACTIVE.
