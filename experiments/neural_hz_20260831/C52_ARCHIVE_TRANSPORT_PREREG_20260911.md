# C52 archive transport: use the existing authenticated numeric decoder

The fresh v2 mathematical transaction passed. Its138785-byte protocol4 artifact
is identical to v1's saved mathematical data. A separate read-only replay
recomputed the chain's two source proofs and six toy points, then the strict
owner ledger rejected a restored eq_uids array with unknown_numpy_external_buffer.
That replay did NOT establish restored whole-state ownership. The v2 fresh
construction/proof/measurement success is unchanged; its artifact is not edited.

The existing C41 authenticated decoder supports NumPy protocol5 _frombuffer,
not protocol4 reconstruction. Use that existing format/decoder, with NO new
owner adapter, loader special case, mathematical change or worker retry:
authenticate/read the complete old artifact, export all fields to a new
exclusive protocol5 file, retire every old numeric array, then load the new
file with the unchanged C41 decoder inside one measured transaction. Recompute
all six source proofs,48 toy points and complete source/state ownership and
strict storage comparisons. A transport success cannot replace any native,
actual-target, full-suite or whole-original-request gate.

All source loading, export, old-array retirement, new decode and checks stay
inside the same1GiB HWM/trace-metadata window, withAS16GiB,CPU1/GPU0,240s wall,
whole256M/nested200M and64M entries. Retain every source/reference/compact/proof
field; no source loading outside tracing or old-buffer acceptance. Any failure
is recorded and stops this transport version. Output is exclusive under
results/c52_math_archive_transport_20260911_v1. Do not alter the old artifact
or extend this bounded format correction into a loader-development campaign.
