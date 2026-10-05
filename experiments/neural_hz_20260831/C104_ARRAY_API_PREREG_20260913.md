# C104: same exact row program through the NumPy array C API

C103 is CLOSED: all2762 tests and1.88x local row payment pass, but actual
source106.818s/RSSgrowth1170534400B exceeds1GiB. No source/native/solver
admission. C103's separate20000-array experiment confirms1440112B retained
exporter metadata after buffer release, with only88B on the second pass.
NumPy2.4.4 buffer.c explicitly frees exporter data on array destruction. This
is a demonstrated overhead mechanism, not a full attribution of allocator RSS.

Replace C103's array buffer-protocol access with the standard NumPy C array
API. Keep the EXACT same two numeric scans, finite/nonzero checks, window,
head, ldexp/inverse, original bounded powers/RHS and private one-use ownership.
The call holds its Python argument tuple and GIL throughout; inspect actual
ndarray type,dtype,endian,dimension,shape,stride and output writability, and
read/write only the private caller's fresh outputs. No buffer export, private
_buffer_info access, manual cache free, array-pointer lifetime extension,
hidden native heap, changed coefficient or weakened numeric guard.

The standard local NumPy2.4.4 headers/API are already installed. Compile with
the same local GCC/O2/no-fast-math policy; freeze every new header, compiler,
source and binary dependency plus the inherited environment. No installation
or new compiler framework. All logical work prices and two owning output
arrays remain. The new ABI type checks replace the six old buffer descriptors
and format/shape checks; there is no new per-coefficient scan or retained
per-array exporter metadata. This claim must still pass real measurements.

All2762 inherited tests plus179 same new row/ownership/power/full-source tests
must collect+execute<=60s. The179 new tests rerun the same C97/original radix
and C98 complete-state oracles against C104. The C103 failed test draft remains
archived, not an inherited passing test. C103/C104 implementation sources are
new and separate; no old file is rewritten.

Reuse C103's fixed four ordinary shapes/2000rows each, complete output equality,
full C41 tracer/RSS, one256M diagnostic pool and unchanged32N+64 numeric and
8N+64 comparison prices. Require aggregate>=1.25x and no shape regression
against C97 before the network; NOT an F-speed or physical-HZ claim.

Then one fresh original Tiny target, complete original C100 source/input/native/
final proof identity binding, all four C102 measured root walks, full LIVE/
reference, native ingestion, ordinary base+property MILP and concrete witness
checks. Keep45s solver/240sworker,256Mwhole/200Mbranch,64Mentries,both1GiB,
AS16GiB,CPU1/GPU0,shared16384aux/131072entries/16Mextra work. Retain100965local/
7357circuit/199unit inverses and1350binaries. Source hashes must match the same
full original proof; otherwise reject, never substitute an archived HZ.

Any failure closes C104v1 without an unchanged retry/cap increase. Historical
and C103 results read-only; new exclusive logs/results/exit and source freeze.
No default/production/charter/commit/push, LP status/ray/marginal, attack/PGD/
BaB/split/backward/dual rescue, binary pivot or convex replacement. C32 native
payload/authentication remains separately paid; not all CPU/hash work lies in
generation charges. Formal1870/all13/full2413 and separateE0CIFAR25/Tiny36/full400
retention gates unchanged. Full Neural-HZ/PLDI/generalization goal ACTIVE.
