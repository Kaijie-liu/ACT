# C106 local pre-freeze test draft correction

First local test run:34passed/14failed in1.36s, process exit1. No measured target
or supervisor had started, and no source qualification is claimed from it.
Twelve failures are a nonexistent test helper mask_rows; the actual constructor
accepts row_mask. Two fail in the ORIGINAL C104 call, before new code: repeated
source loses a selected circuit coordinate; shared outputs require coalescence
outside the old quotient class. They were never passing admitted old examples.

Both v1 test texts remain unchanged as diagnostic drafts. V2 uses the original
row_mask constructor and requires the exact old rejection reason on both old
and new whole sources for those rejected graphs. Ordinary accepted circuit-HZ
full fingerprint checks remain. All row-level shared/repeated sum equality and
Fraction bound checks remain. No numerical source or gate is weakened, and no
previously passing inherited test is omitted. The complete supervisor freezes
both drafts and this record, and executes all2941 inherited tests plus48new v2
tests. No benchmark score, new topology acceptance or baseline promotion.
