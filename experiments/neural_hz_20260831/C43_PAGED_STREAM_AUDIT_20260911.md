# C43 complete:6.75M integer literals,1.30M values; no decoding or solve

redu-hz / f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac; production hash
15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75 unchanged.
Formal1870/2413,13-family baseline and E061/400=CIFAR25+Tiny36 unchanged.
Goal ACTIVE/incomplete. This turn made concrete PROGRESS: C42 exposed a
metadata capacity limit, then a materially different bounded representation
completed the entire source statistic. No HZ capability gain is claimed.

## New bounded representation and frozen execution

C42's1M-per-value dictionary was not enlarged or rerun. C43 stores exact
EXPORTED frequency statistics in256-element uint8 pages. All slots/zeroing
are paid and counted. Counts saturate only at the exported16+ bin; total
occurrences/distinctness remain exact. At most250000 Python page keys and
64M numeric slots are permitted. Dense sources use fewer Python objects;
sparse sources can fail earlier. CPU/GPU/AS/work/time/transient limits unchanged.

24 focused tests1.46s (repeat1.44s), including1000001 distinct dense values
in3907 pages/1000192 actual slots, all histogram transitions and exact complete
agreement with C42's dictionary oracle. The one development fixture failure
was an oversized protocol0 line that BOTH versions correctly reject; that
guard is now explicit and unchanged.

625 frozen source/dependency hashes,68 test files;1784 tests passed52.12s.
One actual nonexecuting stream scan, worker0 in45.97054117638618s; supervisor
normal completion in100.18898054864258s including tests. No source/provenance
drift or old C34/C42 anchor mutation. All opaque payloads passed through the
unchanged reusable64KiB buffer. No unpickler/global/reducer/HZ/solver execution.

## Complete original artifact statistic

Entire506666963-byte C34 pickle through STOP/EOF; SHA
cb5170da0473c01f5304ed307105df0f4aa09c4115af330126fd062ee7bbce3a exactly matches.
Protocol5,897 frames,6802551 opcodes. Integer literal occurrences6749859,
distinct values1300568. Memo indices and bool opcodes are NOT integer literals.
Only7333 occurrences/258 distinct values fall in the labelled[-5,256] range.
Outside it:6742526 occurrences,1300310 values,5442216 repeated occurrences.

Complete outside-range value multiplicity categories:
1:50;2:7;3:7910;4-7:1198208;8-15:94121;16+:14.
Individual counts above16 were not retained; these exported categories and
the occurrence/distinct/repetition totals are exact.

Literal integer opcodes:6438173 BININT,304371 BININT2,7315 BININT1.
There are13319 MEMOIZE,12128 BINGET+LONG_BINGET,340 EMPTY_LIST and2828 APPENDS
opcodes. These are stream counts, not a complete decoded object census.

Opaque payload bytes473442131:BYTEARRAY8=386204244;BINBYTES=87163324;
SHORT_BINBYTES=9280;SHORT_BINUNICODE=65283. The largest opaque argument is
87627504 bytes, but the largest read buffer remained65536 bytes. These payloads
were not decoded or attributed to NumPy/Torch objects merely by opcode type.
Objects inside opaque storage archives remain uninspected.

## Measured resources

Exact paged metadata:5144 pages,1316864 allocated uint8 slots/bytes, including
unused zeros. Whole work184890472/256M; no nested source or native branch.
Measured43.68580474983901s; entryRSS641286144, lifetime HWM before AND after
697921536, conservative growth56635392 bytes. Trace peak2275117 plus17888
end tracer metadata; BOTH original1GiB checks passed. No source HZ was loaded.
These are diagnostic costs, not a native speed/payment result.

## What this changes, and what it does not prove

The complete record identifies substantial repeated integer-valued metadata
as a concrete construction opportunity. It does NOT establish object identity
semantics, safe interning, which fields contain those values, the full decoded
heap, or an actual RSS saving from a changed loader. It cannot convert C41's
failed whole-source decode or C40's failed checkpoint gate into passes.

Read-only source inspection finds a relevant candidate site: Layer.in_vars/
out_vars are integer lists in act/back_end/core.py, and _LayerGraphBuilder's
_alloc_ids in act/pipeline/verification/torch2act.py constructs list(range()).
That site is a hypothesis for a typed interval/shared-metadata representation,
NOT evidence that every counted integer belongs to it. No production edit or
such new representation has been implemented in this turn.

Next requires a genuinely paid source-construction change with complete typed
semantics/alias/source proofs, not another unchanged loader or an assumption
that all equal Python integers can be merged. Full source/whole-LIVE/witness/
native-runtime gates and all2413/13-family formal replay requirements remain.
Current C34 native headroom943763 and C40 diagnostic headroom381747 are NOT
this census's headroom. Do not append millions of free metadata operations.

All seven actual result artifacts are retained. Read the accompanying metadata
handoff. No C44 code or target exists; no actual matrices/score/default changed.

Census e024facd1d81d3efbbbf7afd8a65e8f1d067889462bf1ddff8b46b6f7f548941;
result6abbb950e41e7531c2683bf9e5afa4d7d28fe3c4bd6ac025ea6b5136b947371a;
exit0185d03c3ae2a6e9a2df7b61ff68dcde44b91e54cfeb3de5fe4428e40c99511f.
