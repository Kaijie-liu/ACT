# C75: complete actual native decode independently fails both transient gates

C75 is terminal: supervisor91100 exit1 after4.570889518s; worker3.246165413s.
The complete C74 native checkpoint was independently authenticated and decoded
in3.132612710s. It remained strongly held when the unchanged1GiB gate rejected.
The failure precedes root collection, owner layout, source/native rebinding or
inverse reconstruction. It does not invalidate C74's already completed fresh
native/LIVE result, and it does not qualify C74's failed complete restore.

EntryRSS581472256B; lifetimeHWM2148925440B; growth1567453184B, exceeding1GiB
by493711360B. Final measured traced peak775900040B plus tracer metadata350632288B
is1126532328B, exceeding1GiB by52790504B. The observer's immediate post-decode
metadata value was350632096B; that boundary independently failed as well.
No timer increase, dropped roots, unchecked partial decode or source trimming.

C41 restored394 numeric reducers, with exactly one readonly backing copied:
200 entries/1600B. Encoded object aliases, readonly backing checks and complete
file-byte authentication remain intact. Therefore a large readonly numeric
copy is not the observed cause. Diagnostic work16308. Complete checkpoint
534241302B SHA5586296e7fccef66956dff042d29ca93b999d3aafc1a0285cea17863b089743f
is unchanged, as are all frozen sources and provenance.

C74's later60s restore timeout remains unprofiled beyond this boundary, but an
unchanged retry is already ruled out by the independent decode resource failure.
C71 found the same kind of full-checkpoint decoding failure for the older C25
archive; C72's smaller OFFLINE proof dependency was not a fix for full LIVE
restoration. Do not reuse that boundary change to claim this gate passed.

Next work should identify the complete checkpoint's actual allocation/alias
population and implement an exact memory-bounded restoration or serialization
only if evidence supports it. A read-only streaming opcode/storage inventory
can distinguish numeric payloads, tensor serialization buffers and repeated
Python scalar/container metadata without materializing another full graph.
These are hypotheses, not a diagnosed object-type cause. No specific allocator
or Python container is yet established as the dominant source of the excess.

Preserve every model/cache/caller/bounds/source/phase/proof field and its semantic
and numeric sharing requirements. Keep both1GiB metrics,64M,60s and source
authentication intact. No disabling/pausing/filtering the tracer, warmed entry,
increased limits, lazy eviction or a component-only substitute. Any new format
needs ordinary exact full-field/alias tests and the actual complete restore,
not a new corner-case framework or a repeated expensive original-network run.

All1841 unchanged C74 tests were reused solely for unchanged math/runtime/decoder
code; this observer is not a new proof or solved case. No native/solver rerun.
Formal1870/all13 and separate E0 CIFAR25/Tiny36 unchanged. Full goal ACTIVE.
