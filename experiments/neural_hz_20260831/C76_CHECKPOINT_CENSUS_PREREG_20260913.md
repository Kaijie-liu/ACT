# C76: complete streaming serialization census of the actual native checkpoint

Previous goal turn is PROGRESS: C74 adds actual original-network/native/full
LIVE evidence; C75 locates a decisive complete-decode resource failure. Scores
stay formal1870/all13, separate E0 CIFAR25/Tiny36. No new external blocker.

Before implementing a new decoder, inspect every opcode in the complete C74v2
native checkpoint, SHA5586296e7fccef66956dff042d29ca93b999d3aafc1a0285cea17863b089743f.
Use the installed standard pickletools parser without executing constructors
or restoring a model. Authenticate the complete original file before and after.
This is read-only diagnosis, not a substitute for the failed complete restore.

Record all opcode counts, serialized byte-payload populations/lengths and the
integer scalar population, including exact distinct values in [0,2^24) using a
fixed2MiB bitset. Other integers are counted/range-reported, never silently
discarded. Bound short-string metadata to8192 distinct strings and128 characters
each; count all other strings. Keep only the16 largest byte-payload descriptors,
not their bodies. No opaque object deletion or incomplete archive claim.

Pay8 per opcode,8 per integer census update, the2MiB bitset initialization and
ceil(payload_bytes/8) per byte payload. Full file hashing/bytes read are reported
separately, as in the existing authenticated checkpoint reader. Diagnostic
whole256M,64M entries,CPU1/GPU0,AS16GiB,both1GiB,hard60s/soft45s stay. Emit regular
boundary counts and retain timeout/failure honestly; no retuned prices or caps.

The census distinguishes candidate allocation causes; none is assumed in advance.
No HZ, source, solver, native execution, constructor, witness or default changes.
Reuse1841 unchanged qualification tests only for unchanged math/runtime/decoder
dependencies. All new writes go exclusively to results/c76_checkpoint_census_20260913_v1.
Freeze source/provenance first; logs/events/results/exit automatically retained.
After evidence, return to an exact full restoration with every model/cache/
caller/bounds/source/phase/proof field and all original qualification requirements.
