# C76: complete stream is dominated by repeated integer scalar encodings

Complete authenticated534241302-byte stream scanned without executing any
constructor. All6809900 opcodes are counted;6749869 encode Python integers.
6749790 integers lie in [0,2^24), with only1300550 distinct values there;79 are
outside.7878 occurrences are in the usual small cached range. Integer min-1,
max2086371328. There are only53 BINFLOAT opcodes,16162 MEMOIZE operations and
12999 BINGET/LONG_BINGET references. Original explicit PUT/BINPUT opcodes are
absent. This identifies a large repeated scalar population, not missing HZ data.

Byte payloads:393 BYTEARRAY8 bodies total384556328B;609 BINBYTES bodies total
116296389B;69 short byte bodies14418B. All payloads and opcodes are included;
the16 largest descriptors and all bounded short strings are recorded. Numeric
payloads, tensor payloads and immutable scalar repeats remain separate evidence.
The count alone does not quantify each allocator's actual contribution.

Worker39.760444054s, logical diagnostic173184144; RSSgrowth194924544B and
traced177403442+174160B pass unchanged limits. Supervisor72157 exit0 after
40.883660883s, no source/provenance drift. Input was fully hashed before and
after. No unpickle, network, source/HZ mutation, solver or restore admission.

Next hypothesis: an exact wire transform adds explicit memo references for
repeated immutable integers. Preserve all array and mutable-object opcodes and
their original memo references. Original MEMOIZE must become explicit original
indices so inserted scalar memo entries cannot shift any original reference.
Frame sizes must be recomputed, not copied over a changed byte stream. Scalar
values/types and every other field/constructor remain identical; only extra
sharing among equal built-in immutable ints is allowed. No ndarray/tensor or
mutable-container deduplication. Actual decode and full source/native/inverse
qualification under the original gates are still required; this is not yet a
memory saving or a solved case. Formal1870/all13 and E0 CIFAR25/Tiny36 unchanged.
