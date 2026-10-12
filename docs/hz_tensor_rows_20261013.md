# Captured tensor row intake results

The [frozen intake component](hz_tensor_rows_design_20261013.md) now reads captured
binary64 parameters one affine row at a time, without creating a whole tensor of
rationals. It retains the original H2 source/tensor/state identities. This closes
one intake implementation gap, not trained-model admission, source-to-HZ lowering,
an output proof or any G1–G6 goal.

## Implementation and controls

[TensorRows](../scoped_source/tensor_rows.py) independently validates the ordered
graph and parameter roles/shapes, then yields immutable exact rows. Both weight
and bias streams must exhaust, including their whole-tensor hashes. The last row
alone is insufficient for a completion receipt. Member inventory is a bounded,
deadline-aware two-level scan; it does not inherit the old unrestricted `rglob`.
Public metadata cannot alias internal descriptors. A new replay rechecks bytes.

The reader owns at most two raw chunks and one coefficient row, in addition to
bounded metadata/input coordinates. This excludes rows retained by a caller and
is not a measured peak-RSS claim. A static review caught an exponent-literal
allocation risk before execution; the parser now rejects scientific exponent
strings before constructing a Fraction. Canonical integer/fraction strings and
bounded plain decimal strings remain accepted by this opt-in component.

The archived execution passed 20 unit controls and three process calls: unit
controls, fixed source creation/checking, and relocated checking under `-I -S`.
Small 40-byte and 1-MiB chunk sources agree exactly with the old scalar-node path;
the new reader never calls old weight `tensor()` or `nodes()` materializers.
Controls cover final-hash mismatch with correct chunk hashes, last-row partial
consumption, close/exception/deadline, altered roles/semantics/properties, corrupted
members, nonfinite values and metadata aliases.

The fixed large synthetic source has 3072 inputs and two experts. Its first expert
weight has shape 256×3072; all entries are nonzero dyadic values from a fixed formula.

| Complete large-source census | Observed |
|---|---:|
| Parameters checked | 799494 |
| Center coordinates checked | 3072 |
| Affine rows | 262 |
| Tensor inventory including center | 9 |
| Raw chunk reads | 14 |
| Raw tensor bytes | 6420528 |
| Source bytes including manifest | 6426146 |
| Maximum raw chunk / coefficient row | 1048576 bytes / 3072 values |

A separate read-only reviewer used a standard-library script, without importing
the new reader or fixture/oracle functions, to check every saved parameter and
center value against the analytic formula and original binary64 bytes. All chunk,
tensor, state, request and source hashes/censuses agree. The small relocated copy
has identical bytes and receipt, with no numerical packages loaded. This is not
independent human review or independent network verification.

## Costs and retained execution identity

Raw R1 is `/data1/Kane/MOE/runs/hz_tensor_rows_20261013_r1`, 6603562 regular-file
bytes, below the 64-MiB stage cap. [Compact result](hz_tensor_rows_20261013_r1.json)
binds the archived terminal and dense receipt. Nine executed files are copied and
hashed in `execution.json`; each child had its own 300-second outer cutoff.

Dense fixture generation took 0.3763 s; row traversal plus the independent analytic
oracle took 1.2305 s, with 1.6070 s combined. Three child calls total 1.9968 s of
parent-observed time. The outer record is 2.0040 s **before terminal serialization**;
it excludes the subsequent read-only audit (~0.849 s). None is a production
end-to-end speedup measurement, and no CPU/GPU timing comparison occurred.

Post-execution static review found the wrapper could write PASS when its terminal
file alone crossed the archive cap. R1 is far below the cap and remains unchanged.
The current wrapper accounts for its own serialized terminal before publishing
success; two pure boundary controls pass. Its revised wrapper/test file hashes
therefore differ from the preserved R1 execution snapshots. The reader and all
three mathematical source objects were not rerun or changed. The final local
regression command passes **42 tests** (22 intake/archive, 11 source, 9 store):

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m unittest scripts.test_tensor_rows scripts.test_hz_store_source_units scripts.test_hz_request_store_units -v
```

## Decision and remaining work

Stop this intake width/timing study. E=2, top-2 has one route pair: it is neither a
route-changing control nor the full eight-expert trained model. Source bytes are
already captured; live capture still materializes base64. The declaration's
correspondence to a real checkpoint remains unproved here.

Next, freeze the source-to-store connection and bounded affine consumption. Current
producer/checker require whole layers and 262144 nonzeros per layer; Store retains
all rows with a separate 1048576-nonzero limit. The registered dense 256×3072 layer
alone exceeds the first limit. Do not pass a lazy list into the old checker or
simply increase its caps. Preserve exact original coefficients, factor allocation
order, compensation, inherited constraints and complete endpoints. New matrices
require new evidence. Convolution/pooling remain a separate operator task.

No trained model, solver, HZ propagation, CUDA, sealed input or new output query
ran. The physical GPU refusal and all historical comparison/guarantee boundaries
remain unchanged. All six goal gates are OPEN.
