# Separate saved-source parsing reuse comparison — frozen before execution

Implementation `f98e3f5d0a3d9ed81334374585e28bc6f441c5bb`.
207 controls PASS,86.111s;601 source/protocol bindings intact.
Config `configs/backend_controls/parsed_source_reuse_r1.json`,SHA256
`f499fb93eef7fb01a35ec43d40c455f9b45f6763846c8da8bbdd0f1e84536d13`.
Freeze verifier PASS: exact source files, expected full checker result,
control identities, environment, resources and12-call roster match.

Only the two already-saved synthetic sources/constructions from the repaired
diagnostic are inputs. Per object:off/on,on/off,off/on;fresh private cache per
call. No native queries, propagation, source rebuild, checkpoint or dataset.
Single300s clock including read/hash/check/snapshot/freeze/copy/publication/
receipt/cleanup;2s reserve,sampled8GiB,2threads,no GPU. No retries or expansion.

This measures a checking segment, NOT full-MoE latency or new positive bounds.
No production default is changed. The no-reuse arm calls the original parser,
not an artificially burdened parser. The reuse arm retains all predicates and
pays all extra safety overhead. Stop if those costs erase the saved parsing.

Destination: `data/moe/results/parsed_source_reuse_20260925_r1` (absent at freeze).
After this freeze is committed/pushed:

```sh
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -m parsed_source_reuse.run execute
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/audit_parsed_source_reuse.py
```

Independent saved audit uses the ORIGINAL checker, not the new cache, to
recheck source equality. Include all failure terminals and separately cost
the later audit. Preserve historical studies and sealed real objects.
See [protocol](parsed_source_reuse_protocol_20260925_r1.md) and
[controls](parsed_source_reuse_controls_20260925_r1.json).
