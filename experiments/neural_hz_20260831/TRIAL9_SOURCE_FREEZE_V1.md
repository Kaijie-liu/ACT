# Trial 9 Source Freeze V1

Trial 9 PID 251298 started on 2026-08-31 at 11:14:30 local with `/proc` start
ticks 698719726. Its provenance is branch `redu-hz`, repository commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`, and candidate SHA-256
`16f35a7cd7cb6d8f592d345be736af3a0348571f3ff00bc8ea3d8235a9cd2d89`.

The candidate digest is reproduced exactly as `shadow_worker._provenance` does:
for each of the nine paths in its fixed order, append the UTF-8 relative path
bytes and then the complete file bytes to one SHA-256 stream. Individual file
hashes are frozen in `TRIAL9_SOURCE_FREEZE_SHA256SUMS` and verify from this
directory with:

```text
sha256sum -c TRIAL9_SOURCE_FREEZE_SHA256SUMS
```

While the bound worker remains alive, none of these nine files may be edited.
New design work is restricted to uniquely named files in this experiment
directory. The exit sealer independently requires the same combined candidate
digest in any completed result JSON. This freeze is custody/provenance only;
Trial 9 has formal gain zero and cannot change the 1,870/2,413 baseline.
