# C103 closed-source memory diagnosis only

Frozen C103 source execution has terminated: source106.818s, RSS growth
1170534400B>1073741824B, traced455117449+901920B. No source admission or solver.
The version remains CLOSED regardless of the following diagnostic outcome.

Hypothesis: using NumPy's buffer protocol on every retained row array leaves
per-array exporter metadata alive after PyBuffer_Release. Installed NumPy is
2.4.4; its local ndarraytypes.h includes the private _buffer_info field. The
upstream NumPy source explains that extra buffer information is freed during
array_dealloc, not bf_releasebuffer. This is a hypothesis about a contribution
to C103's measured growth, not yet a complete explanation of process RSS.

One<=60s AS16GiB CPU1/GPU0 diagnostic creates20000 ordinary owning float64
length3 zero arrays under the full C41 tracer/RSS boundary. It records traced
current allocation immediately before, after one memoryview/release pass and
after a second identical pass, while ALL original arrays remain held. Require
all numeric values/dtypes/shapes unchanged; separately report whether metadata
persists and whether the second pass reuses it. One shared256M work pool pays
128 per array per complete pass,64 per allocation and32 per full value check.
All arrays/diagnostic outputs are Python-owned; no private pointer mutation,
prewarm, reset, tracer exclusion, freed-input benchmark or target retry.

This diagnostic cannot admit C103 or explain unmeasured allocator RSS. On a
positive result the bounded next implementation is the public NumPy C array
API, preserving all exact row equations and guards, with new frozen headers/
binary and full source requalification. No new compiler framework, dependency
installation, removed proof, cap change or serialization campaign is allowed.
Formal1870/all13 and externalE0CIFAR25/Tiny36 unchanged; full goal ACTIVE.

Source: https://github.com/numpy/numpy/blob/v2.4.4/numpy/_core/src/multiarray/buffer.c
