# TLL loader compatibility repair and uniform V2 replay

The 2026-09-05 V1 qualification ended with 32 ERROR records, all StopIteration
before specification parsing or HZ propagation. The current ONNX conversion
stores these networks' constant weights as floating buffers, with no trainable
parameters. The preserved worker's `next(model.parameters())` therefore fails.
V1 is not a mathematical or solver regression; it has no solved result or gain.

V2 copies the original isolated worker into a versioned file and changes only
the dtype query and its source-provenance list. Dtype is taken from the common
floating dtype of parameters and buffers; mixed types fail, and a truly
payload-free model uses the converter's explicit float64 default. Original
worker bytes, Trial9 source freeze, V1 driver and all 32 V1 errors are preserved.
V2 adds loader, converter and its own source hashes to worker provenance.

The HZ rule, sparse representation, 45-second solver budget, 240-second wall
limit, 16 GiB per process and four-way concurrency remain unchanged. Every
one of the same 32 rows is rerun in a fresh exclusive directory. Supervisor
records now also propagate the worker's error message. ERROR prevents family
qualification even on a previously UNKNOWN row. No source or outcome from V1
is overwritten or counted as a gain. Formal baseline remains 1870/2413.
