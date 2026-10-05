# Live schema V2: include OrderedDict state and metadata

Live V1 stopped before candidate construction on the original concrete model's
registered state_dict, an OrderedDict. Its exact failure, source hashes, schema
census and exit remain immutable. It did not run a new ReLU or solver.

V2 registers exactly OrderedDict key/value entries and its optional _metadata
attribute, recursively under the same known-type rules. Unknown extra
attributes still reject. No registered state tensor, module-version metadata
or alias is copied/removed. All numerical compiler, source, memory/resource,
probe admission and scalar-oracle gates remain unchanged. Test tensor identity,
metadata drift, unknown attributes and nested OrderedDicts before the new live
run. This is an input-surface instrumentation correction, not a numerical or
verdict retry. One fresh qualification, 240 seconds/16 GiB, exclusive output
evidence/c5_live_transaction_20260905_v2.json. Stop before ReLU/publication.
