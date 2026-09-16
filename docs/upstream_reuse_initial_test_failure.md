# Initial upstream reuse control invocation

The first direct `python -m unittest upstream_reuse.tests -v` invocation ran
8 tests: 6 passed and 2 errored before cache execution. The two source-cache
tests assumed the analytic fixture contained `common_facts.json`; its actual
file is named by `manifest.common_facts.file`. Both failed with
`FileNotFoundError`. The tests now follow that reference. No mathematical,
acceptance or experiment setting changed. The initial real analytic four-way
differential, AST identity, reserve/error cleanup and warm-validation tests
passed. Numbered full-suite receipts follow this correction.
