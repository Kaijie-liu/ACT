# C20 complete product-class census: diagnostic PASS

815 tests7.15s, worker exit0. Supervisor32.2333239717409s; worker23.015816058963537s.
Source/provenance driftfalse; both complete successful original C9/C10 archived
dictionaries retained and their original HZ/portable source identities unchanged.
No products, candidate HZ, generator, native ingestion or solver executed.
Formal1870/2413 and separate E061/400 unchanged, gain0.

Complete106721 local aliases and322936 original product hits match independent
C10 counts. Only357 local aliases have power-two ratios, but their fanout
accounts for216572 hits; selecting by alias count instead of actual hit count
would mischaracterize payment. Every right-power-two hit has ratio magnitude
2^-1. Count right-general106364; of these100732 have power-two LEFT coefficients.
Either operand is power-two in317304/322936 (98.2560%);5632 are general on both
sides.15149 hits have a negative right operand. All322936 hits are EQ; no INEQ
hits on this target, though tests cover EQ/INEQ/binary consumers.

Complete hit-row population43256:6396 all-right-dyadic rows,36860 all-right-
general rows, zero mixed-right rows. The primitive must still support mixed
rows uniformly. The saved exponent/hit-width histograms cover all hits before
frontier selection, not just100603 ultimately selected aliases.

Diagnostic work112595472:104845008 complete local-alias geometry and7750464
classification. Measured19.397237348370254s, conservative HWM growth87822336
bytes, traced peak13113082 plus9728 metadata; unchanged1GiB gate passed.
Absolute process peak1698116KiB includes both archived dictionaries. These are
diagnostic, not generator or live-state resource results.

New product specialization is now worth implementing, but tariff and full
construction remain unproved. A simple right-only shortcut leaves106364 general
calls; the independently preregistered left-dyadic count shows that only5632
actually require general exact arithmetic. Preserve full input/output window
validation, all cached numeric product bits and the entire old frontier.
Do not infer a solved benchmark, wider-target permission or a passed generation
budget from these counts. C18 remains CLOSED and cannot be edited or rerun.

Artifacts in results/c20_product_census_20260911_v1/:

- preregistered.json16984c2501e2c1138610a4e88a937bad95b4e2a9bdc06b4c9ebae6e589e770b8
- product_classes.jsone00c98f86b0617655099c294bd477be2fe55f1845740754f12d2313bc052bdd1
- result.jsoncea6ea5da1fd383ebf2f4d23f8f255ec88b539a8f044f63940fbf8f0901ef917
- exit.json9126b8f9e9b7a98f83375b4de634ffb76cf718249cac3c7ad6b84ae5308bc2ab
- tests.loge3b93f6d11c0dfd985adc8cdf2ea3067dbe36ed44b7d8c1a7a2658e5bd8a42dd
- worker.log5e0f0e97dfe4e976b2a81e8c71c2c5f33a3ac7ea583489174478658e9c8b7963

No worker remains. Any next generator uses a NEW frozen source version and
the full original-expression/ownership/physical/native gates. Current progress
is complete evidence deciding the next implementation, not formal gain.
