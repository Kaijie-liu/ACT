# Ordered flow comparison and negative innovation result

The ordered-flow component passed its complete registered test population and tightened the selected production Softmax-to-PV control. Its entire measured gain was also obtained by the known prefix plus grouped-value reference, which used fewer columns and rows. This closes the claim of an independent innovation advantage on this control; it does not complete the Neural-HZ domain objective or change any formal score.

## Executed evidence

The unique [run](../../results/d113_ordered_probability_flow_20261002_v1/exit.json) finished with supervisor and test exit codes 0. Session 60089 is terminal. All 3853 tests in 190 files passed; all 3849 predecessor tests remained the ordered prefix. Pytest reported 47.62 seconds and 13 inherited warnings. The complete test subprocess took 48.88367719762027 seconds, below the unchanged 60-second cap; supervisor wall time was 59.720812076702714 seconds. Source drift and input drift were empty and provenance drift was false.

The two inherited D112 controls were rerun only into this run's inherited_d112_controls directory. Their complete JSON hashes exactly match the old records: 7e394331bfd9c24520af784d3cc6d357a22ee0cf7fb125cc893b8fd4d3c32b52 and 9e298c4194796c1d00d4348d54c199add3f35f3a60934c60833add7312638e15. Old frozen files were not overwritten. The explicit output relocation was registered before execution and did not remove tests or alter their assertions.

## Same objective and full formulation sizes

These are the five floating LP diagnostic maxima of the same output difference on the four-token, single-channel fixture. They are not trained-network certificates or validated adversarial inputs. A includes the common experimental probability/value bindings in addition to the full selected production wrapper.

| Arm | DeltaY upper | Continuous columns | EQ rows | LE rows |
| --- | ---: | ---: | ---: | ---: |
| A production component and common binding | 1.9382653810132005 | 19 | 6 | 72 |
| B endpoint and fixed sector reference | 0.44770273928717785 | 26 | 9 | 238 |
| B_prefix known prefix rows | 0.3889852131737861 | 26 | 9 | 241 |
| B_group known prefix and grouped product | 1.1102230246251565e-16 | 27 | 10 | 249 |
| C shared ordered flow | 1.1102230246251565e-16 | 30 | 15 | 270 |

All five diagnostic solves returned optimal status. The largest equality residual was below 1.2e-15 and largest inequality residual below 3.2e-15. These floating residuals are not an exact rational optimality certificate or native floating soundness proof. The tiny positive upper bounds for B_group and C are numerical zero at the registered 1e-7 tolerance, not a strict negative certificate.

The [complete control record](../../results/d113_ordered_probability_flow_20261002_v1/production_flow_control.json) includes all solver assignments, statuses, residuals, dimensions and relation receipts. It reports B-C=0.44770273928717774 and B_prefix-C=0.388985213173786, but B_group-C=0.0. Compared with the equally precise B_group, C adds three continuous columns, five EQ rows and 21 LE rows. Thus neither measured precision nor this formulation size favors C over the strongest registered reference.

## Mathematical interpretation

The rational non-implication test passed: the specified endpoint relaxation admits a point with negative second prefix flow, while the certified ordering relation excludes it. This establishes that those finite endpoint rows do not already encode every prefix implication. It does not separate the candidate from all known relational abstractions or establish a true network counterexample.

The output control shows why prefix constraints alone are an insufficient attribution reference. For V=(v,v,0,0), DeltaY=(z_1+z_2)*v. The known prefix gives z_1+z_2<=0, and its grouped product envelope with v in [1,2] proves DeltaY<=0. C obtains the same fact as DeltaY=-F_2*v. Independent per-token product envelopes can miss this aggregation even when the prefix inequality is present. The gain is therefore accounted for by known order information consumed through a known grouped-product identity, not by a new probability theorem or an independently stronger domain.

The subsequent [general substitution audit](PROJECTION_AUDIT.md) proves a paper-level equivalence for arbitrary fixed token/channel counts when the comparison retains exactly the same certified bounds and aggregated product envelopes. It explains which representation difference remains and does not claim a global impossibility result for Neural-HZ innovation. No flow-elimination implementation, extra numerical experiment or post-freeze source change was made for that audit.

## Decision and remaining goal

Keep this default-off implementation as a reusable component and strong-reference fixture. Do not promote it into model or GPU execution on the current justification. Do not seek an innovation claim merely by omitting B_group, enlarging the same fixture, or compressing the flow columns. A subsequent definition candidate needs a distinct, falsifiable structural or compositional contribution against an explicit fair reference, including end-to-end costs; it need not differ from every conceivable equivalent HZ encoding.

The completed gates are conditional mathematical/component gates only. Actual model/source/phase binding, native HZ admission, concrete decoder integration, outward-rounded floating lowering, GPU computation and full physical-resource qualification remain false. The control uses no native binary phase columns; the preservation fixture does not substitute for original-model phase qualification. No child ReLU was executed.

Formal score remains 1870/2413, consisting of 1063 CERT and 807 validated ADV. Independent E0 remains CIFAR100 25 plus TinyImageNet 36, 61/400. New formal gain is zero. Untouched historical results do not establish candidate-level 13-family zero regression or either full replay. The full Goal remains active and incomplete.

Date 2026-10-02 Australia/Sydney; branch redu-hz; commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac; tracked binary diff SHA256 remains 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. Only new isolated experiment files and this new run were written. No production or historical changes, commit, push, remote backup or background worker remain from this run. The documentation workflow keeps the exact semantic obligations, executed bounds, negative attribution result and unqualified stages separate.
