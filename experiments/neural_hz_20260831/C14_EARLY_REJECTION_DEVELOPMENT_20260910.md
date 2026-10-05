# C14 pre-target development

New isolated implementation changes exact conjunction evaluation order, not
the C13 mathematical acceptance predicate. Source includes before-operation
WorkPool charging, fixed8-term coefficient blocks, tri-state guard arrays and
atomic full-population coverage. Initially33 focused tests pass; before target
freeze added the stronger case that a budget-exhausted prefix contains a true
admissible definition yet still cannot be returned/published. No target was
run to choose order, block size, tariffs or capacities.

Code inspection moved consumer-multiplier arithmetic inside the precharged
scalar stage before tests. Exact product/norm/window/collision helpers are
frozen, previously independently tested routines. Eager C13 toy decisions are
used as an oracle, with exact equality of final admissibility and every guard
that C14 actually evaluates. Tests ensure no later norm/collision runs after
an earlier proved failure, and every term in a computed block is charged.
All prior failed versions and data remain untouched. No production edits.
