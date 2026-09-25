"""Unchanged checker predicates with privately bound read-only parser data."""
from parsed_source_reuse.check import check as copying_check, clone
from readonly_source.cache import SourceParser

# Same scope validation, deadline, four original predicates and final cleanup.
# Schema stays SOURCE_PARSE_REUSE_CHECK_R1; parser.policy identifies the view.
check = clone(copying_check, {'SourceParser': SourceParser})
