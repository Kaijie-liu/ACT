"""Same identity/LRU/lifecycle bytecode; change only stored representation.

The legacy timer names remain comparable: freeze = seal once; copy = constant
time read-only borrow (no recursive copying). Full JSON snapshot + hash + byte
comparison on EVERY hit is deliberately retained. No global monkey patch.
"""
from parsed_source_reuse.cache import SourceParser as CopyingParser
from parsed_source_reuse.check import clone
from readonly_source.view import seal, borrow

POLICY = 'READONLY_EXACT_SOURCE_VIEW_R1'


class SourceParser(CopyingParser):
    unpack = clone(CopyingParser.unpack, {'freeze': seal, 'thaw': borrow, 'POLICY': POLICY})
    stats = clone(CopyingParser.stats, {'POLICY': POLICY})
