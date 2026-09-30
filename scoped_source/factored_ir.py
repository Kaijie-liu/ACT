"""Mechanical block storage and named-row assembly, no source lowering rules."""
from fractions import Fraction as F
from scoped_source.factored_io import referenced, MEMBER_LIMIT
from scoped_source.sparse_ir import lp_record


def block(root, ref):
    if set(ref) != {'name','file','bytes','sha256'} or type(ref['bytes']) is not int or not 0 < ref['bytes'] <= MEMBER_LIMIT:
        raise ValueError('source block reference')
    value = referenced(root,ref)
    if set(value) != {'bounds','rows'}: raise ValueError('source block fields')
    return value


def assemble(root, refs, names, guards, tick):
    """Caller independently derives the ordered names and guards."""
    bounds={}; rows=[]
    for name in names:
        tick(); value=block(root,refs[name]); bounds[name]=tuple(map(F,value['bounds'])); rows.extend(value['rows'])
    rows.extend(guards)
    result=lp_record(names,bounds,rows,{},F(0)); tick(); return result
