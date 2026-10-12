"""Portable stdlib V3 source + all-candidate output check; no solver."""
from pathlib import Path
import time

from scoped_source.factored_io import read
from scoped_source.hz_request_store import Work,compact,decode,require,sha
from scoped_source.check_scaled_references import CheckedScaledForest
from scoped_source.scaled_endpoints import check_proof


def verify(root,expected,*,deadline=None):
    started=time.monotonic(); work=Work(started+300 if deadline is None else deadline); root=Path(root)
    raw=read(root,'binding.json',8192,sha(expected)); binding=decode(raw)
    require(raw==compact(binding) and type(binding) is dict and set(binding)=={'schema','source','forest','contract','proof'}
            and binding['schema']=='CAPTURED_SCALED_COMPLETE_BINDING_V3','complete portable binding')
    for name in ('source','forest','contract','proof'): sha(binding[name])
    f=CheckedScaledForest(read(root,'forest.json',128*2**20,binding['forest']),root/'source',binding['source'],
                         expected=binding['forest'],work=work)
    c=decode(read(root,'contract.json',8192,binding['contract']))
    p=decode(read(root,'proof.json',16*2**20,binding['proof']))
    result=check_proof(f,c,p,expected_contract=binding['contract'],expected_proof=binding['proof'],work=work)
    require({p.name for p in root.iterdir()}=={'source','binding.json','forest.json','contract.json','proof.json'},'portable complete file inventory')
    work.tick(); return dict(binding=expected,source=f.source,forest=f.identity,
        stats=dict(f.receipt['stats']),result=result,work=dict(work.counts),seconds=time.monotonic()-started)
