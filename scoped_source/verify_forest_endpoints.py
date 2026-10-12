"""Relocatable stdlib source + complete endpoint check; no native imports."""
from pathlib import Path
import time

from scoped_source.factored_io import read
from scoped_source.hz_request_store import Work, compact, decode, digest, require, sha
from scoped_source.forest_endpoints import CheckedForest, check_proof


def verify(root,expected,*,deadline=None):
    started=time.monotonic(); work=Work(started+300 if deadline is None else deadline); root=Path(root)
    sha(expected)
    raw=read(root,'binding.json',8192,expected); binding=decode(raw)
    require(raw==compact(binding) and set(binding)=={'schema','source','forest','contract','proof'} and
            binding['schema']=='CAPTURED_FOREST_COMPLETE_BINDING_V1', 'complete binding fields')
    for name in ('source','forest','contract','proof'): sha(binding[name])
    forest=CheckedForest(read(root,'forest.json',128*2**20,binding['forest']),root/'source',binding['source'],
                         expected=binding['forest'],work=work)
    contract=decode(read(root,'contract.json',8192,binding['contract']))
    proof=decode(read(root,'proof.json',16*2**20,binding['proof']))
    result=check_proof(forest,contract,proof,expected_contract=binding['contract'],expected_proof=binding['proof'],work=work)
    require({p.name for p in root.iterdir()}=={'source','binding.json','forest.json','contract.json','proof.json'},
            'portable bundle complete inventory')
    work.tick(); return dict(binding=expected,source=forest.source,forest=forest.identity,
        stats=dict(forest.receipt['stats']),result=result,work=dict(work.counts),seconds=time.monotonic()-started)
