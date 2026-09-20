"""Build a NEW compensated Conv/ReLU prefix, never retrofit an old LP proof."""
import argparse
import json
from pathlib import Path
import shutil
import time

from router_source.capture import ROOT,sha
from router_source.build import save
from router_source.checker import inputs,tensor
from upstream_source.checker import hz,conv_operator
from source_enclosure.format import identity
from source_enclosure.produce import box,redundant_guards,affine,relu,join

OLD=ROOT/'data/moe/results/upstream_source_conv98_20260920_v1/relocated'
OLD_HASH='bdf39f969664a5e1b2b49eb16ad7da1872bc36aa6fc1351a601c14510079c6d9'


def build(root,old=OLD,old_hash=OLD_HASH):
    start=time.monotonic()
    if sha(old/'manifest.json')!=old_hash:raise ValueError('old source bundle identity')
    parent=json.loads((old/'manifest.json').read_bytes())
    for name,h in parent['files'].items():
        path=(old/name).resolve()
        if not path.is_relative_to(old.resolve()) or sha(path)!=h:raise ValueError('old source changed')
    destination=root/'relocated';destination.mkdir();pair=parent['pair']
    names=['router_source.json','router_proof.json','router_hz.json','input_hz.json','experts.json']
    names += [f'expert{i}_conv0.json' for i in pair]
    for name in names:shutil.copyfile(old/name,destination/name)
    source=json.loads((destination/'router_source.json').read_bytes())
    shape,_,_,image=inputs(source,parent['request'])
    metadata=json.loads((destination/'experts.json').read_bytes())
    if [v['expert'] for v in metadata]!=pair:raise ValueError('expert coverage')
    timings={};states={};certificates={}
    def store_state(name,value):
        begin=time.monotonic();save(destination/(name+'.state.json'),value)
        timings['state_serialization_seconds']=timings.get('state_serialization_seconds',0)+time.monotonic()-begin
        states[name]=identity(value)
    begin=time.monotonic();entry=box(image['lower'],image['upper']);timings['input_seconds']=time.monotonic()-begin
    store_state('input',entry)
    router=hz(json.loads((destination/'router_hz.json').read_bytes()))
    begin=time.monotonic();guarded=redundant_guards(entry,router['Auc'],router['Aub'],router['ub'])
    timings['guard_seconds']=time.monotonic()-begin;store_state('guard',guarded)
    ends=[]
    for item in metadata:
        i=item['expert'];tag=f'expert{i}/conv0';begin=time.monotonic()
        wi,w=tensor(item['weight']);_,b=tensor(item['bias'])
        op,b=conv_operator(shape,item['graph'],w,wi['shape'],b)
        nominal=json.loads((destination/item['output_file']).read_bytes())
        output,proof=affine(guarded,op,b,nominal,tag)
        timings[f'expert{i}_affine_seconds']=time.monotonic()-begin
        name=f'expert{i}_affine';store_state(name,output);certificates[name]=proof
        begin=time.monotonic();end,proof=relu(output,f'expert{i}/relu1')
        timings[f'expert{i}_relu_seconds']=time.monotonic()-begin
        name=f'expert{i}_relu';store_state(name,end);certificates[name]=proof;ends.append(end)
    begin=time.monotonic();result,proof=join(guarded,*ends);timings['join_seconds']=time.monotonic()-begin
    store_state('join',result);certificates['join']=proof
    save(destination/'trace.json',{'schema':'COMPENSATED_PREFIX_TRACE_V1','states':states,'certificates':certificates})
    for src,dst in [('source_enclosure/format.py','proof_format.py'),('source_enclosure/check.py','step_check.py'),
                    ('source_enclosure/verify.py','verify_prefix.py'),('upstream_source/checker.py','local_check.py'),
                    ('router_source/checker.py','router_check.py')]:shutil.copyfile(ROOT/src,destination/dst)
    manifest={'schema':'COMPENSATED_EXPERT_PREFIX_BUNDLE_V1','request':parent['request'],'pair':pair,
        'endpoint':'TWO_EXPERTS_AFTER_FIRST_CONV_RELU_NOT_CLASSIFICATION',
        'source_manifest_sha256':old_hash,'source_files':{n:parent['files'][n] for n in names},
        'files':{p.name:sha(p) for p in sorted(destination.iterdir())}}
    save(destination/'manifest.json',manifest)
    save(root/'generation.json',{'manifest_sha256':sha(destination/'manifest.json'),
        'whole_build_seconds_before_publication':time.monotonic()-start,'timings':timings,
        'bundle_bytes':sum(p.stat().st_size for p in destination.iterdir()),
        'native_solver_calls':0,'network_forward_calls':0,'checkpoint_loads':0,
        'new_complete_network_propagations':0,'new_compensated_prefixes':2,
        'old_LP_certificates_used':0,'scope':'Saved source bytes to NEW input/guard/Conv/ReLU/shared-frame enclosure. Not whole-request verification time.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    build(p.parse_args().root.resolve())
