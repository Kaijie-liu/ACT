"""Actual successful schema deltas, not real-model admission or a heap bound."""
from scripts.assess_paged_result_contract import schemas, length, Size, array
from scoped_source.direct_method import method
from scoped_source.direct_lifecycle_package import code_files
from scoped_source.direct_supervised import FILES


def assess():
    s=schemas()
    # The symbolic proposal shares nested method/snapshot objects at every copy.
    m=s['manifest']['method']; m.clear(); m.update(method('pair_scoped',1,shared=True,paged_results=True))
    s['manifest']['schema']='PAGED_LIFECYCLE_MANIFEST_V1'
    s['bundle']['schema']='PAGED_LIFECYCLE_BUNDLE_V1'
    c=s['candidate_report']; c['schema']='PAGED_LIFECYCLE_REPORT_V1'
    c['result']=c.pop('summary')
    r=c['replay']; r['result']=r.pop('summary')
    r.update(result_context=s['context'],result_pages_read=32)
    c['snapshots']['result_stream']=dict(pages=32,rows=252,page_bytes=array(32,65536),index_bytes=5431)
    s['invocation']['schema']='PAGED_SUPERVISED_INVOCATION_V1'
    s['invocation']['spec']['schema']='PAGED_SUPERVISED_STUB_V1'
    p=s['parent_receipt']; p['schema']='PAGED_SUPERVISED_RECEIPT_V1'
    p['accepted']['result']=p['accepted'].pop('summary')
    names=('contract','manifest','bundle','inner_terminal','candidate_report','invocation','parent_receipt')
    sizes={n:length(s[n]) for n in names}
    sizes.update(result_pages=s['page_bytes'],result_index=length(s['index']),progress=s['progress'],worker_log=0)
    assert len(code_files(m))==s['code_files']==28
    assert len(FILES)==s['implementation_files']==47
    return dict(schema='PAGED_IMPLEMENTED_SUCCESS_LAYOUT_V1',members_upper=sizes,metadata_upper=sum(sizes.values()),
                page_upper=length(s['page']),index_upper=length(s['index']),
                code_files=28,implementation_files=47,artifact_files=s['artifact_files'],
                whole_request_files=s['whole_request_files'],bundle_members=s['bundle_members'],
                page_read_passes_online=10,real_model_admitted=False,heap_bound_closed=False,
                hypotheses=['existing symbolic uint64 counters and 24-character clock encoding',
                            'generated safe ASCII paths at most 200 characters',
                            'successful complete layout only; bounded failures are separate',
                            'Work retains its old metric, not all arithmetic or I/O operations'])


if __name__=='__main__':
    import json
    print(json.dumps(assess(),sort_keys=True))
