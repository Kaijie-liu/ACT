"""Stdlib-only checker for execution-budget journals, NOT a SAFE checker."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def check(path, *, identity=None, killed=False):
    lines = path.read_bytes().splitlines(keepends=True)
    tail = 0
    if lines and not lines[-1].endswith(b'\n'):
        if not killed: raise ValueError('truncated journal without kill')
        tail = len(lines.pop())
    previous = '0'*64
    events, stack, natives, properties = [], [], {}, {}
    work, total, complete = None, None, False
    clock = -1.
    for index, line in enumerate(lines):
        event = json.loads(line)
        sha = event.pop('sha256')
        raw = json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if (event['seq'] != index or event['previous'] != previous
                or hashlib.sha256(raw).hexdigest() != sha):
            raise ValueError('journal hash/sequence differs')
        previous = sha
        kind = event['kind']
        if index == 0:
            if kind != 'IDENTITY' or (identity is not None and event['identity'] != identity):
                raise ValueError('wrong identity')
        else:
            now = event['clock_elapsed']
            if not math.isfinite(now) or now < clock: raise ValueError('nonmonotonic clock')
            clock = now
        if complete: raise ValueError('events after WORK_COMPLETE')
        if kind == 'CONTRACT_INSTALLED':
            if work is not None or event['version'] != 2 or event['total'] != 300 or event['reserve'] != 5:
                raise ValueError('unsupported budget contract')
            total, work = event['total'], event['work_deadline']
            if work != total-event['reserve']: raise ValueError('reserve not subtracted')
        elif kind in ('LOCAL_BEGIN','PROPERTY_BEGIN'):
            if work is None or not math.isfinite(event['deadline']) or event['deadline'] > min([work]+[d for _,d in stack]):
                raise ValueError('local deadline expanded')
            stack.append((index, event['deadline']))
            if kind == 'PROPERTY_BEGIN':
                scope = event['scope']
                if (not scope or not scope['pairs'] or not scope['row']
                        or len({tuple(p) for p in scope['pairs']}) != len(scope['pairs'])
                        or any(len(p)!=2 or p!=sorted(set(p)) for p in scope['pairs'])
                        or not all(math.isfinite(x) for x in scope['row']+[scope['constant']])):
                    raise ValueError('invalid property scope')
                properties[index] = {'scope':scope, 'result':None, 'replay':None}
        elif kind in ('LOCAL_END','LOCAL_RAISE','PROPERTY_RESULT','PROPERTY_RAISE'):
            if not stack or stack.pop()[0] != event['token']: raise ValueError('unbalanced local scope')
            if kind == 'PROPERTY_RESULT':
                properties[event['token']]['result'] = event['result']
                if event['evidence_role'] != 'INTERMEDIATE_NOT_REQUEST_VERDICT':
                    raise ValueError('property promoted')
        elif kind == 'GRANT':
            if (work is None or not 0 < event['allocation'] <= work
                    or event['deadline'] > work+1e-9 or event['obligations'] < 1):
                raise ValueError('invalid grant')
        elif kind == 'NATIVE_READY':
            if (work is None or not math.isfinite(event['requested']) or event['requested'] <= 0
                    or event['deadline'] > min([work]+[d for _,d in stack])
                    or not math.isfinite(event['deadline'])):
                raise ValueError('invalid native deadline')
            natives[index] = {'ready':event, 'return':None}
        elif kind in ('NATIVE_RETURN','NATIVE_RAISE','NATIVE_SKIPPED'):
            token = event['token']
            if token not in natives or natives[token]['return'] is not None:
                raise ValueError('orphan/duplicate native return')
            ready = natives[token]['ready']
            if kind != 'NATIVE_SKIPPED':
                if (not all(math.isfinite(event[k]) for k in ('entered','effective'))
                        or not ready['clock_elapsed'] <= event['entered'] <= clock
                        or not .001 <= event['effective'] <= ready['requested']
                        or event['entered']+event['effective'] > ready['deadline']+1e-9):
                    raise ValueError('native allocation exceeded live deadline')
            natives[token]['return'] = event
        elif kind == 'PROPERTY_REPLAY':
            item = properties.get(event['token'])
            if item is None or item['result'] is None or item['replay'] is not None:
                raise ValueError('orphan/duplicate replay')
            item['replay'] = event['result']
        elif kind == 'WORK_COMPLETE':
            if stack or any(n['return'] is None for n in natives.values()):
                raise ValueError('completion with open calls')
            if event['status']=='SAFE' and clock >= work: raise ValueError('late SAFE')
            if event['status'] not in ('SAFE','UNSAFE','UNKNOWN','TIMEOUT'):
                raise ValueError('invalid completion status')
            complete = True
        elif kind not in ('IDENTITY','STATE'):
            raise ValueError('unrecognized journal event')
        events.append(event)
    if not events or work is None or (not complete and not killed):
        raise ValueError('incomplete non-killed journal')
    return {'status':'PASS', 'issues':[], 'events':len(events), 'work_complete':complete,
            'partial_tail_bytes':tail, 'open_scopes':[s for s,_ in stack],
            'native_calls':len(natives), 'native_unreturned':sum(n['return'] is None for n in natives.values()),
            'native_skipped':sum(n['return'] is not None and n['return']['kind']=='NATIVE_SKIPPED' for n in natives.values()),
            'properties':properties, 'journal_can_establish_SAFE':False,
            'limit':'Checks execution accounting, not network/HZ/LP/MILP soundness; complete terminal/package remains required.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--killed', action='store_true')
    args=parser.parse_args()
    try: result=check(args.journal,killed=args.killed)
    except Exception as exc: result={'status':'FAIL','issues':[str(exc)]}
    print(json.dumps(result,sort_keys=True))
    raise SystemExit(0 if result['status']=='PASS' else 1)
