"""Linux owned descendant cleanup, including children with separate sessions."""
import os
from pathlib import Path
import signal


def info(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();parts=raw[raw.rfind(')')+2:].split()
        return {'pid':int(pid),'ppid':int(parts[1]),'start':int(parts[19]),'state':parts[0]}
    except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError,ValueError):return None


def descendants(root):
    table={int(p.name):info(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdecimal()}
    owned={root};changed=True
    while changed:
        new={pid for pid,v in table.items() if v and v['ppid'] in owned}
        changed=not new<=owned;owned|=new
    return [v for pid,v in table.items() if pid in owned and v]


def signal_if_same(record,sig):
    current=info(record['pid'])
    if current and current['start']==record['start']:
        try:os.kill(record['pid'],sig)
        except ProcessLookupError:pass


def stop_owned(root_record):
    """Stop spawning first, then kill only verified descendants/identities.

    Do not kill a broad process group or any other user's solver.
    """
    current=info(root_record['pid'])
    if not current or current['start']!=root_record['start']:return []
    signal_if_same(root_record,signal.SIGSTOP)
    records={r['pid']:r for r in descendants(root_record['pid'])}
    for _ in range(3):
        for record in records.values():signal_if_same(record,signal.SIGSTOP)
        found={r['pid']:r for r in descendants(root_record['pid'])}
        if found.keys()<=records.keys():break
        records.update(found)
    for pid,record in records.items():
        if pid!=root_record['pid']:signal_if_same(record,signal.SIGKILL)
    signal_if_same(root_record,signal.SIGKILL)
    return list(records.values())
