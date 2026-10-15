"""Closed ownership transfer of this request's newly generated stdlib source.

No historical/external source intake, links, copy fallback or deletion.
Partial failures stay in the request tree and its outer census.
"""
import os
from pathlib import Path
import stat
import time

from scoped_source.hz_request_store import POLICY, require


class OwnedSource:
    __slots__=('root','source','work','pid','inventory','seconds','inode','_live')
    def __init__(self): raise TypeError('fixed source issuer only')
    def __setattr__(self,k,v): raise AttributeError('owned source identity')
    def __delattr__(self,k): raise AttributeError('owned source identity')


def capture_owned(request_root, case, work):
    from scoped_source.direct_supervised import fixed_source, census
    root=Path(request_root)
    require(root.is_absolute() and root.resolve()==root and root.is_dir()
            and root.is_relative_to('/data1/Kane/MOE') and root!=Path('/data1/Kane/MOE')
            and not any(p.is_symlink() for p in (root,*root.parents)), 'owned request root')
    source_root=root/'source'; require(not source_root.exists(), 'fresh fixed source only')
    start=time.monotonic(); work.tick(); source=fixed_source(source_root,case); work.tick()
    inventory=census(source_root,work.deadline)
    for name in inventory['files']:
        require((source_root/name).stat().st_nlink==1, 'source no hardlink alias')
    token=object.__new__(OwnedSource)
    for k,v in dict(root=source_root,source=source,work=work,pid=os.getpid(),inventory=inventory,
                    seconds=time.monotonic()-start,inode=source_root.stat().st_ino,_live=True).items(): object.__setattr__(token,k,v)
    return token


def transfer(token, archive, *, fault='', member_limit=POLICY['members']):
    from scoped_source.direct_supervised import census
    require(type(token) is OwnedSource and token._live and token.pid==os.getpid()
            and token.work is archive.work and token.root.parent==archive.root.parent,
            'same request/source owner/Work')
    require(fault in ('','before_transfer','after_transfer'), 'transfer fault')
    require(type(member_limit) is int and 0<member_limit<=POLICY['members'], 'remaining transfer members')
    require(not token.root.is_symlink() and token.root.is_dir() and token.root.stat().st_ino==token.inode
            and not any(p.is_symlink() for p in token.root.parents), 'owned source directory inode/no alias')
    work=archive.work; work.tick(); current=census(token.root,work.deadline)
    require(current==token.inventory, 'owned source unchanged')
    dest=archive._path('package/source')
    require(not dest.exists() and not dest.is_symlink() and token.root.stat().st_dev==archive.root.stat().st_dev,
            'same-filesystem unused destination')
    require(archive.bytes+current['bytes']<=archive.limit-archive.reserve
            and len(archive.attempted)+len(current['files'])<=member_limit, 'transfer aggregate/member capacity')
    for name in current['files']:
        p=token.root/name; st=p.stat(follow_symlinks=False)
        require(stat.S_ISREG(st.st_mode) and st.st_nlink==1, 'source nonaliased regular member')
        require('package/source/'+name not in archive.attempted, 'transfer no overwrite')
    if fault=='before_transfer': raise RuntimeError('fixed before transfer')
    dest.parent.mkdir(parents=True,exist_ok=True); work.tick(); begin=time.monotonic()
    os.rename(token.root,dest)  # private request ownership; no cross-device fallback
    object.__setattr__(token,'_live',False)
    # Register all retained bytes before any cooperative failure/deadline point.
    for name,size in current['files'].items():
        path='package/source/'+name; h=current['hashes'][name]
        archive.attempted.add(path); archive.bytes+=size
        archive.entries[path]=dict(file=path,bytes=size,sha256=h)
        archive.prefixes[path]=dict(bytes=size,created=True,sha256=h)
        archive.events.append(dict(file=path,bytes=size,seconds=0.,bytes_fully_written=True,
                                   timely_published=False,complete=False,ownership_transfer=True))
    work.tick()
    require(not token.root.exists() and not dest.is_symlink() and dest.is_dir() and dest.stat().st_ino==token.inode
            and census(dest,work.deadline)==current, 'transfer identity/one source copy')
    require(all((dest/name).stat().st_nlink==1 for name in current['files']), 'transferred source no alias')
    if fault=='after_transfer': raise RuntimeError('fixed after transfer')
    for event in archive.events[-len(current['files']):]:
        event.update(timely_published=True,complete=True)
    work.tick()
    return dict(bytes=current['bytes'],members=len(current['files']),seconds=time.monotonic()-begin,
                copies=1,source=token.source)
