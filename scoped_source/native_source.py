"""Default-off byte acquisition of a declared source, not checkpoint capture.

The issuer copies, never moves, the external input. The existing builder and
independent checker still perform all mathematical source traversals.
"""
import hashlib
import os
from pathlib import Path
import stat
import time

from scoped_source.hz_request_store import POLICY, require, sha, digest
from scoped_source.tensor_rows import TensorRows
from scoped_source.shared_archive import OwnedSource
from scoped_source.wire_archive import Archive
from scoped_source.direct_supervised import census


def directory(path):
    """Pin each absolute directory component without following symlinks."""
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p, 'canonical source/request directory')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in p.parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = nxt
        return fd
    except BaseException:
        os.close(fd); raise


def regular(root_fd, name):
    parts = Path(name).parts
    require(type(name) is str and str(Path(name)) == name and len(parts) in (1, 2)
            and (len(parts) == 1 and name == 'manifest.json' or len(parts) == 2 and parts[0] == 't')
            and all(p not in ('', '.', '..') for p in parts), 'closed source member path')
    parent = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent); parent = nxt
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            s = os.fstat(fd)
            require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'unaliased regular source file')
            return fd, s
        except BaseException:
            os.close(fd); raise
    finally:
        os.close(parent)


def capture_declared_owned(request_root, source_root, source, work, *, enabled=False,
                           reserve=3*2**20):
    require(enabled is True, 'declared source intake opt-in')
    root, incoming = Path(request_root), Path(source_root)
    require(root.is_relative_to('/data1/Kane/MOE') and root != Path('/data1/Kane/MOE')
            and not root.is_relative_to(incoming) and not incoming.is_relative_to(root),
            'independent input and private request directories')
    require(type(reserve) is int and 8192 <= reserve < POLICY['total_bytes'], 'intake metadata reserve')
    sha(source); work.tick(); start = time.monotonic()
    request_fd = directory(root); source_fd = None
    try:
        source_fd = directory(incoming)
        outside = census(root, work.deadline)
        reader = TensorRows(incoming, source, deadline=work.deadline)
        require(reader.request['experts'] <= 8 and reader.request['classes'] <= 10,
                'unchanged complete endpoint dimensions')
        manifest, st = regular(source_fd, 'manifest.json')
        try:
            require(0 < st.st_size <= 4*2**20, 'bounded pinned manifest')
            parts = []; count = 0
            while count < st.st_size:
                work.tick(); chunk = os.read(manifest, min(65536, st.st_size-count))
                require(chunk, 'pinned manifest truncated')
                parts.append(chunk); count += len(chunk)
            require(not os.read(manifest, 1), 'pinned manifest grew')
            raw = b''.join(parts); del parts
            require(len(raw) == st.st_size <= 4*2**20 and digest(raw) == reader._manifest_digest,
                    'pinned manifest identity')
        finally:
            os.close(manifest)
        refs = {'manifest.json': (len(raw), digest(raw))}
        for tensor in reader._tensors.values():
            for c in tensor.chunks:
                require(c.file not in refs, 'source unique physical member')
                refs[c.file] = (c.bytes, c.sha256)
        require(set(refs) == set(reader._files) and
                outside['bytes']+sum(n for n, _ in refs.values())+reserve <= POLICY['total_bytes']
                and len(outside['files'])+len(refs)+8 <= POLICY['members'], 'whole intake capacity')
        acquired = Archive(root/'source', work,
            limit=POLICY['total_bytes']-outside['bytes'], reserve=reserve)
        inodes = set()
        for name, (size, expected) in sorted(refs.items()):
            work.visit('row_visits'); fd, before = regular(source_fd, name)
            created = False; count = 0; h = hashlib.sha256(); begin = time.monotonic()
            try:
                key = (before.st_dev, before.st_ino)
                require(key not in inodes and before.st_size == size, 'source inode/size binding')
                inodes.add(key)
                path = acquired._path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                acquired.attempted.add(name)
                with path.open('xb', buffering=0) as out:
                    created = True
                    while count < size:
                        work.tick(); chunk = os.read(fd, min(65536, size-count))
                        require(chunk, 'source truncated')
                        h.update(chunk); pos = 0
                        while pos < len(chunk):
                            work.tick(); n = out.write(memoryview(chunk)[pos:])
                            require(n and n > 0, 'source short copy')
                            count += n; acquired.bytes += n; pos += n
                        work.visit('coefficient_visits')
                    require(not os.read(fd, 1), 'source grew')
                    os.fsync(out.fileno()); work.tick()
                after = os.fstat(fd)
                require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                        and after.st_nlink == 1 and h.hexdigest() == expected, 'source changed/hash mismatch')
                acquired.entries[name] = dict(file=name, bytes=count, sha256=expected)
            finally:
                os.close(fd)
                acquired.prefixes[name] = dict(bytes=count, created=created,
                                               sha256=h.hexdigest() if count == size else None)
                acquired.events.append(dict(file=name, bytes=count, seconds=time.monotonic()-begin,
                                            complete=name in acquired.entries))
        acquired.audit()
        for directory_path, directory_fd in ((incoming,source_fd),(root,request_fd)):
            pinned = os.fstat(directory_fd); named = directory_path.stat(follow_symlinks=False)
            require((pinned.st_dev,pinned.st_ino) == (named.st_dev,named.st_ino)
                    and directory_path.resolve() == directory_path
                    and not any(p.is_symlink() for p in (directory_path,*directory_path.parents)),
                    'source/request directory identity unchanged')
        # Recheck copied schema/roster; deliberately do not add a third rows() proof pass.
        copy = TensorRows(root/'source', source, deadline=work.deadline)
        require(copy.request == reader.request and copy._files == reader._files, 'copied declaration binding')
        all_files = census(root, work.deadline)
        require(all_files['bytes']+reserve <= POLICY['total_bytes'], 'intake final reserve')
        inventory = census(root/'source', work.deadline)
        require(all((root/'source'/n).stat().st_nlink == 1 for n in inventory['files']),
                'copied source has no new hardlink alias')
        token = object.__new__(OwnedSource)
        for k, v in dict(root=root/'source', source=source, work=work, pid=os.getpid(),
                        inventory=inventory, seconds=time.monotonic()-start,
                        inode=(root/'source').stat().st_ino, _live=True).items():
            object.__setattr__(token, k, v)
        return token
    finally:
        if source_fd is not None: os.close(source_fd)
        os.close(request_fd)
