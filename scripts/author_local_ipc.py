"""Short private IPC location and AF_UNIX-only Python socket policy."""
import os
from pathlib import Path
import socket
import tempfile


def install():
    base=Path('/data1/Kane/MOE/tmp')
    base.mkdir(exist_ok=True)
    path=tempfile.mkdtemp(prefix='re-',dir=base)
    os.environ['TMPDIR']=path
    tempfile.tempdir=path
    connect=socket.socket.connect
    create=socket.create_connection
    def local_only(sock,address):
        if sock.family != socket.AF_UNIX:
            raise RuntimeError('network connections prohibited in author control')
        return connect(sock,address)
    def forbidden(*args,**kwargs):
        raise RuntimeError('network connections prohibited in author control')
    socket.socket.connect=local_only
    socket.create_connection=forbidden
    return path,connect,create
