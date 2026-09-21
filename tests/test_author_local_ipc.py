import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from author_local_ipc import install


class IPC(unittest.TestCase):
    def test_short_local_socket_works_network_rejects(self):
        before=tempfile.tempdir
        old=os.environ.get('TMPDIR')
        path,connect,create=install()
        endpoint=str(Path(path)/'listener')
        try:
            self.assertLess(len(endpoint.encode()),100)
            with socket.socket(socket.AF_UNIX) as server:
                server.bind(endpoint);server.listen(1)
                with socket.socket(socket.AF_UNIX) as client:client.connect(endpoint)
            with socket.socket(socket.AF_INET) as net:
                with self.assertRaises(RuntimeError):net.connect(('127.0.0.1',1))
        finally:
            socket.socket.connect=connect;socket.create_connection=create
            tempfile.tempdir=before
            if old is None:os.environ.pop('TMPDIR',None)
            else:os.environ['TMPDIR']=old
            Path(endpoint).unlink(missing_ok=True)
            Path(path).rmdir()


if __name__=='__main__':unittest.main()
