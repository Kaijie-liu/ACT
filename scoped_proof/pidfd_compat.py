"""Local Linux pidfd ABI bridge; no monkeypatch, numeric syscall or PID fallback.

Only null siginfo/zero flags and signal 0 or SIGKILL are supported. Loading this
module performs no native calls. The OS, libc and ctypes ABI remain trusted.
"""
import ctypes
import os
import signal
import sys


def _integer(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError('invalid '+name)


class _Libc:
    """Explicit ABI, also testable with a private local fake symbol table."""
    def __init__(self, library):
        self.open = library.pidfd_open
        self.send = library.pidfd_send_signal
        self.open.argtypes = [ctypes.c_int, ctypes.c_uint]
        self.open.restype = ctypes.c_int
        self.send.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        self.send.restype = ctypes.c_int

    @staticmethod
    def call(function, *args):
        ctypes.set_errno(0)
        result = function(*args)
        code = ctypes.get_errno()  # Capture before any further native call.
        if result < 0:
            if code <= 0:
                raise RuntimeError('pidfd failure without errno')
            raise OSError(code, os.strerror(code))
        return result

    def pidfd_open(self, pid, flags):
        return self.call(self.open, pid, flags)

    def pidfd_send_signal(self, fd, sig, info, flags):
        result = self.call(self.send, fd, sig, info, flags)
        if result != 0:
            raise RuntimeError('unexpected pidfd_send_signal return')


class Pidfd:
    def __init__(self):
        if sys.platform != 'linux':
            raise RuntimeError('Linux pidfd required')
        if callable(getattr(os, 'pidfd_open', None)) and callable(getattr(signal, 'pidfd_send_signal', None)):
            self._open = os.pidfd_open
            self._send = signal.pidfd_send_signal
            self.backend = 'PYTHON_PIDFD'
        else:
            # Resolve exported libc names through this interpreter's link map.
            # No architecture-dependent syscall numbers or fallback to kill().
            bridge = _Libc(ctypes.CDLL(None, use_errno=True))
            self._open = bridge.pidfd_open
            self._send = bridge.pidfd_send_signal
            self.backend = 'LIBC_PIDFD'

    def pidfd_open(self, pid, flags=0):
        _integer(pid, 1, 2**31-1, 'pid')
        _integer(flags, 0, 0, 'flags')
        fd = self._open(pid, flags)
        try:
            _integer(fd, 0, 2**31-1, 'returned fd')
            if os.get_inheritable(fd):
                raise RuntimeError('pidfd must be CLOEXEC')
        except BaseException:
            os.close(fd)
            raise
        return fd

    def pidfd_send_signal(self, fd, sig, info=None, flags=0):
        _integer(fd, 0, 2**31-1, 'fd')
        _integer(sig, 0, int(signal.SIGKILL), 'signal')
        if sig not in (0, int(signal.SIGKILL)) or info is not None:
            raise ValueError('only signal 0/SIGKILL and null siginfo')
        _integer(flags, 0, 0, 'flags')
        self._send(fd, sig, info, flags)

    def probe(self):
        """Non-destructive self probe of BOTH operations, before child creation.

        Future target-specific/resource failures still fail closed; this is not
        a guarantee that a later open/send cannot fail.
        """
        fd = self.pidfd_open(os.getpid())
        try:
            self.pidfd_send_signal(fd, 0)
        finally:
            os.close(fd)
        return self.backend
