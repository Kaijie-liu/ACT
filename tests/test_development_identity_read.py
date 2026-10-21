"""Bounded initial-stat compatibility; no native workers, solver or model."""
import time
import unittest
from scoped_proof import owned_session as owned


def stat(*,pid=101,state='R',sid=101,start=55):
    return (f'{pid} (fixture) '+' '.join(
        [state,'1','101',str(sid)]+['0']*15+[str(start)])+'\n').encode()


class Reader:
    def __init__(self,*values):
        self.values=list(values);self.reads=0
    def pids(self):return iter((101,))
    def read(self,pid,name,cap):
        self.reads+=1
        value=self.values.pop(0)
        if isinstance(value,BaseException):raise value
        if callable(value):return value()
        return value


class InitialDevelopmentIdentityTests(unittest.TestCase):
    def scan(self,reader):
        return owned._observe(101,time.monotonic()+5,reader,
                              development_policy=owned.DEVELOPMENT_POLICY)

    def test_one_reread_uses_whole_valid_identity(self):
        for bad in (stat(state='RR'),stat(sid=-1),stat(start=-1)):
            r=Reader(bad,stat(start=77),b'VmRSS: 12 kB\n',stat(start=77))
            result=self.scan(r)
            self.assertEqual(result['live'],[101]);self.assertEqual(result['rss'],12288)
            self.assertEqual(r.reads,4)

    def test_disappearance_uses_existing_vanished_entry_rule(self):
        for error in (FileNotFoundError(),ProcessLookupError()):
            r=Reader(stat(sid=-1),error)
            self.assertEqual(self.scan(r)['live'],[]);self.assertEqual(r.reads,2)

    def test_persistent_invalid_and_bad_second_read_reject(self):
        for second in (stat(sid=-1),stat(pid=102),b'101 (short) R\n',PermissionError()):
            r=Reader(stat(sid=-1),second)
            with self.assertRaises(owned._ObservationFailure):self.scan(r)
            self.assertEqual(r.reads,2)

    def test_syntax_pid_length_permission_fail_without_retry(self):
        for first in (stat(pid=102),b'101 (short) R\n',b'x'*8193,PermissionError()):
            r=Reader(first)
            with self.assertRaises(owned._ObservationFailure):self.scan(r)
            self.assertEqual(r.reads,1)

    def test_valid_first_is_still_bound_to_confirmed_sid_and_start(self):
        for last in (stat(sid=102),stat(start=56),stat(sid=-1)):
            r=Reader(stat(),b'VmRSS: 12 kB\n',last)
            with self.assertRaises(owned._ObservationFailure):self.scan(r)
            self.assertEqual(r.reads,3)

    def test_recovered_foreign_identity_does_not_read_status(self):
        r=Reader(stat(sid=-1),stat(sid=102))
        self.assertEqual(self.scan(r)['live'],[]);self.assertEqual(r.reads,2)

    def test_default_and_cleanup_remain_strict(self):
        for call in (lambda r:owned._identity(101,r),
                     lambda r:owned._observe_cleanup(101,time.monotonic()+5,r)):
            r=Reader(stat(sid=-1))
            with self.assertRaises((ValueError,owned._ObservationFailure)):call(r)
            self.assertEqual(r.reads,1)

    def test_deadline_before_and_after_reread(self):
        r=Reader(stat(sid=-1))
        with self.assertRaises(TimeoutError):
            owned._identity(101,r,initial_development_deadline=time.monotonic()-1)
        self.assertEqual(r.reads,0)
        # A reader can consume its deadline. The second result must not pass.
        deadline=time.monotonic()+.01
        def delayed():
            time.sleep(.02);return stat()
        r=Reader(stat(sid=-1),delayed)
        with self.assertRaises(TimeoutError):
            owned._identity(101,r,initial_development_deadline=deadline)
        self.assertLessEqual(r.reads,2)


if __name__=='__main__':unittest.main()
