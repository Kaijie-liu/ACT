"""Nested inclusive/exclusive accounting; never sum inclusive parent/child costs."""
import time


class Timers:
    def __init__(self,*,clock=time.monotonic,tick=lambda:None):
        self.clock=clock;self.tick=tick;self.stack=[];self.values={}

    def call(self,name,fn,*a,**kw):
        self.tick();frame=[self.clock(),0.];self.stack.append(frame)
        try:result=fn(*a,**kw)
        finally:
            elapsed=self.clock()-frame[0];self.stack.pop()
            if self.stack:self.stack[-1][1]+=elapsed
            row=self.values.setdefault(name,{'calls':0,'inclusive_seconds':0.,'exclusive_seconds':0.})
            row['calls']+=1;row['inclusive_seconds']+=elapsed;row['exclusive_seconds']+=max(0.,elapsed-frame[1])
        self.tick();return result
