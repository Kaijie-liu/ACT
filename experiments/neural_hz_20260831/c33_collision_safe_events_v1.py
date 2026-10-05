"""Keep event-stage time and enclosing worker time under distinct keys."""
import math


def record(worker_elapsed_s,event):
    if not math.isfinite(worker_elapsed_s) or worker_elapsed_s<0:
        raise ValueError('invalid enclosing worker time')
    if type(event) is not dict or 'worker_elapsed_s' in event:
        raise ValueError('unregistered event or reserved enclosing-clock collision')
    return {'worker_elapsed_s':worker_elapsed_s,**event}
