"""The clock the time budget runs on: wall-clock time, or this process's CPU time when
CHAMELEON_CPU_CLOCK=1 (eval.bench sets it), so a benchmark case running beside others in
parallel gets the work it would get alone instead of being cut short by the load."""

import os
import time


def now():
    return time.process_time() if os.environ.get("CHAMELEON_CPU_CLOCK") == "1" else time.monotonic()
