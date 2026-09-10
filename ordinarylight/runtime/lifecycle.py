"""Opt-in host timings for expensive renderer lifecycle operations."""

import json
import logging
import time


logger = logging.getLogger("ordinarylight.lifecycle")


def report(event, **details):
    if logger.isEnabledFor(logging.INFO):
        logger.info("%s", json.dumps({"event": event, **details}))


class LifecycleTimer:
    def __init__(self, event, **details):
        self.event, self.details = event, details
        self.started = self.previous = time.perf_counter()
        self.phases = {}

    def mark(self, phase):
        now = time.perf_counter()
        self.phases[phase + "_ms"] = (now - self.previous) * 1000.0
        self.previous = now

    def finish(self):
        report(self.event, **self.details, **self.phases,
               total_ms=(time.perf_counter() - self.started) * 1000.0)


def timed_call(event, callback, *args, lifecycle_details=None, **kwargs):
    timer = LifecycleTimer(event, **(lifecycle_details or {}))
    try:
        return callback(*args, **kwargs)
    finally:
        timer.finish()
