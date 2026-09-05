"""Serialize host state and Vulkan queue/pool use across transport clients."""

from functools import wraps


def serialized(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with self.runtime.lock:
            return function(self, *args, **kwargs)

    return wrapped
