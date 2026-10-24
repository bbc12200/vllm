"""NVTX ranges for Nsight Systems. No-op when CUDA/NVTX is unavailable (e.g. Mac CPU tests).

NVTX push/pop are cheap C calls when no profiler is attached, so these stay in production.
"""
import contextlib
import functools

try:
    import torch

    torch.cuda.nvtx.range_push("nvtx_probe")
    torch.cuda.nvtx.range_pop()
    _push, _pop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop
except Exception:  # no CUDA build, no NVTX library, ...
    _push = _pop = None


@contextlib.contextmanager
def nvtx_range(name: str):
    if _push is None:
        yield
        return
    _push(name)
    try:
        yield
    finally:
        _pop()


def nvtx_annotate(name: str):
    def deco(fn):
        if _push is None:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _push(name)
            try:
                return fn(*args, **kwargs)
            finally:
                _pop()

        return wrapper

    return deco
