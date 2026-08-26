"""Builds and loads the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "src", "capi.mojo")
LIBRARY = os.path.join(ROOT, "dist", "libmojo-pot.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpot_emd": ([I] * 14 + [F], I),
    "mpot_sinkhorn": ([I] * 10 + [F, I, F, I, I], I),
    "mpot_greenkhorn": ([I] * 11 + [F, I, F], I),
    "mpot_dist": ([I] * 7, None),
    "mpot_emd_1d": ([I] * 9 + [F], I),
    "mpot_wasserstein_1d": ([I] * 6 + [F], F),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJOPOT_MOJO")
    if override:
        return override.split()
    executable = shutil.which("mojo")
    if executable:
        return [executable]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "mojo",
        ]
    raise BuildError("mojo not found; set MOJOPOT_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIBRARY)
        and os.path.getmtime(LIBRARY) >= os.path.getmtime(SOURCE)
    ):
        return LIBRARY
    os.makedirs(os.path.dirname(LIBRARY), exist_ok=True)
    command = _mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        SOURCE,
        "-o",
        LIBRARY,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0 or not os.path.exists(LIBRARY):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIBRARY


_library = None
_parallel_device = None
_parallel_ready = False


def lib() -> ctypes.CDLL:
    global _library, _parallel_device, _parallel_ready
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
        try:
            initialize = getattr(
                _library, "KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice"
            )
            initialize.argtypes = []
            initialize.restype = ctypes.c_void_p
            _parallel_device = initialize()
            _parallel_ready = bool(_parallel_device)
        except (AttributeError, OSError):
            _parallel_ready = False
    return _library


def parallel_ready() -> bool:
    lib()
    return _parallel_ready


def f64(value, *, copy: bool = False) -> np.ndarray:
    original = np.asarray(value)
    if np.issubdtype(original.dtype, np.complexfloating):
        raise TypeError("complex values cannot be represented by the float64 kernels")
    if (
        np.issubdtype(original.dtype, np.floating)
        and original.dtype.itemsize > np.dtype(np.float64).itemsize
    ):
        raise TypeError("floating-point inputs wider than float64 are not supported")
    if np.issubdtype(original.dtype, np.integer) and original.size:
        limit = 1 << 53
        if np.any(original > limit) or np.any(original < -limit):
            raise ValueError("integer inputs outside float64's exact range are not supported")
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    if array.dtype not in (np.dtype(np.float64), np.dtype(np.int64)):
        raise TypeError("FFI buffers must have dtype float64 or int64")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise RuntimeError("NumPy returned a null address for a non-empty FFI buffer")
    return address
