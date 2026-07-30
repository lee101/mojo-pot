"""Benchmark mojo-pot against POT on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings

import numpy as np
import ot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojopot as mot  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def transport_problem(n, m, d, seed=0):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.normal(size=(n, d)))
    y = np.ascontiguousarray(rng.normal(size=(m, d)))
    a = np.ascontiguousarray(rng.random(n))
    b = np.ascontiguousarray(rng.random(m))
    a /= a.sum()
    b /= b.sum()
    M = np.ascontiguousarray(ot.dist(x, y))
    M /= M.max()
    return a, b, M, x, y


def machine():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def cases():
    rng = np.random.default_rng(6)
    u = np.ascontiguousarray(rng.normal(size=200_000))
    v = np.ascontiguousarray(rng.normal(size=180_000))
    u_weights = np.ascontiguousarray(rng.random(u.size))
    v_weights = np.ascontiguousarray(rng.random(v.size))
    u_weights /= u_weights.sum()
    v_weights /= v_weights.sum()
    yield (
        "wasserstein_1d (200k x 180k)",
        lambda: mot.wasserstein_1d(u, v, u_weights, v_weights, p=2),
        lambda: ot.wasserstein_1d(u, v, u_weights, v_weights, p=2),
    )

    a, b, M, _, _ = transport_problem(64, 64, 5, seed=1)
    yield (
        "emd (64 x 64)",
        lambda: mot.emd(a, b, M),
        lambda: ot.emd(a, b, M),
    )

    a, b, M, _, _ = transport_problem(256, 256, 6, seed=2)
    yield (
        "sinkhorn (256 x 256)",
        lambda: mot.sinkhorn(a, b, M, 0.1, warn=False),
        lambda: ot.sinkhorn(a, b, M, 0.1, warn=False),
    )

    a, b, M, _, _ = transport_problem(128, 128, 6, seed=3)
    yield (
        "sinkhorn_log (128 x 128)",
        lambda: mot.sinkhorn(
            a, b, M, 0.03, method="sinkhorn_log", numItermax=2000, warn=False
        ),
        lambda: ot.sinkhorn(
            a, b, M, 0.03, method="sinkhorn_log", numItermax=2000, warn=False
        ),
    )

    a, b, M, _, _ = transport_problem(256, 256, 6, seed=4)
    yield (
        "greenkhorn (256 x 256)",
        lambda: mot.sinkhorn(
            a, b, M, 0.1, method="greenkhorn", numItermax=10000, warn=False
        ),
        lambda: ot.sinkhorn(
            a, b, M, 0.1, method="greenkhorn", numItermax=10000, warn=False
        ),
    )

    _, _, _, x, y = transport_problem(2000, 2000, 10, seed=5)
    yield (
        "dist sqeuclidean (2k x 2k x 10)",
        lambda: mot.dist(x, y),
        lambda: ot.dist(x, y),
    )


def main():
    print(f"Machine: {machine()} ({platform.system()} {platform.machine()})")
    print()
    print("| case | mojo-pot | POT | speedup | result |")
    print("|---|---:|---:|---:|---|")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, mojo_function, pot_function in cases():
            mojo_function()
            pot_function()
            mojo_time = timeit(mojo_function)
            pot_time = timeit(pot_function)
            speedup = pot_time / mojo_time
            result = "faster" if speedup >= 1 else "slower"
            print(
                f"| {name} | {mojo_time * 1000:.2f} ms | "
                f"{pot_time * 1000:.2f} ms | {speedup:.2f}x | {result} |"
            )


if __name__ == "__main__":
    main()
