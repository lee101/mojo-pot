# mojo-pot

`mojo-pot` is a standalone Mojo port of compute-heavy balanced optimal
transport solvers from [POT](https://pythonot.github.io/). It provides a
NumPy-facing Python package named `mojopot`; importing it as `ot` makes the
covered functions usable with POT-compatible names and signatures.

This is a focused port, not a replacement for all of POT. The numerical
kernels are compiled as one Mojo shared library, while Python handles input
validation, result objects, and POT-style dispatch.

## Covered API

- `emd` and `emd2`: exact balanced discrete transport using a
  successive-shortest-path min-cost-flow solver
- `emd_1d` and `wasserstein_1d`: monotone one-dimensional couplings and losses
- `sinkhorn` and `sinkhorn2`: classic Sinkhorn-Knopp scaling
- `bregman.sinkhorn_log`: log-domain Sinkhorn for small regularization
- `bregman.greenkhorn`: greedy coordinate scaling
- `dist`: squared-Euclidean and Euclidean pairwise cost matrices
- POT-style `lp` and `bregman` namespaces
- Empty `emd` histograms as uniform weights, marginal checking, transport
  logs, exact dual potentials, and the tested return conventions

The test suite compares every solver against POT 0.9.7.post1 on the same
inputs. Exact transport is checked by primal cost, marginals, dual feasibility,
and strong duality rather than assuming one plan when several plans are
optimal.

Not covered are batched target histograms, non-NumPy backends, warm starts,
verbose iteration output, custom keyword options, multi-process or
multi-threaded exact EMD, Sinkhorn stabilization and epsilon scaling,
unbalanced transport, barycenters, sliced or Gromov-Wasserstein solvers, GPU
execution, and POT's autodifferentiation support. Unsupported options raise
instead of being ignored. Kernel inputs are converted to contiguous float64;
complex values, floats wider than float64, and integers outside float64's
exact range are rejected rather than silently narrowed. Classic Sinkhorn and
pairwise distance evaluation run as a single serial pass: every one of their
inner loops is a bandwidth-bound reduction over the dense cost matrix, well
under two floating-point operations per byte moved, so a worker pool would add
launch overhead without adding arithmetic intensity.

No GPU path is provided because the benchmark targets are not suitable for
one: classic Sinkhorn's repeated matrix-vector passes and pairwise distance
evaluation have well under two floating-point operations per byte moved,
while exact EMD is dominated by branchy residual-graph traversal. Moving these
workloads to a GPU would add transfer and launch overhead without enough
arithmetic intensity to offset it.

## Install

The project pins the Mojo nightly used to build it and installs POT for parity
testing:

```bash
pixi install
pixi run build
```

The build writes `dist/libmojo-pot.so`. The Python wrapper also rebuilds a
missing or stale library on first import.

## Usage

Run this from the repository:

```bash
pixi run python - <<'PY'
import numpy as np
import mojopot as ot

x = np.array([[0.0], [1.0]])
y = np.array([[0.0], [2.0]])
a = np.array([0.4, 0.6])
b = np.array([0.5, 0.5])
M = ot.dist(x, y)

exact_plan = ot.emd(a, b, M)
regularized_plan = ot.sinkhorn(a, b, M, reg=0.1)

print(exact_plan)
print(ot.emd2(a, b, M))
print(regularized_plan.sum(axis=1))
PY
```

`import mojopot as ot` is the intended migration pattern. Calls in the covered
subset keep POT's parameter names, defaults, logs, and namespace layout.

## Benchmarks

Measured on 2026-08-26 with an Intel Xeon E5-2697 v4 at 2.30 GHz
(Linux x86_64). These are best-of-three wall-clock times from `pixi run bench`
on identical float64 inputs:

| case | mojo-pot | POT | speedup | result |
|---|---:|---:|---:|---|
| wasserstein_1d (200k x 180k) | 59.10 ms | 94.44 ms | 1.60x | faster |
| emd (64 x 64) | 3.81 ms | 0.55 ms | 0.14x | slower |
| sinkhorn (256 x 256) | 1.30 ms | 1.77 ms | 1.36x | faster |
| sinkhorn_log (128 x 128) | 14.81 ms | 39.30 ms | 2.65x | faster |
| greenkhorn (256 x 256) | 5.66 ms | 55.52 ms | 9.81x | faster |
| dist sqeuclidean (2k x 2k x 10) | 6.85 ms | 58.53 ms | 8.55x | faster |

POT's mature native network simplex still wins decisively on exact EMD. The
SIMD and cache-contiguous classic Sinkhorn kernel now beats POT for this
benchmark.
Mojo also does well where repeated elementwise reductions or coordinate
updates otherwise incur NumPy temporary and Python-loop overhead. Results
depend on matrix shape, regularization, convergence count, CPU, and BLAS
configuration; rerun `pixi run bench` on the target machine.

## How it works

`src/capi.mojo` is one compilation unit exporting a small C ABI. NumPy owns all
input, result, and scratch allocations. Contiguous row-major float64 buffers
cross `ctypes` as integer addresses and are reconstructed as mutable
`UnsafePointer` values inside the exported Mojo functions. No Mojo allocation
or ownership crosses the FFI boundary.

The exact solver treats the dense cost matrix as an implicit bipartite
residual network. A primal-dual shortest-path traversal uses reduced costs,
node potentials, and a bounded decrease-key heap while retaining reverse arcs
from the current plan. Each augmentation preserves feasibility and
reaches the linear-program optimum. The regularized solvers operate directly
on dense row-major costs. Classic Sinkhorn uses SIMD dot products, scalar tail
loops, and a transposed kernel scratch buffer so both row and column passes are
contiguous. Pairwise distance evaluation likewise uses SIMD feature reductions
with a scalar tail.
Log Sinkhorn uses stable log-sum-exp reductions, and Greenkhorn updates one
maximally violating marginal while maintaining row and column sums
incrementally. The one-dimensional solver sorts the supports in Python and
performs the monotone coupling scan in linear time in Mojo.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test task currently runs 33 numerical and behavioral parity cases against
the real POT package.
