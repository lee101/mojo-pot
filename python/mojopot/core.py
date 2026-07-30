"""POT-compatible balanced optimal transport API."""

from __future__ import annotations

import warnings

import numpy as np

from ._lib import addr, f64, lib, parallel_ready


def _problem(a, b, M):
    cost = f64(M)
    if cost.ndim != 2 or not cost.size:
        raise ValueError("M must be a non-empty two-dimensional cost matrix")
    n, m = cost.shape
    source = np.full(n, 1.0 / n) if np.asarray(a).size == 0 else f64(a)
    target = np.full(m, 1.0 / m) if np.asarray(b).size == 0 else f64(b)
    if source.ndim != 1 or source.shape[0] != n:
        raise ValueError("a must have one entry per row of M")
    if target.ndim != 1 or target.shape[0] != m:
        raise ValueError("b must have one entry per column of M")
    if not np.all(np.isfinite(cost)):
        raise ValueError("M must contain only finite costs")
    if np.any(source < 0) or np.any(target < 0):
        raise ValueError("histograms cannot contain negative weights")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("histograms must contain only finite weights")
    source_mass = float(source.sum())
    target_mass = float(target.sum())
    if source_mass <= 0 or target_mass <= 0:
        raise ValueError("histograms must have positive mass")
    return source, target, cost


def _check_masses(source, target, check_marginals):
    source_mass = float(source.sum())
    target_mass = float(target.sum())
    tolerance = 1e-6 * max(source_mass, target_mass, 1.0)
    if check_marginals and abs(source_mass - target_mass) > tolerance:
        raise AssertionError(
            "a and b vector must have the same sum\n"
            f"Sum of a: {source_mass}, Sum of b: {target_mass}"
        )
    if source_mass != target_mass:
        target = target * (source_mass / target_mass)
    return target


def _dual_from_plan(cost, plan):
    n, m = cost.shape
    support = plan > 0
    component_u = np.full(n, -1, dtype=np.int64)
    component_v = np.full(m, -1, dtype=np.int64)
    u = np.zeros(n)
    v = np.zeros(m)
    component = 0
    for root in range(n + m):
        if root < n:
            if component_u[root] >= 0:
                continue
            component_u[root] = component
            queue = [(True, root)]
        else:
            index = root - n
            if component_v[index] >= 0:
                continue
            component_v[index] = component
            queue = [(False, index)]
        while queue:
            is_source, index = queue.pop()
            if is_source:
                for j in np.flatnonzero(support[index]):
                    if component_v[j] < 0:
                        component_v[j] = component
                        v[j] = cost[index, j] - u[index]
                        queue.append((False, j))
            else:
                for i in np.flatnonzero(support[:, index]):
                    if component_u[i] < 0:
                        component_u[i] = component
                        u[i] = cost[i, index] - v[index]
                        queue.append((True, i))
        component += 1

    shifts = np.zeros(component)
    for _ in range(component):
        changed = False
        for i in range(n):
            ci = component_u[i]
            for j in range(m):
                cj = component_v[j]
                bound = cost[i, j] - u[i] - v[j]
                candidate = shifts[cj] + bound
                if shifts[ci] > candidate:
                    shifts[ci] = candidate
                    changed = True
        if not changed:
            break
    u += shifts[component_u]
    v -= shifts[component_v]
    return u, v


def emd(
    a,
    b,
    M,
    numItermax=100000,
    log=False,
    center_dual=True,
    numThreads=1,
    check_marginals=True,
    potentials_init=None,
):
    """Solve balanced discrete transport exactly using min-cost flow."""
    if potentials_init is not None:
        raise NotImplementedError("potentials_init is not covered by mojo-pot")
    if numThreads != 1:
        raise NotImplementedError("multi-threaded exact EMD is not covered")
    if int(numItermax) != numItermax or numItermax <= 0:
        raise ValueError("numItermax must be a positive integer")
    source, target, cost = _problem(a, b, M)
    target = _check_masses(source, target, check_marginals)
    n, m = cost.shape
    plan = np.empty_like(cost)
    supply = np.empty(n)
    demand = np.empty(m)
    distance = np.empty(n + m)
    predecessor = np.empty(n + m, dtype=np.int64)
    potential = np.empty(n + m)
    visited = np.empty(n + m, dtype=np.int64)
    status = lib().mpot_emd(
        addr(source),
        addr(target),
        addr(cost),
        addr(plan),
        addr(supply),
        addr(demand),
        addr(distance),
        addr(predecessor),
        addr(potential),
        addr(visited),
        n,
        m,
        int(numItermax),
        np.finfo(np.float64).eps * max(float(source.sum()), 1.0) * 32,
    )
    warning = None
    result_code = 1
    if status < 0:
        if status != -(int(numItermax) + 1):
            raise RuntimeError("exact EMD kernel failed to find an augmenting path")
        warning = "numItermax reached before optimality"
        result_code = 3
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
    if not log:
        return plan
    transport_cost = float(np.sum(plan * cost))
    u, v = _dual_from_plan(cost, plan)
    if center_dual:
        shift = (
            float(np.dot(target, v)) - float(np.dot(source, u))
        ) / (float(source.sum()) + float(target.sum()))
        u += shift
        v -= shift
    return plan, {
        "cost": transport_cost,
        "u": u,
        "v": v,
        "warning": warning,
        "result_code": result_code,
    }


def emd2(
    a,
    b,
    M,
    processes=1,
    numItermax=100000,
    log=False,
    return_matrix=False,
    center_dual=True,
    numThreads=1,
    check_marginals=True,
    potentials_init=None,
):
    """Return the exact optimal transport cost."""
    if processes != 1:
        raise NotImplementedError("multi-process emd2 is not covered")
    plan, details = emd(
        a,
        b,
        M,
        numItermax=numItermax,
        log=True,
        center_dual=center_dual,
        numThreads=numThreads,
        check_marginals=check_marginals,
        potentials_init=potentials_init,
    )
    value = details["cost"]
    if return_matrix:
        details["G"] = plan
        return [value, details]
    return (value, details) if log else value


def _sinkhorn_run(
    a,
    b,
    M,
    reg,
    *,
    method,
    numItermax,
    stopThr,
    warn,
    warmstart,
):
    if not np.isfinite(reg) or reg <= 0:
        raise ValueError("reg must be finite and strictly positive")
    if not np.isfinite(stopThr) or stopThr < 0:
        raise ValueError("stopThr must be finite and non-negative")
    if int(numItermax) != numItermax or numItermax <= 0:
        raise ValueError("numItermax must be a positive integer")
    if warmstart is not None:
        raise NotImplementedError("warmstart is not covered by mojo-pot")
    source, target, cost = _problem(a, b, M)
    target = _check_masses(source, target, True)
    n, m = cost.shape
    plan = np.empty_like(cost)
    u = np.empty(n)
    v = np.empty(m)
    error = np.empty(1)
    normalized = method.lower().replace("-", "_")
    if normalized in {"sinkhorn", "sinkhorn_knopp"}:
        kernel_t = np.empty((m, n), dtype=np.float64)
        status = lib().mpot_sinkhorn(
            addr(source),
            addr(target),
            addr(cost),
            addr(plan),
            addr(u),
            addr(v),
            addr(error),
            addr(kernel_t),
            n,
            m,
            float(reg),
            int(numItermax),
            float(stopThr),
            0,
            int(parallel_ready()),
        )
        log_domain = False
    elif normalized in {"sinkhorn_log", "log"}:
        status = lib().mpot_sinkhorn(
            addr(source),
            addr(target),
            addr(cost),
            addr(plan),
            addr(u),
            addr(v),
            addr(error),
            0,
            n,
            m,
            float(reg),
            int(numItermax),
            float(stopThr),
            1,
            0,
        )
        log_domain = True
    elif normalized == "greenkhorn":
        rows = np.empty(n)
        cols = np.empty(m)
        status = lib().mpot_greenkhorn(
            addr(source),
            addr(target),
            addr(cost),
            addr(plan),
            addr(u),
            addr(v),
            addr(rows),
            addr(cols),
            addr(error),
            n,
            m,
            float(reg),
            int(numItermax),
            float(stopThr),
        )
        log_domain = False
    else:
        raise ValueError(
            "covered methods are 'sinkhorn', 'sinkhorn_log', and 'greenkhorn'"
        )
    if status < 0:
        raise FloatingPointError(
            f"{normalized} encountered a zero or non-finite scaling denominator"
        )
    unconverged = status >= int(numItermax) and error[0] > float(stopThr)
    if warn and unconverged:
        warnings.warn(
            "Sinkhorn did not converge; increase numItermax or reg",
            RuntimeWarning,
            stacklevel=3,
        )
    return plan, u, v, float(error[0]), status, log_domain, normalized


def sinkhorn(
    a,
    b,
    M,
    reg,
    method="sinkhorn",
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
    **kwargs,
):
    """Compute an entropically regularized transport plan."""
    if verbose:
        raise NotImplementedError("verbose iteration output is not covered")
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise NotImplementedError(f"unsupported Sinkhorn options: {names}")
    plan, u, v, error, status, log_domain, normalized = _sinkhorn_run(
        a,
        b,
        M,
        reg,
        method=method,
        numItermax=numItermax,
        stopThr=stopThr,
        warn=warn,
        warmstart=warmstart,
    )
    if not log:
        return plan
    if normalized == "greenkhorn":
        return plan, {"u": u, "v": v, "n_iter": max(status, 0)}
    details = {
        "err": [error],
        "niter": max(status - 1, 0),
        "u": np.exp(u) if log_domain else u,
        "v": np.exp(v) if log_domain else v,
    }
    if log_domain:
        details["log_u"] = u
        details["log_v"] = v
    return plan, details


def sinkhorn2(
    a,
    b,
    M,
    reg,
    method="sinkhorn",
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=False,
    warmstart=None,
    **kwargs,
):
    """Return the regularized plan's linear transport cost, as POT does."""
    plan, details = sinkhorn(
        a,
        b,
        M,
        reg,
        method=method,
        numItermax=numItermax,
        stopThr=stopThr,
        verbose=verbose,
        log=True,
        warn=warn,
        warmstart=warmstart,
        **kwargs,
    )
    value = np.float64(np.sum(plan * f64(M)))
    return (value, details) if log else value


def sinkhorn_log(
    a,
    b,
    M,
    reg,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
    **kwargs,
):
    return sinkhorn(
        a,
        b,
        M,
        reg,
        method="sinkhorn_log",
        numItermax=numItermax,
        stopThr=stopThr,
        verbose=verbose,
        log=log,
        warn=warn,
        warmstart=warmstart,
        **kwargs,
    )


def greenkhorn(
    a,
    b,
    M,
    reg,
    numItermax=10000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None,
):
    return sinkhorn(
        a,
        b,
        M,
        reg,
        method="greenkhorn",
        numItermax=numItermax,
        stopThr=stopThr,
        verbose=verbose,
        log=log,
        warn=warn,
        warmstart=warmstart,
    )


def dist(
    x1,
    x2=None,
    metric="sqeuclidean",
    p=2,
    w=None,
    backend="auto",
    nx=None,
    use_tensor=False,
):
    """Pairwise squared-Euclidean or Euclidean distances."""
    del p
    if backend != "auto" or nx is not None or use_tensor:
        raise NotImplementedError("custom backends and tensor output are not covered")
    if w is not None:
        raise NotImplementedError("weighted Minkowski distance is not covered")
    x = f64(x1)
    y = x if x2 is None else f64(x2)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x1 and x2 must be 2D arrays with the same feature count")
    if not x.size or not y.size:
        raise ValueError("x1 and x2 must be non-empty")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x1 and x2 must contain only finite values")
    result = np.empty((x.shape[0], y.shape[0]), dtype=np.float64)
    lib().mpot_dist(
        addr(x),
        addr(y),
        addr(result),
        x.shape[0],
        y.shape[0],
        x.shape[1],
    )
    result[result < 0] = 0
    if metric == "sqeuclidean":
        return result
    if metric == "euclidean":
        np.sqrt(result, out=result)
        return result
    raise NotImplementedError("covered metrics are 'sqeuclidean' and 'euclidean'")


def emd_1d(
    x_a,
    x_b,
    a=None,
    b=None,
    metric="sqeuclidean",
    p=1.0,
    dense=True,
    log=False,
    check_marginals=True,
):
    """Return the monotone optimal coupling between two 1D measures."""
    if not dense:
        raise NotImplementedError("sparse emd_1d output is not covered")
    if metric not in {"sqeuclidean", "euclidean", "cityblock", "minkowski"}:
        raise ValueError(f"Unknown metric '{metric}'.")
    x = f64(x_a).reshape(-1)
    y = f64(x_b).reshape(-1)
    if not x.size or not y.size:
        raise ValueError("x_a and x_b must be non-empty")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x_a and x_b must contain only finite values")
    source = (
        np.full(x.size, 1.0 / x.size)
        if a is None or np.asarray(a).size == 0
        else f64(a)
    )
    target = (
        np.full(y.size, 1.0 / y.size)
        if b is None or np.asarray(b).size == 0
        else f64(b)
    )
    if source.ndim != 1 or source.size != x.size:
        raise ValueError("a must have one weight per x_a location")
    if target.ndim != 1 or target.size != y.size:
        raise ValueError("b must have one weight per x_b location")
    if np.any(source < 0) or np.any(target < 0):
        raise ValueError("histograms cannot contain negative weights")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("histograms must contain only finite weights")
    if source.sum() <= 0 or target.sum() <= 0:
        raise ValueError("histograms must have positive mass")
    target = _check_masses(source, target, check_marginals)
    order_a = np.ascontiguousarray(np.argsort(x, kind="stable"), dtype=np.int64)
    order_b = np.ascontiguousarray(np.argsort(y, kind="stable"), dtype=np.int64)
    sorted_a = np.ascontiguousarray(source[order_a])
    sorted_b = np.ascontiguousarray(target[order_b])
    plan = np.empty((x.size, y.size), dtype=np.float64)
    path_a = np.empty(x.size + y.size, dtype=np.int64)
    path_b = np.empty(x.size + y.size, dtype=np.int64)
    count = lib().mpot_emd_1d(
        addr(sorted_a),
        addr(sorted_b),
        addr(order_a),
        addr(order_b),
        addr(plan),
        addr(path_a),
        addr(path_b),
        x.size,
        y.size,
        np.finfo(np.float64).eps * 32,
    )
    if not log:
        return plan
    distances = np.abs(x[:, None] - y[None, :])
    if metric == "sqeuclidean":
        costs = distances * distances
    elif metric in {"euclidean", "cityblock"}:
        costs = distances
    elif metric == "minkowski":
        costs = distances ** float(p)
    return plan, {
        "cost": float(np.sum(plan * costs)),
        "perms_x_a": path_a[:count].copy(),
        "perms_x_b": path_b[:count].copy(),
    }


def wasserstein_1d(
    u_values,
    v_values,
    u_weights=None,
    v_weights=None,
    p=1,
    require_sort=True,
    return_plans=False,
):
    """Compute the p-Wasserstein loss for unbatched 1D measures."""
    if return_plans:
        raise NotImplementedError("return_plans is not covered")
    u = f64(u_values)
    v = f64(v_values)
    if u.ndim != 1 or v.ndim != 1:
        raise NotImplementedError("batched wasserstein_1d is not covered")
    if not u.size or not v.size:
        raise ValueError("value arrays must be non-empty")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("value arrays must contain only finite values")
    if not np.isfinite(p) or p <= 0:
        raise ValueError("p must be finite and strictly positive")
    a = (
        np.full(u.size, 1.0 / u.size)
        if u_weights is None
        else f64(u_weights)
    )
    b = (
        np.full(v.size, 1.0 / v.size)
        if v_weights is None
        else f64(v_weights)
    )
    if a.ndim != 1 or a.size != u.size or b.ndim != 1 or b.size != v.size:
        raise ValueError("weight arrays must match their value arrays")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("weights cannot be negative")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("weights must contain only finite values")
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("weights must have positive mass")
    b = _check_masses(a, b, True)
    if require_sort:
        order_u = np.argsort(u, kind="stable")
        order_v = np.argsort(v, kind="stable")
        sorted_u = np.ascontiguousarray(u[order_u])
        sorted_v = np.ascontiguousarray(v[order_v])
        sorted_a = np.ascontiguousarray(a[order_u])
        sorted_b = np.ascontiguousarray(b[order_v])
    else:
        sorted_u = np.ascontiguousarray(u)
        sorted_v = np.ascontiguousarray(v)
        sorted_a = np.array(a, order="C", copy=True)
        sorted_b = np.array(b, order="C", copy=True)
    return np.float64(
        lib().mpot_wasserstein_1d(
            addr(sorted_u),
            addr(sorted_v),
            addr(sorted_a),
            addr(sorted_b),
            u.size,
            v.size,
            float(p),
        )
    )
