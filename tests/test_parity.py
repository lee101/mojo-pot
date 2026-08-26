"""Numerical and behavioral parity with POT 0.9."""

import inspect

import numpy as np
import pytest

ot = pytest.importorskip("ot")

import mojopot as mot


def problem(n=7, m=9, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = rng.normal(size=(m, 3))
    a = rng.random(n)
    b = rng.random(m)
    a /= a.sum()
    b /= b.sum()
    M = ot.dist(x, y)
    M /= M.max()
    return a, b, M


def assert_marginals(plan, a, b, atol=1e-9):
    assert np.allclose(plan.sum(axis=1), a, atol=atol)
    assert np.allclose(plan.sum(axis=0), b, atol=atol)
    assert np.all(plan >= -1e-14)


@pytest.mark.parametrize("shape", [(2, 3), (5, 7), (8, 6)])
def test_emd_random_parity(shape):
    a, b, M = problem(*shape, seed=sum(shape))
    ours = mot.emd(a, b, M)
    theirs = ot.emd(a, b, M)
    assert_marginals(ours, a, b)
    assert np.sum(ours * M) == pytest.approx(np.sum(theirs * M), abs=2e-12)


def test_emd_known_transport_plan():
    a = np.array([0.4, 0.6])
    b = np.array([0.5, 0.5])
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    expected = np.array([[0.4, 0.0], [0.1, 0.5]])
    assert np.allclose(mot.emd(a, b, M), expected, atol=1e-15)
    assert mot.emd2(a, b, M) == pytest.approx(0.1)


def test_emd_heap_path_on_dense_problem():
    a, b, M = problem(24, 27, seed=62)
    ours = mot.emd(a, b, M)
    theirs = ot.emd(a, b, M)
    assert_marginals(ours, a, b)
    assert np.sum(ours * M) == pytest.approx(np.sum(theirs * M), abs=2e-12)


def test_emd_empty_histograms_mean_uniform():
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.array_equal(mot.emd([], [], M), ot.emd([], [], M))


def test_emd_log_has_optimal_dual_potentials():
    a, b, M = problem(6, 8, seed=12)
    plan, details = mot.emd(a, b, M, log=True)
    primal = np.sum(plan * M)
    dual = np.dot(a, details["u"]) + np.dot(b, details["v"])
    assert details.keys() == {"cost", "u", "v", "warning", "result_code"}
    assert details["cost"] == pytest.approx(primal, abs=1e-12)
    assert dual == pytest.approx(primal, abs=1e-11)
    assert np.all(details["u"][:, None] + details["v"][None, :] <= M + 1e-12)
    _, reference = ot.emd(a, b, M, log=True)
    assert details["cost"] == pytest.approx(reference["cost"], abs=1e-12)


def test_emd2_log_and_return_matrix_contract():
    a, b, M = problem(4, 5, seed=8)
    value, details = mot.emd2(a, b, M, log=True)
    assert value == details["cost"]
    value2, details2 = mot.emd2(a, b, M, return_matrix=True)
    assert value2 == pytest.approx(value)
    assert_marginals(details2["G"], a, b)


@pytest.mark.parametrize(
    ("metric", "p"),
    [("sqeuclidean", 1.0), ("euclidean", 1.0), ("minkowski", 3.0)],
)
def test_emd_1d_parity(metric, p):
    x = np.array([2.0, -1.0, 0.5, 4.0])
    y = np.array([3.0, 0.0, -2.0])
    a = np.array([0.1, 0.3, 0.4, 0.2])
    b = np.array([0.5, 0.2, 0.3])
    ours, ours_log = mot.emd_1d(x, y, a, b, metric=metric, p=p, log=True)
    theirs, theirs_log = ot.emd_1d(x, y, a, b, metric=metric, p=p, log=True)
    assert np.allclose(ours, theirs, atol=1e-15)
    assert ours_log["cost"] == pytest.approx(theirs_log["cost"], abs=1e-14)
    assert np.array_equal(ours_log["perms_x_a"], theirs_log["perms_x_a"])
    assert np.array_equal(ours_log["perms_x_b"], theirs_log["perms_x_b"])


@pytest.mark.parametrize("p", [1, 2, 3])
def test_wasserstein_1d_parity(p):
    x = np.array([2.0, -1.0, 0.5, 4.0])
    y = np.array([3.0, 0.0, -2.0])
    a = np.array([0.1, 0.3, 0.4, 0.2])
    b = np.array([0.5, 0.2, 0.3])
    assert mot.wasserstein_1d(x, y, a, b, p=p) == pytest.approx(
        ot.wasserstein_1d(x, y, a, b, p=p), abs=1e-14
    )


def test_marginal_validation_matches_pot():
    a, b, M = problem(3, 4)
    b *= 0.5
    with pytest.raises(AssertionError, match="same sum"):
        mot.emd(a, b, M)
    plan = mot.emd(a, b, M, check_marginals=False)
    assert_marginals(plan, a, b * 2)


@pytest.mark.parametrize("method", ["sinkhorn", "sinkhorn_log", "greenkhorn"])
def test_sinkhorn_plan_parity(method):
    a, b, M = problem(8, 11, seed=30)
    options = dict(method=method, numItermax=20000, stopThr=1e-11, warn=False)
    ours = mot.sinkhorn(a, b, M, 0.15, **options)
    theirs = ot.sinkhorn(a, b, M, 0.15, **options)
    assert_marginals(ours, a, b, atol=2e-9)
    assert np.allclose(ours, theirs, atol=2e-9, rtol=2e-8)


def test_sinkhorn2_parity_and_log():
    a, b, M = problem(9, 7, seed=5)
    ours, details = mot.sinkhorn2(a, b, M, 0.1, log=True)
    theirs = ot.sinkhorn2(a, b, M, 0.1)
    assert ours == pytest.approx(theirs, rel=2e-9, abs=2e-11)
    assert {"err", "niter", "u", "v"} <= details.keys()


def test_sinkhorn_simd_tail_parity():
    a, b, M = problem(13, 17, seed=51)
    options = dict(numItermax=20000, stopThr=1e-11, warn=False)
    ours = mot.sinkhorn(a, b, M, 0.12, **options)
    theirs = ot.sinkhorn(a, b, M, 0.12, **options)
    assert np.allclose(ours, theirs, atol=2e-9, rtol=2e-8)


def test_sinkhorn_parallel_threshold():
    n = 1024
    a = np.full(n, 1.0 / n)
    M = np.zeros((n, n))
    plan = mot.sinkhorn(a, a, M, 1.0, warn=False)
    assert_marginals(plan, a, a, atol=1e-14)


def test_log_sinkhorn_stays_finite_at_small_regularization():
    a, b, M = problem(6, 7, seed=14)
    ours = mot.bregman.sinkhorn_log(
        a, b, M, 0.002, numItermax=20000, stopThr=1e-10, warn=False
    )
    theirs = ot.bregman.sinkhorn_log(
        a, b, M, 0.002, numItermax=20000, stopThr=1e-10, warn=False
    )
    assert np.all(np.isfinite(ours))
    assert_marginals(ours, a, b, atol=2e-8)
    assert np.sum(ours * M) == pytest.approx(np.sum(theirs * M), abs=2e-8)


def test_zero_weight_histogram_entries_log_and_greenkhorn():
    a = np.array([0.0, 0.3, 0.7])
    b = np.array([0.4, 0.0, 0.6])
    M = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
    for method in ("sinkhorn_log", "greenkhorn"):
        plan = mot.sinkhorn(
            a, b, M, 0.2, method=method, numItermax=20000, warn=False
        )
        assert_marginals(plan, a, b, atol=2e-8)


def test_bregman_and_lp_namespaces():
    assert mot.lp.emd is mot.emd
    assert mot.lp.emd2 is mot.emd2
    assert mot.bregman.sinkhorn is mot.sinkhorn
    assert mot.bregman.sinkhorn2 is mot.sinkhorn2
    assert mot.bregman.greenkhorn is mot.greenkhorn


def test_public_signatures_match_covered_pot_functions():
    for name in (
        "emd",
        "emd2",
        "emd_1d",
        "wasserstein_1d",
        "sinkhorn",
        "sinkhorn2",
        "dist",
    ):
        assert inspect.signature(getattr(mot, name)) == inspect.signature(getattr(ot, name))
    assert inspect.signature(mot.bregman.sinkhorn_log) == inspect.signature(
        ot.bregman.sinkhorn_log
    )
    assert inspect.signature(mot.bregman.greenkhorn) == inspect.signature(
        ot.bregman.greenkhorn
    )


@pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean"])
def test_dist_parity(metric):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(17, 5))
    y = rng.normal(size=(13, 5))
    assert np.allclose(mot.dist(x, y, metric=metric), ot.dist(x, y, metric=metric))


def test_dist_x2_defaults_to_x1():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(10, 4))
    ours = mot.dist(x)
    assert np.allclose(ours, ot.dist(x))
    assert np.array_equal(np.diag(ours), np.zeros(10))


def test_dist_parallel_threshold_with_simd_tail():
    n = 1024
    x = np.zeros((n, 3))
    y = np.ones((n, 3))
    result = mot.dist(x, y)
    assert np.array_equal(result, np.full((n, n), 3.0))


def test_input_validation_and_uncovered_options():
    a, b, M = problem(3, 4)
    with pytest.raises(ValueError, match="strictly positive"):
        mot.sinkhorn(a, b, M, 0)
    with pytest.raises(ValueError, match="covered methods"):
        mot.sinkhorn(a, b, M, 0.1, method="sinkhorn_stabilized")
    with pytest.raises(NotImplementedError, match="warmstart"):
        mot.sinkhorn(a, b, M, 0.1, warmstart=(a, b))
    with pytest.raises(NotImplementedError, match="unsupported Sinkhorn options"):
        mot.sinkhorn(a, b, M, 0.1, unsupported=True)
    with pytest.raises(NotImplementedError, match="verbose"):
        mot.sinkhorn(a, b, M, 0.1, verbose=True)
    with pytest.raises(NotImplementedError, match="potentials_init"):
        mot.emd(a, b, M, potentials_init=(a, b))
    with pytest.raises(NotImplementedError, match="multi-threaded"):
        mot.emd(a, b, M, numThreads=2)
    with pytest.raises(NotImplementedError, match="custom backends"):
        mot.dist(np.ones((2, 1)), backend="numpy")
    with pytest.raises(ValueError, match="negative"):
        mot.emd(-a, b, M)


def test_ffi_boundary_rejects_unsafe_values_and_dtypes():
    a, b, M = problem(3, 4)
    with pytest.raises(ValueError, match="finite and strictly positive"):
        mot.sinkhorn(a, b, M, np.nan)
    with pytest.raises(ValueError, match="finite and non-negative"):
        mot.sinkhorn(a, b, M, 0.1, stopThr=np.inf)
    with pytest.raises(TypeError, match="complex"):
        mot.emd(a, b, M.astype(np.complex128))
    if hasattr(np, "float128"):
        with pytest.raises(TypeError, match="wider than float64"):
            mot.dist(np.ones((2, 2), dtype=np.float128))
    with pytest.raises(ValueError, match="exact range"):
        mot.dist(np.array([[2**54]], dtype=np.int64))


def test_1d_and_distance_validation_before_ffi():
    with pytest.raises(ValueError, match="finite values"):
        mot.dist([[0.0, np.inf]])
    with pytest.raises(ValueError, match="finite values"):
        mot.emd_1d([0.0, np.nan], [1.0])
    with pytest.raises(ValueError, match="positive mass"):
        mot.emd_1d([0.0], [1.0], [0.0], [1.0])
    with pytest.raises(ValueError, match="finite values"):
        mot.wasserstein_1d([0.0, np.inf], [1.0])
    with pytest.raises(ValueError, match="positive mass"):
        mot.wasserstein_1d([0.0], [1.0], [0.0], [1.0])


def test_numerical_kernel_failures_are_not_silently_returned():
    a = np.array([0.5, 0.5])
    M = np.array([[0.0, 1.0e6], [1.0e6, 0.0]])
    with pytest.raises(FloatingPointError, match="scaling denominator"):
        mot.sinkhorn(a, a, M + 1.0e6, 1.0e-6, method="sinkhorn", warn=False)
    with pytest.raises(FloatingPointError, match="scaling denominator"):
        mot.sinkhorn(a, a, M + 1.0e6, 1.0e-6, method="greenkhorn", warn=False)
