"""Optimal transport solvers accelerated with Mojo."""

from . import bregman, lp
from .core import (
    dist,
    emd,
    emd2,
    emd_1d,
    greenkhorn,
    sinkhorn,
    sinkhorn2,
    sinkhorn_log,
    wasserstein_1d,
)

__version__ = "0.1.0"

__all__ = [
    "bregman",
    "dist",
    "emd",
    "emd2",
    "emd_1d",
    "greenkhorn",
    "lp",
    "sinkhorn",
    "sinkhorn2",
    "sinkhorn_log",
    "wasserstein_1d",
]
