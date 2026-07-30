"""Entropic optimal transport solvers."""

from .core import greenkhorn, sinkhorn, sinkhorn2, sinkhorn_log

__all__ = ["sinkhorn", "sinkhorn2", "sinkhorn_log", "greenkhorn"]
