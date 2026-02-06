"""Controller module for nr_diff_flat."""
from .nr_diff_flat_jax import NR_tracker_flat
from .nr_diff_flat_numpy import nr_diff_flat_numpy

__all__ = ['NR_tracker_flat', 'nr_diff_flat_numpy']
