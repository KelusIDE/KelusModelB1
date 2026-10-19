"""Kelus Model B1: a small language model implemented without ML frameworks."""
import os

# Small CPU matrix multiplications are often faster with a single BLAS thread.
# Set these before NumPy is imported, while respecting explicit user settings.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

__version__ = "0.2.0"
