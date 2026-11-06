"""Kani Seal 3: a small language model implemented without ML frameworks."""
import os



for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

__version__ = "0.2.0"
