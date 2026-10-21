"""Test-process safeguards for macOS scientific-library thread initialization."""
from __future__ import annotations

import os


# Loading PyTorch before Arrow can otherwise make Accelerate/OpenMP initialization
# race and segfault during collection or the first Parquet read on macOS x86.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
