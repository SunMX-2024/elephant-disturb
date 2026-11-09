"""Array helpers used by the paired-site analyses."""

from __future__ import annotations

import numpy as np


def nanravel(values: np.ndarray) -> np.ndarray:
    """Flatten in NumPy's default order and remove NaNs, retaining infinities."""
    values = np.asarray(values).ravel()
    return values[~np.isnan(values)]
