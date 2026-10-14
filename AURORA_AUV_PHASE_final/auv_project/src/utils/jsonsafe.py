"""Strict-JSON sanitiser: numpy -> python, NaN/+-Inf -> None (browsers reject the NaN token)."""
from __future__ import annotations

import math
from enum import Enum

import numpy as np


def jsonsafe(obj):
    """Recursively convert `obj` into something json.dumps(allow_nan=False) accepts."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return jsonsafe(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonsafe(v) for v in obj]
    return str(obj)
