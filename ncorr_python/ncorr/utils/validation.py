"""
Validation utilities.

Equivalent to ncorr_util_isrealbb.m and ncorr_util_isintbb.m
"""

from typing import Union, Tuple, Optional
import numpy as np


def is_real_bounded(
    value: Union[float, int, str],
    lower: float,
    upper: float,
    include_lower: bool = True,
    include_upper: bool = True,
) -> Tuple[bool, Optional[float], str]:
    """
    Check if value is a real number within bounds.

    Args:
        value: Value to check (can be string for user input)
        lower: Lower bound
        upper: Upper bound
        include_lower: Include lower bound (>=) vs (>)
        include_upper: Include upper bound (<=) vs (<)

    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    # Try to parse string
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return False, None, f"'{value}' is not a valid number"

    # Check if finite
    if not np.isfinite(value):
        return False, None, f"Value must be finite, got {value}"

    # Check bounds
    if include_lower:
        if value < lower:
            return False, None, f"Value {value} < {lower} (minimum)"
    else:
        if value <= lower:
            return False, None, f"Value {value} <= {lower} (must be greater)"

    if include_upper:
        if value > upper:
            return False, None, f"Value {value} > {upper} (maximum)"
    else:
        if value >= upper:
            return False, None, f"Value {value} >= {upper} (must be less)"

    return True, float(value), ""


def is_int_bounded(
    value: Union[int, float, str],
    lower: int,
    upper: int,
    include_lower: bool = True,
    include_upper: bool = True,
) -> Tuple[bool, Optional[int], str]:
    """
    Check if value is an integer within bounds.

    Args:
        value: Value to check
        lower: Lower bound
        upper: Upper bound
        include_lower: Include lower bound
        include_upper: Include upper bound

    Returns:
        Tuple of (is_valid, parsed_value, error_message)
    """
    # Try to parse string
    if isinstance(value, str):
        try:
            value = int(float(value))
        except ValueError:
            return False, None, f"'{value}' is not a valid integer"

    # Check if it's an integer
    if isinstance(value, float):
        if not value.is_integer():
            return False, None, f"Value {value} is not an integer"
        value = int(value)

    # Check bounds
    if include_lower:
        if value < lower:
            return False, None, f"Value {value} < {lower} (minimum)"
    else:
        if value <= lower:
            return False, None, f"Value {value} <= {lower} (must be greater)"

    if include_upper:
        if value > upper:
            return False, None, f"Value {value} > {upper} (maximum)"
    else:
        if value >= upper:
            return False, None, f"Value {value} >= {upper} (must be less)"

    return True, int(value), ""


def validate_dic_parameters(params: dict) -> Tuple[bool, str]:
    """
    Validate DIC parameter dictionary.

    Args:
        params: Parameter dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Radius
    valid, _, msg = is_int_bounded(params.get("radius", 30), 10, 200)
    if not valid:
        return False, f"Invalid radius: {msg}"

    # Spacing
    valid, _, msg = is_int_bounded(params.get("spacing", 5), 0, 80)
    if not valid:
        return False, f"Invalid spacing: {msg}"

    # Cutoff diffnorm
    valid, _, msg = is_real_bounded(params.get("cutoff_diffnorm", 1e-2), 1e-8, 1)
    if not valid:
        return False, f"Invalid cutoff_diffnorm: {msg}"

    # Cutoff iteration
    valid, _, msg = is_int_bounded(params.get("cutoff_iteration", 20), 5, 100)
    if not valid:
        return False, f"Invalid cutoff_iteration: {msg}"

    # Threads
    valid, _, msg = is_int_bounded(params.get("total_threads", 1), 1, 128)
    if not valid:
        return False, f"Invalid total_threads: {msg}"

    return True, ""
