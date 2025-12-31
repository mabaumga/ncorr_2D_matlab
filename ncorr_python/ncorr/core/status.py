"""
Status enumeration for operation results.

Equivalent to MATLAB's out.m class.
"""

from enum import IntEnum


class Status(IntEnum):
    """
    Enumeration for operation status results.

    This acts as an enumeration and can be used to check if the output of a
    function was successful, failed, or cancelled.

    Examples:
        >>> status = some_operation()
        >>> if status == Status.SUCCESS:
        ...     print("Operation succeeded")
        >>> elif status == Status.CANCELLED:
        ...     print("Operation was cancelled by user")
    """

    SUCCESS = 1
    FAILED = 0
    CANCELLED = -1

    @classmethod
    def success(cls) -> "Status":
        """Return success status."""
        return cls.SUCCESS

    @classmethod
    def failed(cls) -> "Status":
        """Return failed status."""
        return cls.FAILED

    @classmethod
    def cancelled(cls) -> "Status":
        """Return cancelled status."""
        return cls.CANCELLED
