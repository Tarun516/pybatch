class PyBatchError(Exception):
    """Base exception for all expected PyBatch errors."""


class ValidationError(PyBatchError):
    """Base exception for invalid user-provided data."""


class InvalidArrayError(ValidationError):
    """Raised when an input is not a valid NumPy array."""


class EmptyArrayError(ValidationError):
    """Raised when an operation receives an empty array."""


class InvalidShapeError(ValidationError):
    """Raised when array dimensions are incompatible with an operation."""


class MissingOperandError(ValidationError):
    """Raised when an operation requires another input array."""


class InvalidTopKError(ValidationError):
    """Raised when a top-k value is invalid."""


class NonFiniteValueError(ValidationError):
    """Raised when an input contains NaN or infinity."""


class ZeroNormError(ValidationError):
    """Raised when normalization is requested for a zero vector."""


class InvalidConfigurationError(ValidationError):
    """Raised when engine configuration is invalid."""


class UnsupportedOperationError(PyBatchError):
    """Raised when the requested operation is not supported."""


class InvalidBatchSizeError(ValidationError):
    """Raised when a batch size is not greater than zero."""
