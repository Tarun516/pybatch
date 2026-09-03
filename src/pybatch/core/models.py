from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from pybatch.core.errors import (
    EmptyArrayError,
    InvalidArrayError,
    InvalidConfigurationError,
    InvalidTopKError,
    ValidationError,
)


type FloatArray = NDArray[np.float64]


class Operation(StrEnum):
    """An enum of the supported vector operations."""

    NORMALIZE = "normalize"
    DOT_PRODUCT = "dot_product"
    COSINE_SIMILARITY = "cosine_similarity"
    MATRIX_MULTIPLY = "matrix_multiply"
    TOP_K_SIMILARITY = "top_k_similarity"


def _prepare_array(
    value: FloatArray,
    field_name: str,
) -> FloatArray:
    """Prepare a value for use in a vector operation."""

    # Check if the value is a NumPy array.
    if not isinstance(value, np.ndarray):
        raise InvalidArrayError(f"{field_name} must be a NumPy ndarray.")

    # Check if the value is not empty.
    if value.size == 0:
        raise EmptyArrayError(f"{field_name} cannot be empty.")

    # Convert the value to a NumPy array.
    try:
        array = np.array(
            value,
            dtype=np.float64,
            copy=True,
        )
    # If the value is not a NumPy array, raise an InvalidArrayError.
    except (TypeError, ValueError) as exc:
        raise InvalidArrayError(
            f"{field_name} could not be converted to float64."
        ) from exc

    # Set the write flag to False to make the array read-only.
    array.setflags(write=False)

    return array


@dataclass(frozen=True, slots=True)
class VectorJob:
    """A single vector operation."""

    operation: Operation
    left: FloatArray
    right: FloatArray | None = None
    top_k: int | None = None
    job_id: str = field(default_factory=lambda: uuid4().hex)
    """Post-init hook to prepare the input arrays."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "left",
            _prepare_array(self.left, "left"),
        )

        if self.right is not None:
            object.__setattr__(
                self,
                "right",
                _prepare_array(self.right, "right"),
            )

        if self.top_k is not None and self.top_k <= 0:
            raise InvalidTopKError("top_k must be greater than zero.")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Configuration for the vector engine."""

    zero_norm_epsilon: float = 1e-12
    check_finite: bool = True

    """Post-init hook to validate the configuration."""

    def __post_init__(self) -> None:
        if self.zero_norm_epsilon <= 0:
            raise InvalidConfigurationError(
                "zero_norm_epsilon must be greater than zero."
            )


@dataclass(frozen=True, slots=True)
class TopKResult:
    """A result of a top-k similarity operation."""

    indices: NDArray[np.int64]
    scores: FloatArray


type ResultValue = FloatArray | float | TopKResult


@dataclass(frozen=True, slots=True)
class JobResult[T]:
    """A result of a single vector operation."""

    job_id: str
    operation: Operation
    value: T


@dataclass(frozen=True, slots=True)
class Batch:
    """A batch of vector operations."""

    jobs: tuple[VectorJob, ...]
    """Post-init hook to validate the batch."""

    def __post_init__(self) -> None:
        """Post-init hook to validate the batch."""
        if not self.jobs:
            raise ValidationError("A batch must contain at least one job.")
