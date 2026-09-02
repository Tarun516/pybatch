import numpy as np
from pybatch.core.errors import (
    EmptyArrayError,
    InvalidShapeError,
    InvalidTopKError,
    NonFiniteValueError,
)
from pybatch.core.models import EngineConfig, FloatArray


def validate_array(array: FloatArray, *, name: str, config: EngineConfig) -> None:

    if array.size == 0:
        raise EmptyArrayError(f"{name} cannot be empty")

    if config.check_finite and not np.isfinite(array).all():
        raise NonFiniteValueError(f"{name} contains NaN or infinity")


def validate_vector(vector: FloatArray, *, name: str, config: EngineConfig) -> None:

    validate_array(vector, name=name, config=config)

    if vector.ndim != 1:
        raise InvalidShapeError(
            f"{name} must be a 1D vector, but received a {vector.shape} array"
        )


def validate_matrix(matrix: FloatArray, *, name: str, config: EngineConfig) -> None:

    validate_array(matrix, name=name, config=config)

    if matrix.ndim != 2:
        raise InvalidShapeError(
            f"{name} must be a 2D matrix, but received a {matrix.shape} array"
        )


def validate_same_vector_shape(left: FloatArray, right: FloatArray) -> None:
    if left.shape != right.shape:
        raise InvalidShapeError(
            "Vectors must have identical shapes"
            f"but received {left.shape} and {right.shape}"
        )


def validate_top_k(
    top_k: int,
    *,
    candidate_count: int,
) -> None:
    if top_k <= 0:
        raise InvalidTopKError("top_k must be greater than zero.")

    if top_k > candidate_count:
        raise InvalidTopKError(
            "top_k cannot exceed the number of candidate vectors. "
            f"Received top_k={top_k}, "
            f"candidates={candidate_count}."
        )
