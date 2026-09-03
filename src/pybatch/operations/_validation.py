import numpy as np
from pybatch.core.errors import (
    EmptyArrayError,
    InvalidShapeError,
    InvalidTopKError,
    NonFiniteValueError,
)
from pybatch.core.models import EngineConfig, FloatArray


def validate_array(array: FloatArray, *, name: str, config: EngineConfig) -> None:
    """Validate that an array is a non-empty array.

    Args:
        array: The array to validate.
        name: The name of the array.
        config: The engine configuration.
    """

    if array.size == 0:
        raise EmptyArrayError(f"{name} cannot be empty")

    if config.check_finite and not np.isfinite(array).all():
        raise NonFiniteValueError(f"{name} contains NaN or infinity")


def validate_vector(vector: FloatArray, *, name: str, config: EngineConfig) -> None:
    """Validate that a vector is a 1D array.

    Args:
        vector: The vector to validate.
        name: The name of the vector.
        config: The engine configuration.
    """

    validate_array(vector, name=name, config=config)

    if vector.ndim != 1:
        raise InvalidShapeError(
            f"{name} must be a 1D vector, but received a {vector.shape} array"
        )


def validate_matrix(matrix: FloatArray, *, name: str, config: EngineConfig) -> None:
    """Validate that a matrix is a 2D array.

    Args:
        matrix: The matrix to validate.
        name: The name of the matrix.
        config: The engine configuration.
    """

    validate_array(matrix, name=name, config=config)

    if matrix.ndim != 2:
        raise InvalidShapeError(
            f"{name} must be a 2D matrix, but received a {matrix.shape} array"
        )


def validate_same_vector_shape(left: FloatArray, right: FloatArray) -> None:
    """Validate that two vectors have the same shape.

    Args:
        left: The left vector.
        right: The right vector.
    """

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
    """Validate the top-k parameter.

    Args:
        top_k: The number of top-k similarity scores to return.
        candidate_count: The number of candidate vectors.
    """

    if top_k <= 0:
        raise InvalidTopKError("top_k must be greater than zero.")

    if top_k > candidate_count:
        raise InvalidTopKError(
            "top_k cannot exceed the number of candidate vectors. "
            f"Received top_k={top_k}, "
            f"candidates={candidate_count}."
        )
