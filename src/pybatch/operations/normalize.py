import numpy as np

from pybatch.core.errors import ZeroNormError
from pybatch.core.models import EngineConfig, FloatArray
from pybatch.operations._validation import (
    validate_matrix,
    validate_vector,
)


def normalize(
    vector: FloatArray,
    *,
    config: EngineConfig,
) -> FloatArray:
    """
    Normalize a vector to have a unit norm.

    Args:
        vector: The vector to normalize.
        config: The configuration for the engine.

    Returns:
        The normalized vector.
    """
    validate_vector(
        vector,
        name="vector",
        config=config,
    )

    norm = float(np.linalg.norm(vector))

    if norm <= config.zero_norm_epsilon:
        raise ZeroNormError("Cannot normalize a zero-norm vector.")

    result = vector / norm

    result.setflags(write=False)

    return result


def normalize_rows(
    matrix: FloatArray,
    *,
    config: EngineConfig,
) -> FloatArray:
    """
    Normalize the rows of a matrix to have a unit norm.

    Args:
        matrix: The matrix to normalize.
        config: The configuration for the engine.

    Returns:
        The normalized matrix.
    """
    validate_matrix(
        matrix,
        name="matrix",
        config=config,
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    zero_norm_mask = norms[:, 0] <= config.zero_norm_epsilon

    if np.any(zero_norm_mask):
        zero_indices = np.flatnonzero(zero_norm_mask)

        raise ZeroNormError(
            f"Matrix contains zero-norm rows at indices {zero_indices.tolist()}."
        )

    result = matrix / norms

    result = np.asarray(
        result,
        dtype=np.float64,
    )

    result.setflags(write=False)

    return result
