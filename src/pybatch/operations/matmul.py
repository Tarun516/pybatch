import numpy as np

from pybatch.core.errors import InvalidShapeError
from pybatch.core.models import EngineConfig, FloatArray
from pybatch.operations._validation import validate_matrix


def matrix_multiply(
    left: FloatArray,
    right: FloatArray,
    *,
    config: EngineConfig,
) -> FloatArray:
    """Multiply two matrices.

    Args:
        left: The left matrix.
        right: The right matrix.
        config: The engine configuration.
    """

    validate_matrix(
        left,
        name="left",
        config=config,
    )

    validate_matrix(
        right,
        name="right",
        config=config,
    )

    if left.shape[1] != right.shape[0]:
        raise InvalidShapeError(
            "Matrix dimensions are incompatible for multiplication. "
            f"Received shapes {left.shape} and {right.shape}."
        )

    result = left @ right

    result = np.asarray(
        result,
        dtype=np.float64,
    )

    result.setflags(write=False)

    return result
