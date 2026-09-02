import numpy as np

from pybatch.core.errors import ZeroNormError
from pybatch.core.models import EngineConfig, FloatArray
from pybatch.operations._validation import validate_vector


def normalize(
    vector: FloatArray,
    *,
    config: EngineConfig,
) -> FloatArray:
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
