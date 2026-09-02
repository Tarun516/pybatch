import numpy as np
import pytest

from pybatch.core.errors import (
    EmptyArrayError,
    InvalidConfigurationError,
    InvalidTopKError,
)
from pybatch.core.models import (
    Batch,
    EngineConfig,
    Operation,
    VectorJob,
)


def test_vector_job_copies_input() -> None:
    source = np.array([1.0, 2.0])

    job = VectorJob(
        operation=Operation.NORMALIZE,
        left=source,
    )

    source[0] = 99.0

    assert job.left[0] == 1.0


def test_vector_job_input_is_read_only() -> None:
    job = VectorJob(
        operation=Operation.NORMALIZE,
        left=np.array([1.0, 2.0]),
    )

    with pytest.raises(ValueError):
        job.left[0] = 99.0


def test_vector_job_rejects_empty_array() -> None:
    with pytest.raises(EmptyArrayError):
        VectorJob(
            operation=Operation.NORMALIZE,
            left=np.array([]),
        )


def test_vector_job_rejects_invalid_top_k() -> None:
    with pytest.raises(InvalidTopKError):
        VectorJob(
            operation=Operation.TOP_K_SIMILARITY,
            left=np.array([1.0, 0.0]),
            top_k=0,
        )


def test_engine_config_rejects_invalid_epsilon() -> None:
    with pytest.raises(InvalidConfigurationError):
        EngineConfig(
            zero_norm_epsilon=0,
        )


def test_batch_rejects_empty_jobs() -> None:
    from pybatch.core.errors import ValidationError

    with pytest.raises(ValidationError):
        Batch(jobs=())
