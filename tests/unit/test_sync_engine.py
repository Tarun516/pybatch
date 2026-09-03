import numpy as np
import pytest

from pybatch.core.errors import MissingOperandError
from pybatch.core.models import (
    EngineConfig,
    Operation,
    TopKResult,
    VectorJob,
)
from pybatch.engine import SyncEngine


@pytest.fixture
def engine() -> SyncEngine:
    return SyncEngine()


def test_execute_normalize(
    engine: SyncEngine,
) -> None:
    job = VectorJob(
        operation=Operation.NORMALIZE,
        left=np.array([3.0, 4.0]),
    )

    result = engine.execute(job)

    assert result.job_id == job.job_id
    assert result.operation is Operation.NORMALIZE

    np.testing.assert_allclose(
        result.value,
        np.array([0.6, 0.8]),
    )


def test_execute_dot_product(
    engine: SyncEngine,
) -> None:
    job = VectorJob(
        operation=Operation.DOT_PRODUCT,
        left=np.array([1.0, 2.0, 3.0]),
        right=np.array([4.0, 5.0, 6.0]),
    )

    result = engine.execute(job)

    assert result.job_id == job.job_id
    assert result.operation is Operation.DOT_PRODUCT
    assert result.value == pytest.approx(32.0)


def test_execute_cosine_similarity(
    engine: SyncEngine,
) -> None:
    job = VectorJob(
        operation=Operation.COSINE_SIMILARITY,
        left=np.array([1.0, 0.0]),
        right=np.array([1.0, 0.0]),
    )

    result = engine.execute(job)

    assert result.value == pytest.approx(1.0)


def test_execute_matrix_multiply(
    engine: SyncEngine,
) -> None:
    job = VectorJob(
        operation=Operation.MATRIX_MULTIPLY,
        left=np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
        right=np.array(
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        ),
    )

    result = engine.execute(job)

    np.testing.assert_allclose(
        result.value,
        np.array(
            [
                [19.0, 22.0],
                [43.0, 50.0],
            ]
        ),
    )


def test_execute_top_k_similarity(
    engine: SyncEngine,
) -> None:
    job = VectorJob(
        operation=Operation.TOP_K_SIMILARITY,
        left=np.array([1.0, 0.0]),
        right=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ]
        ),
        top_k=2,
    )

    result = engine.execute(job)

    assert isinstance(
        result.value,
        TopKResult,
    )

    np.testing.assert_array_equal(
        result.value.indices,
        np.array([0, 2]),
    )


def test_dot_product_requires_right_operand(
    engine: SyncEngine,
) -> None:
    job = VectorJob(
        operation=Operation.DOT_PRODUCT,
        left=np.array([1.0, 2.0]),
    )

    with pytest.raises(
        MissingOperandError,
        match="requires a right operand",
    ):
        engine.execute(job)


def test_top_k_requires_top_k_value(
    engine: SyncEngine,
) -> None:
    job = VectorJob(
        operation=Operation.TOP_K_SIMILARITY,
        left=np.array([1.0, 0.0]),
        right=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )

    with pytest.raises(
        MissingOperandError,
        match="requires top_k",
    ):
        engine.execute(job)


def test_engine_uses_custom_config() -> None:
    config = EngineConfig(
        check_finite=False,
    )

    engine = SyncEngine(
        config=config,
    )

    assert engine.config is config


def test_default_config_is_created() -> None:
    engine = SyncEngine()

    assert engine.config == EngineConfig()
