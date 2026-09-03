import numpy as np
import pytest

from pybatch.core.errors import MissingOperandError
from pybatch.core.models import (
    EngineConfig,
    Operation,
    VectorJob,
)
from pybatch.engine.handlers import (
    DotProductHandler,
    NormalizeHandler,
)


@pytest.fixture
def config() -> EngineConfig:
    return EngineConfig()


def test_normalize_handler(
    config: EngineConfig,
) -> None:
    handler = NormalizeHandler()

    job = VectorJob(
        operation=Operation.NORMALIZE,
        left=np.array([3.0, 4.0]),
    )

    result = handler.execute(
        job,
        config=config,
    )

    assert isinstance(
        result,
        np.ndarray,
    )

    np.testing.assert_allclose(
        result,
        np.array([0.6, 0.8]),
    )


def test_dot_product_handler(
    config: EngineConfig,
) -> None:
    handler = DotProductHandler()

    job = VectorJob(
        operation=Operation.DOT_PRODUCT,
        left=np.array([1.0, 2.0]),
        right=np.array([3.0, 4.0]),
    )

    result = handler.execute(
        job,
        config=config,
    )

    assert isinstance(result, float)
    assert result == pytest.approx(11.0)


def test_dot_product_handler_requires_right_operand(
    config: EngineConfig,
) -> None:
    handler = DotProductHandler()

    job = VectorJob(
        operation=Operation.DOT_PRODUCT,
        left=np.array([1.0, 2.0]),
    )

    with pytest.raises(
        MissingOperandError,
    ):
        handler.execute(
            job,
            config=config,
        )


def inspect_handler(
    handler: DotProductHandler,
) -> None:
    value = handler.execute(
        job,
        config=config,
    )

    reveal_type(value)
