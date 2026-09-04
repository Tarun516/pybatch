import numpy as np
import pytest

from pybatch.core.errors import (
    InvalidShapeError,
    ZeroNormError,
)
from pybatch.core.models import (
    EngineConfig,
)
from pybatch.operations import (
    normalize_rows,
)


@pytest.fixture
def config() -> EngineConfig:
    return EngineConfig()


def test_normalize_rows(
    config: EngineConfig,
) -> None:
    matrix = np.array(
        [
            [3.0, 4.0],
            [5.0, 12.0],
        ],
        dtype=np.float64,
    )

    result = normalize_rows(
        matrix,
        config=config,
    )

    expected = np.array(
        [
            [0.6, 0.8],
            [
                5.0 / 13.0,
                12.0 / 13.0,
            ],
        ]
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_normalize_rows_produces_unit_norms(
    config: EngineConfig,
) -> None:
    matrix = np.array(
        [
            [3.0, 4.0],
            [5.0, 12.0],
            [8.0, 15.0],
        ],
        dtype=np.float64,
    )

    result = normalize_rows(
        matrix,
        config=config,
    )

    norms = np.linalg.norm(
        result,
        axis=1,
    )

    np.testing.assert_allclose(
        norms,
        np.ones(3),
    )


def test_normalize_rows_rejects_zero_row(
    config: EngineConfig,
) -> None:
    matrix = np.array(
        [
            [3.0, 4.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ZeroNormError,
        match="indices \\[1\\]",
    ):
        normalize_rows(
            matrix,
            config=config,
        )


def test_normalize_rows_requires_matrix(
    config: EngineConfig,
) -> None:
    vector = np.array(
        [3.0, 4.0],
        dtype=np.float64,
    )

    with pytest.raises(
        InvalidShapeError,
    ):
        normalize_rows(
            vector,
            config=config,
        )


def test_normalize_rows_result_is_read_only(
    config: EngineConfig,
) -> None:
    matrix = np.array(
        [
            [3.0, 4.0],
            [5.0, 12.0],
        ],
        dtype=np.float64,
    )

    result = normalize_rows(
        matrix,
        config=config,
    )

    assert result.flags.writeable is False
