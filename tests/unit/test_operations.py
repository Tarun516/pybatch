import numpy as np
import pytest

from pybatch.core.errors import (
    InvalidShapeError,
    InvalidTopKError,
    NonFiniteValueError,
    ZeroNormError,
)
from pybatch.core.models import EngineConfig
from pybatch.operations import (
    cosine_similarity,
    dot_product,
    matrix_multiply,
    normalize,
    top_k_similarity,
)


@pytest.fixture
def config() -> EngineConfig:
    return EngineConfig()


def test_normalize_returns_unit_vector(
    config: EngineConfig,
) -> None:
    vector = np.array(
        [3.0, 4.0],
        dtype=np.float64,
    )

    result = normalize(
        vector,
        config=config,
    )

    np.testing.assert_allclose(
        result,
        np.array([0.6, 0.8]),
    )

    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_normalize_rejects_zero_vector(
    config: EngineConfig,
) -> None:
    vector = np.array(
        [0.0, 0.0],
        dtype=np.float64,
    )

    with pytest.raises(ZeroNormError):
        normalize(
            vector,
            config=config,
        )


def test_dot_product(
    config: EngineConfig,
) -> None:
    left = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    right = np.array(
        [4.0, 5.0, 6.0],
        dtype=np.float64,
    )

    result = dot_product(
        left,
        right,
        config=config,
    )

    assert result == pytest.approx(32.0)


def test_dot_product_rejects_different_shapes(
    config: EngineConfig,
) -> None:
    left = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    right = np.array(
        [1.0, 2.0],
        dtype=np.float64,
    )

    with pytest.raises(InvalidShapeError):
        dot_product(
            left,
            right,
            config=config,
        )


def test_cosine_similarity_of_identical_vectors(
    config: EngineConfig,
) -> None:
    left = np.array(
        [1.0, 2.0],
        dtype=np.float64,
    )

    right = np.array(
        [1.0, 2.0],
        dtype=np.float64,
    )

    result = cosine_similarity(
        left,
        right,
        config=config,
    )

    assert result == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors(
    config: EngineConfig,
) -> None:
    left = np.array(
        [1.0, 0.0],
        dtype=np.float64,
    )

    right = np.array(
        [0.0, 1.0],
        dtype=np.float64,
    )

    result = cosine_similarity(
        left,
        right,
        config=config,
    )

    assert result == pytest.approx(0.0)


def test_matrix_multiply(
    config: EngineConfig,
) -> None:
    left = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    right = np.array(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=np.float64,
    )

    result = matrix_multiply(
        left,
        right,
        config=config,
    )

    expected = np.array(
        [
            [19.0, 22.0],
            [43.0, 50.0],
        ],
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_matrix_multiply_rejects_incompatible_shapes(
    config: EngineConfig,
) -> None:
    left = np.ones(
        (2, 3),
        dtype=np.float64,
    )

    right = np.ones(
        (4, 2),
        dtype=np.float64,
    )

    with pytest.raises(InvalidShapeError):
        matrix_multiply(
            left,
            right,
            config=config,
        )


def test_top_k_similarity(
    config: EngineConfig,
) -> None:
    query = np.array(
        [1.0, 0.0],
        dtype=np.float64,
    )

    candidates = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float64,
    )

    result = top_k_similarity(
        query,
        candidates,
        top_k=2,
        config=config,
    )

    np.testing.assert_array_equal(
        result.indices,
        np.array([0, 2]),
    )

    assert result.scores[0] == pytest.approx(1.0)
    assert result.scores[0] >= result.scores[1]


def test_top_k_rejects_too_large_k(
    config: EngineConfig,
) -> None:
    query = np.array(
        [1.0, 0.0],
        dtype=np.float64,
    )

    candidates = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(InvalidTopKError):
        top_k_similarity(
            query,
            candidates,
            top_k=3,
            config=config,
        )


def test_operation_rejects_non_finite_values(
    config: EngineConfig,
) -> None:
    vector = np.array(
        [1.0, np.nan],
        dtype=np.float64,
    )

    with pytest.raises(NonFiniteValueError):
        normalize(
            vector,
            config=config,
        )
