import numpy as np

from pybatch.core.errors import ZeroNormError
from pybatch.core.models import (
    EngineConfig,
    FloatArray,
    TopKResult,
)
from pybatch.operations._validation import (
    validate_matrix,
    validate_same_vector_shape,
    validate_top_k,
    validate_vector,
)


def dot_product(
    left: FloatArray,
    right: FloatArray,
    *,
    config: EngineConfig,
) -> float:
    """Compute the dot product between two vectors.

    Args:
        left: The left vector.
        right: The right vector.
        config: The engine configuration.
    """

    validate_vector(
        left,
        name="left",
        config=config,
    )

    validate_vector(
        right,
        name="right",
        config=config,
    )

    validate_same_vector_shape(
        left,
        right,
    )

    return float(np.dot(left, right))


def cosine_similarity(
    left: FloatArray,
    right: FloatArray,
    *,
    config: EngineConfig,
) -> float:
    """Compute the cosine similarity between two vectors.

    Args:
        left: The left vector.
        right: The right vector.
        config: The engine configuration.
    """

    validate_vector(
        left,
        name="left",
        config=config,
    )

    validate_vector(
        right,
        name="right",
        config=config,
    )

    validate_same_vector_shape(
        left,
        right,
    )

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))

    if left_norm <= config.zero_norm_epsilon:
        raise ZeroNormError("left vector has zero norm.")

    if right_norm <= config.zero_norm_epsilon:
        raise ZeroNormError("right vector has zero norm.")

    dot = float(np.dot(left, right))

    return dot / (left_norm * right_norm)


def top_k_similarity(
    query: FloatArray,
    candidates: FloatArray,
    *,
    top_k: int,
    config: EngineConfig,
) -> TopKResult:
    """Compute the top-k similarity scores between a query vector and a set of candidate vectors.

    Args:
        query: The query vector.
        candidates: The candidate vectors.
        top_k: The number of top-k similarity scores to return.
        config: The engine configuration.
    """

    validate_vector(
        query,
        name="query",
        config=config,
    )

    validate_matrix(
        candidates,
        name="candidates",
        config=config,
    )

    if candidates.shape[1] != query.shape[0]:
        from pybatch.core.errors import InvalidShapeError

        raise InvalidShapeError(
            "Candidate vector dimensions must match the query. "
            f"Query dimension={query.shape[0]}, "
            f"candidate dimension={candidates.shape[1]}."
        )

    candidate_count = candidates.shape[0]

    validate_top_k(
        top_k,
        candidate_count=candidate_count,
    )

    query_norm = float(np.linalg.norm(query))

    if query_norm <= config.zero_norm_epsilon:
        raise ZeroNormError("query vector has zero norm.")

    candidate_norms = np.linalg.norm(
        candidates,
        axis=1,
    )

    zero_norm_mask = candidate_norms <= config.zero_norm_epsilon

    if np.any(zero_norm_mask):
        zero_indices = np.flatnonzero(zero_norm_mask)

        raise ZeroNormError(
            "Candidate vectors contain zero-norm rows "
            f"at indices {zero_indices.tolist()}."
        )

    dot_products = candidates @ query

    scores = dot_products / (candidate_norms * query_norm)

    partition_start = scores.size - top_k

    top_indices = np.argpartition(
        scores,
        partition_start,
    )[partition_start:]

    sorted_order = np.argsort(scores[top_indices])[::-1]

    top_indices = top_indices[sorted_order]

    top_scores = scores[top_indices]

    top_indices = np.asarray(
        top_indices,
        dtype=np.int64,
    )

    top_scores = np.asarray(
        top_scores,
        dtype=np.float64,
    )

    top_indices.setflags(write=False)
    top_scores.setflags(write=False)

    return TopKResult(
        indices=top_indices,
        scores=top_scores,
    )
