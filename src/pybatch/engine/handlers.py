from pybatch.core.errors import MissingOperandError
from pybatch.core.models import (
    EngineConfig,
    FloatArray,
    Operation,
    ResultValue,
    VectorJob,
)
from pybatch.operations import (
    cosine_similarity,
    dot_product,
    matrix_multiply,
    normalize,
    top_k_similarity,
)


def _require_right(
    job: VectorJob,
) -> FloatArray:
    """Require the right operand for the operation.

    Args:
        job: The job to execute.
        operation: The operation to execute.
    """
    if job.right is None:
        raise MissingOperandError(f"{job.operation.value} requires a right operand.")

    return job.right


def _require_top_k(
    job: VectorJob,
) -> int:
    """Require the top-k parameter for the operation.

    Args:
        job: The job to execute.
    """
    if job.top_k is None:
        raise MissingOperandError("top_k_similarity requires top_k.")

    return job.top_k


class NormalizeHandler:
    """A handler for the normalize operation."""

    operation = Operation.NORMALIZE
    """The operation to handle."""

    def execute(
        self,
        job: VectorJob,
        *,
        config: EngineConfig,
    ) -> ResultValue:
        return normalize(
            job.left,
            config=config,
        )


class DotProductHandler:
    """A handler for the dot product operation."""

    operation = Operation.DOT_PRODUCT
    """The operation to handle."""

    def execute(
        self,
        job: VectorJob,
        *,
        config: EngineConfig,
    ) -> ResultValue:
        return dot_product(
            job.left,
            _require_right(job),
            config=config,
        )


class CosineSimilarityHandler:
    """A handler for the cosine similarity operation."""

    operation = Operation.COSINE_SIMILARITY
    """The operation to handle."""

    def execute(
        self,
        job: VectorJob,
        *,
        config: EngineConfig,
    ) -> ResultValue:
        return cosine_similarity(
            job.left,
            _require_right(job),
            config=config,
        )


class MatrixMultiplyHandler:
    """A handler for the matrix multiply operation."""

    operation = Operation.MATRIX_MULTIPLY
    """The operation to handle."""

    def execute(
        self,
        job: VectorJob,
        *,
        config: EngineConfig,
    ) -> ResultValue:
        return matrix_multiply(
            job.left,
            _require_right(job),
            config=config,
        )


class TopKSimilarityHandler:
    """A handler for the top-k similarity operation."""

    operation = Operation.TOP_K_SIMILARITY
    """The operation to handle."""

    def execute(
        self,
        job: VectorJob,
        *,
        config: EngineConfig,
    ) -> ResultValue:
        return top_k_similarity(
            job.left,
            _require_right(job),
            top_k=_require_top_k(job),
            config=config,
        )


from pybatch.core.protocols import OperationHandler


def default_handlers() -> tuple[OperationHandler, ...]:
    """Get the default handlers for the operations."""
    return (
        NormalizeHandler(),
        DotProductHandler(),
        CosineSimilarityHandler(),
        MatrixMultiplyHandler(),
        TopKSimilarityHandler(),
    )
