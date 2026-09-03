from pybatch.core.errors import (
    MissingOperandError,
    UnsupportedOperationError,
)
from pybatch.core.models import (
    EngineConfig,
    JobResult,
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


class SyncEngine:
    def __init__(
        self,
        config: EngineConfig | None = None,
    ) -> None:
        self._config = config or EngineConfig()

    @property
    def config(self) -> EngineConfig:
        return self._config

    def execute(
        self,
        job: VectorJob,
    ) -> JobResult:
        value = self._execute_operation(job)

        return JobResult(
            job_id=job.job_id,
            operation=job.operation,
            value=value,
        )

    def _execute_operation(
        self,
        job: VectorJob,
    ) -> ResultValue:
        match job.operation:
            case Operation.NORMALIZE:
                return normalize(
                    job.left,
                    config=self._config,
                )

            case Operation.DOT_PRODUCT:
                right = self._require_right(job)

                return dot_product(
                    job.left,
                    right,
                    config=self._config,
                )

            case Operation.COSINE_SIMILARITY:
                right = self._require_right(job)

                return cosine_similarity(
                    job.left,
                    right,
                    config=self._config,
                )

            case Operation.MATRIX_MULTIPLY:
                right = self._require_right(job)

                return matrix_multiply(
                    job.left,
                    right,
                    config=self._config,
                )

            case Operation.TOP_K_SIMILARITY:
                right = self._require_right(job)
                top_k = self._require_top_k(job)

                return top_k_similarity(
                    job.left,
                    right,
                    top_k=top_k,
                    config=self._config,
                )

            case _:
                raise UnsupportedOperationError(
                    f"Unsupported operation: {job.operation!r}."
                )

    @staticmethod
    def _require_right(
        job: VectorJob,
    ):
        if job.right is None:
            raise MissingOperandError(
                f"{job.operation.value} requires a right operand."
            )

        return job.right

    @staticmethod
    def _require_top_k(
        job: VectorJob,
    ) -> int:
        if job.top_k is None:
            raise MissingOperandError("top_k_similarity requires top_k.")

        return job.top_k
