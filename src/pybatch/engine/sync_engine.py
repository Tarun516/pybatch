from collections.abc import Iterable

from pybatch.core.errors import UnsupportedOperationError
from pybatch.core.models import (
    EngineConfig,
    JobResult,
    Operation,
    ResultValue,
    VectorJob,
)
from pybatch.core.protocols import OperationHandler
from pybatch.engine.handlers import default_handlers


class SyncEngine:
    def __init__(
        self,
        config: EngineConfig | None = None,
        handlers: Iterable[OperationHandler[ResultValue]] | None = None,
    ) -> None:
        """Initialize the engine.

        Args:
            config: The configuration for the engine.
            handlers: The handlers for the engine.
        """
        self._config = config or EngineConfig()

        selected_handlers = default_handlers() if handlers is None else handlers

        self._handlers = self._build_registry(selected_handlers)

    @property
    def config(self) -> EngineConfig:
        """The configuration for the engine."""
        return self._config

    def execute(
        self,
        job: VectorJob,
    ) -> JobResult[ResultValue]:
        """Execute a job.

        Args:
            job: The job to execute.
        """
        handler = self._handlers.get(job.operation)

        if handler is None:
            raise UnsupportedOperationError(
                f"No handler registered for {job.operation.value}."
            )

        value = handler.execute(
            job,
            config=self._config,
        )

        return JobResult(
            job_id=job.job_id,
            operation=job.operation,
            value=value,
        )

    @staticmethod
    def _build_registry(
        handlers: Iterable[OperationHandler[ResultValue]],
    ) -> dict[
        Operation,
        OperationHandler[ResultValue],
    ]:
        registry: dict[
            Operation,
            OperationHandler[ResultValue],
        ] = {}

        for handler in handlers:
            registry[handler.operation] = handler

        return registry
