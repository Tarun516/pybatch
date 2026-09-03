from collections.abc import Iterable

from pybatch.core.errors import UnsupportedOperationError
from pybatch.core.models import (
    EngineConfig,
    JobResult,
    Operation,
    VectorJob,
)
from pybatch.core.protocols import OperationHandler
from pybatch.engine.handlers import default_handlers


class SyncEngine:
    def __init__(
        self,
        config: EngineConfig | None = None,
        handlers: Iterable[OperationHandler] | None = None,
    ) -> None:
        self._config = config or EngineConfig()

        selected_handlers = default_handlers() if handlers is None else handlers

        self._handlers = self._build_registry(selected_handlers)

    @property
    def config(self) -> EngineConfig:
        return self._config

    def execute(
        self,
        job: VectorJob,
    ) -> JobResult:
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
        handlers: Iterable[OperationHandler],
    ) -> dict[Operation, OperationHandler]:
        registry: dict[
            Operation,
            OperationHandler,
        ] = {}

        for handler in handlers:
            registry[handler.operation] = handler

        return registry
