from typing import Protocol

from pybatch.core.models import EngineConfig, Operation, VectorJob


class OperationHandler[T](Protocol):
    """A protocol for handling vector operations."""

    operation: Operation
    """The operation to handle."""

    def execute(self, job: VectorJob, *, config: EngineConfig) -> T:
        """Execute the operation.

        Args:
            job: The job to execute.
            config: The configuration for the operation.
        """
        ...
