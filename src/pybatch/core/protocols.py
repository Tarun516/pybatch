from typing import Protocol

from pybatch.core.models import EngineConfig, Operation, ResultValue, VectorJob


class OperationHandler(Protocol):
    """A protocol for handling vector operations."""

    operation: Operation
    """The operation to handle."""

    def execute(self, job: VectorJob, *, config: EngineConfig) -> ResultValue:
        """Execute the operation.

        Args:
            job: The job to execute.
            config: The configuration for the operation.
        """
        ...
