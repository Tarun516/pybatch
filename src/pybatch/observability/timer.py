import logging
from time import perf_counter_ns
from types import TracebackType
from typing import Self

logger = logging.getLogger("pybatch.timing")


class ExecutionTimer:
    """A context manager for measuring execution time."""

    def __init__(
        self,
        name: str,
    ) -> None:
        """Initialize a timer.

        Args:
            name: The name used in the timing log message.
        """
        self.name = name
        self._started_at: int | None = None
        self.elapsed_ms: float | None = None

    def __enter__(
        self,
    ) -> Self:
        """Start the timer and return this instance."""
        self._started_at = perf_counter_ns()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Record and log elapsed time without suppressing exceptions."""
        if self._started_at is None:
            return

        elapsed_ns = perf_counter_ns() - self._started_at

        self.elapsed_ms = elapsed_ns / 1_000_000

        logger.debug(
            "%s completed in %.3f ms",
            self.name,
            self.elapsed_ms,
        )
