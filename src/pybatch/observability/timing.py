import logging
from collections.abc import Callable
from functools import wraps
from time import perf_counter_ns


logger = logging.getLogger(
    "pybatch.timing"
)

def timed[**P, R](
    func: Callable[P, R],
) -> Callable[P, R]:
    @wraps(func)
    def wrapper(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        if not logger.isEnabledFor(
            logging.DEBUG
        ):
            return func(
                *args,
                **kwargs,
            )

        started_at = perf_counter_ns()

        try:
            return func(
                *args,
                **kwargs,
            )
        finally:
            elapsed_ns = (
                perf_counter_ns()
                - started_at
            )

            elapsed_ms = (
                elapsed_ns / 1_000_000
            )

            logger.debug(
                "%s completed in %.3f ms",
                func.__qualname__,
                elapsed_ms,
            )

    return wrapper