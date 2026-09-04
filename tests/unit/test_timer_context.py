import logging

import pytest

from pybatch.observability import (
    ExecutionTimer,
)


def test_execution_timer_records_duration() -> None:
    with ExecutionTimer("test operation") as timer:
        total = sum(range(1_000))

    assert total > 0

    assert timer.elapsed_ms is not None

    assert timer.elapsed_ms >= 0


def test_execution_timer_logs_duration(
    caplog,
) -> None:
    with (
        caplog.at_level(
            logging.DEBUG,
            logger="pybatch.timing",
        ),
        ExecutionTimer("test operation"),
    ):
        sum(range(1_000))

    assert "test operation completed in" in caplog.text


def test_execution_timer_does_not_suppress_exceptions() -> None:
    with (
        pytest.raises(
            RuntimeError,
            match="boom",
        ),
        ExecutionTimer("failing operation"),
    ):
        raise RuntimeError("boom")
