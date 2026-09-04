import logging

from pybatch.observability import timed


def test_timed_preserves_return_value() -> None:
    @timed
    def add(
        left: int,
        right: int,
    ) -> int:
        return left + right

    result = add(
        2,
        3,
    )
    print(result)
    assert result == 5


def test_timed_preserves_function_metadata() -> None:
    @timed
    def example() -> None:
        """Example function."""

    assert example.__name__ == "example"
    print(example.__doc__)
    assert example.__doc__ == "Example function."


def test_timed_logs_execution_time(
    caplog,
) -> None:
    @timed
    def example() -> int:
        return 42

    with caplog.at_level(
        logging.DEBUG,
        logger="pybatch.timing",
    ):
        result = example()

    assert result == 42
    print(caplog.text)
    assert "example completed in" in caplog.text
