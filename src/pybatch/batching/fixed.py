from collections.abc import Iterable, Iterator

from pybatch.core.errors import InvalidBatchSizeError
from pybatch.core.models import Batch, VectorJob


def iter_batches(
    jobs: Iterable[VectorJob],
    *,
    batch_size: int,
) -> Iterator[Batch]:
    """Iterate over the jobs in fixed-size batches.

    Args:
        jobs: The jobs to iterate over.
        batch_size: The size of the batches.
    """
    if batch_size <= 0:
        raise InvalidBatchSizeError(
            f"Batch size must be greater than zero, got {batch_size}"
        )

    pending: list[VectorJob] = []

    for job in jobs:
        pending.append(job)

        if len(pending) == batch_size:
            yield Batch(
                jobs=tuple(pending),
            )
            pending.clear()

    if pending:
        yield Batch(
            jobs=tuple(pending),
        )
