import numpy as np
import pytest

from pybatch.batching import iter_batches
from pybatch.core.errors import InvalidBatchSizeError
from pybatch.core.models import (
    Operation,
    VectorJob,
)


def make_job(
    value: float,
) -> VectorJob:
    return VectorJob(
        operation=Operation.NORMALIZE,
        left=np.array(
            [value, 1.0],
            dtype=np.float64,
        ),
    )


def test_iter_batches_groups_jobs() -> None:
    jobs = [
        make_job(1.0),
        make_job(2.0),
        make_job(3.0),
        make_job(4.0),
    ]

    batches = list(
        iter_batches(
            jobs,
            batch_size=2,
        )
    )

    assert len(batches) == 2

    assert batches[0].jobs == (
        jobs[0],
        jobs[1],
    )

    assert batches[1].jobs == (
        jobs[2],
        jobs[3],
    )


def test_iter_batches_emits_partial_final_batch() -> None:
    jobs = [
        make_job(1.0),
        make_job(2.0),
        make_job(3.0),
    ]

    batches = list(
        iter_batches(
            jobs,
            batch_size=2,
        )
    )

    assert len(batches) == 2

    assert len(batches[0].jobs) == 2
    assert len(batches[1].jobs) == 1

    assert batches[1].jobs == (jobs[2],)


def test_iter_batches_accepts_generator_input() -> None:
    jobs = (make_job(float(index)) for index in range(4))

    batches = list(
        iter_batches(
            jobs,
            batch_size=2,
        )
    )

    assert len(batches) == 2

    assert all(len(batch.jobs) == 2 for batch in batches)


def test_iter_batches_empty_input_produces_no_batches() -> None:
    batches = list(
        iter_batches(
            [],
            batch_size=2,
        )
    )

    assert batches == []


@pytest.mark.parametrize(
    "batch_size",
    [
        0,
        -1,
        -10,
    ],
)
def test_iter_batches_rejects_invalid_batch_size(
    batch_size: int,
) -> None:
    with pytest.raises(
        InvalidBatchSizeError,
    ):
        list(
            iter_batches(
                [],
                batch_size=batch_size,
            )
        )
