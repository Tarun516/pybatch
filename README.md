# PyBatch

**A small Python runtime for learning how real numerical systems are built — from typed jobs and dispatch to lazy batching, observability, and NumPy vectorization.**

PyBatch started with a simple question:

> What actually sits between “I have some numerical work” and “the machine executes it efficiently”?

Instead of jumping straight into `asyncio`, multiprocessing, queues, GPUs, or model serving, PyBatch builds that runtime layer step by step.

Today, it can represent numerical jobs, validate them, route them through typed operation handlers, execute them synchronously, batch them lazily, stream batch results, measure execution time, and perform vectorized NumPy operations.

---

## What PyBatch looks like

```text
VectorJob
   │
   ▼
SyncEngine
   │
   ▼
OperationHandler registry
   │
   ▼
Concrete handler
   │
   ▼
NumPy operation
   │
   ▼
JobResult
```

For multiple jobs:

```text
Job source
   │
   ▼
iter_batches()
   │
   ▼
Lazy Batch stream
   │
   ▼
SyncEngine
   │
   ▼
Lazy result-batch stream
```

The project deliberately separates three concerns:

- **Jobs describe what should happen.**
- **Operations define numerical semantics.**
- **The engine decides how work is executed.**

That separation is the foundation for adding more advanced execution strategies later without rewriting the whole system.

---

## Current capabilities

### Structured numerical jobs

PyBatch represents work using typed domain objects rather than loose dictionaries or raw function calls.

Supported operations currently include:

- vector normalization
- dot product
- cosine similarity
- matrix multiplication
- top-k similarity

Inputs are normalized into `float64` NumPy arrays, copied at the boundary, and marked read-only to keep jobs stable after construction.

---

### Typed handler-based dispatch

The engine does not contain one giant `if`/`match` block for every operation.

Each operation has a handler satisfying a shared `OperationHandler[T]` protocol:

```text
NORMALIZE          → NormalizeHandler
DOT_PRODUCT        → DotProductHandler
COSINE_SIMILARITY  → CosineSimilarityHandler
MATRIX_MULTIPLY    → MatrixMultiplyHandler
TOP_K_SIMILARITY   → TopKSimilarityHandler
```

This keeps dispatch extensible while preserving precise return types inside concrete handlers.

---

### Lazy batching

PyBatch can group any `Iterable[VectorJob]` into fixed-size batches without materializing the entire input first.

```python
batches = iter_batches(
    jobs,
    batch_size=32,
)
```

Because batching is generator-based, input is consumed only when downstream code asks for the next batch.

This means PyBatch can work naturally with lists, generators, streams, or other iterable job sources.

---

### Lazy batch execution

The synchronous engine can execute one batch at a time and expose a lazy stream of result batches:

```python
result_batches = engine.iter_batch_results(
    batches,
)
```

Nothing is forced to run ahead unnecessarily.

The consumer pulls the next result batch, the engine pulls the next batch, and the batcher pulls only enough jobs to satisfy that request.

That makes the current pipeline a small but real example of pull-based processing.

---

### Observability primitives

PyBatch currently includes two timing mechanisms.

Use a decorator when the whole function should be measured:

```python
@timed
def execute_batch(...):
    ...
```

Use a context manager when only a specific block should be measured:

```python
with ExecutionTimer("numpy work") as timer:
    result = run_operation()
```

The timing layer uses `perf_counter_ns()` and Python logging rather than printing directly from library code.

---

### NumPy vectorization

PyBatch now includes a vectorized row-normalization operation:

```python
normalized = normalize_rows(
    matrix,
    config=config,
)
```

Instead of asking Python to loop over vectors one by one, the operation computes row norms in one NumPy call and relies on broadcasting:

```text
matrix shape: (N, D)
      │
      ├── norm(axis=1, keepdims=True)
      │
      ▼
norms shape: (N, 1)
      │
      ▼
matrix / norms
```

This is an important distinction in PyBatch:

```text
Logical batching
= group multiple jobs together

Vectorization
= perform numerical work across many values together
```

PyBatch supports the first generally and has started introducing the second at the operation layer.

---

## Example

```python
import numpy as np

from pybatch.batching import iter_batches
from pybatch.core.models import Operation, VectorJob
from pybatch.engine import SyncEngine


jobs = (
    VectorJob(
        operation=Operation.NORMALIZE,
        left=np.array([value, 1.0]),
    )
    for value in range(1, 6)
)

engine = SyncEngine()

batches = iter_batches(
    jobs,
    batch_size=2,
)

for result_batch in engine.iter_batch_results(batches):
    for result in result_batch:
        print(result.job_id, result.value)
```

The interesting part is not the output. It is the execution shape:

```text
produce jobs lazily
      ↓
group one batch
      ↓
execute that batch
      ↓
return its results
      ↓
pause until the caller asks again
```

---

## Project structure

```text
src/pybatch/
├── core/             # domain models, errors, protocols
├── operations/       # numerical operations + validation
├── engine/           # handlers and synchronous execution
├── batching/         # lazy fixed-size batching
└── observability/    # decorators and timing scopes
```

Tests live under:

```text
tests/unit/
```

---

## Development

PyBatch uses Python 3.14+ and `uv`.

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest -v
```

Lint:

```bash
uv run ruff check .
```

Check formatting:

```bash
uv run ruff format --check .
```

Run static type checking:

```bash
uv run mypy src
```

---

## Design principles

PyBatch intentionally follows a few rules:

- build a correct synchronous core before adding concurrency
- separate data, execution, and numerical semantics
- prefer domain-specific errors over leaking dependency behavior
- preserve precise types where possible, widen only at heterogeneous boundaries
- accept the smallest useful interface (`Iterable` instead of requiring `list`)
- use laziness when work does not need to be materialized eagerly
- keep library observability opt-in and non-invasive
- vectorize numerical work instead of pushing element-by-element loops through Python when possible

---

## Current status

PyBatch is still intentionally small.

It is **not yet** an async runtime, dynamic batch scheduler, thread/process executor, or model-serving system.

What it does have is the foundation those systems need:

```text
typed work
+ validation
+ dispatch
+ lazy pipelines
+ batching
+ observability
+ numerical execution
```

The goal is to understand each layer well enough that the later concurrency and inference-runtime pieces do not feel like magic.