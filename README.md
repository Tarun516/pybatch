# PyBatch

PyBatch is a Python vector-processing runtime designed to execute numerical jobs such as vector normalization, dot products, cosine similarity, matrix multiplication, and top-k similarity search.

The project is built around a simple idea:

> Represent numerical work as structured jobs, validate inputs at clear boundaries, and execute them through a reusable engine.

PyBatch is intentionally designed as a runtime rather than a collection of disconnected NumPy utility functions.

---

## What PyBatch Does

A computation is represented using a `VectorJob`.

Example:

```python
job = VectorJob(
    operation=Operation.COSINE_SIMILARITY,
    left=left_vector,
    right=right_vector,
)
```

The job describes:

```text
what operation should run
what input data should be used
what optional parameters are required
```

Execution logic is kept separate from the job itself.

Conceptually:

```text
VectorJob
    │
    ▼
Execution Engine
    │
    ▼
Operation
    │
    ▼
NumPy
    │
    ▼
JobResult
```

This separation allows the same job model to be reused by different execution strategies.

---

## Supported Operations

PyBatch currently provides the following numerical operations.

### Vector Normalization

Converts a vector into a unit vector while preserving its direction.

```python
normalize(vector, config=config)
```

Example:

```text
[3, 4]

   ↓ normalize

[0.6, 0.8]
```

---

### Dot Product

Computes the scalar dot product between two vectors.

```python
dot_product(
    left,
    right,
    config=config,
)
```

For:

```text
left  = [1, 2, 3]
right = [4, 5, 6]
```

the result is:

```text
32
```

---

### Cosine Similarity

Measures the directional similarity between two vectors.

```python
cosine_similarity(
    left,
    right,
    config=config,
)
```

Cosine similarity is commonly used in:

* embedding search
* semantic retrieval
* recommendation systems
* vector databases
* RAG systems

---

### Matrix Multiplication

Performs standard 2D matrix multiplication using NumPy.

```python
matrix_multiply(
    left,
    right,
    config=config,
)
```

Matrix dimensions must satisfy:

```text
(A × B) @ (B × C)

        ↓

(A × C)
```

---

### Top-K Similarity Search

Compares a query vector against multiple candidate vectors and returns the highest-scoring matches.

```python
result = top_k_similarity(
    query,
    candidates,
    top_k=5,
    config=config,
)
```

The result contains:

```python
result.indices
result.scores
```

Internally, similarity scores are calculated using vectorized matrix operations instead of Python loops.

For example:

```text
Candidates
(N × D)

    @

Query
(D)

    ↓

Scores
(N)
```

---

# Architecture

The project separates domain representation, validation, numerical operations, and execution.

```text
                    PyBatch

                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼

      Core         Operations       Engine

        │              │              │
        │              │              │
 VectorJob         normalize      execution
 JobResult         dot_product    dispatch
 Batch             cosine
 EngineConfig      top-k
 Operation         matmul
 Errors
```

---

# Domain Model

## `VectorJob`

Represents a unit of numerical work.

```text
VectorJob
│
├── job_id
├── operation
├── left
├── right
└── top_k
```

Example:

```python
job = VectorJob(
    operation=Operation.DOT_PRODUCT,
    left=left,
    right=right,
)
```

`VectorJob` contains data only.

It does not know how the computation will be executed.

---

## `Operation`

Operations are represented using `StrEnum`.

```python
Operation.NORMALIZE
Operation.DOT_PRODUCT
Operation.COSINE_SIMILARITY
Operation.MATRIX_MULTIPLY
Operation.TOP_K_SIMILARITY
```

Using an enum gives the runtime a controlled set of valid operations instead of relying on arbitrary strings.

---

## `EngineConfig`

Contains numerical and validation configuration shared by operations.

Current configuration includes:

```python
EngineConfig(
    zero_norm_epsilon=1e-12,
    check_finite=True,
)
```

### `zero_norm_epsilon`

Defines the threshold below which a vector is considered to have effectively zero magnitude.

### `check_finite`

Controls whether inputs containing:

```text
NaN
+Infinity
-Infinity
```

are rejected.

---

## `JobResult`

Represents the output produced by executing a job.

```text
JobResult
│
├── job_id
├── operation
└── value
```

Depending on the operation, `value` may contain:

```text
float
NumPy array
TopKResult
```

---

## `Batch`

Represents a fixed collection of jobs.

```python
batch = Batch(
    jobs=(
        job_a,
        job_b,
        job_c,
    )
)
```

A tuple is used so the contents of a batch cannot be accidentally modified after creation.

---

# Validation

Validation is separated from numerical computation.

Shared validation logic lives under:

```text
operations/_validation.py
```

It handles things such as:

```text
empty arrays
invalid dimensions
incompatible vector shapes
invalid matrix shapes
NaN / infinity
invalid top_k
```

This keeps operation implementations focused on the mathematical computation itself.

For example:

```text
dot_product()

validate left vector
validate right vector
validate compatible shape
        ↓
perform dot product
```

---

# Error Handling

PyBatch exposes its own error hierarchy instead of leaking raw NumPy errors as its public contract.

```text
PyBatchError
│
├── UnsupportedOperationError
│
└── ValidationError
    │
    ├── InvalidArrayError
    ├── EmptyArrayError
    ├── InvalidShapeError
    ├── MissingOperandError
    ├── InvalidTopKError
    ├── NonFiniteValueError
    ├── ZeroNormError
    └── InvalidConfigurationError
```

For example, incompatible matrix dimensions produce:

```text
InvalidShapeError
```

rather than exposing an implementation-specific NumPy traceback.

---

# NumPy Design

PyBatch currently standardizes numerical inputs using:

```python
NDArray[np.float64]
```

represented internally through:

```python
type FloatArray = NDArray[np.float64]
```

Inputs stored inside `VectorJob` are copied and marked read-only.

This prevents code outside the job from mutating data after the job has already been created.

Example:

```python
source = np.array([1.0, 2.0])

job = VectorJob(
    operation=Operation.NORMALIZE,
    left=source,
)

source[0] = 100
```

The data inside `job.left` remains unchanged.

---

# Vectorized Computation

PyBatch avoids Python loops for numerical workloads where NumPy can operate on entire arrays.

For example, computing similarity against many candidate vectors uses:

```python
candidates @ query
```

instead of:

```python
for candidate in candidates:
    score = np.dot(candidate, query)
```

If:

```text
candidates.shape = (10000, 768)

query.shape = (768,)
```

then:

```text
(10000 × 768)

       @

(768)

       ↓

(10000)
```

produces one score for every candidate vector.

This keeps heavy numerical computation inside optimized native numerical libraries rather than executing each arithmetic operation through the Python interpreter.

---

# Repository Structure

```text
pybatch/
│
├── pyproject.toml
├── uv.lock
├── README.md
│
├── src/
│   └── pybatch/
│       │
│       ├── core/
│       │   ├── errors.py
│       │   └── models.py
│       │
│       ├── operations/
│       │   ├── _validation.py
│       │   ├── normalize.py
│       │   ├── similarity.py
│       │   └── matmul.py
│       │
│       └── engine/
│
└── tests/
    └── unit/
        ├── test_models.py
        └── test_operations.py
```

---

# Tech Stack

```text
Language
Python 3.14+

Numerical Computing
NumPy

Package / Environment Management
uv

Testing
pytest

Linting / Formatting
Ruff

Static Type Checking
mypy
```

---

# Setup

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Run formatting:

```bash
uv run ruff format .
```

Run type checking:

```bash
uv run mypy src
```

---

# Core Design Principles

PyBatch follows a few simple design rules:

```text
explicit domain models
        ↓
clear validation boundaries
        ↓
domain-specific errors
        ↓
numerical computation
        ↓
separate execution layer
```

A job describes **what work needs to happen**.

An operation defines **the numerical behavior**.

An engine determines **how that work gets executed**.

That separation keeps the numerical semantics independent from the execution strategy.
