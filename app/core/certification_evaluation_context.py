from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_CERTIFICATION_EVALUATION_ACTIVE: ContextVar[bool] = ContextVar(
    "certification_evaluation_active",
    default=False,
)


def certification_evaluation_active() -> bool:
    return _CERTIFICATION_EVALUATION_ACTIVE.get() is True


@contextmanager
def certification_evaluation_context() -> Iterator[None]:
    """Thread/task-local boundary that suppresses local evaluation effects."""
    token = _CERTIFICATION_EVALUATION_ACTIVE.set(True)
    try:
        yield
    finally:
        _CERTIFICATION_EVALUATION_ACTIVE.reset(token)
