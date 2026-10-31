"""Registry of experiment tasks, keyed by ``experiment.task`` in the config."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig

TaskFn = Callable[[DictConfig], dict[str, Any]]
TASKS: dict[str, TaskFn] = {}


def task(name: str) -> Callable[[TaskFn], TaskFn]:
    def register(fn: TaskFn) -> TaskFn:
        TASKS[name] = fn
        return fn

    return register
