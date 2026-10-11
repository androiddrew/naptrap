"""Explicit model registry; future model families can provide their own loaders."""

from __future__ import annotations

from typing import Callable

from .types import KeywordModel


def _phonmatchnet(**kwargs) -> KeywordModel:
    from .models.phonmatchnet import PhonMatchNet
    return PhonMatchNet.from_bundle(**kwargs)


_FACTORIES: dict[str, Callable[..., KeywordModel]] = {"phonmatchnet": _phonmatchnet}


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def register_model(name: str, factory: Callable[..., KeywordModel]) -> None:
    """Register a new family without changing existing implementations."""
    if not isinstance(name, str) or not name or name != name.strip().lower():
        raise ValueError("Model names must be nonempty, lowercase and whitespace-trimmed")
    if not callable(factory):
        raise TypeError("A model factory must be callable")
    if name in _FACTORIES:
        raise ValueError("Model is already registered: " + name)
    _FACTORIES[name] = factory


def load_model(name: str = "phonmatchnet", **kwargs) -> KeywordModel:
    try:
        factory = _FACTORIES[name]
    except KeyError as error:
        raise ValueError(f"Unknown model {name!r}; available: {', '.join(available_models())}") from error
    return factory(**kwargs)
