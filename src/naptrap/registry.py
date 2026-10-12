"""Model registry: maps a family name to the function that loads it."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from .types import KeywordModel


def _phonmatchnet(**kwargs) -> KeywordModel:
    from .models.phonmatchnet import PhonMatchNet
    return PhonMatchNet.from_pretrained(**kwargs)


_FACTORIES: Dict[str, Callable[..., KeywordModel]] = {"phonmatchnet": _phonmatchnet}


def available_models() -> Tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def register_model(name: str, factory: Callable[..., KeywordModel]) -> None:
    """Register an additional model family under a new lowercase name."""
    if not isinstance(name, str) or not name or name != name.strip().lower():
        raise ValueError("Model names must be nonempty, lowercase and whitespace-trimmed")
    if not callable(factory):
        raise TypeError("A model factory must be callable")
    if name in _FACTORIES:
        raise ValueError("Model is already registered: " + name)
    _FACTORIES[name] = factory


def load_model(name: str = "phonmatchnet", **kwargs) -> KeywordModel:
    """Load a registered model family; keyword arguments go to its loader.

    For ``phonmatchnet`` see :meth:`naptrap.PhonMatchNet.from_pretrained`.
    """
    try:
        factory = _FACTORIES[name]
    except KeyError as error:
        raise ValueError(f"Unknown model {name!r}; available: {', '.join(available_models())}") from error
    return factory(**kwargs)
