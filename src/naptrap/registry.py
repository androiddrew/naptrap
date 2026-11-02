"""Model registry: maps a family name to the function that loads it."""

from __future__ import annotations

import os
from typing import Callable, Dict, Tuple

from .types import KeywordModel

EXPERIMENTAL_ENV = "NAPTRAP_EXPERIMENTAL"


def _phonmatchnet(**kwargs) -> KeywordModel:
    from .models.phonmatchnet import PhonMatchNet
    return PhonMatchNet.from_pretrained(**kwargs)


def _zp_kws(**kwargs) -> KeywordModel:
    from .models.zp_kws import ZpKws
    return ZpKws.from_pretrained(**kwargs)


_FACTORIES: Dict[str, Callable[..., KeywordModel]] = {"phonmatchnet": _phonmatchnet}
# Unreleased families, available only when named in NAPTRAP_EXPERIMENTAL
# (comma-separated), e.g. NAPTRAP_EXPERIMENTAL=zp-kws.
_EXPERIMENTAL: Dict[str, Callable[..., KeywordModel]] = {"zp-kws": _zp_kws}


def _enabled() -> Dict[str, Callable[..., KeywordModel]]:
    names = {name.strip().lower() for name in os.environ.get(EXPERIMENTAL_ENV, "").split(",")}
    return {**_FACTORIES, **{name: factory for name, factory in _EXPERIMENTAL.items() if name in names}}


def available_models() -> Tuple[str, ...]:
    return tuple(sorted(_enabled()))


def register_model(name: str, factory: Callable[..., KeywordModel]) -> None:
    """Register an additional model family under a new lowercase name."""
    if not isinstance(name, str) or not name or name != name.strip().lower():
        raise ValueError("Model names must be nonempty, lowercase and whitespace-trimmed")
    if not callable(factory):
        raise TypeError("A model factory must be callable")
    if name in _FACTORIES or name in _EXPERIMENTAL:
        raise ValueError("Model is already registered: " + name)
    _FACTORIES[name] = factory


def load_model(name: str = "phonmatchnet", **kwargs) -> KeywordModel:
    """Load a registered model family; keyword arguments go to its loader.

    For ``phonmatchnet`` see :meth:`naptrap.PhonMatchNet.from_pretrained`.
    """
    factories = _enabled()
    if name not in factories:
        if name in _EXPERIMENTAL:
            raise ValueError(f"Model {name!r} is experimental; set {EXPERIMENTAL_ENV}={name} to enable it")
        raise ValueError(f"Unknown model {name!r}; available: {', '.join(sorted(factories))}")
    return factories[name](**kwargs)
