from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Optional, TypeVar

from redis.asyncio import Redis

from app.core.logging import get_logger
from app.core.settings import Settings
from app.features.intent.ports import ClaimParserPort
from app.features.search.ports import SearchPort
from app.features.verification.ports import ClaimVerifierPort

logger = get_logger(__name__)

T = TypeVar("T")


def _load_symbol(dotted_path: str) -> Any:
    module_path, _, symbol_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid dotted path: {dotted_path}")
    module = import_module(module_path)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ValueError(f"Symbol not found: {dotted_path}") from exc


def _instantiate(dotted_path: str, *, settings: Settings, redis: Optional[Redis]) -> Any:
    cls = _load_symbol(dotted_path)
    if not callable(cls):
        raise ValueError(f"Adapter is not callable: {dotted_path}")

    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        # If signature is not available, call without kwargs.
        return cls()

    kwargs: dict[str, Any] = {}
    for name in sig.parameters.keys():
        if name == "settings":
            kwargs[name] = settings
        elif name == "redis":
            kwargs[name] = redis

    return cls(**kwargs)


@dataclass
class Container:
    settings: Settings
    redis: Optional[Redis]

    _search: Optional[SearchPort] = None
    _intent: Optional[ClaimParserPort] = None
    _verifier: Optional[ClaimVerifierPort] = None

    def search(self) -> SearchPort:
        if self._search is None:
            self._search = _instantiate(self.settings.search_adapter, settings=self.settings, redis=self.redis)
            logger.info(f"[DI] Search adapter: {self.settings.search_adapter}")
        return self._search

    def intent(self) -> ClaimParserPort:
        if self._intent is None:
            self._intent = _instantiate(self.settings.intent_adapter, settings=self.settings, redis=self.redis)
            logger.info(f"[DI] Intent adapter: {self.settings.intent_adapter}")
        return self._intent

    def verifier(self) -> ClaimVerifierPort:
        if self._verifier is None:
            self._verifier = _instantiate(self.settings.verifier_adapter, settings=self.settings, redis=self.redis)
            logger.info(f"[DI] Verifier adapter: {self.settings.verifier_adapter}")
        return self._verifier
