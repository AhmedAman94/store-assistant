"""Run PydanticAI coroutines from synchronous code, wherever that code happens to be.

`Agent.run_sync` starts an event loop in the calling thread, which fails inside hosts that
already have one running (Streamlit, Jupyter, some test runners). Instead we keep one
background thread with its own event loop for the whole process and submit work to it. The
Anthropic HTTP client stays bound to that single loop, so repeated calls never see a closed or
foreign loop.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _lock:
        if _loop is None or _loop.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="store-assistant-loop", daemon=True
            )
            thread.start()
            _loop = loop
        return _loop


def run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Block until `coro` completes on the shared background loop and return its result."""
    future = asyncio.run_coroutine_threadsafe(coro, _get_loop())
    return future.result()
