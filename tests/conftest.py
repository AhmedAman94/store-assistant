"""Shared fixtures. No test here calls a real model; see test_live.py for the opt-in live check.

Settings are built with `_env_file=None` and every Settings environment variable is removed, so
the suite behaves identically regardless of the developer's `.env` or shell.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from store_assistant.config import Settings
from store_assistant.db import Repository
from store_assistant.session import SessionState
from store_assistant.tools import Deps

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
    monkeypatch.chdir(tmp_path)  # no .env in reach


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        db_path=tmp_path / "test.db",
        store_passphrase="open sesame",
        llm_model="test:test",
        anthropic_api_key=None,
        logfire_token=None,
    )


@pytest.fixture
def repo(settings: Settings) -> Iterator[Repository]:
    repository = Repository(settings.db_path)
    yield repository
    repository.close()


@pytest.fixture
def state() -> SessionState:
    return SessionState(conversation_id=1)


@pytest.fixture
def deps(settings: Settings, repo: Repository, state: SessionState) -> Deps:
    return Deps(settings=settings, repo=repo, state=state)


@dataclass
class FakeCtx:
    """Stand-in for RunContext: the tools only read `ctx.deps`."""

    deps: Deps


@pytest.fixture
def ctx(deps: Deps) -> RunContext[Deps]:
    return cast(RunContext[Deps], FakeCtx(deps=deps))


def say(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def call(tool_name: str, **args: Any) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])


def scripted_model(*responses: ModelResponse, name: str = "scripted") -> FunctionModel:
    """A model that replays the given responses in order, then answers 'ok' forever."""
    queue = deque(responses)

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return queue.popleft() if queue else say("ok")

    return FunctionModel(function, model_name=name)


def text_model(text: str) -> FunctionModel:
    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return say(text)

    return FunctionModel(function, model_name="text")
