from pydantic_ai.models.test import TestModel

from store_assistant.agent import INSTRUCTIONS, build_agent, model_settings_for
from store_assistant.config import Settings
from store_assistant.tools import Deps


def test_instructions_never_contain_the_passphrase(settings: Settings) -> None:
    assert settings.store_passphrase.lower() not in INSTRUCTIONS.lower()


def test_anthropic_models_get_caching_and_effort(settings: Settings) -> None:
    settings.llm_effort = "low"
    anthropic = model_settings_for(settings, "anthropic:claude-opus-5-5")
    assert anthropic["max_tokens"] == settings.max_output_tokens
    assert anthropic.get("anthropic_cache") is True
    assert anthropic.get("anthropic_effort") == "low"
    other = model_settings_for(settings, "test:test")
    assert "anthropic_cache" not in other


def test_agent_exposes_exactly_the_five_tools(deps: Deps, settings: Settings) -> None:
    model = TestModel(call_tools=[])
    agent = build_agent(settings, model)
    agent.run_sync("hello", deps=deps)
    params = model.last_model_request_parameters
    assert params is not None
    assert sorted(t.name for t in params.function_tools) == [
        "end_session",
        "flag_off_topic",
        "get_store",
        "save_store",
        "verify_passphrase",
    ]
