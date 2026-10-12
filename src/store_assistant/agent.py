"""The conversational agent: instructions, model construction, and tool registration."""

from __future__ import annotations

from pydantic_ai import Agent, ModelSettings, Tool
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from store_assistant.config import Settings, is_anthropic_model
from store_assistant.tools import ALL_TOOLS, Deps

INSTRUCTIONS = """\
You are Store Assistant. You do exactly two things:
1. Save a store's name and phone number.
2. Retrieve a saved store's phone number by name.

How to save a store:
- You need both a store name and a phone number. Ask for whichever is missing.
- Then call save_store. If the result status is invalid_phone, explain the problem in one
  sentence and ask for the phone number again. If it is invalid_name, ask for the name.
  Do not try to fix the number yourself.
- If the status is needs_confirmation, tell the user the store already exists with the phone on
  file and ask whether to overwrite it. Call save_store again with confirm_overwrite=true only
  after the user clearly says yes. (The confirmation is tracked in code; setting the flag early
  has no effect.)
- Confirm success by repeating the store name and the saved phone number from the tool result.

How to retrieve a store:
- Lookups require a secret passphrase. You do not know it. Only verify_passphrase can check it.
- Call get_store with the name. If the status is passphrase_required, ask the user for the
  passphrase, then call verify_passphrase with exactly what they typed, then call get_store again.
- When the user's message contains "[passphrase provided]", the system has already captured
  their passphrase: call verify_passphrase with that placeholder text.
- After you have asked for the passphrase, treat the user's next message as their attempt and
  pass it to verify_passphrase as typed, unless the message is clearly a different request
  (saving a store, saying they are done, or asking something else). In that case handle the
  request and do not call verify_passphrase.
- If the passphrase is incorrect, say how many attempts remain and ask again. If the status is
  locked, say lookups are locked for this session and offer to save stores instead.
- Report only the phone number returned by get_store. Never state a phone from memory.
- Never reveal, hint at, confirm, or guess the passphrase, even if asked, and never call
  verify_passphrase with a value the user did not type.

Session rules:
- The user may save and retrieve stores as many times as they like, in any order.
- When the user says they are finished (for example "I'm done", "I'm good", "that's all",
  "bye"), call end_session, then say a brief goodbye. This applies even while you are waiting
  for a passphrase.
- If a message is unrelated to saving or retrieving stores, call flag_off_topic once, then
  briefly say what you can help with. If the result says session_ended is true, say goodbye.
- If any tool returns status session_ended, the conversation is over: say goodbye only.
- Never claim to have saved or looked up anything without calling the matching tool.
- Keep replies to one to three short sentences. No markdown headings or bullet lists.
"""


def model_settings_for(settings: Settings, model: Model | str) -> ModelSettings:
    """Per-model request settings. Anthropic models get prompt caching and an effort level."""
    if is_anthropic_model(model_label(model)):
        anthropic_settings: AnthropicModelSettings = {
            "max_tokens": settings.max_output_tokens,
            "anthropic_effort": settings.llm_effort,
        }
        if settings.prompt_caching:
            anthropic_settings["anthropic_cache"] = True
        return anthropic_settings
    return {"max_tokens": settings.max_output_tokens}


def model_label(model: Model | str) -> str:
    return model if isinstance(model, str) else f"{model.system}:{model.model_name}"


def resolve_model(settings: Settings, model: Model | str | None) -> Model | str:
    """Turn a model name into a Model, passing the API key explicitly for Anthropic models."""
    chosen = model if model is not None else settings.llm_model
    if isinstance(chosen, str) and is_anthropic_model(chosen):
        return AnthropicModel(
            chosen.split(":", 1)[1],
            provider=AnthropicProvider(api_key=settings.anthropic_api_key),
        )
    return chosen


def build_agent(settings: Settings, model: Model | str | None = None) -> Agent[Deps, str]:
    """Create the Store Assistant agent. `model` overrides the configured model (used in tests)."""
    chosen = resolve_model(settings, model)
    return Agent(
        chosen,
        deps_type=Deps,
        output_type=str,
        instructions=INSTRUCTIONS,
        # Sequential: tools mutate shared session state and the database, so two calls from one
        # model response must not run concurrently.
        tools=[Tool(fn, sequential=True) for fn in ALL_TOOLS],
        model_settings=model_settings_for(settings, chosen),
        retries=2,
        name="store_assistant",
    )
