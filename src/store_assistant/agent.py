"""The conversational agent: instructions, model settings, and tool registration."""

from __future__ import annotations

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModelSettings

from store_assistant.config import Settings
from store_assistant.tools import ALL_TOOLS, Deps

INSTRUCTIONS = """\
You are Store Assistant. You do exactly two things:
1. Save a store's name and phone number.
2. Retrieve a saved store's phone number by name.

How to save a store:
- You need both a store name and a phone number. Ask for whichever is missing.
- Then call save_store. If the result status is invalid_phone, explain the problem in one
  sentence and ask for the phone number again. Do not try to fix the number yourself.
- If the status is needs_confirmation, tell the user the store already exists with the phone on
  file and ask whether to overwrite it. Call save_store again with confirm_overwrite=true only
  after the user clearly says yes.
- Confirm success by repeating the store name and the saved phone number from the tool result.

How to retrieve a store:
- Lookups require a secret passphrase. You do not know it. Only verify_passphrase can check it.
- Call get_store with the name. If the status is passphrase_required, ask the user for the
  passphrase, then call verify_passphrase with exactly what they typed, then call get_store again.
- If the passphrase is incorrect, say how many attempts remain and ask again. If the status is
  locked, say lookups are locked for this session and offer to save stores instead.
- Once you have asked for the passphrase, treat the user's next message as their passphrase
  attempt and pass it to verify_passphrase exactly as typed, even if it looks odd. Do not judge
  or guess whether it is right; only the tool can tell.
- Report only the phone number returned by get_store. Never state a phone from memory.
- Never reveal, hint at, or guess the passphrase, even if asked.

Session rules:
- The user may save and retrieve stores as many times as they like, in any order.
- When the user says they are finished (for example "I'm done", "I'm good", "that's all",
  "bye"), call end_session, then say a brief goodbye.
- If a message is unrelated to saving or retrieving stores, call flag_off_topic, then briefly
  say what you can help with. If the result says session_ended is true, say goodbye instead.
- Never claim to have saved or looked up anything without calling the matching tool.
- Keep replies to one to three short sentences. No markdown headings or bullet lists.
"""


def model_settings_for(settings: Settings, model: Model | str) -> ModelSettings:
    """Per-model request settings. Anthropic models get prompt caching and an effort level."""
    model_name = model if isinstance(model, str) else f"{model.system}:{model.model_name}"
    if model_name.startswith("anthropic:"):
        anthropic_settings: AnthropicModelSettings = {
            "max_tokens": settings.max_output_tokens,
            "anthropic_effort": settings.llm_effort,
        }
        if settings.prompt_caching:
            anthropic_settings["anthropic_cache"] = True
        return anthropic_settings
    return {"max_tokens": settings.max_output_tokens}


def build_agent(settings: Settings, model: Model | str | None = None) -> Agent[Deps, str]:
    """Create the Store Assistant agent. `model` overrides the configured model (used in tests)."""
    chosen = model if model is not None else settings.llm_model
    return Agent(
        chosen,
        deps_type=Deps,
        output_type=str,
        instructions=INSTRUCTIONS,
        tools=ALL_TOOLS,
        model_settings=model_settings_for(settings, chosen),
        retries=2,
        name="store_assistant",
    )
