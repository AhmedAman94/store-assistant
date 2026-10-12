"""A rule-based stand-in for the language model, used by `--demo` and by end-to-end tests.

It makes zero API calls. It reads the conversation the same way a real model would (the message
history PydanticAI passes to every model) and decides which tool to call with simple keyword
rules, then phrases a reply from the tool's structured result. Everything else, including the
tools, guardrails, database, and summary flow, is the real code path.
"""

from __future__ import annotations

import re
from typing import Any

import phonenumbers
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

DONE_RE = re.compile(
    r"\b(i'?m done|i am done|i'?m good|i am good|that'?s all|that is all|nothing else|"
    r"no thanks|goodbye|bye|done|quit|exit)\b",
    re.I,
)
YES_RE = re.compile(r"^\s*(y|yes|yeah|yep|yup|sure|ok|okay|confirm|go ahead|do it|please)\b", re.I)
NO_RE = re.compile(r"^\s*(n|no|nope|nah|cancel|keep|don'?t)\b", re.I)
LOOKUP_RE = re.compile(r"\b(?:look\s*up|lookup|find|retrieve|fetch|search(?: for)?)\s+(.+)", re.I)
LOOKUP_PHONE_RE = re.compile(
    r"\b(?:what(?:'s| is)|get|show|give me|tell me)\s+(?:the\s+)?"
    r"(?:phone\s+(?:number\s+)?|number\s+|contact\s+)(?:for|of)\s+(.+)",
    re.I,
)
SAVE_RE = re.compile(r"\b(?:save|add|store|remember|record|create|register)\s+(.+)", re.I)
ASK_PHONE_RE = re.compile(r"phone number for ([^?.!]+?)\?$")
ASK_PASS_RE = re.compile(r"passphrase to look up ([^?.!]+?)\.$")
ASK_OVERWRITE_RE = re.compile(r"Overwrite ([^?.!]+?) with (\+?[\d().\- ]+?)\?$")


def _text(parts: list[Any]) -> str:
    return " ".join(p.content for p in parts if isinstance(p, TextPart)).strip()


def _last_assistant_text(messages: list[ModelMessage]) -> str:
    for message in reversed(messages[:-1]):
        if isinstance(message, ModelResponse):
            text = _text(list(message.parts))
            if text:
                return text
    return ""


def _find_tool_call(messages: list[ModelMessage], tool_call_id: str) -> ToolCallPart | None:
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart) and part.tool_call_id == tool_call_id:
                    return part
    return None


def _as_dict(content: Any) -> dict[str, Any]:
    if hasattr(content, "model_dump"):
        return dict(content.model_dump())
    if isinstance(content, dict):
        return dict(content)
    return {"message": str(content)}


def extract_phone(text: str) -> tuple[str | None, str]:
    """Return (phone text, remaining text) using libphonenumber's matcher, with a loose fallback."""
    for match in phonenumbers.PhoneNumberMatcher(
        text, "US", leniency=phonenumbers.Leniency.POSSIBLE
    ):
        raw = match.raw_string
        return raw, text.replace(raw, " ")
    loose = re.search(r"\+?\d[\d\s().-]{4,}\d", text)
    if loose:
        return loose.group(0), text.replace(loose.group(0), " ")
    return None, text


def _clean_name(candidate: str) -> str:
    name = candidate.strip(" ,.:;-\"'")
    name = re.sub(r"^(?:a|an|the|new|my)\s+", "", name, flags=re.I)
    name = re.sub(r"^(?:store|shop)\s+(?:called|named)\s+", "", name, flags=re.I)
    name = re.sub(r"^(?:called|named)\s+", "", name, flags=re.I)
    name = re.sub(
        r"^(?:the\s+)?(?:phone\s+)?(?:number|phone)\s+(?:for|of)\s+", "", name, flags=re.I
    )
    name = re.sub(r"^(?:for|of)\s+", "", name, flags=re.I)
    name = re.split(r"\s+(?:with|whose|at|phone|number)\b|,|:", name, maxsplit=1, flags=re.I)[0]
    name = re.sub(r"(?:'s)?\s+(?:phone|number|phone number)\s*$", "", name, flags=re.I)
    name = re.sub(r"^(?:the\s+)?store\s+", "", name, flags=re.I)
    return name.strip(" ,.:;-\"'?")


def _decide(messages: list[ModelMessage]) -> ModelResponse:
    last = messages[-1]
    assert isinstance(last, ModelRequest)
    tool_returns = [p for p in last.parts if isinstance(p, ToolReturnPart)]
    if tool_returns:
        return _after_tool(messages, tool_returns[-1])
    user_parts = [p for p in last.parts if isinstance(p, UserPromptPart)]
    text = user_parts[-1].content if user_parts else ""
    text = text if isinstance(text, str) else ""
    return _on_user_text(messages, text)


def _on_user_text(messages: list[ModelMessage], text: str) -> ModelResponse:
    previous = _last_assistant_text(messages)

    if m := ASK_PHONE_RE.search(previous):
        name = m.group(1)
        phone, _ = extract_phone(text)
        if phone:
            return _call("save_store", {"name": name, "phone": phone})
        return _say(f"I still need a phone number. What's the phone number for {name}?")

    if m := ASK_PASS_RE.search(previous):
        return _call("verify_passphrase", {"passphrase": text.strip()})

    if m := ASK_OVERWRITE_RE.search(previous):
        name, phone = m.group(1), m.group(2)
        if YES_RE.match(text):
            return _call("save_store", {"name": name, "phone": phone, "confirm_overwrite": True})
        if NO_RE.match(text):
            return _say(f"Okay, I left {name} unchanged. Anything else?")

    if DONE_RE.search(text):
        return _call("end_session", {})

    if m := LOOKUP_PHONE_RE.search(text) or LOOKUP_RE.search(text):
        name = _clean_name(m.group(1))
        if name:
            return _call("get_store", {"name": name})
        return _say("Which store should I look up?")

    if m := SAVE_RE.search(text):
        phone, rest = extract_phone(text)
        save_match = SAVE_RE.search(rest)
        name = _clean_name(save_match.group(1)) if save_match else ""
        if name and phone:
            return _call("save_store", {"name": name, "phone": phone})
        if name:
            return _say(f"Sure. What's the phone number for {name}?")
        return _say("Sure. What's the store name and phone number?")

    return _call("flag_off_topic", {})


def _after_tool(messages: list[ModelMessage], tool_return: ToolReturnPart) -> ModelResponse:
    data = _as_dict(tool_return.content)
    status = data.get("status")
    name = data.get("store_name", "")
    call = _find_tool_call(messages, tool_return.tool_call_id)
    args = call.args_as_dict() if call is not None else {}

    if tool_return.tool_name == "save_store":
        if status in ("saved", "updated"):
            return _say(f"{data.get('message', '')} Anything else?")
        if status == "needs_confirmation":
            return _say(
                f"{name} already exists with phone {data.get('existing_phone')}. "
                f"Overwrite {name} with {args.get('phone', '')}?"
            )
        return _say(f"{data.get('message', '')} What's the phone number for {name}?")

    if tool_return.tool_name == "verify_passphrase":
        pending = ASK_PASS_RE.search(_last_assistant_text(messages))
        pending_name = pending.group(1) if pending else ""
        if status in ("verified", "already_verified") and pending_name:
            return _call("get_store", {"name": pending_name})
        if status == "incorrect":
            return _say(
                f"{data.get('message', '')} Please enter the passphrase to look up {pending_name}."
            )
        if status == "locked":
            return _say("Lookups are locked for this session. I can still save stores for you.")
        return _say("Passphrase accepted. Which store should I look up?")

    if tool_return.tool_name == "get_store":
        if status == "found":
            return _say(f"{name}'s phone number is {data.get('phone')}. Anything else?")
        if status == "passphrase_required":
            return _say(f"Please enter the passphrase to look up {name}.")
        if status == "locked":
            return _say("Lookups are locked for this session. I can still save stores for you.")
        return _say(f"I couldn't find a store named {name}. Would you like to save it?")

    if tool_return.tool_name == "flag_off_topic":
        if data.get("session_ended"):
            return _say("This conversation has stayed off-topic, so I'll wrap up here. Goodbye!")
        remaining = data.get("strikes_remaining", 0)
        return _say(
            "I can only help save or look up store phone numbers. "
            f"({remaining} more off-topic message(s) will end the session.)"
        )

    if tool_return.tool_name == "end_session":
        return _say("Goodbye! I've saved a summary of our conversation.")

    return _say("Okay. What would you like to do next?")


def _say(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _call(tool_name: str, args: dict[str, Any]) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])


def make_demo_model() -> FunctionModel:
    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return _decide(messages)

    return FunctionModel(function, model_name="demo")


def make_demo_summary_model() -> FunctionModel:
    """Writes a summary from the operations block in the summarizer's prompt."""

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ""
        for part in messages[-1].parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                prompt = part.content
        ops_match = re.search(r"Operations performed:\n(.*?)\n\n", prompt, re.S)
        why_match = re.search(r"The conversation ended because (.+?)\.$", prompt.strip())
        ops = [
            line[2:]
            for line in (ops_match.group(1).splitlines() if ops_match else [])
            if line.startswith("- ")
        ]
        ops = [op for op in ops if op != "(none)"]
        body = "; ".join(ops) if ops else "no store operations were completed"
        why = why_match.group(1) if why_match else "the session ended"
        return _say(f"In this session: {body}. The conversation ended because {why}.")

    return FunctionModel(function, model_name="demo-summary")
