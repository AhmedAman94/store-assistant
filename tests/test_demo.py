"""End-to-end: the rule-based demo model drives a full conversation through the real runner."""

from store_assistant.config import Settings
from store_assistant.db import Repository
from store_assistant.demo_model import extract_phone, make_demo_model, make_demo_summary_model
from store_assistant.runner import ConversationRunner


def test_demo_conversation_covers_every_requirement(settings: Settings, repo: Repository) -> None:
    runner = ConversationRunner(
        settings, repo, model=make_demo_model(), summary_model=make_demo_summary_model()
    )
    runner.start()
    replies = {}
    script = [
        ("name_only", "Save a store called Trader Joe's"),
        ("bad_phone", "555-123"),
        ("good_phone", "(415) 555-2671"),
        ("one_shot_save", "Add Whole Foods with phone 212-555-0199"),
        ("lookup", "What's the phone number for Trader Joe's?"),
        ("wrong_pass", "wrong one"),
        ("right_pass", "open sesame"),
        ("lookup2", "look up Whole Foods"),
        ("missing", "find Safeway"),
        ("dup", "save Whole Foods 646-555-0123"),
        ("confirm", "yes"),
        ("off1", "what's the weather like?"),
        ("done", "I'm done"),
    ]
    for key, text in script:
        result = runner.send(text)
        replies[key] = result
        if result.ended:
            break

    assert "phone number for Trader Joe's" in replies["name_only"].reply
    assert "not a valid phone number" in replies["bad_phone"].reply
    assert "+1 415-555-2671" in replies["good_phone"].reply
    assert "Whole Foods" in replies["one_shot_save"].reply
    assert "passphrase" in replies["lookup"].reply.lower()
    assert "2 attempt(s) remaining" in replies["wrong_pass"].reply
    assert "+14155552671" in replies["right_pass"].reply
    assert "+12125550199" in replies["lookup2"].reply
    assert "couldn't find" in replies["missing"].reply
    assert "Overwrite Whole Foods" in replies["dup"].reply
    assert "Updated" in replies["confirm"].reply
    assert "only help" in replies["off1"].reply
    assert replies["done"].ended and replies["done"].end_reason == "user_done"

    stores = {s.name: s.phone_e164 for s in repo.list_stores()}
    assert stores == {"Trader Joe's": "+14155552671", "Whole Foods": "+16465550123"}
    conversation = repo.get_conversation(1)
    assert conversation is not None and conversation.summary
    assert "Trader Joe's" in conversation.summary
    assert "open sesame" not in conversation.summary


def test_demo_model_does_not_end_on_store_names_containing_done(
    settings: Settings, repo: Repository
) -> None:
    runner = ConversationRunner(
        settings, repo, model=make_demo_model(), summary_model=make_demo_summary_model()
    )
    runner.start()
    result = runner.send("save Done Deal Pawn 415-555-0109")
    assert not result.ended and "Done Deal Pawn" in result.reply


def test_demo_model_ends_session_when_done_is_said_at_passphrase_prompt(
    settings: Settings, repo: Repository
) -> None:
    runner = ConversationRunner(
        settings, repo, model=make_demo_model(), summary_model=make_demo_summary_model()
    )
    runner.start()
    runner.send("look up Costco")
    result = runner.send("I'm done")
    assert result.ended and result.end_reason == "user_done"


def test_extract_phone_handles_common_formats() -> None:
    assert extract_phone("call 415-555-2671 please")[0] == "415-555-2671"
    assert extract_phone("+44 20 7946 0958")[0] == "+44 20 7946 0958"
    assert extract_phone("no number here")[0] is None
