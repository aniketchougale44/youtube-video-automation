"""The chat agent: what it routes to, and what it refuses to let a model decide.

The model picks WHICH tool runs; it never composes an answer and it has no tool that publishes.
That split is what these tests pin, because the two failure modes are very different in cost. A
misrouted question is answerable with "no, I meant X". A misrouted publish puts a video on
somebody's channel, and there is no undo for an audience that already saw it.
"""
import types
import uuid

import pytest

from api.routes import chat as chat_mod
from api.routes.chat import _review_text


def _card(**over) -> dict:
    card = {
        "id": "11111111-1111-1111-1111-111111111111",
        "short_id": "11111111",
        "title": "Counting to Ten with Friendly Animals",
        "description": "A gentle counting song.",
        "script_summary": "Count along!",
        "render_url": "/media/run/final.mp4",
        "thumbnails": [{"index": 0, "url": "/media/run/thumb0.png"}],
        "compliance": {
            "copyright_risk_score": 3,
            "av_sync_ok": True,
            "spec_check_ok": True,
            "passed": True,
            "blocking_reasons": [],
            "community_guidelines_flags": [],
        },
    }
    card.update(over)
    return card


_USER = types.SimpleNamespace(
    id=uuid.UUID("99999999-9999-9999-9999-999999999999"), email="probe@example.com"
)


@pytest.fixture
def routed(monkeypatch):
    """Drives the tool choice from the test instead of the model."""
    box = {"choice": None}
    monkeypatch.setattr(chat_mod, "choose_tool", lambda *a, **k: box["choice"])
    return box


@pytest.fixture
def effects(monkeypatch):
    """Captures what would have been handed to Celery."""
    seen = {"resume": [], "runs": []}
    monkeypatch.setattr(
        chat_mod, "_resume",
        lambda run_id, approved, notes="": seen["resume"].append((run_id, approved, notes)),
    )
    monkeypatch.setattr(chat_mod, "_enqueue", lambda run_id: seen["runs"].append(run_id))
    monkeypatch.setattr(
        chat_mod, "create_run",
        lambda db, **kw: types.SimpleNamespace(id=uuid.uuid4(), **kw),
    )
    # Per-account limits are enforced before a run is created and are covered by their own tests;
    # here they would just query a database these tests do not have.
    monkeypatch.setattr(chat_mod, "check_can_start_run", lambda db, user_id: None)
    return seen


def _pending(monkeypatch, card):
    monkeypatch.setattr(chat_mod, "_pending_review", lambda db, user_id=None: card)


# --------------------------------------------------------------------- the review text


def test_review_text_leads_with_the_title_and_offers_the_three_decisions():
    text = _review_text(_card())
    assert "Counting to Ten with Friendly Animals" in text
    assert "upload it" in text and "scrap it" in text and "change the title" in text


def test_review_text_surfaces_a_compliance_failure_with_its_reasons():
    text = _review_text(_card(compliance={
        "copyright_risk_score": 9, "av_sync_ok": True, "spec_check_ok": True,
        "passed": False, "blocking_reasons": ["third-party music"],
        "community_guidelines_flags": [],
    }))
    assert "did **not** pass" in text and "third-party music" in text


def test_review_text_warns_when_there_is_no_file_to_play():
    assert "nothing to play" in _review_text(_card(render_url=None))


def test_review_text_survives_an_unreadable_checkpoint():
    assert "could not read the compliance result" in _review_text(_card(compliance=None))


# --------------------------------------------------------------------- publishing


@pytest.mark.parametrize("phrase", ["upload it", "publish it", "yes", "looks good", "Ship it."])
def test_an_explicit_confirmation_publishes_without_consulting_the_model(
    monkeypatch, routed, effects, phrase
):
    """The model is not asked at all on this path. It cannot cause a publish by misreading a
    sentence, because it never sees one."""
    _pending(monkeypatch, _card())
    routed["choice"] = ("GetHelp", {})  # would be used if the agent were reached; it must not be

    reply = chat_mod._answer(phrase, None, _USER)

    assert effects["resume"] == [("11111111-1111-1111-1111-111111111111", True, phrase)]
    assert reply.intent == "APPROVE"


@pytest.mark.parametrize("phrase", ["scrap it", "reject", "no", "discard it"])
def test_an_explicit_refusal_scraps_without_consulting_the_model(monkeypatch, routed, effects, phrase):
    _pending(monkeypatch, _card())
    routed["choice"] = ("GetHelp", {})

    chat_mod._answer(phrase, None, _USER)

    assert effects["resume"] == [("11111111-1111-1111-1111-111111111111", False, phrase)]


def test_no_tool_the_model_can_choose_resumes_a_run(monkeypatch, effects):
    """The guarantee, asserted on behaviour rather than on tool names.

    Resuming is what publishes, so this drives every tool the model is allowed to pick -- with a
    video sitting there waiting to be published -- and checks none of them reach it. A new tool
    added later that can publish fails here, which is the point.
    """
    _pending(monkeypatch, _card())
    monkeypatch.setattr(chat_mod, "_ideas_answer", lambda db: [])
    monkeypatch.setattr(chat_mod, "_status_answer", lambda db, user_id=None: "status")
    monkeypatch.setattr(chat_mod, "list_runs", lambda db, **kw: [])

    assert set(chat_mod._HANDLERS) == {t.__name__ for t in chat_mod._TOOLS}, (
        "every offered tool needs a handler, and every handler needs to be an offered tool"
    )

    for name, handler in chat_mod._HANDLERS.items():
        args = {"topic": "a topic", "story": "word " * 40}
        handler(args, None, _USER)
        assert effects["resume"] == [], f"{name} resumed a run — that publishes"


@pytest.mark.parametrize(
    "message",
    [
        "upload a video of story for children",
        "upload a story for kids",
        "publish a new video about counting to ten for toddlers",
    ],
)
def test_a_request_describing_new_content_never_publishes(monkeypatch, routed, effects, message):
    """These open with a publish verb and once reached APPROVE. They describe something that does
    not exist yet, so the only acceptable outcome is that nothing is published."""
    _pending(monkeypatch, _card())
    routed["choice"] = ("StartVideo", {"topic": "a story for kids"})

    chat_mod._answer(message, None, _USER)

    assert effects["resume"] == [], f"{message!r} must not publish"


def test_a_confirmation_with_nothing_waiting_does_nothing(monkeypatch, routed, effects):
    _pending(monkeypatch, None)
    routed["choice"] = ("GetHelp", {})

    reply = chat_mod._answer("upload it", None, _USER)

    assert effects["resume"] == []
    assert "Nothing is waiting" in reply.reply


# --------------------------------------------------------------------- routing


def test_a_named_topic_starts_a_run_owned_by_the_caller(monkeypatch, routed, effects):
    _pending(monkeypatch, None)
    routed["choice"] = ("StartVideo", {"topic": "counting to ten"})

    reply = chat_mod._answer("make a video about counting to ten", None, _USER)

    assert len(effects["runs"]) == 1
    assert "counting to ten" in reply.reply
    assert reply.run_id


def test_a_topicless_start_asks_rather_than_inventing_a_subject(monkeypatch, routed, effects):
    _pending(monkeypatch, None)
    routed["choice"] = ("StartVideo", {"topic": ""})

    reply = chat_mod._answer("make me a video", None, _USER)

    assert effects["runs"] == []
    assert "What should the video be about" in reply.reply


def test_a_story_too_short_to_use_is_refused_before_a_run_is_created(monkeypatch, routed, effects):
    _pending(monkeypatch, None)
    routed["choice"] = ("StartFromStory", {"story": "once upon a time"})

    reply = chat_mod._answer("once upon a time", None, _USER)

    assert effects["runs"] == []
    assert "too short" in reply.reply


def test_showing_the_review_returns_the_card_for_the_page_to_render(monkeypatch, routed):
    card = _card()
    _pending(monkeypatch, card)
    routed["choice"] = ("ShowPendingReview", {})

    reply = chat_mod._answer("let me see the video", None, _USER)

    assert reply.review == card and reply.run_id == card["id"]


def test_an_unknown_tool_name_falls_back_to_help_rather_than_failing(monkeypatch, routed):
    _pending(monkeypatch, None)
    routed["choice"] = ("SomeToolThatDoesNotExist", {})

    assert "Here's what I can do" in chat_mod._answer("???", None, _USER).reply


def test_when_the_model_declines_to_route_the_operator_still_gets_help(monkeypatch, routed):
    _pending(monkeypatch, None)
    routed["choice"] = None

    assert "Here's what I can do" in chat_mod._answer("hello", None, _USER).reply


def test_status_is_answered_from_the_database_when_the_model_is_down(monkeypatch):
    """The question someone asks when things look wrong is "what's happening" -- which is exactly
    when the model is most likely to be the thing that is wrong."""
    monkeypatch.setattr(chat_mod, "_pending_review", lambda db, user_id=None: None)
    monkeypatch.setattr(
        chat_mod, "choose_tool",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("every provider failed")),
    )
    monkeypatch.setattr(chat_mod, "_status_answer", lambda db, user_id=None: "2 runs in flight")

    reply = chat_mod._answer("what is happening", None, _USER)

    assert reply.reply == "2 runs in flight"
    assert reply.intent == "STATUS"
