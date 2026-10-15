"""Covers the chat review loop: a finished video is shown, then approved, revised or scrapped
from the conversation.

The DB and Celery seams are stubbed -- what matters here is the routing. A misrouted intent in
this endpoint means either a video publishes when the operator asked for a change, or a change
silently does nothing to a video about to go public, so each intent asserts on the *effect*
(which task got handed what) rather than on the wording of the reply.
"""
import pytest

from api.routes import chat as chat_mod
from api.routes.chat import ChatRequest, _Intent, _review_text


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


@pytest.fixture
def intents(monkeypatch):
    """Drives the classifier from the test instead of the LLM."""
    box = {}

    def _fake(prompt, output_model, system=None):
        return box["parsed"]

    monkeypatch.setattr(chat_mod, "call_structured", _fake)
    return box


@pytest.fixture
def effects(monkeypatch):
    """Captures what would have been handed to Celery."""
    seen = {"resume": [], "revise": []}
    monkeypatch.setattr(
        chat_mod, "_resume",
        lambda run_id, approved, notes="": seen["resume"].append((run_id, approved, notes)),
    )
    monkeypatch.setattr(
        chat_mod, "_revise",
        lambda run_id, title="", description="": seen["revise"].append((run_id, title, description)),
    )
    return seen


def _pending(monkeypatch, card):
    monkeypatch.setattr(chat_mod, "_pending_review", lambda db: card)


# --------------------------------------------------------------------- _review_text


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
    assert "did **not** pass" in text
    assert "third-party music" in text


def test_review_text_warns_when_there_is_no_file_to_play():
    # An approve on a run with no render is an upload of nothing -- say so before they click.
    assert "nothing to play" in _review_text(_card(render_url=None))


def test_review_text_survives_an_unreadable_checkpoint():
    assert "could not read the compliance result" in _review_text(_card(compliance=None))


# --------------------------------------------------------------------- routing


def test_review_returns_the_card_for_the_dashboard_to_render(monkeypatch, intents):
    card = _card()
    _pending(monkeypatch, card)
    intents["parsed"] = _Intent(intent="REVIEW")

    reply = chat_mod.chat(ChatRequest(message="show me the video"), db=None)

    assert reply.intent == "REVIEW"
    assert reply.review == card
    assert reply.run_id == card["id"]


def test_approve_resumes_the_run_as_approved(monkeypatch, intents, effects):
    _pending(monkeypatch, _card())
    intents["parsed"] = _Intent(intent="APPROVE")

    reply = chat_mod.chat(ChatRequest(message="upload it"), db=None)

    assert effects["resume"] == [("11111111-1111-1111-1111-111111111111", True, "upload it")]
    assert reply.intent == "APPROVE"


@pytest.mark.parametrize(
    "message",
    [
        "upload a video of story for children",  # observed classifying as APPROVE
        "upload a story for kids",
        "upload a video about dinosaurs",
        "publish a new video about counting to ten for toddlers",
    ],
)
def test_a_request_describing_new_content_never_publishes(monkeypatch, intents, effects, message):
    """The guard that stops "upload a story for kids" publishing the video already parked.

    These all open with a publish verb and were seen reaching APPROVE. Each one describes
    something that does not exist yet, so the only acceptable outcome is to ask -- an approval
    that rests on the classifier parsing a sentence correctly is an unwanted upload waiting to
    happen.
    """
    _pending(monkeypatch, _card())
    intents["parsed"] = _Intent(intent="APPROVE")

    reply = chat_mod.chat(ChatRequest(message=message), db=None)

    assert effects["resume"] == [], f"{message!r} must not publish"
    assert "want to be sure which you mean" in reply.reply
    assert reply.review is not None


@pytest.mark.parametrize("message", ["upload it", "publish it", "yes", "looks good, send it", "ship it"])
def test_a_bare_confirmation_still_publishes(monkeypatch, intents, effects, message):
    """The guard must not make approving tedious -- a short confirmation goes straight through."""
    _pending(monkeypatch, _card())
    intents["parsed"] = _Intent(intent="APPROVE")

    chat_mod.chat(ChatRequest(message=message), db=None)

    assert effects["resume"] == [("11111111-1111-1111-1111-111111111111", True, message)]


def test_reject_resumes_the_run_as_not_approved(monkeypatch, intents, effects):
    _pending(monkeypatch, _card())
    intents["parsed"] = _Intent(intent="REJECT")

    chat_mod.chat(ChatRequest(message="scrap it"), db=None)

    assert effects["resume"] == [("11111111-1111-1111-1111-111111111111", False, "scrap it")]


def test_revise_declines_instead_of_touching_a_parked_run(monkeypatch, intents, effects):
    """In-place editing is not wired up. What matters is that asking for one cannot publish.

    The mechanism that was tried wrote the checkpoint, which destroyed the pending interrupt, and
    rebuilding it meant re-entering the approval gate -- where an already-queued decision sends
    the run to upload. So a revise must do nothing at all until the edit can ride along in the
    ApprovalDecision.
    """
    _pending(monkeypatch, _card())
    intents["parsed"] = _Intent(intent="REVISE", new_title="Ten Little Animals")

    reply = chat_mod.chat(ChatRequest(message="change the title to Ten Little Animals"), db=None)

    assert effects["resume"] == [], "a revise must never resume the graph"
    assert effects["revise"] == [], "a revise must not write the parked checkpoint"
    assert "can't edit" in reply.reply
    assert "Ten Little Animals" in reply.reply  # their words are echoed back, not silently dropped


def test_revise_without_replacement_text_asks_instead_of_inventing_one(monkeypatch, intents, effects):
    _pending(monkeypatch, _card())
    intents["parsed"] = _Intent(intent="REVISE")

    reply = chat_mod.chat(ChatRequest(message="the title is wrong"), db=None)

    assert effects["revise"] == [] and effects["resume"] == []
    assert "what should it say instead" in reply.reply
    assert reply.review is not None  # the card stays on screen while they decide


@pytest.mark.parametrize("intent", ["APPROVE", "REJECT", "REVIEW", "REVISE"])
def test_decisions_are_inert_when_nothing_is_waiting(monkeypatch, intents, effects, intent):
    _pending(monkeypatch, None)
    intents["parsed"] = _Intent(intent=intent, new_title="Whatever")

    reply = chat_mod.chat(ChatRequest(message="upload it"), db=None)

    assert effects["resume"] == [] and effects["revise"] == []
    assert "Nothing is waiting" in reply.reply
