"""Isolation between accounts, at the layer where it is actually enforced.

The danger in this app is not only that one account reads another's data -- it is that approving
a review *publishes a video*. If an approval could ever reach somebody else's run, one user could
push another user's video to a channel they do not own. So the queries that feed the review
queue, and the credentials the upload runs under, are the things pinned here.

No database: these assert on the SQL that gets built and on the credential context, which is
where the mistakes would be.
"""
import uuid

import pytest

from tools import youtube as youtube_tool

# --------------------------------------------------------------------- credential scoping


def test_an_upload_runs_under_the_credentials_it_was_given():
    with youtube_tool.acting_for("their-client", "their-secret", "their-refresh", "UCtheirs"):
        creds = youtube_tool._oauth_credentials()
    assert creds["client_id"] == "their-client"
    assert creds["refresh_token"] == "their-refresh"
    assert creds["channel_id"] == "UCtheirs"


def test_credentials_do_not_leak_past_the_block_they_were_set_for(monkeypatch):
    """A worker process handles one task after another. If a run's credentials outlived the run,
    the next task -- possibly another user's -- would upload with them."""
    import types

    monkeypatch.setattr(
        youtube_tool, "get_settings",
        lambda: types.SimpleNamespace(
            youtube_client_id="env-client", youtube_client_secret="env-secret",
            youtube_refresh_token="env-refresh", youtube_channel_id="UCenv",
        ),
    )
    with youtube_tool.acting_for("user-client", "user-secret", "user-refresh", "UCuser"):
        assert youtube_tool._oauth_credentials()["client_id"] == "user-client"
    assert youtube_tool._oauth_credentials()["client_id"] == "env-client"


def test_credentials_are_restored_even_when_the_run_raises(monkeypatch):
    import types

    monkeypatch.setattr(
        youtube_tool, "get_settings",
        lambda: types.SimpleNamespace(
            youtube_client_id="env-client", youtube_client_secret="env-secret",
            youtube_refresh_token="env-refresh", youtube_channel_id="UCenv",
        ),
    )
    with pytest.raises(RuntimeError), youtube_tool.acting_for("u", "s", "r", "UC"):
        raise RuntimeError("the pipeline failed mid-run")
    assert youtube_tool._oauth_credentials()["client_id"] == "env-client"


# --------------------------------------------------------------------- query scoping


def _compiled(stmt) -> str:
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


def test_listing_runs_for_a_user_filters_on_the_owner():
    """The review queue is built from this. An unfiltered query here is what would put one
    account's finished video in another account's approval queue, where approving it publishes."""
    from unittest.mock import MagicMock

    from db.crud import list_runs

    db = MagicMock()
    user_id = uuid.uuid4()
    list_runs(db, user_id=user_id)
    sql = _compiled(db.execute.call_args[0][0])
    # Matching on the WHERE clause, not on the substring "user_id" -- that now appears in the
    # SELECT column list too, so a looser check passes even with no filter at all.
    assert "WHERE runs.user_id" in sql


def test_listing_runs_without_a_user_is_unfiltered_on_purpose():
    """The scheduler and the operator view want every run. This is asserted so that the absence
    of a filter stays a deliberate choice rather than becoming an accident nobody notices."""
    from unittest.mock import MagicMock

    from db.crud import list_runs

    db = MagicMock()
    list_runs(db)
    assert "WHERE" not in _compiled(db.execute.call_args[0][0])


def test_a_run_is_created_owned_by_whoever_asked_for_it():
    from unittest.mock import MagicMock

    from db.crud import create_run

    db = MagicMock()
    user_id = uuid.uuid4()
    run = create_run(db, user_topic="a story for kids", user_id=user_id)
    assert run.user_id == user_id


def test_a_scheduled_run_has_no_owner_rather_than_a_guessed_one():
    from unittest.mock import MagicMock

    from db.crud import create_run

    run = create_run(MagicMock(), user_topic="nightly scheduled video")
    assert run.user_id is None
