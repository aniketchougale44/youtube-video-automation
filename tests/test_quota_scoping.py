"""One account's uploads must not spend another account's YouTube quota.

Google grants the 10,000 units/day per Google Cloud PROJECT. Each user brings their own OAuth
client, so each has their own project and their own full budget. A single shared counter -- which
is what this was -- let the first account to upload drain the day for everybody, quietly undoing
the reason users supply their own credentials at all.

The bucket is keyed on the project number parsed out of the OAuth client id, because that is the
thing Google actually meters: two users who really do share a project share a bucket here too,
which is what Google will do to them regardless.
"""
import pytest

from tools import quota
from tools import youtube as youtube_tool

MIA = "111111111-miaabc.apps.googleusercontent.com"
RAJ = "222222222-rajxyz.apps.googleusercontent.com"


@pytest.fixture
def buckets(monkeypatch):
    """In-memory stand-in for Redis, keyed exactly as the real bucket is."""
    state: dict[str, int] = {}

    class _FakeRedis:
        def get(self, key):
            return state.get(key, 0)

        def decrby(self, key, amount):
            state[key] = max(0, state.get(key, 0) - amount)

        def pipeline(self):
            return self

        def incrby(self, key, amount):
            state[key] = state.get(key, 0) + amount
            return self

        def expire(self, *_a, **_k):
            return self

        def execute(self):
            return None

    monkeypatch.setattr(quota, "_redis", lambda: _FakeRedis())
    return state


def test_the_project_is_read_from_the_client_id():
    assert quota.scope_for_client_id(MIA) == "111111111"
    assert quota.scope_for_client_id(RAJ) == "222222222"
    assert quota.scope_for_client_id("") == quota.DEPLOYMENT


def test_an_unrecognised_client_id_gets_its_own_bucket_not_a_shared_one():
    """Falling back to the shared bucket would pool a credential with everyone else's, which is
    the exact failure this scoping exists to prevent."""
    odd = quota.scope_for_client_id("not-a-google-style-id")
    assert odd not in ("", quota.DEPLOYMENT)
    assert odd != quota.scope_for_client_id("a-different-odd-id")


def test_one_account_upload_does_not_touch_another_accounts_budget(buckets):
    limit = quota.get_settings().youtube_daily_quota_units

    with youtube_tool.acting_for(MIA, "s", "r", "UCmia"):
        quota.consume("videos.insert")
        assert quota.remaining_units() == limit - 1600

    with youtube_tool.acting_for(RAJ, "s", "r", "UCraj"):
        assert quota.remaining_units() == limit, "raj must still have a full day"

    assert quota.remaining_units(quota.DEPLOYMENT) == limit


def test_api_key_reads_are_billed_to_the_deployment_not_the_signed_in_user(buckets):
    """Trend research and competitor search run on YOUTUBE_API_KEY, which belongs to the
    deployment's project. Billing them to the user would charge them for quota Google never took
    from them, and leave the operator's own budget looking untouched until it ran out mid-run."""
    limit = quota.get_settings().youtube_daily_quota_units

    with youtube_tool.acting_for(MIA, "s", "r", "UCmia"):
        quota.consume("search.list", scope=quota.DEPLOYMENT)
        assert quota.remaining_units() == limit, "the user's own budget is untouched"

    assert quota.remaining_units(quota.DEPLOYMENT) == limit - 100


def test_a_refund_returns_units_to_the_account_that_was_charged(buckets):
    limit = quota.get_settings().youtube_daily_quota_units

    with (
        youtube_tool.acting_for(MIA, "s", "r", "UCmia"),
        pytest.raises(RuntimeError),
        quota.reserve("videos.insert"),
    ):
        raise RuntimeError("upload failed")

    with youtube_tool.acting_for(MIA, "s", "r", "UCmia"):
        assert quota.remaining_units() == limit, "mia's units came back"
    with youtube_tool.acting_for(RAJ, "s", "r", "UCraj"):
        assert quota.remaining_units() == limit, "raj was never charged or credited"


def test_metered_refunds_the_same_bucket_it_charged(buckets, monkeypatch):
    """The charge and the refund are resolved once. Re-deriving the scope on the way out would,
    if the acting account changed in between, credit units to somebody who was never billed."""
    limit = quota.get_settings().youtube_daily_quota_units

    @quota.metered("videos.insert")
    def _upload():
        raise RuntimeError("boom")

    with youtube_tool.acting_for(MIA, "s", "r", "UCmia"), pytest.raises(RuntimeError):
        _upload()

    with youtube_tool.acting_for(MIA, "s", "r", "UCmia"):
        assert quota.remaining_units() == limit
    with youtube_tool.acting_for(RAJ, "s", "r", "UCraj"):
        assert quota.remaining_units() == limit


def test_exhausting_one_account_does_not_block_another(buckets):
    """The headline failure: one user's uploads must never stop another user publishing."""
    with youtube_tool.acting_for(MIA, "s", "r", "UCmia"):
        for _ in range(6):  # 6 x 1600 = 9600 of 10000
            quota.consume("videos.insert")
        assert not quota.can_afford("videos.insert")
        with pytest.raises(quota.QuotaExceededError):
            quota.consume("videos.insert")

    with youtube_tool.acting_for(RAJ, "s", "r", "UCraj"):
        assert quota.can_afford("videos.insert"), "raj's day is unaffected by mia's uploads"
