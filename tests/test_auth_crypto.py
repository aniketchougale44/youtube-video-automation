"""Security properties of accounts: password storage, credential encryption, session tokens.

These are the parts where a mistake is quiet. A broken render is obvious the moment you watch the
video; a session cookie that can be edited into another account's, or a refresh token sitting in
the database as plaintext, looks exactly like a working system until someone takes a copy of it.

Pure functions only, no database -- the HTTP flow is exercised against a real Postgres separately.
"""
import time

import pytest

from core import crypto


@pytest.fixture(autouse=True)
def _fixed_secret(monkeypatch):
    """Pins APP_SECRET_KEY so these never depend on the deployment's real key."""
    import types

    monkeypatch.setattr(
        crypto, "get_settings",
        lambda: types.SimpleNamespace(app_secret_key="test-secret-key-for-unit-tests-only"),
    )


# --------------------------------------------------------------------------- passwords


def test_a_password_is_not_recoverable_from_what_is_stored():
    stored = crypto.hash_password("correct horse battery staple")
    assert "correct horse battery staple" not in stored
    assert crypto.verify_password("correct horse battery staple", stored)
    assert not crypto.verify_password("correct horse battery stapl", stored)


def test_the_same_password_hashes_differently_every_time():
    """Per-password salt. Without it, identical passwords share a hash, and one cracked entry
    reveals every account that chose the same password."""
    a = crypto.hash_password("same-password-twice")
    b = crypto.hash_password("same-password-twice")
    assert a != b
    assert crypto.verify_password("same-password-twice", a)
    assert crypto.verify_password("same-password-twice", b)


@pytest.mark.parametrize("stored", ["", "garbage", "scrypt$only-two", "bcrypt$aaaa$bbbb", "$$"])
def test_a_corrupt_password_row_fails_the_login_rather_than_the_request(stored):
    assert crypto.verify_password("anything", stored) is False


# --------------------------------------------------------------------------- stored secrets


def test_a_stored_credential_is_not_readable_from_the_ciphertext():
    token = crypto.encrypt_secret("1//0g-a-real-looking-refresh-token")
    assert "refresh-token" not in token
    assert crypto.decrypt_secret(token) == "1//0g-a-real-looking-refresh-token"


def test_an_absent_credential_is_not_stored_as_encrypted_emptiness():
    assert crypto.encrypt_secret("") == ""
    assert crypto.decrypt_secret("") == ""


def test_a_credential_from_another_key_reads_as_absent_rather_than_raising(monkeypatch):
    """After APP_SECRET_KEY is rotated, old credentials are unreadable. The useful outcome is
    "this user has no credential, ask them to reconnect" -- not a worker crashing every run."""
    import types

    token = crypto.encrypt_secret("secret-under-the-old-key")
    monkeypatch.setattr(
        crypto, "get_settings",
        lambda: types.SimpleNamespace(app_secret_key="a-completely-different-secret-key"),
    )
    assert crypto.decrypt_secret(token) == ""


# --------------------------------------------------------------------------- sessions


def test_a_session_round_trips_to_the_user_who_was_issued_it():
    token = crypto.mint_session("11111111-1111-1111-1111-111111111111")
    assert crypto.read_session(token) == "11111111-1111-1111-1111-111111111111"


@pytest.mark.parametrize(
    "token",
    ["", "not-a-token", "gAAAAABmmmmmmmmmmmmmmmmmmmmmmmmmmmm", "a.b.c"],
)
def test_a_cookie_that_was_not_minted_here_is_refused(token):
    with pytest.raises(crypto.SessionInvalid):
        crypto.read_session(token)


def test_an_edited_cookie_is_refused():
    """Fernet authenticates as well as encrypts, so flipping bytes cannot promote a cookie into
    another account's session -- it just stops being a valid token."""
    token = crypto.mint_session("11111111-1111-1111-1111-111111111111")
    tampered = token[:-8] + ("A" * 8 if not token.endswith("A" * 8) else "B" * 8)
    with pytest.raises(crypto.SessionInvalid):
        crypto.read_session(tampered)


def test_a_session_past_its_ttl_is_refused(monkeypatch):
    """Expiry is a property of the token, checked by Fernet, not something a route has to
    remember to test."""
    token = crypto.mint_session("11111111-1111-1111-1111-111111111111")
    real_time = time.time
    monkeypatch.setattr(
        time, "time", lambda: real_time() + crypto.SESSION_MAX_AGE_SECONDS + 60
    )
    with pytest.raises(crypto.SessionInvalid):
        crypto.read_session(token)


def test_a_session_cannot_be_forged_without_the_secret(monkeypatch):
    """A token minted under a different APP_SECRET_KEY must not be accepted here -- that is the
    whole basis for trusting the user id inside it."""
    import types

    monkeypatch.setattr(
        crypto, "get_settings",
        lambda: types.SimpleNamespace(app_secret_key="an-attackers-own-secret"),
    )
    forged = crypto.mint_session("22222222-2222-2222-2222-222222222222")

    monkeypatch.setattr(
        crypto, "get_settings",
        lambda: types.SimpleNamespace(app_secret_key="test-secret-key-for-unit-tests-only"),
    )
    with pytest.raises(crypto.SessionInvalid):
        crypto.read_session(forged)
