"""Password hashing, secret encryption and session tokens.

All three are built on `cryptography`, which the project already depends on, so multi-user auth
adds no new packages. Everything here derives from one secret, APP_SECRET_KEY: rotate it and
every session is invalidated and every stored credential becomes unreadable, which is the
correct behaviour for a key believed to be compromised, and a footgun otherwise. Back it up.

What is stored where, and why:

- Passwords are scrypt hashes with a per-password salt. They are never decryptable -- there is
  no code path in this file that turns a stored password back into one.
- A user's Google client secret and YouTube refresh token ARE decryptable, because the worker
  has to present them to Google to upload on that user's behalf. They are encrypted at rest with
  Fernet (AES-128-CBC + HMAC) so a database dump alone does not hand over anybody's channel.
- Sessions are Fernet tokens holding a user id and an issue time. Fernet authenticates as well as
  encrypts, so a cookie cannot be edited into another user's session, and its own TTL check makes
  expiry a property of the token rather than something the application has to remember to test.
"""
import base64
import json
import os
import time

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from core.settings import get_settings

# scrypt parameters. n=2**14 is the interactive-login figure from the scrypt paper: a fraction of
# a second per hash here, while making an offline guessing run over a stolen table expensive.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_KEY_BYTES = 32

SESSION_MAX_AGE_SECONDS = 14 * 24 * 3600


class SessionInvalid(Exception):
    """Raised when a session cookie is missing, tampered with, or past its TTL."""


def _fernet() -> Fernet:
    """Fernet instance keyed from APP_SECRET_KEY.

    The secret is accepted as free text so operators are not forced to produce base64 by hand;
    it is hashed to exactly the 32 bytes Fernet requires rather than truncated or padded, so a
    short secret still produces a full-width key (it does not make a weak secret strong -- see
    the length check in settings).
    """
    import hashlib

    secret = get_settings().app_secret_key.encode("utf-8")
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret).digest()))


# --------------------------------------------------------------------------- passwords


def hash_password(password: str) -> str:
    """-> "scrypt$<salt_b64>$<hash_b64>". One-way; there is deliberately no inverse."""
    salt = os.urandom(_SALT_BYTES)
    derived = Scrypt(salt=salt, length=_KEY_BYTES, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P).derive(
        password.encode("utf-8")
    )
    return f"scrypt${base64.b64encode(salt).decode()}${base64.b64encode(derived).decode()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time via scrypt's own verify, and False rather than an exception on a malformed
    record -- a corrupted row should fail the login, not 500 the endpoint."""
    try:
        scheme, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        Scrypt(
            salt=base64.b64decode(salt_b64), length=_KEY_BYTES,
            n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
        ).verify(password.encode("utf-8"), base64.b64decode(hash_b64))
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- stored secrets


def encrypt_secret(plaintext: str) -> str:
    """For values the worker must later present to Google. Empty in, empty out, so an optional
    credential that was never supplied is not stored as an encrypted empty string."""
    if not plaintext:
        return ""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_secret(ciphertext: str) -> str:
    """Returns "" for anything this key cannot read, rather than raising.

    A credential encrypted under a rotated APP_SECRET_KEY is unrecoverable, and the useful
    behaviour is for the caller to see "no credential" and ask the user to reconnect, not for a
    background job to crash on every run.
    """
    if not ciphertext:
        return ""
    try:
        return _fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return ""


# --------------------------------------------------------------------------- sessions


def mint_session(user_id: str) -> str:
    return _fernet().encrypt(
        json.dumps({"uid": str(user_id), "iat": int(time.time())}).encode("utf-8")
    ).decode("utf-8")


def read_session(token: str) -> str:
    """-> user id. Raises SessionInvalid for a missing, tampered, or expired cookie."""
    if not token:
        raise SessionInvalid("no session cookie")
    try:
        raw = _fernet().decrypt(token.encode("utf-8"), ttl=SESSION_MAX_AGE_SECONDS)
    except InvalidToken as exc:
        raise SessionInvalid("session cookie invalid or expired") from exc
    try:
        uid = json.loads(raw)["uid"]
    except Exception as exc:
        raise SessionInvalid("session payload unreadable") from exc
    if not uid:
        raise SessionInvalid("session has no user")
    return uid
