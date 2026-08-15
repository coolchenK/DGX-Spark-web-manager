import base64
import hashlib
import hmac
import secrets
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from cryptography.fernet import Fernet, InvalidToken
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer


def hash_api_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_api_key() -> str:
    return f"dgx_{secrets.token_urlsafe(32)}"


def mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


class SecretBox:
    def __init__(self, secret_key: str):
        digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(digest))

    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt(self, value: str) -> str:
        try:
            return self._fernet.decrypt(value.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Encrypted secret could not be decrypted") from exc


class PasswordManager:
    def __init__(self, password: str):
        self._hasher = PasswordHasher()
        self._password_hash = self._hasher.hash(password)

    def verify(self, password: str) -> bool:
        try:
            return self._hasher.verify(self._password_hash, password)
        except VerifyMismatchError:
            return False


class SessionManager:
    def __init__(self, secret_key: str, ttl_seconds: int):
        self._serializer = URLSafeTimedSerializer(secret_key, salt="dgx-manager-session")
        self._ttl_seconds = ttl_seconds

    def create(self, payload: dict[str, Any]) -> str:
        return self._serializer.dumps(payload)

    def load(self, token: str) -> dict[str, Any] | None:
        try:
            value = self._serializer.loads(token, max_age=self._ttl_seconds)
        except (BadSignature, SignatureExpired):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def constant_time_equal(left: str | None, right: str | None) -> bool:
        return bool(left and right and hmac.compare_digest(left, right))

