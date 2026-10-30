import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import AdminUser
from app.db.session import get_db

_PASSWORD_ITERATIONS = 310_000
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("Passwords must contain at least 12 characters.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${_PASSWORD_ITERATIONS}${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, encoded: str | None) -> bool:
    if not encoded:
        return False
    try:
        algorithm, iterations, salt_value, digest_value = encoded.split("$", maxsplit=3)
        if algorithm != "pbkdf2_sha256":
            return False
        expected = _b64decode(digest_value)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), _b64decode(salt_value), int(iterations)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def create_access_token(user_id: UUID, *, expires_in: int = 8 * 60 * 60) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": str(user_id), "iat": now, "exp": now + expires_in}
    signing_input = f"{_json_segment(header)}.{_json_segment(payload)}"
    signature = hmac.new(
        settings.jwt_secret.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    return f"{signing_input}.{_b64encode(signature)}"


def decode_access_token(token: str) -> UUID:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired access token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        header_value, payload_value, signature_value = token.split(".", maxsplit=2)
        signing_input = f"{header_value}.{payload_value}"
        expected = hmac.new(
            settings.jwt_secret.encode(), signing_input.encode(), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, _b64decode(signature_value)):
            raise unauthorized
        header = json.loads(_b64decode(header_value))
        payload = json.loads(_b64decode(payload_value))
        if header.get("alg") != "HS256" or int(payload["exp"]) <= int(time.time()):
            raise unauthorized
        return UUID(payload["sub"])
    except (AttributeError, UnicodeDecodeError, ValueError, KeyError, TypeError):
        raise unauthorized from None


async def get_current_admin(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> AdminUser:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = decode_access_token(credentials.credentials)
    user = await session.get(AdminUser, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired access token")
    return user


def require_admin(user: AdminUser) -> None:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")


def _json_segment(value: dict) -> str:
    return _b64encode(json.dumps(value, separators=(",", ":")).encode())


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
