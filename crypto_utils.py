"""
Encrypts/decrypts sensitive fields (OAuth access & refresh tokens) before
they touch the database. Access tokens are as sensitive as passwords —
anyone who reads them from a DB dump could post on a user's behalf, so they
must never be stored in plain text.

Uses Fernet (AES-128 in CBC mode + HMAC) from the `cryptography` library —
the standard, well-audited choice for "encrypt small secrets with one key"
use cases like this.

Generate a key once with:
  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
Put it in TOKEN_ENCRYPTION_KEY in your .env — and keep it safe. Losing it
means every stored token becomes permanently unreadable.
"""

from cryptography.fernet import Fernet, InvalidToken
from config import Config

_fernet = None


def _get_fernet():
    global _fernet
    if _fernet is None:
        if not Config.TOKEN_ENCRYPTION_KEY:
            raise EnvironmentError(
                "TOKEN_ENCRYPTION_KEY is not set. Generate one with:\n"
                "  python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"\n"
                "and put it in your .env file."
            )
        _fernet = Fernet(Config.TOKEN_ENCRYPTION_KEY.encode())
    return _fernet


def encrypt(plain_text: str) -> str:
    if plain_text is None:
        return None
    return _get_fernet().encrypt(plain_text.encode()).decode()


def decrypt(cipher_text: str) -> str:
    if cipher_text is None:
        return None
    try:
        return _get_fernet().decrypt(cipher_text.encode()).decode()
    except InvalidToken:
        raise ValueError("Could not decrypt value — wrong TOKEN_ENCRYPTION_KEY or corrupted data.")
