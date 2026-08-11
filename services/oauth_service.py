"""
Official OAuth "Connect Account" flows.

Each function here builds the real authorization URL for that platform's
official OAuth flow, or exchanges a returned auth code for an access token.
These are the same flows used by any legitimate third-party app — the user
is always redirected to the platform's own login page and explicitly
approves the permissions your app is requesting.
"""

import base64
import hashlib
import os
import requests
from urllib.parse import urlencode

from config import Config

REDIRECT_BASE = Config.BASE_URL


# ============================================================================
# GOOGLE / YOUTUBE
# ============================================================================
def google_authorize_url(state):
    params = {
        "client_id": Config.GOOGLE_OAUTH_CLIENT_ID,
        "redirect_uri": f"{REDIRECT_BASE}/connect/google/callback",
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/youtube.upload "
                 "https://www.googleapis.com/auth/userinfo.profile",
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)


def google_exchange_code(code):
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": Config.GOOGLE_OAUTH_CLIENT_ID,
            "client_secret": Config.GOOGLE_OAUTH_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": f"{REDIRECT_BASE}/connect/google/callback",
        },
        timeout=15,
    )
    return resp.json()


# ============================================================================
# FACEBOOK / INSTAGRAM (Instagram Business accounts connect via Facebook Login)
# ============================================================================
def facebook_authorize_url(state):
    params = {
        "client_id": Config.FACEBOOK_APP_ID,
        "redirect_uri": f"{REDIRECT_BASE}/connect/facebook/callback",
        "scope": "pages_show_list,pages_read_engagement,instagram_basic,public_profile",
        "response_type": "code",
        "state": state,
    }
    return "https://www.facebook.com/v19.0/dialog/oauth?" + urlencode(params)


def facebook_exchange_code(code):
    resp = requests.get(
        "https://graph.facebook.com/v19.0/oauth/access_token",
        params={
            "client_id": Config.FACEBOOK_APP_ID,
            "client_secret": Config.FACEBOOK_APP_SECRET,
            "redirect_uri": f"{REDIRECT_BASE}/connect/facebook/callback",
            "code": code,
        },
        timeout=15,
    )
    return resp.json()


# ============================================================================
# X / TWITTER (OAuth 2.0 with PKCE)
# ============================================================================
def _pkce_pair():
    verifier = base64.urlsafe_b64encode(os.urandom(40)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


def x_authorize_url(state):
    verifier, challenge = _pkce_pair()
    params = {
        "response_type": "code",
        "client_id": Config.X_CLIENT_ID,
        "redirect_uri": f"{REDIRECT_BASE}/connect/x/callback",
        "scope": "tweet.read users.read offline.access",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    url = "https://twitter.com/i/oauth2/authorize?" + urlencode(params)
    return url, verifier  # caller must stash `verifier` in the session for the callback


def x_exchange_code(code, verifier):
    resp = requests.post(
        "https://api.twitter.com/2/oauth2/token",
        data={
            "code": code,
            "grant_type": "authorization_code",
            "client_id": Config.X_CLIENT_ID,
            "redirect_uri": f"{REDIRECT_BASE}/connect/x/callback",
            "code_verifier": verifier,
        },
        auth=(Config.X_CLIENT_ID, Config.X_CLIENT_SECRET),
        timeout=15,
    )
    return resp.json()


# ============================================================================
# TIKTOK
# ============================================================================
def tiktok_authorize_url(state):
    params = {
        "client_key": Config.TIKTOK_CLIENT_KEY,
        "redirect_uri": f"{REDIRECT_BASE}/connect/tiktok/callback",
        "response_type": "code",
        "scope": "user.info.basic,video.list",
        "state": state,
    }
    return "https://www.tiktok.com/v2/auth/authorize/?" + urlencode(params)


def tiktok_exchange_code(code):
    resp = requests.post(
        "https://open.tiktokapis.com/v2/oauth/token/",
        data={
            "client_key": Config.TIKTOK_CLIENT_KEY,
            "client_secret": Config.TIKTOK_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": f"{REDIRECT_BASE}/connect/tiktok/callback",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    return resp.json()


# ============================================================================
# TELEGRAM (Login Widget — different pattern: Telegram calls YOUR callback
# with signed user data instead of a redirect-with-code flow)
# ============================================================================
def telegram_verify_login(auth_data: dict) -> bool:
    """
    Verifies the signed payload the Telegram Login Widget sends to your
    callback. See: https://core.telegram.org/widgets/login#checking-authorization
    """
    if not Config.TELEGRAM_BOT_TOKEN:
        return False

    received_hash = auth_data.get("hash")
    check_fields = {k: v for k, v in auth_data.items() if k != "hash"}
    data_check_string = "\n".join(f"{k}={check_fields[k]}" for k in sorted(check_fields))

    import hmac
    secret_key = hashlib.sha256(Config.TELEGRAM_BOT_TOKEN.encode()).digest()
    computed_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()

    return computed_hash == received_hash
