"""
Fetches a connected user's own content for VIEW-ONLY display in the dashboard.

Deliberately no download/export functions live here. Every function returns
either a playback/embed URL (so the browser streams from the platform's own
CDN, same as visiting the site directly) or a link back to the original post.
Nothing is copied, cached, or re-hosted — that would risk copyright/ToS
violations, which this project avoids entirely.
"""

import requests


def get_youtube_uploads(access_token, max_results=12):
    # Step 1: find the "uploads" playlist for the authenticated channel
    channels_resp = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"part": "contentDetails", "mine": "true"},
        timeout=15,
    ).json()

    items = channels_resp.get("items", [])
    if not items:
        return []

    uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    playlist_resp = requests.get(
        "https://www.googleapis.com/youtube/v3/playlistItems",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"part": "snippet", "playlistId": uploads_playlist_id, "maxResults": max_results},
        timeout=15,
    ).json()

    videos = []
    for item in playlist_resp.get("items", []):
        snip = item["snippet"]
        vid = snip["resourceId"]["videoId"]
        videos.append({
            "title": snip.get("title"),
            "thumbnail": snip.get("thumbnails", {}).get("medium", {}).get("url"),
            "embed_url": f"https://www.youtube.com/embed/{vid}",
            "watch_url": f"https://www.youtube.com/watch?v={vid}",
        })
    return videos


def get_facebook_pages(access_token):
    resp = requests.get(
        "https://graph.facebook.com/v19.0/me/accounts",
        params={"access_token": access_token},
        timeout=15,
    ).json()
    return resp.get("data", [])


def get_facebook_page_posts(page_id, page_access_token, limit=10):
    resp = requests.get(
        f"https://graph.facebook.com/v19.0/{page_id}/posts",
        params={
            "access_token": page_access_token,
            "fields": "message,permalink_url,created_time,full_picture",
            "limit": limit,
        },
        timeout=15,
    ).json()
    return resp.get("data", [])


def get_instagram_media(ig_user_id, access_token, limit=12):
    resp = requests.get(
        f"https://graph.facebook.com/v19.0/{ig_user_id}/media",
        params={
            "access_token": access_token,
            "fields": "caption,media_type,media_url,permalink,thumbnail_url,timestamp",
            "limit": limit,
        },
        timeout=15,
    ).json()
    return resp.get("data", [])


def get_tiktok_videos(access_token, open_id, max_count=12):
    resp = requests.post(
        "https://open.tiktokapis.com/v2/video/list/",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        params={"fields": "id,title,cover_image_url,share_url,embed_link"},
        json={"max_count": max_count},
        timeout=15,
    ).json()
    return resp.get("data", {}).get("videos", [])


def get_x_recent_posts(access_token, user_id, max_results=10):
    resp = requests.get(
        f"https://api.twitter.com/2/users/{user_id}/tweets",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"max_results": max_results, "tweet.fields": "created_at,public_metrics"},
        timeout=15,
    ).json()
    return resp.get("data", [])


def get_telegram_channel_messages(bot_token, chat_id, limit=10):
    """
    NOTE: the Bot API doesn't offer a generic "list recent messages" endpoint —
    it only receives messages via webhooks/getUpdates as they happen, or via
    forwarding. This helper reads from getUpdates as a best-effort view of
    recent activity in chats the bot is part of.
    """
    resp = requests.get(
        f"https://api.telegram.org/bot{bot_token}/getUpdates",
        params={"limit": limit},
        timeout=15,
    ).json()
    messages = [
        u["channel_post"] for u in resp.get("result", [])
        if "channel_post" in u and str(u["channel_post"].get("chat", {}).get("id")) == str(chat_id)
    ]
    return messages
