"""
YouTube-only search — using the official YouTube Data API v3.

Returns results shaped like YouTube's own search page: thumbnail, title,
channel name + avatar + link, view count, "time ago", and video duration —
plus the video's own embed URL so it can be played inline, and its real
watch_url so "open on YouTube" always works.

IMPORTANT (Jinja gotcha that caused the crash you hit): never name a dict
key "items" if you're going to access it with dot notation in a template.
`{{ data.items }}` resolves to Python's built-in `dict.items` method before
it looks for a dict key called "items" — that's what caused
`TypeError: 'builtin_function_or_method' object is not iterable`. This
version uses the key "results" everywhere instead.

This makes 3 calls to the YouTube Data API per search:
  1. search.list       -> find matching video IDs
  2. videos.list        -> get view counts + duration for those IDs
  3. channels.list       -> get channel avatar for the video's uploader
That's ~100 quota units per search on the default 10,000/day quota — i.e.
roughly 100 searches/day before you hit the daily cap.
"""

import re
from datetime import datetime, timezone

import requests
from config import Config


def _format_duration(iso_duration: str) -> str:
    """Converts YouTube's ISO 8601 duration (e.g. 'PT4M13S') to '4:13'."""
    if not iso_duration:
        return ""
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration)
    if not match:
        return ""
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _format_count(n) -> str:
    """1234567 -> '1.2M', 4200 -> '4.2K'."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "0"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _time_ago(iso_date: str) -> str:
    if not iso_date:
        return ""
    published = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
    delta = datetime.now(timezone.utc) - published
    days = delta.days
    if days < 1:
        hours = delta.seconds // 3600
        if hours < 1:
            return f"{max(delta.seconds // 60, 1)} minutes ago"
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 365:
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def search_youtube(query: str, max_results: int = 12) -> dict:
    if not Config.GOOGLE_API_KEY:
        return {
            "available": False,
            "reason": "GOOGLE_API_KEY is not set in .env — get one from console.cloud.google.com "
                       "and enable 'YouTube Data API v3'.",
            "results": [],
        }

    # 1. search.list — find matching videos
    search_resp = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "key": Config.GOOGLE_API_KEY,
            "q": query,
            "part": "snippet",
            "type": "video",
            "maxResults": max_results,
        },
        timeout=15,
    ).json()

    if "error" in search_resp:
        return {"available": False, "reason": search_resp["error"].get("message", "YouTube search error"), "results": []}

    video_ids = [
        item["id"]["videoId"] for item in search_resp.get("items", [])
        if item.get("id", {}).get("videoId")
    ]
    if not video_ids:
        return {"available": True, "results": []}

    # 2. videos.list — view count, like count, duration for each result
    videos_resp = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={
            "key": Config.GOOGLE_API_KEY,
            "id": ",".join(video_ids),
            "part": "snippet,contentDetails,statistics",
        },
        timeout=15,
    ).json()

    if "error" in videos_resp:
        return {"available": False, "reason": videos_resp["error"].get("message", "YouTube videos.list error"), "results": []}

    video_items = videos_resp.get("items", [])

    # 3. channels.list — channel avatar for each uploader (deduplicated)
    channel_ids = list({v["snippet"]["channelId"] for v in video_items if v.get("snippet", {}).get("channelId")})
    channel_map = {}
    if channel_ids:
        channels_resp = requests.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"key": Config.GOOGLE_API_KEY, "id": ",".join(channel_ids), "part": "snippet"},
            timeout=15,
        ).json()
        for c in channels_resp.get("items", []):
            channel_map[c["id"]] = c

    results = []
    for v in video_items:
        snip = v.get("snippet", {})
        stats = v.get("statistics", {})
        content = v.get("contentDetails", {})
        channel = channel_map.get(snip.get("channelId"), {})
        channel_snip = channel.get("snippet", {})

        results.append({
            "video_id": v["id"],
            "title": snip.get("title"),
            "description": (snip.get("description") or "")[:160],
            "thumbnail": snip.get("thumbnails", {}).get("medium", {}).get("url"),
            "channel_title": snip.get("channelTitle"),
            "channel_id": snip.get("channelId"),
            "channel_url": f"https://www.youtube.com/channel/{snip.get('channelId')}",
            "channel_avatar": channel_snip.get("thumbnails", {}).get("default", {}).get("url"),
            "published_ago": _time_ago(snip.get("publishedAt")),
            "view_count": _format_count(stats.get("viewCount")) if "viewCount" in stats else None,
            "duration": _format_duration(content.get("duration")),
            "embed_url": f"https://www.youtube.com/embed/{v['id']}?autoplay=1",
            "watch_url": f"https://www.youtube.com/watch?v={v['id']}",
        })

    return {"available": True, "results": results}









# """
# Aggregated search across platforms — using ONLY official public APIs.

# Honest note on platform coverage (this matters, so read it before wiring up
# your keys): not every platform offers a public "search everything" API to
# third-party apps. Where one doesn't exist, this module says so explicitly
# instead of faking results or scraping the site (scraping breaks most
# platforms' Terms of Service and is not something this project does).

# Real, working public search:
#   - Google        -> Custom Search JSON API (needs API key + Search Engine ID)
#   - YouTube       -> YouTube Data API v3 search.list (fully public, no login needed)
#   - X / Twitter   -> API v2 recent search (needs a Bearer Token with
#                      Elevated/Pro access — the Essential/free tier does not
#                      include search)

# No public "search all content" API (by design, for privacy reasons):
#   - Facebook  -> Graph API public content search was removed in 2018.
#                  Once a user connects their account, we can only show
#                  THEIR OWN posts/pages, not search all of Facebook.
#   - Instagram -> Only hashtag search is available, and only for a connected
#                  Business/Creator account, capped at ~30 hashtag queries/week.
#   - TikTok    -> No public search API for regular apps. The Display API
#                  only returns a connected user's own videos.
#   - Telegram  -> No official "search all of Telegram" API. A bot can only
#                  read messages in chats/channels it has been added to.
# """

# import requests
# from config import Config


# def search_google(query, max_results=5):
#     if not (Config.GOOGLE_API_KEY and Config.GOOGLE_CSE_ID):
#         return {"available": False, "reason": "Google API key / Search Engine ID not configured.", "items": []}

#     resp = requests.get(
#         "https://www.googleapis.com/customsearch/v1",
#         params={
#             "key": Config.GOOGLE_API_KEY,
#             "cx": Config.GOOGLE_CSE_ID,
#             "q": query,
#             "num": max_results,
#         },
#         timeout=15,
#     )
#     data = resp.json()
#     if "error" in data:
#         return {"available": False, "reason": data["error"].get("message", "Google search error"), "items": []}

#     items = [
#         {
#             "title": item.get("title"),
#             "link": item.get("link"),
#             "snippet": item.get("snippet"),
#             "thumbnail": (item.get("pagemap", {}).get("cse_thumbnail", [{}])[0].get("src")),
#         }
#         for item in data.get("items", [])
#     ]
#     return {"available": True, "items": items}


# def search_youtube(query, max_results=6):
#     if not Config.GOOGLE_API_KEY:
#         return {"available": False, "reason": "Google API key not configured.", "items": []}

#     resp = requests.get(
#         "https://www.googleapis.com/youtube/v3/search",
#         params={
#             "key": Config.GOOGLE_API_KEY,
#             "q": query,
#             "part": "snippet",
#             "type": "video",
#             "maxResults": max_results,
#         },
#         timeout=15,
#     )
#     data = resp.json()
#     if "error" in data:
#         return {"available": False, "reason": data["error"].get("message", "YouTube search error"), "items": []}

#     items = []
#     for item in data.get("items", []):
#         vid = item["id"]["videoId"]
#         snip = item["snippet"]
#         items.append({
#             "title": snip.get("title"),
#             "channel": snip.get("channelTitle"),
#             "thumbnail": snip.get("thumbnails", {}).get("medium", {}).get("url"),
#             "video_id": vid,
#             "embed_url": f"https://www.youtube.com/embed/{vid}",
#             "watch_url": f"https://www.youtube.com/watch?v={vid}",
#         })
#     return {"available": True, "items": items}


# def search_x(query, max_results=10):
#     if not Config.X_BEARER_TOKEN:
#         return {"available": False, "reason": "X Bearer Token not configured (needs Elevated/Pro API access).", "items": []}

#     resp = requests.get(
#         "https://api.twitter.com/2/tweets/search/recent",
#         headers={"Authorization": f"Bearer {Config.X_BEARER_TOKEN}"},
#         params={
#             "query": query,
#             "max_results": max(10, min(max_results, 100)),
#             "tweet.fields": "created_at,author_id,public_metrics",
#         },
#         timeout=15,
#     )
#     data = resp.json()
#     if "errors" in data or resp.status_code >= 300:
#         return {"available": False, "reason": data.get("title", "X search error — check API access tier"), "items": []}

#     items = [
#         {
#             "text": t.get("text"),
#             "created_at": t.get("created_at"),
#             "url": f"https://x.com/i/web/status/{t.get('id')}",
#             "metrics": t.get("public_metrics", {}),
#         }
#         for t in data.get("data", [])
#     ]
#     return {"available": True, "items": items}


# def search_facebook(query):
#     return {
#         "available": False,
#         "reason": (
#             "Facebook removed public content search from the Graph API in 2018 "
#             "for privacy reasons. Connect your account to view your own Pages/posts instead."
#         ),
#         "items": [],
#     }


# def search_instagram(query):
#     return {
#         "available": False,
#         "reason": (
#             "Instagram only supports hashtag search (not general search), and only "
#             "for a connected Business/Creator account, capped at ~30 queries/week. "
#             "Connect your account and use the Instagram dashboard's hashtag lookup."
#         ),
#         "items": [],
#     }


# def search_tiktok(query):
#     return {
#         "available": False,
#         "reason": (
#             "TikTok does not offer a public search API to third-party apps. "
#             "Connect your account to view your own uploaded videos via the Display API."
#         ),
#         "items": [],
#     }


# def search_telegram(query):
#     return {
#         "available": False,
#         "reason": (
#             "Telegram has no official 'search everything' API. A bot can only read "
#             "messages in chats/channels it has been explicitly added to as an admin."
#         ),
#         "items": [],
#     }


# def aggregate_search(query, platforms):
#     """Run search across every requested platform and return a dict keyed by platform."""
#     handlers = {
#         "google": search_google,
#         "youtube": search_youtube,
#         "x": search_x,
#         "facebook": search_facebook,
#         "instagram": search_instagram,
#         "tiktok": search_tiktok,
#         "telegram": search_telegram,
#     }
#     results = {}
#     for platform in platforms:
#         handler = handlers.get(platform)
#         if handler:
#             try:
#                 results[platform] = handler(query)
#             except Exception as e:
#                 results[platform] = {"available": False, "reason": str(e), "items": []}
#     return results
