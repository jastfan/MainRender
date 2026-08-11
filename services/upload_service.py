"""
Upload functions for each platform's official API. Unlike the OAuth/search/
content services above, these take the access token as a direct argument —
once a user connects their account (services/oauth_service.py), the token
you stored for them gets passed in here at upload time.
"""

import time
import requests


def upload_to_youtube(token_data, video_path, title, description="", tags=None, privacy_status="public"):
    import googleapiclient.discovery
    from google.oauth2.credentials import Credentials
    from googleapiclient.http import MediaFileUpload
    from config import Config

    creds = Credentials(
        token=token_data.get("access_token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=Config.GOOGLE_OAUTH_CLIENT_ID,
        client_secret=Config.GOOGLE_OAUTH_CLIENT_SECRET,
    ) 
    youtube = googleapiclient.discovery.build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {"title": title, "description": description, "tags": tags or []},
        "status": {"privacyStatus": privacy_status, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(video_path, chunksize=5 * 1024 * 1024, resumable=True, mimetype="video/*")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk(num_retries=5)

    video_id = response.get("id")
    return {"platform": "youtube", "id": video_id, "url": f"https://youtu.be/{video_id}"}



def upload_to_youtube_stream(token_data, video_stream, title, description="", tags=None,
                              privacy_status="public", category_id="22"):
    import googleapiclient.discovery
    from google.oauth2.credentials import Credentials
    from googleapiclient.http import MediaIoBaseUpload
    from config import Config

    creds = Credentials(
        token=token_data.get("access_token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=Config.GOOGLE_OAUTH_CLIENT_ID,
        client_secret=Config.GOOGLE_OAUTH_CLIENT_SECRET,
    )
    youtube = googleapiclient.discovery.build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {"title": title[:100], "description": description[:5000], "tags": tags or [], "categoryId": category_id},
        "status": {"privacyStatus": privacy_status, "selfDeclaredMadeForKids": False},
    }
    media = MediaIoBaseUpload(video_stream, chunksize=5 * 1024 * 1024, resumable=True, mimetype="video/*")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk(num_retries=5)

    video_id = response.get("id")
    return {"platform": "youtube", "id": video_id, "url": f"https://youtu.be/{video_id}"}



def upload_to_facebook(page_id, page_access_token, video_path, description="", as_reel=True):
    endpoint = f"https://graph.facebook.com/v19.0/{page_id}/videos"
    params = {"access_token": page_access_token, "description": description}
    if as_reel:
        params["video_type"] = "REELS"

    with open(video_path, "rb") as f:
        result = requests.post(endpoint, params=params, files={"source": f}, timeout=300).json()

    if "error" in result:
        raise RuntimeError(f"[Facebook] Upload failed: {result['error']}")
    return {"platform": "facebook", "id": result.get("id")}



def upload_to_facebook_stream(page_id, page_access_token, video_stream, description="", as_reel=True):
    endpoint = f"https://graph.facebook.com/v19.0/{page_id}/videos"
    params = {"access_token": page_access_token, "description": description}
    if as_reel:
        params["video_type"] = "REELS"
    result = requests.post(endpoint, params=params, files={"source": video_stream}, timeout=300).json()
    if "error" in result:
        raise RuntimeError(f"[Facebook] Upload failed: {result['error']}")
    return {"platform": "facebook", "id": result.get("id")}



def upload_to_instagram(ig_user_id, access_token, video_url, caption=""):
    base = "https://graph.facebook.com/v19.0"

    container = requests.post(
        f"{base}/{ig_user_id}/media",
        params={"media_type": "REELS", "video_url": video_url, "caption": caption, "access_token": access_token},
        timeout=120,
    ).json()
    if "error" in container:
        raise RuntimeError(f"[Instagram] Container failed: {container['error']}")

    container_id = container["id"]
    waited = 0
    while waited < 300:
        status = requests.get(
            f"{base}/{container_id}",
            params={"fields": "status_code", "access_token": access_token},
        ).json()
        if status.get("status_code") == "FINISHED":
            break
        if status.get("status_code") == "ERROR":
            raise RuntimeError(f"[Instagram] Processing failed: {status}")
        time.sleep(5)
        waited += 5

    publish = requests.post(
        f"{base}/{ig_user_id}/media_publish",
        params={"creation_id": container_id, "access_token": access_token},
    ).json()
    if "error" in publish:
        raise RuntimeError(f"[Instagram] Publish failed: {publish['error']}")
    return {"platform": "instagram", "id": publish["id"]}


def upload_to_x(oauth1_session, media_path, text=""):
    """oauth1_session: an authenticated tweepy.API instance for the connected user."""
    media = oauth1_session.media_upload(filename=media_path, media_category="tweet_video")
    tweet = oauth1_session.update_status(status=text, media_ids=[media.media_id])
    return {"platform": "x", "id": tweet.id, "url": f"https://x.com/i/web/status/{tweet.id}"}


def upload_to_pinterest(access_token, board_id, media_url, title, description="", is_video=False):
    payload = {
        "board_id": board_id,
        "title": title,
        "description": description,
        "media_source": {"source_type": "video_url" if is_video else "image_url", "url": media_url},
    }
    result = requests.post(
        "https://api.pinterest.com/v5/pins",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    ).json()
    if "id" not in result:
        raise RuntimeError(f"[Pinterest] Pin creation failed: {result}")
    return {"platform": "pinterest", "id": result["id"]}


def upload_to_tiktok(access_token, video_path, title=""):
    """
    TikTok's Content Posting API uses a two-step init + upload process.
    This posts to the user's inbox as a draft (TikTok requires users to
    review/confirm posting from within the app for most API access levels).
    """
    init_resp = requests.post(
        "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json={"source_info": {"source": "FILE_UPLOAD"}},
        timeout=30,
    ).json()

    upload_url = init_resp.get("data", {}).get("upload_url")
    if not upload_url:
        raise RuntimeError(f"[TikTok] Init failed: {init_resp}")

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    requests.put(
        upload_url,
        headers={"Content-Type": "video/mp4", "Content-Range": f"bytes 0-{len(video_bytes)-1}/{len(video_bytes)}"},
        data=video_bytes,
        timeout=300,
    )
    return {"platform": "tiktok", "id": init_resp.get("data", {}).get("publish_id"), "note": "Sent to TikTok inbox for the user to review and post."}



def upload_to_tiktok_stream(access_token, video_stream, title=""):
    init_resp = requests.post(
        "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json={"source_info": {"source": "FILE_UPLOAD"}}, timeout=30,
    ).json()
    upload_url = init_resp.get("data", {}).get("upload_url")
    if not upload_url:
        raise RuntimeError(f"[TikTok] Init failed: {init_resp}")
    video_bytes = video_stream.read()
    requests.put(upload_url, headers={"Content-Type": "video/mp4",
                 "Content-Range": f"bytes 0-{len(video_bytes)-1}/{len(video_bytes)}"},
                 data=video_bytes, timeout=300)
    return {"platform": "tiktok", "id": init_resp.get("data", {}).get("publish_id")}