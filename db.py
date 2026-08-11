"""
MongoDB data layer.

Four collections, each with a clear job:

  users               - one document per person using the platform
  connected_accounts   - one document per (user, platform) OAuth connection,
                          tokens stored ENCRYPTED, never in plain text
  videos               - one document per uploaded video (storage location,
                          transcript, AI-generated title/description/hashtags)
  posts                - one document per (video, platform) — this is the
                          "full report" the user sees: status, timestamps,
                          the live post URL once published, or the exact
                          error if it failed

Nothing here talks to the OAuth providers or upload APIs directly — this
module is pure storage/retrieval, kept separate so it's easy to reason about
and easy to swap later (e.g. to Postgres) without touching business logic.
"""

from datetime import datetime, timezone
from bson import ObjectId
from pymongo import MongoClient, ASCENDING

from config import Config
import crypto_utils

_client = None
_db = None


def get_db():
    global _client, _db
    if _db is None:
        _client = MongoClient(Config.MONGODB_URI)
        _db = _client[Config.MONGODB_DB_NAME]
        _ensure_indexes(_db)
    return _db



def check_connection() -> dict:
    """
    Actually verifies the MongoDB connection is alive (a client object being
    created does NOT mean the server is reachable). Pings the server and
    returns a simple status dict:

        {"connected": True, "message": "..."}
        {"connected": False, "message": "..."}
    """
    try:
        client = MongoClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return {
            "connected": True,
            "message": f"Connected to MongoDB at '{Config.MONGODB_URI}', database '{Config.MONGODB_DB_NAME}'.",
        }
    except Exception as e:
        return {
            "connected": False,
            "message": f"Could not connect to MongoDB at '{Config.MONGODB_URI}': {e}",
        }


def _ensure_indexes(db):
    db.users.create_index([("email", ASCENDING)], unique=True)
    db.connected_accounts.create_index([("user_id", ASCENDING), ("platform", ASCENDING)], unique=True)
    db.videos.create_index([("user_id", ASCENDING), ("uploaded_at", ASCENDING)])
    db.posts.create_index([("video_id", ASCENDING)])
    db.posts.create_index([("user_id", ASCENDING), ("status", ASCENDING)])


def _now():
    return datetime.now(timezone.utc)


# ============================================================================
# USERS
# ============================================================================
def create_or_get_user(email: str, name: str = "") -> dict:
    db = get_db()
    existing = db.users.find_one({"email": email})
    if existing:
        return existing

    user_doc = {"email": email, "name": name, "created_at": _now()}
    result = db.users.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id
    return user_doc


def get_user(user_id) -> dict | None:
    return get_db().users.find_one({"_id": ObjectId(user_id)})


# ============================================================================
# CONNECTED ACCOUNTS  (OAuth tokens — always encrypted at rest)
# ============================================================================
def save_connected_account(user_id, platform: str, token_data: dict, platform_profile: dict | None = None):
    """
    token_data: whatever the OAuth provider returned — typically has
    access_token, refresh_token, expires_in, etc. Access & refresh tokens
    are encrypted before storage; everything else (scope, token type) is
    kept as-is since it isn't sensitive on its own.
    """
    db = get_db()

    stored = dict(token_data)
    if "access_token" in stored:
        stored["access_token"] = crypto_utils.encrypt(stored["access_token"])
    if "refresh_token" in stored:
        stored["refresh_token"] = crypto_utils.encrypt(stored["refresh_token"])

    doc = {
        "user_id": ObjectId(user_id),
        "platform": platform,
        "tokens": stored,
        "profile": platform_profile or {},
        "connected_at": _now(),
        "updated_at": _now(),
    }

    db.connected_accounts.update_one(
        {"user_id": ObjectId(user_id), "platform": platform},
        {"$set": doc},
        upsert=True,
    )


def get_connected_account(user_id, platform: str) -> dict | None:
    """Returns the account doc with tokens DECRYPTED, ready to use for API calls."""
    doc = get_db().connected_accounts.find_one({"user_id": ObjectId(user_id), "platform": platform})
    if not doc:
        return None

    tokens = dict(doc["tokens"])
    if "access_token" in tokens:
        tokens["access_token"] = crypto_utils.decrypt(tokens["access_token"])
    if "refresh_token" in tokens:
        tokens["refresh_token"] = crypto_utils.decrypt(tokens["refresh_token"])
    doc["tokens"] = tokens
    return doc


def list_connected_platforms(user_id) -> list[str]:
    db = get_db()
    return [doc["platform"] for doc in db.connected_accounts.find({"user_id": ObjectId(user_id)}, {"platform": 1})]


def disconnect_account(user_id, platform: str):
    get_db().connected_accounts.delete_one({"user_id": ObjectId(user_id), "platform": platform})


# ============================================================================
# VIDEOS
# ============================================================================
def create_video(user_id, filename: str, storage_key: str, storage_url: str, size_bytes: int = 0) -> dict:
    db = get_db()
    doc = {
        "user_id": ObjectId(user_id),
        "filename": filename,
        "storage_key": storage_key,
        "storage_url": storage_url,
        "size_bytes": size_bytes,
        "transcript": None,
        "ai_title": None,
        "ai_description": None,
        "ai_hashtags": [],
        "status": "uploaded",  # uploaded -> transcribing -> generating -> ready -> scheduled -> done
        "uploaded_at": _now(),
        "updated_at": _now(),
    }
    result = db.videos.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def update_video(video_id, **fields):
    fields["updated_at"] = _now()
    get_db().videos.update_one({"_id": ObjectId(video_id)}, {"$set": fields})


def get_video(video_id) -> dict | None:
    return get_db().videos.find_one({"_id": ObjectId(video_id)})


def list_user_videos(user_id, limit: int = 50) -> list[dict]:
    return list(
        get_db().videos.find({"user_id": ObjectId(user_id)}).sort("uploaded_at", -1).limit(limit)
    )


# ============================================================================
# POSTS  (per-platform upload jobs — this IS the "full report")
# ============================================================================
def create_post_job(user_id, video_id, platform: str, scheduled_time=None) -> dict:
    db = get_db()
    doc = {
        "user_id": ObjectId(user_id),
        "video_id": ObjectId(video_id),
        "platform": platform,
        "status": "pending",  # pending -> processing -> posted | failed
        "scheduled_time": scheduled_time,
        "posted_at": None,
        "platform_post_id": None,
        "platform_post_url": None,
        "error_message": None,
        "retry_count": 0,
        "created_at": _now(),
        "updated_at": _now(),
    }
    result = db.posts.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def update_post_job(post_id, **fields):
    fields["updated_at"] = _now()
    get_db().posts.update_one({"_id": ObjectId(post_id)}, {"$set": fields})


def mark_post_success(post_id, platform_post_id: str, platform_post_url: str):
    update_post_job(
        post_id,
        status="posted",
        posted_at=_now(),
        platform_post_id=platform_post_id,
        platform_post_url=platform_post_url,
        error_message=None,
    )


def mark_post_failed(post_id, error_message: str):
    db = get_db()
    post = db.posts.find_one({"_id": ObjectId(post_id)})
    retry_count = (post.get("retry_count", 0) + 1) if post else 1
    update_post_job(post_id, status="failed", error_message=error_message, retry_count=retry_count)


def get_pending_posts(limit: int = 50) -> list[dict]:
    """Used by the background worker to find jobs ready to run."""
    return list(
        get_db().posts.find({"status": "pending"}).sort("created_at", 1).limit(limit)
    )


def get_video_posts(video_id) -> list[dict]:
    return list(get_db().posts.find({"video_id": ObjectId(video_id)}))


def get_user_report(user_id) -> list[dict]:
    """app.route
    The full report: every video the user uploaded, with the status of
    every platform it was (or will be) posted to. This is exactly what
    powers the dashboard's "what's going on" view.
    """
    db = get_db()
    videos = list_user_videos(user_id)
    report = []
    for video in videos:
        posts = get_video_posts(video["_id"])
        report.append({"video": video, "posts": posts})
    return report


def create_queued_video(user_id, filename, storage_key, storage_url, size_bytes, scheduled_time, title="", caption="", platforms=None):
    db = get_db()
    doc = {
        "user_id": ObjectId(user_id),
        "filename": filename,
        "storage_key": storage_key,
        "storage_url": storage_url,
        "size_bytes": size_bytes,
        "ai_title": title,
        "ai_description": caption,
        "status": "queued",
        "scheduled_time": scheduled_time,
        "platforms": platforms or [],  # kaunse platforms pe post karna hai, bulk upload waqt select honge
        "uploaded_at": _now(),
        "updated_at": _now(),
    }
    result = db.videos.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def get_due_videos():
    """Scheduler ye function call karega — jo bhi video ka time aa gaya hai."""
    db = get_db()
    now = _now()
    return list(db.videos.find({"status": "queued", "scheduled_time": {"$lte": now}}))



# ============================================================================
# PENDING CLIPS  (staging area — jab tak review na ho, R2/YouTube ko pata nahi)
# ============================================================================
def create_pending_clip(user_id, render_job_id, clip_id, source_title, source_description,
                         source_hashtags, index, duration, preview_url):
    db = get_db()
    doc = {
        "user_id": ObjectId(user_id),
        "render_job_id": render_job_id,
        "clip_id": clip_id,
        "source_title": source_title,
        "source_description": source_description,
        "source_hashtags": source_hashtags,
        "index": index,
        "duration": duration,
        "preview_url": preview_url,
        "status": "pending_review",
        "created_at": _now(),
        "updated_at": _now(),
    }
    result = db.pending_clips.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def list_pending_clips(user_id):
    return list(
        get_db().pending_clips.find({"user_id": ObjectId(user_id)}).sort("created_at", -1)
    )


def get_pending_clip(pending_clip_id):
    return get_db().pending_clips.find_one({"_id": ObjectId(pending_clip_id)})


def delete_pending_clip(pending_clip_id):
    get_db().pending_clips.delete_one({"_id": ObjectId(pending_clip_id)})


def mark_job_synced(job_id):
    get_db().synced_jobs.insert_one({"job_id": job_id, "synced_at": _now()})


def get_synced_job_ids():
    return {d["job_id"] for d in get_db().synced_jobs.find({}, {"job_id": 1})}


def get_all_users():
    return list(get_db().users.find({}, {"_id": 1}))


def track_render_job(user_id, job_id):
    db = get_db()
    db.render_jobs.insert_one({"user_id": ObjectId(user_id), "job_id": job_id, "done": False, "created_at": _now()})


def get_pending_render_jobs():
    db = get_db()
    return [{"user_id": str(r["user_id"]), "job_id": r["job_id"]} for r in db.render_jobs.find({"done": False})]


def mark_render_job_done(job_id):
    db = get_db()
    db.render_jobs.update_one({"job_id": job_id}, {"$set": {"done": True}})
    
if __name__ == "__main__":
    result = check_connection()
    print(result["message"])
    print("CONNECTED" if result["connected"] else "NOT CONNECTED")