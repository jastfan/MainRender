"""
Uploads video files to S3-compatible storage (AWS S3, Cloudflare R2,
Backblaze B2, DigitalOcean Spaces — all speak the same S3 API) and returns
a public URL. This is what feeds `storage_url` into db.create_video(), and
is required for Instagram/Pinterest which only accept a public HTTPS link.
"""

import io
import os
import threading
import uuid
import boto3
from botocore.client import Config as BotoConfig

from config import Config


# ════════════════════════════════════════════════════════════════════════
# RESUMABLE MULTIPART UPLOAD
# ════════════════════════════════════════════════════════════════════════
# Plain upload_fileobj() (below) sends the whole file in one logical
# transfer — if the connection drops halfway (very common on slow/patchy
# internet), boto3 has nothing left to hand back and the caller has to
# start over from byte 0.
#
# These functions instead drive S3/R2's multipart upload API by hand, part
# by part, and remember (in `_multipart_state`, keyed by OUR OWN upload_id
# — not S3's) exactly which part numbers already succeeded. A retry that
# passes the SAME upload_id reuses the same S3-side multipart upload and
# skips every part already recorded as done, so it only re-sends the bytes
# that never made it the first time. Bucket + Key + S3's own UploadId all
# live in this dict; nothing is written to disk, so this resume state does
# NOT survive a server restart (see publish_module.py's notes on that).
# ════════════════════════════════════════════════════════════════════════
MULTIPART_PART_SIZE = 10 * 1024 * 1024   # 10 MB/part — safely above S3/R2's 5 MB minimum

_multipart_state = {}
_multipart_lock = threading.Lock()


class _ProgressReader:
    """Wraps one part's bytes so botocore's HTTP layer reads it in small
    increments (which it always does when the request body is a file-like
    object, instead of handing it the whole 10 MB at once) — that's what
    lets us report REAL progress continuously *during* a single part's
    upload, not just once the whole part finishes. Without this, the
    progress bar/speed only move in big 10 MB jumps and go blank in
    between, which is exactly what looked broken before."""
    def __init__(self, data, on_read):
        self._buf = io.BytesIO(data)
        self._on_read = on_read
        self._len = len(data)

    def __len__(self):
        return self._len

    def read(self, size=-1):
        chunk = self._buf.read(size)
        if chunk and self._on_read:
            self._on_read(len(chunk))
        return chunk

    def seek(self, *a, **kw):
        return self._buf.seek(*a, **kw)

    def tell(self):
        return self._buf.tell()


def start_or_resume_multipart(upload_id: str, filename: str, user_id: str, total_bytes: int,
                               part_size: int = MULTIPART_PART_SIZE) -> dict:
    """First call for an upload_id creates a fresh S3 multipart upload.
    Any later call with the SAME upload_id just returns the already-open
    one (with whatever parts have already been recorded) — that's the
    resume."""
    with _multipart_lock:
        state = _multipart_state.get(upload_id)
        if state:
            return state
    if not Config.S3_BUCKET_NAME:
        raise EnvironmentError("S3_BUCKET_NAME is not set. Configure storage in your .env file.")
    ext = os.path.splitext(filename)[1] or ".mp4"
    key = f"videos/{user_id}/{uuid.uuid4().hex}{ext}"
    resp = _client().create_multipart_upload(
        Bucket=Config.S3_BUCKET_NAME, Key=key, ContentType="video/mp4", ACL="public-read",
    )
    state = {
        "bucket": Config.S3_BUCKET_NAME, "key": key, "s3_upload_id": resp["UploadId"],
        "parts": {}, "part_size": part_size, "total_bytes": total_bytes,
    }
    with _multipart_lock:
        # another thread may have created one concurrently — keep whichever landed first
        state = _multipart_state.setdefault(upload_id, state)
    return state


def upload_part_resumable(upload_id: str, part_number: int, chunk_bytes: bytes, progress_cb=None) -> str:
    """Uploads ONE part. Idempotent for a part_number already recorded as
    done — a given part's byte offset/contents are deterministic from the
    source file, so re-uploading it would be wasted work, not a correctness
    issue, but skipping it is exactly the point of resuming.

    `progress_cb`, if given, is called repeatedly with the number of NEW
    bytes read (not a running total) as this part streams out — real,
    continuous, per-chunk progress instead of one jump at the end."""
    with _multipart_lock:
        state = _multipart_state.get(upload_id)
        if state and part_number in state["parts"]:
            return state["parts"][part_number]
    if not state:
        raise RuntimeError("No multipart upload registered for this upload_id.")
    body = _ProgressReader(chunk_bytes, progress_cb) if progress_cb else chunk_bytes
    resp = _client().upload_part(
        Bucket=state["bucket"], Key=state["key"], PartNumber=part_number,
        UploadId=state["s3_upload_id"], Body=body,
    )
    etag = resp["ETag"]
    with _multipart_lock:
        state["parts"][part_number] = etag
    return etag


def complete_multipart(upload_id: str) -> dict:
    with _multipart_lock:
        state = _multipart_state.get(upload_id)
    if not state:
        raise RuntimeError("No multipart upload registered for this upload_id.")
    parts = [{"ETag": etag, "PartNumber": pn} for pn, etag in sorted(state["parts"].items())]
    _client().complete_multipart_upload(
        Bucket=state["bucket"], Key=state["key"], UploadId=state["s3_upload_id"],
        MultipartUpload={"Parts": parts},
    )
    public_url = f"{Config.S3_PUBLIC_BASE_URL.rstrip('/')}/{state['key']}" if Config.S3_PUBLIC_BASE_URL else \
                 f"https://{Config.S3_BUCKET_NAME}.s3.{Config.S3_REGION}.amazonaws.com/{state['key']}"
    result = {"storage_key": state["key"], "public_url": public_url, "size_bytes": state["total_bytes"]}
    with _multipart_lock:
        _multipart_state.pop(upload_id, None)
    return result


def abort_multipart(upload_id: str):
    """Cancels an in-progress multipart upload and discards its state —
    used when a user drops a pending upload for good, so R2/S3 doesn't
    keep billing for the orphaned parts forever."""
    with _multipart_lock:
        state = _multipart_state.pop(upload_id, None)
    if state:
        try:
            _client().abort_multipart_upload(Bucket=state["bucket"], Key=state["key"], UploadId=state["s3_upload_id"])
        except Exception:
            pass


def _client():
    return boto3.client(
        "s3",
        region_name=Config.S3_REGION,
        aws_access_key_id=Config.S3_ACCESS_KEY_ID,
        aws_secret_access_key=Config.S3_SECRET_ACCESS_KEY,
        endpoint_url=Config.S3_ENDPOINT_URL,  # None = real AWS S3
        config=BotoConfig(signature_version="s3v4"),
    )


def upload_video_file(local_path: str, user_id: str) -> dict:
    """
    Uploads a local video file and returns {storage_key, public_url, size_bytes}.
    Object key is namespaced by user so files are easy to audit/clean up per account.
    """
    if not Config.S3_BUCKET_NAME:
        raise EnvironmentError("S3_BUCKET_NAME is not set. Configure storage in your .env file.")

    ext = os.path.splitext(local_path)[1] or ".mp4"
    key = f"videos/{user_id}/{uuid.uuid4().hex}{ext}"
    size_bytes = os.path.getsize(local_path)

    _client().upload_file(
        local_path,
        Config.S3_BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": "video/mp4", "ACL": "public-read"},
    )

    if Config.S3_PUBLIC_BASE_URL:
        public_url = f"{Config.S3_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    else:
        public_url = f"https://{Config.S3_BUCKET_NAME}.s3.{Config.S3_REGION}.amazonaws.com/{key}"

    return {"storage_key": key, "public_url": public_url, "size_bytes": size_bytes}


def delete_video_file(storage_key: str):
    if not Config.S3_BUCKET_NAME:
        return
    _client().delete_object(Bucket=Config.S3_BUCKET_NAME, Key=storage_key)


def get_video_stream(storage_key: str):
    """
    R2 se video fetch karke seekable BytesIO stream deta hai.
    NOTE: boto3's get_object()["Body"] ek non-seekable StreamingBody hai.
    Resumable uploads (YouTube/TikTok/Facebook) chunk retry ke waqt stream.seek()
    call karte hain, jo StreamingBody par fail hoke "seek" error deta hai.
    Isliye pura video memory me BytesIO me load karte hain (seekable).
    """
    import io
    response = _client().get_object(Bucket=Config.S3_BUCKET_NAME, Key=storage_key)
    return io.BytesIO(response["Body"].read())


def upload_video_stream(file_obj, filename: str, user_id: str, progress_cb=None) -> dict:
    """
    progress_cb, if given, is boto3's standard transfer Callback: it gets
    invoked repeatedly with the number of NEW bytes sent since the last
    call (not a running total) — this is real transfer progress from the
    S3/R2 client itself, not a simulated/fake progress bar. May be called
    from more than one thread for large (multipart) uploads, so keep any
    accumulation on the caller side thread-safe.
    """
    if not Config.S3_BUCKET_NAME:
        raise EnvironmentError("S3_BUCKET_NAME is not set.")
    ext = os.path.splitext(filename)[1] or ".mp4"
    key = f"videos/{user_id}/{uuid.uuid4().hex}{ext}"
    file_obj.seek(0, os.SEEK_END)
    size_bytes = file_obj.tell()
    file_obj.seek(0)
    _client().upload_fileobj(file_obj, Config.S3_BUCKET_NAME, key,
                              ExtraArgs={"ContentType": "video/mp4", "ACL": "public-read"},
                              Callback=progress_cb)
    public_url = f"{Config.S3_PUBLIC_BASE_URL.rstrip('/')}/{key}" if Config.S3_PUBLIC_BASE_URL else \
                 f"https://{Config.S3_BUCKET_NAME}.s3.{Config.S3_REGION}.amazonaws.com/{key}"
    return {"storage_key": key, "public_url": public_url, "size_bytes": size_bytes}