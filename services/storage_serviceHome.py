"""
Uploads video files to S3-compatible storage (AWS S3, Cloudflare R2,
Backblaze B2, DigitalOcean Spaces — all speak the same S3 API) and returns
a public URL. This is what feeds `storage_url` into db.create_video(), and
is required for Instagram/Pinterest which only accept a public HTTPS link.
"""

import os
import uuid
import boto3
from botocore.client import Config as BotoConfig

from config import Config


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


def upload_video_stream(file_obj, filename: str, user_id: str) -> dict:
    if not Config.S3_BUCKET_NAME:
        raise EnvironmentError("S3_BUCKET_NAME is not set.")
    ext = os.path.splitext(filename)[1] or ".mp4"
    key = f"videos/{user_id}/{uuid.uuid4().hex}{ext}"
    file_obj.seek(0, os.SEEK_END)
    size_bytes = file_obj.tell()
    file_obj.seek(0)
    _client().upload_fileobj(file_obj, Config.S3_BUCKET_NAME, key,
                              ExtraArgs={"ContentType": "video/mp4", "ACL": "public-read"})
    public_url = f"{Config.S3_PUBLIC_BASE_URL.rstrip('/')}/{key}" if Config.S3_PUBLIC_BASE_URL else \
                 f"https://{Config.S3_BUCKET_NAME}.s3.{Config.S3_REGION}.amazonaws.com/{key}"
    return {"storage_key": key, "public_url": public_url, "size_bytes": size_bytes}