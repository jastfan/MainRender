"""
content_pipeline.py — connects render_service (RenderDetect.py) to
platform_hub in two stages, so nothing reaches R2/YouTube without review:

  Stage 1 (sync_finished_jobs):
      polls render_service ("kuch naya bana kya?") — never sends a URL.
      Any finished job not yet seen gets its clips recorded in Mongo's
      `pending_clips` collection, with title/description/hashtags pulled
      straight from render_service. Nothing is uploaded to R2 or queued
      for posting yet.

  Stage 2 (approve_clip_to_upload):
      called once a human (via the Review Studio page) approves ONE
      specific clip. Only then: final export -> R2 upload -> Mongo
      `videos` record -> eligible for the scheduler to post.

Later, once an automatic quality/relevance scorer exists (whisper + AI),
a scheduled job can call approve_clip_to_upload() directly for clips that
pass the quality bar, skipping manual review entirely for those. Neither
function below needs to change for that — only what calls them does.
"""
from datetime import datetime, timezone, timedelta

from services import render_client, storage_service
import db


# ============================================================================
# STAGE 1 — poll render_service, land finished clips in staging only
# ============================================================================
def sync_finished_jobs(user_id):
    """
    Checks render_service for jobs that finished since the last check, and
    stages every clip from those jobs into `pending_clips` for review.
    Never sends a URL — render_service already has whatever URL the user
    gave it directly, on its own UI.
    """
    already_synced = db.get_synced_job_ids()
    staged = []

    for job in render_client.list_all_jobs(user_id):
        if not job["done"] or job.get("error") or job["job_id"] in already_synced:
            continue

        status = render_client.poll_cut_status_once(job["job_id"], user_id)
        source_title = status.get("title") or "Untitled"
        source_description = status.get("description") or ""
        source_hashtags = [f"#{t}" for t in (status.get("tags") or [])]

        for clip in status["clips"]:
            clip_id = clip["clip_id"]
            doc = db.create_pending_clip(
                user_id=user_id,
                render_job_id=job["job_id"],
                clip_id=clip_id,
                source_title=source_title,
                source_description=source_description,
                source_hashtags=source_hashtags,
                index=clip.get("index"),
                duration=clip.get("duration"),
                preview_url=f"{render_client.BASE_URL}/media/{clip_id}",
            )
            staged.append(doc)

        db.mark_job_synced(job["job_id"])

    return staged


# ============================================================================
# STAGE 2 — one approved clip: final export -> R2 -> queued video record
# ============================================================================
def approve_clip_to_upload(user_id, pending_clip_id, title=None, caption="", platforms=None,
                            gap_hours=24, index=0, base_time=None, settings=None):
    """
    Takes ONE pending clip the user approved (optionally with edited
    title/caption/platforms from the Review Studio form), renders the
    final file, uploads it to R2, and creates the queued-video record the
    scheduler will pick up. Removes the clip from staging once done.
    """
    pending = db.get_pending_clip(pending_clip_id)
    if not pending or str(pending["user_id"]) != str(user_id):
        raise ValueError("Clip not found or doesn't belong to this user.")

    clip_id = pending["clip_id"]

    export_id = render_client.start_export(clip_id, settings=settings or {})
    export_status = render_client.poll_export_status(export_id)

    fname = export_status["path"].split("/")[-1] if export_status.get("path") else None
    if not fname:
        raise RuntimeError("Export finished but no output file was returned.")

    file_stream = render_client.download_file_stream(fname)
    storage_result = storage_service.upload_video_stream(file_stream, fname, user_id)

    base_time = base_time or datetime.now(timezone.utc)
    scheduled_time = base_time + timedelta(hours=gap_hours * (index + 1))

    video_doc = db.create_queued_video(
        user_id=user_id,
        filename=fname,
        storage_key=storage_result["storage_key"],
        storage_url=storage_result["public_url"],
        size_bytes=storage_result["size_bytes"],
        scheduled_time=scheduled_time,
        title=title or pending["source_title"],
        caption=caption or "",
        platforms=platforms or [],
    )

    db.delete_pending_clip(pending_clip_id)

    return video_doc