from apscheduler.schedulers.background import BackgroundScheduler
import db
from services import upload_service, storage_service, content_service
from services import content_pipeline
from datetime import datetime, timezone, timedelta

def _post_to_platform(platform, token, storage_key, title, caption):
    if platform == "youtube":
        video_stream = storage_service.get_video_stream(storage_key)
        return upload_service.upload_to_youtube_stream(token, video_stream, title, caption)

    access_token = token.get("access_token") if isinstance(token, dict) else token

    if platform == "facebook":
        pages = content_service.get_facebook_pages(access_token)
        if not pages:
            raise RuntimeError("No Facebook Page found for this account.")
        page = pages[0]
        video_stream = storage_service.get_video_stream(storage_key)
        return upload_service.upload_to_facebook_stream(page["id"], page["access_token"], video_stream, caption)

    if platform == "tiktok":
        video_stream = storage_service.get_video_stream(storage_key)
        return upload_service.upload_to_tiktok_stream(access_token, video_stream, title)

    raise RuntimeError(f"Scheduled posting not implemented for {platform} yet.")


def process_due_videos():
    for video in db.get_due_videos():
        db.update_video(video["_id"], status="posting")
        had_failure = False

        for platform in video.get("platforms", []):
            post_job = db.create_post_job(video["user_id"], video["_id"], platform)
            db.update_post_job(post_job["_id"], status="processing")
            try:
                account = db.get_connected_account(video["user_id"], platform)
                if not account:
                    raise RuntimeError(f"{platform} is not connected.")
                result = _post_to_platform(platform, account["tokens"], video["storage_key"],
                                            video.get("ai_title") or video["filename"],
                                            video.get("ai_description") or "")
                db.mark_post_success(post_job["_id"], platform_post_id=str(result.get("id", "")), platform_post_url=result.get("url", ""))
            except Exception as e:
                db.mark_post_failed(post_job["_id"], str(e))
                had_failure = True

        db.update_video(video["_id"], status="posted_with_errors" if had_failure else "posted")


# def sync_all_users():
#     for user in db.get_all_users():
#         try:
#             content_pipeline.sync_finished_jobs(str(user["_id"]))
#         except Exception as e:
#             print(f"sync_finished_jobs failed for user {user['_id']}: {e}")


def process_pending_render_jobs():
    from services import render_client
    for tracked in db.get_pending_render_jobs():
        try:
            status = render_client.poll_cut_status_once(tracked["job_id"], tracked["user_id"])
        except Exception as e:
            print(f"render job {tracked['job_id']} status check failed: {e}")
            continue
        if status.get("error"):
            db.mark_render_job_done(tracked["job_id"])
            continue
        if not status.get("done"):
            continue

        for clip in status["clips"]:
            if clip.get("duration", 0) < 3:   # basic quality gate
                continue
            try:
                export_id = render_client.start_export(clip["clip_id"])
                export_status = render_client.poll_export_status(export_id)
                fname = export_status["path"].split("/")[-1]
                file_stream = render_client.download_file_stream(fname)
                storage_result = storage_service.upload_video_stream(file_stream, fname, tracked["user_id"])
                db.create_queued_video(
                    user_id=tracked["user_id"], filename=fname,
                    storage_key=storage_result["storage_key"], storage_url=storage_result["public_url"],
                    size_bytes=storage_result["size_bytes"],
                    scheduled_time=datetime.now(timezone.utc) + timedelta(hours=24),
                    title=status.get("title") or fname, caption=status.get("description") or "",
                    platforms=["youtube"],
                )
                render_client.cleanup_clip(fname)
            except Exception as e:
                print(f"clip {clip.get('clip_id')} auto-upload failed: {e}")

        db.mark_render_job_done(tracked["job_id"])

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(process_due_videos, "interval", minutes=1)
    scheduler.add_job(process_pending_render_jobs, "interval", minutes=1)
    # scheduler.add_job(sync_all_users, "interval", minutes=2)
    scheduler.start()
    return scheduler









# from apscheduler.schedulers.background import BackgroundScheduler
# import db
# import os
# from services import upload_service, video_utils


# def process_due_videos():
#     due_videos = db.get_due_videos()
#     for video in due_videos:
#         db.update_video(video["_id"], status="posting")

#         # R2 se local mein download karna padega upload se pehle
#         # (ye function storage_service mein add karna hoga)
#         local_path = f"/tmp/{video['filename']}"
#         storage_service.download_video_file(video["storage_key"], local_path)

#         for platform in video.get("platforms", []):
#             account = db.get_connected_account(video["user_id"], platform)
#             if not account:
#                 continue
#             try:
#                 result = upload_service.upload_to_youtube(
#                     account["tokens"], local_path, video["ai_title"], video["ai_description"]
#                 )
#                 db.mark_post_success_for_video(video["_id"], platform, result)
#             except Exception as e:
#                 db.mark_post_failed_for_video(video["_id"], platform, str(e))

#         db.update_video(video["_id"], status="posted")
#         os.remove(local_path)


# def start_scheduler():
#     scheduler = BackgroundScheduler()
#     scheduler.add_job(process_due_videos, "interval", minutes=10)
#     scheduler.start()
#     return scheduler