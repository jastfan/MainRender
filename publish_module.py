#!/usr/bin/env python3
# publish_module.py — "Publish Studio": a self-contained Flask Blueprint that
# plugs into RenderDetect.py the same way downloader_bp / editor_bp do.
#
# ══════════════════════════════════════════════════════════════════════════
# WHAT CHANGED IN THIS REVISION
# ══════════════════════════════════════════════════════════════════════════
#
# 1) THE BIG BUG — "select 4 hours, but it publishes right away":
#
#    In the old bulk-publish flow, "Gap between posts" only controlled the
#    SPACING BETWEEN items — it never delayed the FIRST one. With "Start"
#    left blank (which defaults to "now"), item #0 was always scheduled at
#    `base_time + gap * 0` == right now. So picking a 4-hour gap and
#    hitting "Publish Selected" published the first video immediately and
#    only pushed the *second* one out by 4 hours — which is exactly the
#    "publishes on the same time I publish, not after the set hours"
#    behavior you ran into.
#
#    THE FIX: _bulk_schedule_params() now applies a `first_offset` — when
#    no explicit start time is given, item #0 is `base_time + gap * 1`,
#    item #1 is `+gap * 2`, etc. "Publish every 4 hours" now means exactly
#    that: first one 4 hours away, then every 4 hours after. If you DO
#    pick an explicit start time via the calendar, that time IS slot #0
#    and every next item is +gap after it (no extra offset needed, since
#    you already told it exactly when to start).
#
# 2) Hardened timezone handling as a second, related fix: <input
#    type="datetime-local"> gives back a plain string with NO timezone
#    info, in the BROWSER's local time. The old code sent that raw string
#    to the server, which had to guess what timezone it was in. The page's
#    JS now converts every picked time to a real, unambiguous UTC instant
#    before it leaves the browser — see toUTCISO() in the <script>:
#    `new Date(localString).toISOString()` correctly reads a plain
#    "YYYY-MM-DDTHH:MM" as browser-local time and turns it into a UTC ISO
#    string ending in "Z". The server just parses that, no guessing. This
#    removes an entire class of "off by my UTC offset" scheduling bugs on
#    top of the fix in (1).
#
# 3) The whole scheduling UI is rebuilt as a compact, YouTube Studio-style
#    calendar + time popover ("Schedule" chip -> pick a date on a real
#    calendar grid + a time -> Confirm), with quick presets (Now, +1h,
#    +4h, +6h, Tomorrow 9AM) so you rarely need the calendar at all. Bulk
#    publishing gets a live preview of exactly when every selected video
#    will go out before you commit — no more guessing at the math.
#
# ── everything else (how it plugs into RenderDetect.py) is unchanged ────
#   - Every clip that finishes exporting in RenderDetect (EXPORT_JOBS
#     status == "done") automatically shows up here.
#   - Single OR bulk publish; a "New Uploads" tab for files that never
#     went through the render pipeline.
#   - AI title + description generation reuses hook_detector2's Gemini
#     key pool.
#   - Publishing = upload straight to R2 (storage_service) + a `videos`
#     record in Mongo with a scheduled_time. The background scheduler
#     (services/scheduler.py, polls every 1 minute) is what actually
#     posts it once scheduled_time <= now.
#
# Wired up from RenderDetect.py's init_render(), same as downloader/editor:
#     from publish_module import publish_bp, init_publish
#     init_publish(BASE, get_export_jobs=lambda: EXPORT_JOBS, get_clips=lambda: CLIPS)
#     main_app.register_blueprint(publish_bp)

import json
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from flask import Blueprint, request, jsonify, session, send_file, Response

import db
from services import storage_service

publish_bp = Blueprint("publish_bp", __name__)

# ── wired up by init_publish() ──────────────────────────────────────────
_OUTPUT_DIR = None          # Path to RenderDetect's shorts_final/ dir
_get_export_jobs = None     # callable -> RenderDetect's EXPORT_JOBS dict
_get_clips = None           # callable -> RenderDetect's CLIPS dict

# Exports the user has already published — filtered out of the feed so a
# card doesn't linger (or get double-published) after it's been queued.
_published_export_ids = set()
_published_lock = threading.Lock()

PLATFORM_CHOICES = ["youtube", "facebook", "tiktok"]  # same set review.html offers

PLATFORM_META = {
    "youtube":  {"label": "YouTube",  "emoji": "▶️", "color": "#FF0033"},
    "facebook": {"label": "Facebook", "emoji": "📘", "color": "#1877F2"},
    "tiktok":   {"label": "TikTok",   "emoji": "🎵", "color": "#25F4EE"},
}


# ════════════════════════════════════════════════════════════════════════
# Upload progress tracking — real byte-level progress for every publish
# (single card, bulk-selected cards, and the New Uploads file picker).
#
# There are two legs to a "publish" and this tracks both honestly:
#   1) BROWSER -> SERVER   (only exists for a fresh file from "New Uploads" —
#      a single/bulk-card publish has no file to send, the clip is already
#      on the server from the render pipeline). This leg can ONLY be
#      measured accurately on the client, via XMLHttpRequest's real
#      `upload.onprogress` event — that's what actually seeing bytes leave
#      the browser looks like, so that part lives in the page's <script>.
#   2) SERVER -> CLOUD STORAGE (R2/S3)  (every publish goes through this).
#      This is tracked HERE, server-side, from boto3's real transfer
#      callback (bytes actually handed to the S3 client), so it reflects
#      the real network conditions between this server and the bucket, not
#      a fake timer. The page polls /api/publish/upload_progress/<id> to
#      read it live.
# ════════════════════════════════════════════════════════════════════════
_UPLOAD_PROGRESS = {}
_progress_lock = threading.Lock()
_PROGRESS_TTL_SEC = 15 * 60      # stale entries get swept out eventually
_SPEED_WINDOW_SEC = 3.0          # rolling window used for "current" speed


def _progress_sweep_locked():
    """Drop old entries. Caller must already hold _progress_lock."""
    cutoff = time.time() - _PROGRESS_TTL_SEC
    dead = [k for k, v in _UPLOAD_PROGRESS.items() if v.get("updated_at", 0) < cutoff]
    for k in dead:
        _UPLOAD_PROGRESS.pop(k, None)


def _new_progress(upload_id, filename="", total_bytes=0, stage="queued"):
    """(Re)register an upload_id before work starts, so a poller that hits
    the endpoint a moment early sees an honest 'queued' state instead of a
    blank/404."""
    now = time.time()
    with _progress_lock:
        _progress_sweep_locked()
        _UPLOAD_PROGRESS[upload_id] = {
            "upload_id": upload_id, "filename": filename, "stage": stage,
            "bytes_done": 0, "bytes_total": total_bytes,
            "speed_bps": 0.0, "peak_bps": 0.0, "eta_seconds": None,
            "slow": False, "message": "",
            "started_at": now, "updated_at": now,
            "_samples": [(now, 0)],
        }
    return upload_id


def _set_progress(upload_id, **fields):
    if not upload_id:
        return
    now = time.time()
    with _progress_lock:
        entry = _UPLOAD_PROGRESS.get(upload_id)
        if not entry:
            entry = {
                "upload_id": upload_id, "filename": fields.get("filename", ""),
                "stage": "queued", "bytes_done": 0, "bytes_total": 0,
                "speed_bps": 0.0, "peak_bps": 0.0, "eta_seconds": None,
                "slow": False, "message": "", "started_at": now, "updated_at": now,
                "_samples": [(now, 0)],
            }
            _UPLOAD_PROGRESS[upload_id] = entry
        entry.update({k: v for k, v in fields.items()})
        entry["updated_at"] = now

        # current speed = slope over a short rolling window, NOT the
        # lifetime average — that's what lets us actually notice "internet
        # just got slow" instead of smoothing it away.
        samples = entry["_samples"]
        samples.append((now, entry["bytes_done"]))
        cutoff = now - _SPEED_WINDOW_SEC
        while len(samples) > 2 and samples[0][0] < cutoff:
            samples.pop(0)
        if len(samples) >= 2 and (samples[-1][0] - samples[0][0]) > 0.05:
            dt = samples[-1][0] - samples[0][0]
            dbytes = samples[-1][1] - samples[0][1]
            speed = dbytes / dt
        else:
            elapsed = now - entry["started_at"]
            speed = (entry["bytes_done"] / elapsed) if elapsed > 0.2 else entry["speed_bps"]
        entry["speed_bps"] = max(speed, 0.0)
        entry["peak_bps"] = max(entry.get("peak_bps", 0.0), entry["speed_bps"])

        remaining = max(entry["bytes_total"] - entry["bytes_done"], 0)
        entry["eta_seconds"] = (remaining / entry["speed_bps"]) if entry["speed_bps"] > 1024 else None

        if entry["stage"] == "uploading" and entry["bytes_total"] > 2_000_000:
            slow_abs = entry["speed_bps"] < 60_000                                    # < ~60 KB/s
            slow_rel = entry["peak_bps"] > 250_000 and entry["speed_bps"] < entry["peak_bps"] * 0.25
            entry["slow"] = bool(slow_abs or slow_rel)
        else:
            entry["slow"] = False


def _progress_public(entry):
    if not entry:
        return None
    pct = 0
    if entry["bytes_total"] > 0:
        pct = round(min(entry["bytes_done"] / entry["bytes_total"], 1.0) * 100, 1)
    elif entry["stage"] == "done":
        pct = 100
    return {
        "upload_id": entry["upload_id"], "filename": entry["filename"], "stage": entry["stage"],
        "bytes_done": entry["bytes_done"], "bytes_total": entry["bytes_total"], "pct": pct,
        "speed_bps": round(entry["speed_bps"], 1), "eta_seconds": entry["eta_seconds"],
        "slow": entry["slow"], "message": entry["message"],
    }


@publish_bp.route("/api/publish/upload_progress/<upload_id>")
def api_upload_progress(upload_id):
    """Polled by the page (every ~450ms) while a publish is in flight."""
    with _progress_lock:
        snap = _progress_public(_UPLOAD_PROGRESS.get(upload_id))
    if not snap:
        # Not an error — a bulk item may just not have had its turn yet.
        return jsonify({"upload_id": upload_id, "stage": "queued", "pct": 0,
                         "bytes_done": 0, "bytes_total": 0, "speed_bps": 0,
                         "eta_seconds": None, "slow": False, "message": ""})
    return jsonify(snap)


def _store_and_queue(user_id, local_path, filename, size_bytes, title, caption,
                      platforms, scheduled_time, upload_id, is_temp_file=False):
    """Shared tail for every publish path: stream the LOCAL file at
    `local_path` up to cloud storage using RESUMABLE multipart upload (see
    services/storage_service.py), then write the Mongo `videos` record.

    Resumability: this call registers `upload_id` in `_resumable_registry`
    BEFORE attempting the storage upload. If the storage upload fails
    partway (dropped connection, slow internet, etc.) the exception
    propagates, the registry entry is deliberately left in place, and a
    later call to /api/publish/resume_upload with the same upload_id will
    re-enter this exact function and skip every part already uploaded —
    see _upload_resumable_and_get_result() below. Only a fully successful
    run clears the registry (and deletes the temp file, if any)."""
    _register_resumable(
        upload_id, local_path=str(local_path), filename=filename, size_bytes=size_bytes,
        user_id=user_id, title=title, caption=caption, platforms=platforms or [],
        scheduled_time=scheduled_time.isoformat(), is_temp_file=is_temp_file,
    )
    _set_progress(upload_id, filename=filename, bytes_total=size_bytes, stage="uploading", message="")

    try:
        storage_result = _upload_resumable_and_get_result(local_path, filename, user_id, size_bytes, upload_id)
    except Exception as e:
        _set_progress(upload_id, stage="error", message=str(e))
        raise  # registry entry stays — that's what makes /resume_upload possible

    _set_progress(upload_id, stage="saving", bytes_done=size_bytes)
    video_doc = db.create_queued_video(
        user_id=user_id, filename=filename,
        storage_key=storage_result["storage_key"], storage_url=storage_result["public_url"],
        size_bytes=storage_result["size_bytes"], scheduled_time=scheduled_time,
        title=title, caption=caption, platforms=platforms or [],
    )
    _set_progress(upload_id, stage="done", message="")
    _clear_resumable(upload_id)   # success — deletes temp file (if any) and drops registry entry
    return video_doc


def _upload_resumable_and_get_result(local_path, filename, user_id, size_bytes, upload_id):
    """Drives storage_service's manual multipart API part-by-part so a
    failed attempt can resume instead of restarting. Reopens the file
    fresh (cheap — it's a local temp/export file, not the network) and
    seeks to each part's offset, which is what lets a retry skip straight
    past parts already confirmed uploaded."""
    part_size = storage_service.MULTIPART_PART_SIZE
    state = storage_service.start_or_resume_multipart(upload_id, filename, user_id, size_bytes, part_size)
    total_parts = max(1, -(-max(size_bytes, 1) // part_size))  # ceil, at least 1 part

    already_done = len(state["parts"])
    if already_done:
        done_bytes = sum(min(part_size, size_bytes - (pn - 1) * part_size) for pn in state["parts"])
        _set_progress(upload_id, bytes_done=done_bytes,
                       message=f"Resuming — {already_done}/{total_parts} part(s) already uploaded.")
    else:
        _set_progress(upload_id, message="")

    with open(local_path, "rb") as f:
        for part_number in range(1, total_parts + 1):
            if part_number in state["parts"]:
                continue  # already uploaded on a previous, failed attempt — don't re-send it
            offset = (part_number - 1) * part_size
            f.seek(offset)
            chunk = f.read(part_size)

            # live, continuous progress WHILE this part streams out — not
            # just one jump when the whole 10 MB part finishes. `progress_cb`
            # fires on every small read botocore's HTTP layer does internally
            # (typically tens of KB at a time), so % / speed / ETA move
            # smoothly and reflect the actual network throughput second to
            # second, exactly like the "real" progress bar should.
            base_bytes = _UPLOAD_PROGRESS.get(upload_id, {}).get("bytes_done", 0)
            sent_so_far = [0]

            def _on_chunk_read(n, _base=base_bytes, _sent=sent_so_far):
                _sent[0] += n
                _set_progress(upload_id, bytes_done=_base + _sent[0], message="")

            storage_service.upload_part_resumable(upload_id, part_number, chunk, progress_cb=_on_chunk_read)

    return storage_service.complete_multipart(upload_id)


# ── resumable-upload registry: what a /resume_upload retry needs, without
#    the browser having to send the file (or even the form fields) again ──
_UPLOAD_TMP_DIR = Path(tempfile.gettempdir()) / "publish_studio_pending_uploads"
_resumable_registry = {}
_resumable_lock = threading.Lock()


def _register_resumable(upload_id, **fields):
    with _resumable_lock:
        _resumable_registry[upload_id] = fields


def _get_resumable(upload_id):
    with _resumable_lock:
        return _resumable_registry.get(upload_id)


def _clear_resumable(upload_id):
    with _resumable_lock:
        entry = _resumable_registry.pop(upload_id, None)
    if entry and entry.get("is_temp_file"):
        try:
            Path(entry["local_path"]).unlink(missing_ok=True)
        except Exception:
            pass


def init_publish(base_dir, get_export_jobs, get_clips):
    """Call once at startup from RenderDetect.init_render()."""
    global _OUTPUT_DIR, _get_export_jobs, _get_clips
    _OUTPUT_DIR = Path(base_dir) / "shorts_final"
    _get_export_jobs = get_export_jobs
    _get_clips = get_clips


def _current_user_id():
    return session.get("user_id")


def _require_user():
    uid = _current_user_id()
    if not uid:
        return None, (jsonify({"error": "Not logged in. Log in at /login first, then reopen this tab."}), 401)
    return uid, None


# ════════════════════════════════════════════════════════════════════════
# AI title/description generation — reuses hook_detector2's Gemini pool
# ════════════════════════════════════════════════════════════════════════
def _ai_generate(source_title="", context_hint=""):
    """Returns {"title", "description", "hashtags"}. Falls back to something
    sane (never raises) so a Gemini hiccup never blocks publishing."""
    fallback = {
        "title": (source_title or "Untitled Short")[:95],
        "description": context_hint or "",
        "hashtags": [],
    }
    try:
        from pydantic import BaseModel, Field
        import hook_detector2 as hd

        class TitleDescription(BaseModel):
            title: str = Field(description="Catchy, scroll-stopping title for a vertical short-form video, under 90 characters, no clickbait lies")
            description: str = Field(description="2-4 sentence engaging caption/description for the post, natural tone, no keyword stuffing")
            hashtags: list[str] = Field(description="5-8 relevant hashtags, lowercase, WITHOUT the # symbol")

        prompt = f"""Write a title + description + hashtags for a short-form vertical
video (YouTube Shorts / Reels / TikTok) about to be published.

Source/original title: "{source_title or 'Unknown'}"
Extra context: {context_hint or 'none'}

Rules:
- Title: punchy, curiosity-driven, under 90 characters, no emojis spam (max 1).
- Description: 2-4 sentences, sounds human, ends with a soft call to engage.
- Hashtags: 5-8, lowercase, relevant, no spaces, without the # symbol.
"""

        def _call(client, model):
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": TitleDescription,
                    "temperature": 0.7,
                    "max_output_tokens": 512,
                },
            )

        result = hd.pool.call(hd.STAGE1_MODEL_CANDIDATES, _call)
        data = json.loads(result.text)
        return {
            "title": (data.get("title") or fallback["title"])[:95],
            "description": data.get("description") or "",
            "hashtags": data.get("hashtags") or [],
        }
    except Exception as e:
        fallback["description"] = fallback["description"] or f"(AI generation unavailable: {e})"
        return fallback


@publish_bp.route("/api/publish/ai_generate", methods=["POST"])
def api_ai_generate():
    body = request.json or {}
    source_title = body.get("source_title", "")
    context_hint = body.get("context_hint", "")
    return jsonify(_ai_generate(source_title, context_hint))


# ════════════════════════════════════════════════════════════════════════
# Feed of exported clips waiting to be published
# ════════════════════════════════════════════════════════════════════════
@publish_bp.route("/api/publish/exports")
def api_publish_exports():
    jobs = _get_export_jobs() if _get_export_jobs else {}
    clips = _get_clips() if _get_clips else {}
    out = []
    with _published_lock:
        published = set(_published_export_ids)

    for export_id, job in jobs.items():
        if job.get("status") != "done" or export_id in published:
            continue
        clip_id = job.get("clip_id")
        clip_info = clips.get(clip_id, {}) if clip_id else {}
        out.append({
            "export_id": export_id,
            "clip_id": clip_id,
            "title": job.get("title") or clip_info.get("title") or export_id,
            "duration": clip_info.get("duration"),
            "preview_url": f"/api/publish/media/{Path(job['path']).name}",
            "download_url": job.get("url"),
        })
    # newest first
    out.reverse()
    return jsonify({
        "exports": out,
        "platforms": db.list_connected_platforms(_current_user_id()) if _current_user_id() else [],
        "platform_meta": PLATFORM_META,
    })


@publish_bp.route("/api/publish/media/<fname>")
def api_publish_media(fname):
    """Inline (non-attachment) file serving so <video> preview/scrubbing
    works — RenderDetect's own /api/download/<fname> forces a download."""
    p = _OUTPUT_DIR / Path(fname).name
    if not p.exists():
        return "Not found", 404
    return send_file(p, as_attachment=False, conditional=True)


# ════════════════════════════════════════════════════════════════════════
# Scheduling helpers
# ════════════════════════════════════════════════════════════════════════
def _schedule_time_from(scheduled_time_str):
    """Parse a timestamp coming from the browser into an aware UTC datetime.

    The page's JS always sends a proper UTC-normalized ISO string ending in
    "Z" (built with `new Date(...).toISOString()` — see toUTCISO() in
    PUBLISH_PAGE's <script>), which is what fixes the old "publishes
    immediately instead of waiting" bug: a naive "YYYY-MM-DDTHH:MM" string
    from <input type="datetime-local"> is in the BROWSER'S local timezone,
    not UTC, and blindly relabeling it as UTC could put it in the past for
    anyone east of UTC — so the scheduler would fire on its very next tick.

    If we ever do receive a naive string (no "Z"/offset) we still have to
    pick *something* rather than crash — we treat it as UTC and log nothing
    special, same safe fallback as before, but this path should not be hit
    from our own UI anymore.
    """
    if not scheduled_time_str:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(str(scheduled_time_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _bulk_schedule_params(explicit_start_str, gap_minutes_raw):
    """Shared math for both bulk endpoints so single-clip bulk and file
    bulk-upload behave identically.

    Returns (base_time, gap_minutes, first_offset).

    - If the caller picked an explicit start time (via the calendar), that
      IS item #0's publish time, and each following item is +gap after it.
    - If they left it blank ("start from now"), item #0 goes out one full
      gap from now — "publish every 4 hours" means the first one is 4
      hours away too, not immediate — and each following item is another
      +gap after that.
    """
    try:
        gap_minutes = max(float(gap_minutes_raw), 0.0)
    except (TypeError, ValueError):
        gap_minutes = 60.0
    base_time = _schedule_time_from(explicit_start_str)
    first_offset = 0 if explicit_start_str else 1
    return base_time, gap_minutes, first_offset


# ════════════════════════════════════════════════════════════════════════
# Publish ONE exported clip (single, or one call of a bulk batch)
# ════════════════════════════════════════════════════════════════════════
def _publish_export(user_id, export_id, title, caption, platforms, scheduled_time, upload_id=None):
    upload_id = upload_id or uuid.uuid4().hex
    jobs = _get_export_jobs()
    job = jobs.get(export_id)
    if not job or job.get("status") != "done":
        _set_progress(upload_id, stage="error", message="That export isn't finished (or doesn't exist) anymore.")
        raise ValueError("That export isn't finished (or doesn't exist) anymore.")

    src_path = Path(job["path"])
    if not src_path.exists():
        _set_progress(upload_id, stage="error", message="Exported file is missing from disk.")
        raise ValueError("Exported file is missing from disk.")

    size_bytes = src_path.stat().st_size
    _new_progress(upload_id, filename=src_path.name, total_bytes=size_bytes, stage="reading")

    video_doc = _store_and_queue(
        user_id=user_id, local_path=src_path, filename=src_path.name, size_bytes=size_bytes,
        title=title or job.get("title") or src_path.stem, caption=caption or "",
        platforms=platforms, scheduled_time=scheduled_time, upload_id=upload_id, is_temp_file=False,
    )

    with _published_lock:
        _published_export_ids.add(export_id)

    return video_doc


@publish_bp.route("/api/publish/upload", methods=["POST"])
def api_publish_upload():
    """Single-clip publish — the 'Approve/Publish' button on one card."""
    user_id, err = _require_user()
    if err:
        return err

    body = request.json or {}
    export_id = body.get("export_id")
    if not export_id:
        return jsonify({"error": "export_id is required"}), 400
    upload_id = body.get("upload_id") or uuid.uuid4().hex

    try:
        scheduled_time = _schedule_time_from(body.get("scheduled_time"))
        video_doc = _publish_export(
            user_id=user_id,
            export_id=export_id,
            title=body.get("title", ""),
            caption=body.get("caption", ""),
            platforms=body.get("platforms") or [],
            scheduled_time=scheduled_time,
            upload_id=upload_id,
        )
        return jsonify({
            "ok": True,
            "video_id": str(video_doc["_id"]),
            "scheduled_time": video_doc["scheduled_time"].isoformat(),
            "upload_id": upload_id,
        })
    except Exception as e:
        return jsonify({"error": str(e), "upload_id": upload_id}), 400


@publish_bp.route("/api/publish/upload_bulk_exports", methods=["POST"])
def api_publish_upload_bulk_exports():
    """Bulk publish — several already-exported clips selected at once.
    Each item gets queued `gap_minutes` apart — see _bulk_schedule_params()
    for exactly how the first item's offset is decided."""
    user_id, err = _require_user()
    if err:
        return err

    body = request.json or {}
    items = body.get("items") or []   # [{export_id, title, caption}, ...]
    platforms = body.get("platforms") or []
    explicit_start = body.get("scheduled_time")
    base_time, gap_minutes, first_offset = _bulk_schedule_params(explicit_start, body.get("gap_minutes", 60))

    # Pre-register every item as "queued" BEFORE the loop starts, so the
    # page can start polling all of them immediately and see an honest
    # "waiting its turn" bar for #2, #3... instead of nothing — items are
    # deliberately processed one at a time here (not in parallel), same as
    # the scheduling math above assumes.
    for item in items:
        item["_upload_id"] = item.get("upload_id") or uuid.uuid4().hex
        _new_progress(item["_upload_id"], filename=item.get("title") or item.get("export_id") or "", stage="queued")

    results, errors = [], []
    for i, item in enumerate(items):
        export_id = item.get("export_id")
        upload_id = item["_upload_id"]
        try:
            scheduled_time = base_time + timedelta(minutes=gap_minutes * (i + first_offset))
            video_doc = _publish_export(
                user_id=user_id,
                export_id=export_id,
                title=item.get("title", ""),
                caption=item.get("caption", ""),
                platforms=item.get("platforms") or platforms,
                scheduled_time=scheduled_time,
                upload_id=upload_id,
            )
            results.append({"export_id": export_id, "video_id": str(video_doc["_id"]),
                             "scheduled_time": scheduled_time.isoformat(), "upload_id": upload_id})
        except Exception as e:
            errors.append({"export_id": export_id, "error": str(e), "upload_id": upload_id})

    return jsonify({"queued": results, "errors": errors})


# ════════════════════════════════════════════════════════════════════════
# Single-file browser upload — used by "New Uploads". Deliberately ONE
# file per HTTP request (not batched like bulk_upload_files below) so the
# browser's own upload.onprogress reflects THIS file only, and a slow or
# failed file never blocks/holds up its siblings.
# ════════════════════════════════════════════════════════════════════════
@publish_bp.route("/api/publish/upload_file", methods=["POST"])
def api_publish_upload_file():
    user_id, err = _require_user()
    if err:
        return err

    video_file = request.files.get("video_file")
    if not video_file or not video_file.filename:
        return jsonify({"error": "No file provided."}), 400

    upload_id = request.form.get("upload_id") or uuid.uuid4().hex
    title = request.form.get("title") or video_file.filename
    caption = request.form.get("caption", "")
    platforms = request.form.getlist("platforms")
    scheduled_time = _schedule_time_from(request.form.get("scheduled_time"))

    # Save to a local temp file FIRST (instead of streaming straight into
    # storage). This is what makes New Uploads resumable without asking
    # the browser to resend the file on retry: the bytes already made it
    # here — from now on any failure is on the server->storage leg, and
    # /api/publish/resume_upload can pick that back up using this same
    # local copy, no re-upload from the browser needed.
    _UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _UPLOAD_TMP_DIR / f"{upload_id}{Path(video_file.filename).suffix or '.mp4'}"
    _new_progress(upload_id, filename=video_file.filename, total_bytes=0, stage="reading")
    try:
        video_file.save(str(tmp_path))
        size_bytes = tmp_path.stat().st_size
        video_doc = _store_and_queue(
            user_id=user_id, local_path=tmp_path, filename=video_file.filename, size_bytes=size_bytes,
            title=title, caption=caption, platforms=platforms, scheduled_time=scheduled_time,
            upload_id=upload_id, is_temp_file=True,
        )
        return jsonify({
            "ok": True, "video_id": str(video_doc["_id"]),
            "scheduled_time": video_doc["scheduled_time"].isoformat(), "upload_id": upload_id,
        })
    except Exception as e:
        return jsonify({"error": str(e), "upload_id": upload_id}), 400


@publish_bp.route("/api/publish/resume_upload", methods=["POST"])
def api_resume_upload():
    """Retry after a failed publish/upload — resumes the SAME multipart
    transfer from the next unfinished part instead of re-uploading bytes
    that already made it to storage. Needs nothing from the client but the
    upload_id: no file re-send, no form fields re-send — everything needed
    was saved server-side in _resumable_registry on the first attempt."""
    user_id, err = _require_user()
    if err:
        return err

    body = request.json or {}
    upload_id = body.get("upload_id")
    if not upload_id:
        return jsonify({"error": "upload_id is required"}), 400

    entry = _get_resumable(upload_id)
    if not entry:
        return jsonify({"error": "Nothing to resume for this upload (the server may have restarted "
                                  "since the first attempt). Please upload the file again."}), 404
    if entry["user_id"] != user_id:
        return jsonify({"error": "Not authorized to resume this upload."}), 403

    local_path = Path(entry["local_path"])
    if not local_path.exists():
        _clear_resumable(upload_id)
        return jsonify({"error": "The source file is no longer available on the server. "
                                  "Please upload it again."}), 404

    try:
        scheduled_time = datetime.fromisoformat(entry["scheduled_time"])
        video_doc = _store_and_queue(
            user_id=user_id, local_path=local_path, filename=entry["filename"],
            size_bytes=entry["size_bytes"], title=entry["title"], caption=entry["caption"],
            platforms=entry["platforms"], scheduled_time=scheduled_time,
            upload_id=upload_id, is_temp_file=entry.get("is_temp_file", False),
        )
        return jsonify({
            "ok": True, "video_id": str(video_doc["_id"]),
            "scheduled_time": video_doc["scheduled_time"].isoformat(), "upload_id": upload_id,
        })
    except Exception as e:
        return jsonify({"error": str(e), "upload_id": upload_id}), 400


# ════════════════════════════════════════════════════════════════════════
# "New Uploads" tab — bulk uploading files that never went through the
# render pipeline (mirrors app.py's /bulk_upload, just embedded here too)
# ════════════════════════════════════════════════════════════════════════
@publish_bp.route("/api/publish/bulk_upload_files", methods=["POST"])
def api_publish_bulk_upload_files():
    user_id, err = _require_user()
    if err:
        return err

    files = request.files.getlist("video_files")
    titles = request.form.getlist("titles")
    captions = request.form.getlist("captions")
    platforms = request.form.getlist("platforms")
    explicit_start = request.form.get("scheduled_time")
    base_time, gap_minutes, first_offset = _bulk_schedule_params(explicit_start, request.form.get("gap_minutes", 60))

    if not files:
        return jsonify({"error": "Choose at least one video file."}), 400

    queued, errors = [], []
    for i, video_file in enumerate(files):
        if not video_file.filename:
            continue
        try:
            storage_result = storage_service.upload_video_stream(video_file.stream, video_file.filename, user_id)
            scheduled_time = base_time + timedelta(minutes=gap_minutes * (i + first_offset))
            title = titles[i] if i < len(titles) and titles[i] else video_file.filename
            caption = captions[i] if i < len(captions) else ""
            video_doc = db.create_queued_video(
                user_id=user_id, filename=video_file.filename,
                storage_key=storage_result["storage_key"], storage_url=storage_result["public_url"],
                size_bytes=storage_result["size_bytes"], scheduled_time=scheduled_time,
                title=title, caption=caption, platforms=platforms,
            )
            queued.append({"filename": video_file.filename, "video_id": str(video_doc["_id"]),
                            "scheduled_time": scheduled_time.isoformat()})
        except Exception as e:
            errors.append({"filename": video_file.filename, "error": str(e)})

    return jsonify({"queued": queued, "errors": errors})


@publish_bp.route("/api/publish/platforms")
def api_publish_platforms():
    uid = _current_user_id()
    return jsonify({
        "platforms": db.list_connected_platforms(uid) if uid else [],
        "logged_in": bool(uid),
        "platform_meta": PLATFORM_META,
        "server_now": datetime.now(timezone.utc).isoformat(),
    })


# ════════════════════════════════════════════════════════════════════════
# The page itself
# ════════════════════════════════════════════════════════════════════════
PUBLISH_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Publish Studio</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');
*{box-sizing:border-box;}
:root{
  --bg:#0a0c11; --panel:#12141c; --panel2:#181b26; --panel3:#1f232f; --border:#262a38;
  --text:#eef0f6; --dim:#8a90a4; --dim2:#5c6178;
  --accent:#7c6cff; --accent2:#22d3c4; --accent-grad:linear-gradient(135deg,#7c6cff,#22d3c4);
  --ok:#3DD68C; --fail:#FF6259; --warn:#FFB648;
}
body{margin:0;background:
    radial-gradient(1100px 500px at 12% -8%, rgba(124,108,255,.16), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(34,211,196,.10), transparent 55%),
    var(--bg);
  color:var(--text);font-family:'Inter',sans-serif;padding:22px 26px 70px;min-height:100vh;}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;font-weight:700;letter-spacing:-0.01em;margin:0 0 4px;}
h1{display:flex;align-items:center;gap:10px;}
h1 .beam{width:9px;height:28px;border-radius:5px;background:var(--accent-grad);display:inline-block;}
.hint{color:var(--dim);font-size:13.5px;margin:0 0 20px;max-width:860px;line-height:1.55;}
.top-tabs{display:flex;gap:8px;margin-bottom:20px;border-bottom:1px solid var(--border);padding-bottom:12px;}
.top-tab-btn{background:var(--panel);border:1px solid var(--border);color:var(--dim);padding:9px 16px;
  border-radius:9px;font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:13.5px;cursor:pointer;transition:.15s;}
.top-tab-btn:hover{color:var(--text);border-color:#3a3f52;}
.top-tab-btn.active{background:var(--accent-grad);color:#0a0c11;border-color:transparent;box-shadow:0 4px 18px -6px rgba(124,108,255,.55);}
.view{display:none;} .view.active{display:block;animation:fadeIn .18s ease;}
@keyframes fadeIn{from{opacity:0;transform:translateY(3px);}to{opacity:1;transform:none;}}
.login-banner{background:var(--panel2);border:1px solid var(--fail);border-radius:10px;padding:14px 16px;
  color:var(--text);font-size:13.5px;margin-bottom:18px;}
.login-banner a{color:var(--accent2);}
.empty{color:var(--dim);font-size:14px;border:1px dashed var(--border);border-radius:12px;padding:38px 18px;text-align:center;}

/* ── bulk scheduling bar ─────────────────────────────────────────── */
.bulk-bar{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:14px 16px;margin-bottom:16px;}
.bulk-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;font-size:13px;margin-bottom:10px;}
.bulk-row:last-child{margin-bottom:0;}
.bulk-row label.lbl{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em;}
.select-all-wrap{display:flex;align-items:center;gap:7px;color:var(--text);cursor:pointer;}
.select-all-wrap input{accent-color:var(--accent);width:16px;height:16px;}
.sel-count{color:var(--accent2);font-weight:600;font-family:'IBM Plex Mono',monospace;font-size:12px;}
.interval-input{display:flex;align-items:center;gap:6px;background:var(--panel2);border:1px solid var(--border);
  border-radius:9px;padding:4px 4px 4px 10px;}
.interval-input input[type=number]{width:52px;background:transparent;border:none;color:var(--text);font-size:14px;
  font-family:'IBM Plex Mono',monospace;font-weight:600;text-align:right;}
.interval-input input[type=number]:focus{outline:none;}
.interval-input select{background:var(--panel3);border:1px solid var(--border);color:var(--text);border-radius:6px;
  padding:6px 8px;font-size:12.5px;}
.preset-chip{background:var(--panel2);border:1px solid var(--border);color:var(--dim);border-radius:999px;
  padding:5px 11px;font-size:12px;cursor:pointer;font-family:'Space Grotesk',sans-serif;font-weight:600;transition:.12s;}
.preset-chip:hover{color:var(--text);border-color:#3a3f52;}
.preset-chip.active{background:color-mix(in srgb, var(--accent) 22%, transparent);border-color:var(--accent);color:#cfc9ff;}
.sched-trigger{display:flex;align-items:center;gap:7px;background:var(--panel2);border:1px solid var(--border);
  border-radius:9px;padding:8px 12px;font-size:13px;cursor:pointer;font-family:'Inter',sans-serif;color:var(--text);}
.sched-trigger:hover{border-color:var(--accent);}
.sched-trigger .rel{color:var(--accent2);font-size:11.5px;font-family:'IBM Plex Mono',monospace;}
.publish-selected-btn{margin-left:auto;font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:13.5px;
  border:none;border-radius:10px;padding:11px 20px;cursor:pointer;background:var(--accent-grad);color:#0a0c11;
  box-shadow:0 6px 20px -6px rgba(124,108,255,.5);}
.publish-selected-btn:disabled{opacity:.45;cursor:wait;box-shadow:none;}
.publish-selected-btn:hover:not(:disabled){filter:brightness(1.07);}
.bulk-preview{margin-top:2px;border-top:1px dashed var(--border);padding-top:10px;display:none;}
.bulk-preview.show{display:block;}
.bulk-preview-title{font-size:11.5px;color:var(--dim2);text-transform:uppercase;letter-spacing:.04em;font-weight:700;margin-bottom:7px;}
.bulk-preview-list{display:flex;flex-direction:column;gap:5px;max-height:180px;overflow-y:auto;}
.bulk-preview-item{display:flex;align-items:center;gap:9px;font-size:12.5px;color:var(--dim);
  background:var(--panel2);border-radius:7px;padding:6px 10px;}
.bulk-preview-item .n{width:20px;height:20px;border-radius:6px;background:var(--panel3);color:var(--accent2);
  font-family:'IBM Plex Mono',monospace;font-size:11px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.bulk-preview-item .nm{flex:1;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.bulk-preview-item .when{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--text);white-space:nowrap;}
.bulk-preview-item .rel{color:var(--accent2);font-size:11px;white-space:nowrap;}

/* ── cards ────────────────────────────────────────────────────────── */
.grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(270px, 1fr));gap:18px;}
.card{background:var(--panel);border:1px solid var(--border);border-radius:16px;overflow:hidden;display:flex;flex-direction:column;
  transition:border-color .15s, transform .15s, box-shadow .15s;}
.card:hover{transform:translateY(-2px);box-shadow:0 10px 28px -14px rgba(0,0,0,.6);}
.card.selected{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset;}
.card.just-highlighted{animation:pulseHi 1.6s ease 2;}
@keyframes pulseHi{0%{box-shadow:0 0 0 0 rgba(124,108,255,.55);}70%{box-shadow:0 0 0 12px rgba(124,108,255,0);}100%{box-shadow:0 0 0 0 rgba(124,108,255,0);}}
.card-top{display:flex;align-items:center;gap:8px;padding:11px 12px;border-bottom:1px solid var(--border);background:var(--panel2);}
.card-top input[type=checkbox]{accent-color:var(--accent);width:16px;height:16px;cursor:pointer;}
.video-wrap{width:100%;aspect-ratio:9/16;background:#000;max-height:420px;position:relative;}
.video-wrap video{width:100%;height:100%;object-fit:contain;display:block;background:#000;}
.dur-badge{position:absolute;bottom:8px;right:8px;background:rgba(0,0,0,.72);color:#fff;font-size:11px;font-weight:600;
  padding:3px 7px;border-radius:6px;font-family:'IBM Plex Mono',monospace;}
.card-body{padding:14px;display:flex;flex-direction:column;gap:10px;}
.card-body label{font-size:11px;font-weight:700;color:var(--dim2);text-transform:uppercase;letter-spacing:.03em;margin-bottom:-5px;}
.card-body input[type=text],.card-body textarea{
  background:var(--panel2);border:1px solid var(--border);border-radius:9px;color:var(--text);
  font-family:'Inter',sans-serif;font-size:13px;padding:8px 10px;width:100%;transition:border-color .12s;}
.card-body input[type=text]:focus,.card-body textarea:focus{outline:none;border-color:var(--accent);}
.card-body textarea{min-height:54px;resize:vertical;}
.ai-row{display:flex;gap:6px;}
.ai-btn{flex-shrink:0;background:var(--panel2);border:1px solid var(--accent2);color:var(--accent2);
  border-radius:8px;padding:7px 10px;font-size:12px;cursor:pointer;font-family:'Space Grotesk',sans-serif;font-weight:600;}
.ai-btn:hover{background:color-mix(in srgb, var(--accent2) 15%, transparent);}
.ai-btn:disabled{opacity:.5;cursor:wait;}
.chip-row{display:flex;flex-wrap:wrap;gap:6px;}
.chip{display:flex;align-items:center;gap:5px;background:var(--panel2);border:1px solid var(--border);
  border-radius:999px;padding:5px 10px 5px 8px;font-size:11.5px;cursor:pointer;transition:.12s;}
.chip:hover{border-color:#3a3f52;}
.chip:has(input:checked){border-color:var(--chip-c, var(--accent));background:color-mix(in srgb, var(--chip-c, var(--accent)) 16%, transparent);}
.chip input{accent-color:var(--chip-c, var(--accent));}
.publish-btn{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:13px;border:none;border-radius:10px;
  padding:11px 10px;cursor:pointer;background:var(--ok);color:#062015;margin-top:2px;transition:filter .12s;}
.publish-btn:hover{filter:brightness(1.08);}
.publish-btn:disabled{opacity:.55;cursor:wait;}
.status-line{font-size:11.5px;color:var(--dim);min-height:14px;}
.status-line.ok{color:var(--ok);} .status-line.err{color:var(--fail);}
.dropzone{border:1.5px dashed var(--border);border-radius:14px;padding:38px 18px;text-align:center;color:var(--dim);
  font-size:13.5px;cursor:pointer;margin-bottom:16px;background:var(--panel);transition:.15s;}
.dropzone:hover{border-color:#3a3f52;}
.dropzone.drag{border-color:var(--accent);color:var(--text);background:color-mix(in srgb, var(--accent) 6%, var(--panel));}
.file-row{display:grid;grid-template-columns:1.1fr 1.3fr 1.6fr;gap:10px;align-items:start;background:var(--panel);
  border:1px solid var(--border);border-radius:12px;padding:10px 12px;margin-bottom:10px;}
.file-row .fname{font-size:12.5px;color:var(--dim);word-break:break-all;padding-top:6px;}
.file-row input,.file-row textarea{background:var(--panel2);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:7px 9px;font-size:12.5px;width:100%;}
.file-row textarea{min-height:44px;resize:vertical;}
.queue-btn{font-family:'Space Grotesk',sans-serif;font-weight:700;background:var(--accent-grad);
  color:#0a0c11;border:none;border-radius:10px;padding:11px 18px;font-size:13.5px;cursor:pointer;
  box-shadow:0 6px 20px -6px rgba(124,108,255,.5);}
.queue-btn:disabled{opacity:.5;cursor:wait;box-shadow:none;}
.small-dim{font-size:12px;color:var(--dim);}

/* ── schedule popover (the "YouTube-style" calendar+time picker) ────── */
.sched-pop{position:fixed;z-index:9999;width:296px;background:var(--panel);border:1px solid var(--border);
  border-radius:16px;box-shadow:0 22px 60px -18px rgba(0,0,0,.75);padding:14px;font-family:'Inter',sans-serif;
  animation:popIn .12s ease;}
@keyframes popIn{from{opacity:0;transform:translateY(-4px) scale(.98);}to{opacity:1;transform:none;}}
.sched-pop-quick{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;}
.sched-pop-quick button{background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:999px;
  padding:6px 11px;font-size:11.5px;cursor:pointer;font-family:'Space Grotesk',sans-serif;font-weight:600;}
.sched-pop-quick button:hover{border-color:var(--accent);color:var(--accent2);}
.cal-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;}
.cal-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:13.5px;}
.cal-nav{background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:7px;
  width:26px;height:26px;cursor:pointer;font-size:14px;line-height:1;}
.cal-nav:hover{border-color:var(--accent);}
.cal-dow{color:var(--dim2);font-size:10.5px;font-weight:700;margin-bottom:2px;}
.cal-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:3px;text-align:center;}
.cal-cell{padding:6px 0;border-radius:8px;font-size:12.5px;cursor:pointer;color:var(--text);}
.cal-cell.empty{cursor:default;}
.cal-cell:not(.empty):not(.past):hover{background:var(--panel2);}
.cal-cell.today{color:var(--accent2);font-weight:700;}
.cal-cell.sel{background:var(--accent-grad);color:#0a0c11;font-weight:700;}
.cal-cell.past{color:var(--dim2);opacity:.35;cursor:not-allowed;}
.time-row{display:flex;align-items:center;justify-content:center;gap:6px;margin:14px 0 12px;}
.time-row input[type=number]{width:42px;background:var(--panel2);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:8px 0;text-align:center;font-family:'IBM Plex Mono',monospace;font-size:15px;font-weight:600;}
.time-row input[type=number]:focus{outline:none;border-color:var(--accent);}
.time-row .colon{color:var(--dim);font-weight:700;}
.ampm-toggle{display:flex;border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-left:4px;}
.ampm-btn{background:var(--panel2);border:none;color:var(--dim);padding:8px 10px;font-size:11.5px;font-weight:700;cursor:pointer;}
.ampm-btn.active{background:var(--accent-grad);color:#0a0c11;}
.sched-pop-actions{display:flex;gap:8px;}
.sched-pop-actions button{flex:1;border:none;border-radius:9px;padding:9px 0;font-size:12.5px;font-weight:700;
  font-family:'Space Grotesk',sans-serif;cursor:pointer;}
.sched-pop-cancel{background:var(--panel2);color:var(--dim);}
.sched-pop-cancel:hover{color:var(--text);}
.sched-pop-confirm{background:var(--accent-grad);color:#0a0c11;}
.sched-pop-confirm:hover{filter:brightness(1.07);}
.sched-pop-overlay{position:fixed;inset:0;z-index:9998;background:rgba(5,6,10,.35);backdrop-filter:blur(1px);}

/* ── upload progress bar ─────────────────────────────────────────── */
.prog-wrap{margin-top:8px;background:var(--panel2);border:1px solid var(--border);border-radius:10px;
  padding:9px 10px 8px;display:none;}
.prog-wrap.show{display:block;}
.prog-top{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:11px;margin-bottom:6px;}
.prog-stage{display:flex;align-items:center;gap:6px;color:var(--dim);font-weight:600;font-family:'Space Grotesk',sans-serif;}
.prog-stage.prog-slow{color:var(--warn);}
.prog-stage .dot{width:7px;height:7px;border-radius:50%;background:var(--dim2);flex-shrink:0;}
.prog-wrap.st-sending .prog-stage .dot,.prog-wrap.st-reading .prog-stage .dot,
.prog-wrap.st-uploading .prog-stage .dot{background:var(--accent);animation:pulseDot 1s ease infinite;}
.prog-wrap.st-saving .prog-stage .dot{background:var(--warn);animation:pulseDot .7s ease infinite;}
.prog-wrap.st-done .prog-stage .dot{background:var(--ok);animation:none;}
.prog-wrap.st-error .prog-stage .dot{background:var(--fail);animation:none;}
.prog-wrap.st-queued .prog-stage .dot{background:var(--dim2);animation:none;}
@keyframes pulseDot{0%,100%{opacity:1;}50%{opacity:.35;}}
.prog-pct{font-family:'IBM Plex Mono',monospace;color:var(--text);font-weight:600;flex-shrink:0;}
.prog-bar-track{height:7px;border-radius:99px;background:var(--panel3);overflow:hidden;position:relative;}
.prog-bar-fill{height:100%;border-radius:99px;background:var(--accent-grad);width:0%;transition:width .18s ease;}
.prog-wrap.st-error .prog-bar-fill{background:var(--fail);}
.prog-wrap.st-done .prog-bar-fill{background:var(--ok);}
.prog-wrap.st-queued .prog-bar-fill{background:var(--dim2);}
.prog-bottom{display:flex;justify-content:space-between;gap:8px;font-size:10.5px;color:var(--dim);margin-top:5px;
  font-family:'IBM Plex Mono',monospace;}
.prog-msg{font-size:11px;color:var(--fail);margin-top:6px;line-height:1.4;}
.prog-retry{margin-top:7px;background:var(--panel3);border:1px solid var(--fail);color:var(--fail);border-radius:7px;
  padding:5px 10px;font-size:11.5px;cursor:pointer;font-family:'Space Grotesk',sans-serif;font-weight:600;}
.prog-retry:hover{background:color-mix(in srgb, var(--fail) 14%, transparent);}

/* ── global "active uploads" tray — always visible, no scrolling to find
   which card/file is uploading and how fast ─────────────────────────── */
.upload-tray{position:fixed;right:22px;bottom:22px;width:322px;max-height:min(60vh,480px);
  background:var(--panel);border:1px solid var(--border);border-radius:14px;
  box-shadow:0 20px 50px -18px rgba(0,0,0,.65);z-index:9997;display:none;flex-direction:column;
  overflow:hidden;font-family:'Inter',sans-serif;}
.upload-tray.show{display:flex;}
.tray-head{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;
  background:var(--panel2);border-bottom:1px solid var(--border);font-family:'Space Grotesk',sans-serif;
  font-weight:700;font-size:13px;flex-shrink:0;}
.tray-count{background:var(--accent-grad);color:#0a0c11;border-radius:999px;padding:2px 9px;
  font-size:11.5px;font-weight:700;font-family:'IBM Plex Mono',monospace;}
.tray-list{overflow-y:auto;padding:8px;display:flex;flex-direction:column;gap:8px;}
.tray-item{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:8px 9px 7px;}
.tray-item-top{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:5px;}
.tray-item-title{font-size:12px;color:var(--text);font-weight:600;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;flex:1;}
.tray-item-x{background:none;border:none;color:var(--dim);cursor:pointer;font-size:13px;line-height:1;
  padding:2px 4px;flex-shrink:0;}
.tray-item-x:hover{color:var(--fail);}
.tray-item .prog-wrap{margin-top:0;padding:0;border:none;background:transparent;}
</style>
</head>
<body>

<div class="upload-tray" id="uploadTray">
  <div class="tray-head"><span>⬆️ Active uploads</span><span class="tray-count" id="trayCount">0</span></div>
  <div class="tray-list" id="trayList"></div>
</div>


<h1><span class="beam"></span>Publish Studio</h1>
<p class="hint">
  Every clip you export lands here automatically. Edit the title/caption (or let AI write them),
  pick platforms and a time, then hit <strong>Publish</strong> — it uploads straight to storage and
  gets queued; the scheduler posts it the moment its time comes, in your own local time zone.
  Tick several cards to publish them as an evenly-spaced batch, or use <strong>New Uploads</strong>
  to bulk-publish files that never went through the render pipeline at all.
</p>

<div id="loginBanner" class="login-banner" style="display:none;">
  Not logged in — Publish Studio needs a hub account to know which platforms to post to.
  <a href="/login" target="_top">Log in here</a>, then reopen this tab.
</div>

<div class="top-tabs">
  <button type="button" class="top-tab-btn active" id="tabExportsBtn" onclick="showView('exports')">🎬 Exported Clips</button>
  <button type="button" class="top-tab-btn" id="tabUploadsBtn" onclick="showView('uploads')">⬆️ New Uploads</button>
</div>

<!-- ═══════════════ TAB 1: exported clips feed ═══════════════ -->
<div class="view active" id="view-exports">
  <div class="bulk-bar">
    <div class="bulk-row">
      <label class="select-all-wrap"><input type="checkbox" id="selectAllChk" onchange="toggleSelectAll(this)"> Select all</label>
      <span class="sel-count" id="selCount">0 selected</span>
      <label class="lbl" style="margin-left:8px;">Every</label>
      <div class="interval-input">
        <input type="number" min="1" id="bulkIntervalNum" value="4" oninput="onIntervalChange('bulk')">
        <select id="bulkIntervalUnit" onchange="onIntervalChange('bulk')">
          <option value="minutes">min</option>
          <option value="hours" selected>hours</option>
          <option value="days">days</option>
        </select>
      </div>
      <div id="bulkPresetChips" class="chip-row"></div>
    </div>
    <div class="bulk-row">
      <label class="lbl">Start</label>
      <button type="button" class="sched-trigger" id="bulkStartTrigger" onclick="openPicker('bulk')">
        <span id="bulkStartLabel">🕒 Now — first video goes out after one interval</span>
      </button>
      <div class="chip-row" id="bulkPlatformChips"></div>
      <button type="button" class="publish-selected-btn" id="bulkPublishBtn" onclick="publishSelected()">🚀 Publish Selected</button>
    </div>
    <div class="bulk-preview" id="bulkPreview">
      <div class="bulk-preview-title">Schedule preview</div>
      <div class="bulk-preview-list" id="bulkPreviewList"></div>
    </div>
  </div>
  <div id="exportsEmpty" class="empty" style="display:none;">No exported clips waiting to publish. Export something in the Studio tab and it'll show up here within a few seconds.</div>
  <div class="grid" id="exportsGrid"></div>
</div>

<!-- ═══════════════ TAB 2: bulk upload new files ═══════════════ -->
<div class="view" id="view-uploads">
  <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
    📁 Click or drag &amp; drop video files here — multiple at once for bulk upload
    <input type="file" id="fileInput" accept="video/*" multiple style="display:none;" onchange="handleFiles(this.files)">
  </div>
  <div id="fileRows"></div>
  <div class="bulk-bar" id="uploadsBar" style="display:none;">
    <div class="bulk-row">
      <label class="lbl">Every</label>
      <div class="interval-input">
        <input type="number" min="1" id="uploadIntervalNum" value="4" oninput="onIntervalChange('upload')">
        <select id="uploadIntervalUnit" onchange="onIntervalChange('upload')">
          <option value="minutes">min</option>
          <option value="hours" selected>hours</option>
          <option value="days">days</option>
        </select>
      </div>
      <div id="uploadPresetChips" class="chip-row"></div>
    </div>
    <div class="bulk-row">
      <label class="lbl">Start</label>
      <button type="button" class="sched-trigger" id="uploadStartTrigger" onclick="openPicker('upload')">
        <span id="uploadStartLabel">🕒 Now — first video goes out after one interval</span>
      </button>
      <div class="chip-row" id="uploadPlatformChips"></div>
      <button type="button" class="queue-btn" id="queueAllBtn" onclick="queueAllUploads()">🚀 Queue All</button>
    </div>
    <div class="bulk-preview" id="uploadPreview">
      <div class="bulk-preview-title">Schedule preview</div>
      <div class="bulk-preview-list" id="uploadPreviewList"></div>
    </div>
  </div>
  <p class="status-line" id="uploadsStatus"></p>
</div>

<script>
let PLATFORMS = [];
let PLATFORM_META = {};
let LOGGED_IN = false;
let pendingFiles = []; // {file, title, caption}
let highlightExportId = null;

// bulk/upload scheduling state — startIso === null means "start from now"
const bulkState   = { startIso: null };
const uploadState = { startIso: null };

// ════════════════════════════════════════════════════════════════════
// UPLOAD PROGRESS ENGINE — real, not simulated.
//   • "sending" leg (browser -> server) comes from XMLHttpRequest's real
//     upload.onprogress event — only exists for a fresh file (New Uploads).
//   • "uploading"/"saving" legs (server -> cloud storage -> Mongo) come
//     from polling /api/publish/upload_progress/<id>, which itself is fed
//     by boto3's real transfer callback server-side.
// A publish that has BOTH legs shows one bar: 0–50% = bytes actually
// leaving the browser, 50–100% = the server's real progress pushing those
// same bytes into storage. A publish with only the storage leg (single /
// bulk "Publish" on an already-rendered clip) just uses 0–100% for that.
// ════════════════════════════════════════════════════════════════════
function genUploadId(){
  try{ return crypto.randomUUID().replace(/-/g,''); }
  catch(e){ return 'u'+Date.now().toString(16)+Math.random().toString(16).slice(2); }
}
function fmtBytes(n){
  if(n === null || n === undefined || isNaN(n)) return '0 B';
  const u = ['B','KB','MB','GB']; let i=0, v=n;
  while(v>=1024 && i<u.length-1){ v/=1024; i++; }
  return (i===0 ? Math.round(v) : v.toFixed(1)) + ' ' + u[i];
}
function fmtSpeed(bps){ return (bps && bps>0) ? fmtBytes(bps)+'/s' : '—'; }
function fmtEta(s){
  if(s===null || s===undefined || !isFinite(s)) return '—';
  s = Math.round(s);
  if(s < 1) return 'almost done';
  if(s < 60) return s+'s left';
  const m = Math.floor(s/60), sec = s%60;
  if(m < 60) return m+'m '+String(sec).padStart(2,'0')+'s left';
  const h = Math.floor(m/60);
  return h+'h '+(m%60)+'m left';
}
const STAGE_LABEL = {
  queued:'Waiting…', sending:'Uploading…', reading:'Preparing file…',
  uploading:'Uploading to cloud storage…', saving:'Saving…',
  done:'Done ✅', error:'Failed',
};
const _progState = {}; // uploadId -> {els:[el,...], hasSendLeg, sendPct, serverPct, pollTimer, stopped, title}

function _progInnerHtml(hasSendLeg){
  return `
    <div class="prog-top">
      <span class="prog-stage"><span class="dot"></span><span class="stageTxt">${hasSendLeg ? STAGE_LABEL.sending : STAGE_LABEL.queued}</span></span>
      <span class="prog-pct">0%</span>
    </div>
    <div class="prog-bar-track"><div class="prog-bar-fill"></div></div>
    <div class="prog-bottom"><span class="prog-detail">—</span><span class="prog-eta">—</span></div>
    <div class="prog-msg" style="display:none;"></div>`;
}

// Every upload also gets a mirror entry in the fixed "Active uploads" tray
// (bottom-right, always on screen) — so during a bulk publish/upload you
// can see title + live speed/% for every item at a glance without
// scrolling down to hunt for its card.
function ensureTray(){
  const tray = document.getElementById('uploadTray');
  return tray;
}
function trayUpdateCount(){
  const tray = document.getElementById('uploadTray');
  if(!tray) return;
  const n = document.querySelectorAll('#trayList .tray-item').length;
  document.getElementById('trayCount').textContent = n;
  tray.classList.toggle('show', n > 0);
}
function trayAdd(uploadId, title, hasSendLeg){
  ensureTray();
  const list = document.getElementById('trayList');
  if(!list) return null;
  let item = document.getElementById('tray-'+uploadId);
  if(!item){
    item = document.createElement('div');
    item.className = 'tray-item';
    item.id = 'tray-'+uploadId;
    item.innerHTML = `<div class="tray-item-top"><div class="tray-item-title"></div>
      <button type="button" class="tray-item-x" onclick="trayRemove('${uploadId}')" title="Dismiss">✕</button></div>
      <div class="prog-wrap show" id="trayprog-${uploadId}"></div>`;
    list.appendChild(item);
  }
  item.querySelector('.tray-item-title').textContent = title || 'Untitled';
  const progEl = document.getElementById('trayprog-'+uploadId);
  progEl.innerHTML = _progInnerHtml(hasSendLeg);
  progEl.className = 'prog-wrap show st-' + (hasSendLeg ? 'sending' : 'queued');
  trayUpdateCount();
  return progEl;
}
function trayRemove(uploadId){
  document.getElementById('tray-'+uploadId)?.remove();
  trayUpdateCount();
}

function initProgress(uploadId, containerId, opts){
  const el = document.getElementById(containerId);
  if(!el) return;
  const hasSendLeg = !!(opts && opts.hasSendLeg);
  const title = (opts && opts.title) || '';
  const els = [el];
  el.innerHTML = _progInnerHtml(hasSendLeg);
  el.className = 'prog-wrap show st-' + (hasSendLeg ? 'sending' : 'queued');
  if(title){
    const trayEl = trayAdd(uploadId, title, hasSendLeg);
    if(trayEl) els.push(trayEl);
  }
  _progState[uploadId] = { els, hasSendLeg, sendPct: 0, serverPct: 0, pollTimer: null, stopped: false, title };
}
function combinedPct(uploadId){
  const st = _progState[uploadId]; if(!st) return 0;
  if(!st.hasSendLeg) return st.serverPct;
  return Math.min(st.sendPct*0.5 + st.serverPct*0.5, 100);
}
function updateProgressUI(uploadId, opts){
  const st = _progState[uploadId]; if(!st) return;
  const stage = opts.stage;
  const p = opts.pct !== undefined ? opts.pct : combinedPct(uploadId);
  st.els.forEach(el=>{
    if(!el || !el.isConnected) return; // card/tray item may have been removed already
    if(stage) el.className = 'prog-wrap show st-' + stage;
    const stageTxt = el.querySelector('.stageTxt');
    if(stage && STAGE_LABEL[stage]) stageTxt.textContent = STAGE_LABEL[stage];
    el.querySelector('.prog-stage').classList.toggle('prog-slow', !!opts.slow);
    if(opts.slow) stageTxt.textContent = (STAGE_LABEL[stage] || stageTxt.textContent) + ' — slow connection';
    el.querySelector('.prog-pct').textContent = Math.round(p) + '%';
    el.querySelector('.prog-bar-fill').style.width = Math.max(0, Math.min(p,100)) + '%';
    if(opts.detail !== undefined) el.querySelector('.prog-detail').textContent = opts.detail;
    if(opts.eta !== undefined) el.querySelector('.prog-eta').textContent = opts.eta;
    const msgEl = el.querySelector('.prog-msg');
    if(opts.message){ msgEl.style.display = 'block'; msgEl.textContent = '⚠ ' + opts.message; }
    else if(opts.message === ''){ msgEl.style.display = 'none'; }
  });
}
function stopProgress(uploadId){
  const st = _progState[uploadId];
  if(st){ st.stopped = true; if(st.pollTimer) clearTimeout(st.pollTimer); }
}

// real client -> server byte progress, straight from the browser's network stack
function trackXhrSend(xhr, uploadId){
  let lastT = performance.now(), lastB = 0, speed = 0;
  xhr.upload.addEventListener('progress', (e)=>{
    if(!e.lengthComputable) return;
    const now = performance.now();
    const dt = (now - lastT) / 1000;
    if(dt > 0.15){
      speed = (e.loaded - lastB) / dt;
      lastT = now; lastB = e.loaded;
    }
    const st = _progState[uploadId]; if(!st) return;
    st.sendPct = (e.loaded / e.total) * 100;
    const eta = speed > 1024 ? (e.total - e.loaded) / speed : null;
    updateProgressUI(uploadId, {
      stage: 'sending', pct: combinedPct(uploadId),
      detail: fmtBytes(e.loaded) + ' / ' + fmtBytes(e.total) + ' · ' + fmtSpeed(speed),
      eta: eta ? fmtEta(eta) : '—',
      slow: speed > 0 && speed < 40000 && e.total > 2_000_000 && e.loaded < e.total,
    });
  });
}

// real server-side (storage-upload) progress, polled live
function pollServerProgress(uploadId, onDone){
  const st = _progState[uploadId];
  if(!st) return;
  const tick = async ()=>{
    if(st.stopped) return;
    try{
      const r = await fetch('/api/publish/upload_progress/' + uploadId);
      const d = await r.json();
      st.serverPct = d.pct || 0;
      updateProgressUI(uploadId, {
        stage: d.stage, pct: combinedPct(uploadId),
        detail: d.bytes_total ? (fmtBytes(d.bytes_done) + ' / ' + fmtBytes(d.bytes_total) + ' · ' + fmtSpeed(d.speed_bps)) : (STAGE_LABEL[d.stage] || ''),
        eta: d.eta_seconds != null ? fmtEta(d.eta_seconds) : '—',
        slow: !!d.slow, message: d.stage === 'error' ? (d.message || 'Upload failed.') : '',
      });
      if(d.stage === 'done' || d.stage === 'error'){
        st.stopped = true;
        onDone && onDone(d);
        return;
      }
    }catch(e){ /* transient network hiccup while polling — just retry */ }
    if(!st.stopped) st.pollTimer = setTimeout(tick, 450);
  };
  tick();
}

function showRetryBtn(containerId, onRetry){
  const el = document.getElementById(containerId);
  if(!el || el.querySelector('.prog-retry')) return;
  const btn = document.createElement('button');
  btn.type = 'button'; btn.className = 'prog-retry'; btn.textContent = '🔁 Retry upload';
  btn.onclick = ()=>{ btn.remove(); onRetry(); };
  el.appendChild(btn);
}

// Shared retry wiring for an already-rendered clip's publish (used by both
// the single "✅ Publish" button AND every item inside "🚀 Publish
// Selected"). Attaching this is what makes the retry button show up THE
// MOMENT that one item's progress bar reports "error" — not after the
// whole bulk request finishes — because it's called straight from that
// item's live poll callback, not from the bulk response at the end.
function attachExportRetry(exportId, uploadId){
  showRetryBtn('prog-'+exportId, ()=>resumeUpload(uploadId, 'prog-'+exportId, (dd)=>{
    if(dd.stage === 'done'){
      setStatus(exportId, '✅ Uploaded — queued', 'ok');
      setTimeout(()=>{ document.getElementById('card-'+exportId)?.remove(); trayRemove(uploadId); }, 1400);
    }else if(dd.stage === 'error'){
      setStatus(exportId, '❌ ' + (dd.message||'Failed'), 'err');
    }
  }, ()=>publishOne(exportId)));
}

// Resumes a failed upload from wherever it stopped — NO file/JSON resend,
// just the upload_id. The server already knows the local file, title,
// caption, platforms, and schedule from the first attempt (see
// _resumable_registry in publish_module.py), and skips every multipart
// part already confirmed uploaded, so a slow-internet failure at 63%
// resumes near 63%, not 0%. `fallbackFn`, if given, runs instead of
// offering another resume when the server has genuinely nothing saved to
// resume from (e.g. the very first attempt never reached it at all).
async function resumeUpload(uploadId, containerId, onDone, fallbackFn){
  const st = _progState[uploadId];
  if(st){ st.stopped = false; }
  updateProgressUI(uploadId, {stage:'uploading', message:''});
  pollServerProgress(uploadId, onDone);
  try{
    const r = await fetch('/api/publish/resume_upload', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({upload_id: uploadId})
    });
    const d = await r.json();
    if(d.error){
      stopProgress(uploadId);
      updateProgressUI(uploadId, {stage:'error', message: d.error});
      const cantResume = /nothing to resume|no longer available/i.test(d.error || '');
      if(cantResume && fallbackFn) fallbackFn();
      else showRetryBtn(containerId, ()=>resumeUpload(uploadId, containerId, onDone, fallbackFn));
      return {ok:false, error:d.error};
    }
    return {ok:true, data:d};
  }catch(e){
    stopProgress(uploadId);
    updateProgressUI(uploadId, {stage:'error', message: e.message});
    showRetryBtn(containerId, ()=>resumeUpload(uploadId, containerId, onDone, fallbackFn));
    return {ok:false, error:e.message};
  }
}

function showView(name){
  document.getElementById('view-exports').classList.toggle('active', name==='exports');
  document.getElementById('view-uploads').classList.toggle('active', name==='uploads');
  document.getElementById('tabExportsBtn').classList.toggle('active', name==='exports');
  document.getElementById('tabUploadsBtn').classList.toggle('active', name==='uploads');
}

// ════════════════════ time helpers ════════════════════
// THE FIX: convert a JS Date to a real, unambiguous UTC instant before it
// ever reaches the server. This is what stops "publish in 4 hours" from
// firing immediately for anyone not already on UTC.
function toUTCISO(date){ return date.toISOString(); }

function fmtWhen(date){
  return date.toLocaleString(undefined, { month:'short', day:'numeric', hour:'numeric', minute:'2-digit' });
}
function fmtRelative(date){
  const diffMs = date.getTime() - Date.now();
  const past = diffMs < 0;
  const abs = Math.abs(diffMs);
  const mins = Math.round(abs/60000);
  let out;
  if(mins < 1) out = 'now';
  else if(mins < 60) out = mins + 'm';
  else if(mins < 1440) out = Math.floor(mins/60) + 'h ' + (mins%60) + 'm';
  else out = Math.floor(mins/1440) + 'd ' + Math.floor((mins%1440)/60) + 'h';
  return (past ? out + ' ago' : 'in ' + out);
}
function intervalMinutes(prefix){
  const n = parseFloat(document.getElementById(prefix+'IntervalNum').value) || 0;
  const unit = document.getElementById(prefix+'IntervalUnit').value;
  const mult = unit === 'minutes' ? 1 : unit === 'hours' ? 60 : 1440;
  return Math.max(n * mult, 1);
}

// ════════════════════ YouTube-style calendar + time popover ════════════════════
let _pickerCtx = null; // {prefix, anchorDate}

function openPicker(prefix){
  closePicker();
  const state = prefix === 'bulk' ? bulkState : uploadState;
  const anchorDate = state.startIso ? new Date(state.startIso) : new Date(Date.now() + 60*60*1000);
  _pickerCtx = { prefix, view: new Date(anchorDate.getFullYear(), anchorDate.getMonth(), 1), selected: new Date(anchorDate) };

  const overlay = document.createElement('div');
  overlay.className = 'sched-pop-overlay';
  overlay.onclick = closePicker;
  document.body.appendChild(overlay);

  const trigger = document.getElementById(prefix+'StartTrigger');
  const rect = trigger.getBoundingClientRect();
  const pop = document.createElement('div');
  pop.className = 'sched-pop';
  pop.id = 'schedPop';
  pop.style.top = Math.min(rect.bottom + 8, window.innerHeight - 420) + 'px';
  pop.style.left = Math.min(rect.left, window.innerWidth - 312) + 'px';
  pop.onclick = (e)=>e.stopPropagation();
  pop.innerHTML = `
    <div class="sched-pop-quick" id="pkQuick"></div>
    <div id="pkCal"></div>
    <div class="time-row">
      <input type="number" min="1" max="12" id="pkHH" value="8">
      <span class="colon">:</span>
      <input type="number" min="0" max="59" id="pkMM" value="00">
      <div class="ampm-toggle">
        <button type="button" class="ampm-btn active" data-v="AM" onclick="setAmPm('AM')">AM</button>
        <button type="button" class="ampm-btn" data-v="PM" onclick="setAmPm('PM')">PM</button>
      </div>
    </div>
    <div class="sched-pop-actions">
      <button type="button" class="sched-pop-cancel" onclick="closePicker()">Cancel</button>
      <button type="button" class="sched-pop-confirm" onclick="confirmPicker()">Set schedule</button>
    </div>`;
  document.body.appendChild(pop);

  // seed time fields + am/pm from the anchor date
  let h24 = anchorDate.getHours();
  const ampm = h24 >= 12 ? 'PM' : 'AM';
  let h12 = h24 % 12; if(h12 === 0) h12 = 12;
  document.getElementById('pkHH').value = h12;
  document.getElementById('pkMM').value = String(anchorDate.getMinutes()).padStart(2,'0');
  setAmPm(ampm);

  renderQuickChips();
  renderCal();
}

function renderQuickChips(){
  const now = new Date();
  const opts = [
    { label:'Now', get:()=>new Date(now.getTime()+2*60000) },
    { label:'+1h', get:()=>new Date(now.getTime()+60*60000) },
    { label:'+4h', get:()=>new Date(now.getTime()+4*60*60000) },
    { label:'+6h', get:()=>new Date(now.getTime()+6*60*60000) },
    { label:'Tomorrow 9AM', get:()=>{ const d=new Date(now); d.setDate(d.getDate()+1); d.setHours(9,0,0,0); return d; } },
  ];
  const wrap = document.getElementById('pkQuick');
  wrap.innerHTML = opts.map((o,i)=>`<button type="button" data-i="${i}">${o.label}</button>`).join('');
  wrap.querySelectorAll('button').forEach((btn,i)=>{
    btn.onclick = ()=>{
      const d = opts[i].get();
      _pickerCtx.selected = d;
      _pickerCtx.view = new Date(d.getFullYear(), d.getMonth(), 1);
      let h24 = d.getHours(); const ampm = h24>=12?'PM':'AM'; let h12=h24%12; if(h12===0) h12=12;
      document.getElementById('pkHH').value = h12;
      document.getElementById('pkMM').value = String(d.getMinutes()).padStart(2,'0');
      setAmPm(ampm);
      renderCal();
    };
  });
}

function setAmPm(v){
  document.querySelectorAll('.ampm-btn').forEach(b=>b.classList.toggle('active', b.dataset.v===v));
}

function renderCal(){
  const ctx = _pickerCtx;
  const y = ctx.view.getFullYear(), m = ctx.view.getMonth();
  const first = new Date(y, m, 1);
  const startWeekday = first.getDay();
  const daysInMonth = new Date(y, m+1, 0).getDate();
  const today = new Date(); today.setHours(0,0,0,0);
  let cells = '';
  for(let i=0;i<startWeekday;i++) cells += `<div class="cal-cell empty"></div>`;
  for(let d=1; d<=daysInMonth; d++){
    const cellDate = new Date(y,m,d);
    const isPast = cellDate < today;
    const isSel = ctx.selected && cellDate.toDateString() === ctx.selected.toDateString();
    const isToday = cellDate.toDateString() === today.toDateString();
    cells += `<div class="cal-cell ${isPast?'past':''} ${isSel?'sel':''} ${isToday?'today':''}" data-d="${d}">${d}</div>`;
  }
  const cal = document.getElementById('pkCal');
  cal.innerHTML = `
    <div class="cal-head">
      <button type="button" class="cal-nav" data-nav="-1">‹</button>
      <span class="cal-title">${ctx.view.toLocaleString(undefined,{month:'long',year:'numeric'})}</span>
      <button type="button" class="cal-nav" data-nav="1">›</button>
    </div>
    <div class="cal-grid cal-dow">${['S','M','T','W','T','F','S'].map(d=>`<div>${d}</div>`).join('')}</div>
    <div class="cal-grid">${cells}</div>`;
  cal.querySelectorAll('.cal-nav').forEach(btn=>btn.onclick=()=>{
    ctx.view.setMonth(ctx.view.getMonth()+parseInt(btn.dataset.nav));
    renderCal();
  });
  cal.querySelectorAll('.cal-cell:not(.empty):not(.past)').forEach(cell=>{
    cell.onclick = ()=>{
      ctx.selected = new Date(ctx.view.getFullYear(), ctx.view.getMonth(), parseInt(cell.dataset.d));
      renderCal();
    };
  });
}

function confirmPicker(){
  const ctx = _pickerCtx;
  if(!ctx || !ctx.selected){ closePicker(); return; }
  let h12 = parseInt(document.getElementById('pkHH').value) || 12;
  let mm = parseInt(document.getElementById('pkMM').value) || 0;
  const ampm = document.querySelector('.ampm-btn.active').dataset.v;
  h12 = Math.min(Math.max(h12,1),12); mm = Math.min(Math.max(mm,0),59);
  let h24 = h12 % 12; if(ampm === 'PM') h24 += 12;
  const d = new Date(ctx.selected.getFullYear(), ctx.selected.getMonth(), ctx.selected.getDate(), h24, mm, 0, 0);

  const state = ctx.prefix === 'bulk' ? bulkState : uploadState;
  state.startIso = toUTCISO(d);
  updateStartLabel(ctx.prefix);
  closePicker();
}

function closePicker(){
  document.getElementById('schedPop')?.remove();
  document.querySelector('.sched-pop-overlay')?.remove();
  _pickerCtx = null;
}

function clearStart(prefix){
  const state = prefix === 'bulk' ? bulkState : uploadState;
  state.startIso = null;
  updateStartLabel(prefix);
}

function updateStartLabel(prefix){
  const state = prefix === 'bulk' ? bulkState : uploadState;
  const label = document.getElementById(prefix+'StartLabel');
  if(!state.startIso){
    label.textContent = '🕒 Now — first video goes out after one interval';
  }else{
    const d = new Date(state.startIso);
    label.innerHTML = `🗓️ ${fmtWhen(d)} <span style="color:var(--accent2);font-family:'IBM Plex Mono',monospace;font-size:11px;">(${fmtRelative(d)})</span> — click to change, or <a href="#" onclick="event.preventDefault();event.stopPropagation();clearStart('${prefix}');" style="color:var(--dim);text-decoration:underline;">reset to Now</a>`;
  }
  refreshPreview(prefix);
}

// ════════════════════ interval preset chips ════════════════════
const INTERVAL_PRESETS = [
  { label:'15 min', mins:15 }, { label:'30 min', mins:30 }, { label:'1 hr', mins:60 },
  { label:'4 hrs', mins:240 }, { label:'6 hrs', mins:360 }, { label:'24 hrs', mins:1440 },
];
function renderPresetChips(prefix){
  const wrap = document.getElementById(prefix+'PresetChips');
  wrap.innerHTML = INTERVAL_PRESETS.map(p=>`<span class="preset-chip" data-m="${p.mins}">${p.label}</span>`).join('');
  wrap.querySelectorAll('.preset-chip').forEach(chip=>{
    chip.onclick = ()=>{
      const mins = parseInt(chip.dataset.m);
      const numEl = document.getElementById(prefix+'IntervalNum');
      const unitEl = document.getElementById(prefix+'IntervalUnit');
      if(mins % 1440 === 0){ numEl.value = mins/1440; unitEl.value = 'days'; }
      else if(mins % 60 === 0){ numEl.value = mins/60; unitEl.value = 'hours'; }
      else { numEl.value = mins; unitEl.value = 'minutes'; }
      onIntervalChange(prefix);
    };
  });
  syncPresetActive(prefix);
}
function syncPresetActive(prefix){
  const mins = intervalMinutes(prefix);
  document.getElementById(prefix+'PresetChips').querySelectorAll('.preset-chip').forEach(c=>{
    c.classList.toggle('active', parseInt(c.dataset.m) === mins);
  });
}
function onIntervalChange(prefix){
  syncPresetActive(prefix);
  refreshPreview(prefix);
}

// ════════════════════ live schedule preview ════════════════════
function computeSlot(prefix, index, startIsoOverride){
  const state = prefix === 'bulk' ? bulkState : uploadState;
  const startIso = startIsoOverride !== undefined ? startIsoOverride : state.startIso;
  const gap = intervalMinutes(prefix);
  const base = startIso ? new Date(startIso) : new Date();
  const firstOffset = startIso ? 0 : 1;
  return new Date(base.getTime() + gap * 60000 * (index + firstOffset));
}

function refreshPreview(prefix){
  if(prefix === 'bulk'){
    const names = [...document.querySelectorAll('#exportsGrid .card')]
      .filter(c => c.querySelector('.cardChk')?.checked)
      .map(c => c.querySelector('.titleInput').value || 'Untitled');
    renderPreviewList('bulk', names);
  }else{
    renderPreviewList('upload', pendingFiles.map(pf => pf.title || pf.file.name));
  }
}
function renderPreviewList(prefix, names){
  const box = document.getElementById(prefix+'Preview');
  const list = document.getElementById(prefix+'PreviewList');
  if(!names.length){ box.classList.remove('show'); return; }
  box.classList.add('show');
  list.innerHTML = names.map((nm, i)=>{
    const d = computeSlot(prefix, i);
    return `<div class="bulk-preview-item"><span class="n">${i+1}</span><span class="nm">${escapeAttr(nm)}</span>
      <span class="when">${fmtWhen(d)}</span><span class="rel">${fmtRelative(d)}</span></div>`;
  }).join('');
}

// ════════════════════ platforms ════════════════════
function platformChipsHtml(namePrefix){
  return PLATFORMS.map(p => {
    const meta = PLATFORM_META[p] || {label:p, emoji:'📤', color:'#7c6cff'};
    return `<label class="chip" style="--chip-c:${meta.color}"><input type="checkbox" class="${namePrefix}-plat" value="${p}"> ${meta.emoji} ${meta.label}</label>`;
  }).join('');
}

async function loadPlatforms(){
  const r = await fetch('/api/publish/platforms');
  const d = await r.json();
  PLATFORMS = d.platforms || [];
  PLATFORM_META = d.platform_meta || {};
  LOGGED_IN = !!d.logged_in;
  document.getElementById('loginBanner').style.display = LOGGED_IN ? 'none' : 'block';
  document.getElementById('bulkPlatformChips').innerHTML = platformChipsHtml('bulk');
  document.getElementById('uploadPlatformChips').innerHTML = platformChipsHtml('upload');
  renderPresetChips('bulk');
  renderPresetChips('upload');
  updateStartLabel('bulk');
  updateStartLabel('upload');
}

function fmtDuration(s){
  if(!s && s !== 0) return '';
  s = Math.round(s);
  return Math.floor(s/60) + ':' + String(s%60).padStart(2,'0');
}

function cardHtml(item){
  const dur = fmtDuration(item.duration);
  return `
  <div class="card" id="card-${item.export_id}" data-export-id="${item.export_id}">
    <div class="card-top">
      <input type="checkbox" class="cardChk" onchange="onCardCheck()">
      <span class="small-dim" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeAttr(item.title)}</span>
    </div>
    <div class="video-wrap">
      <video src="${item.preview_url}" controls preload="metadata" playsinline muted></video>
      ${dur ? `<span class="dur-badge">${dur}</span>` : ''}
    </div>
    <div class="card-body">
      <label>Title</label>
      <div class="ai-row">
        <input type="text" class="titleInput" value="${escapeAttr(item.title)}" oninput="refreshPreview('bulk')">
        <button type="button" class="ai-btn" onclick="aiFillCard('${item.export_id}')">✨ AI</button>
      </div>
      <label>Description / caption</label>
      <textarea class="captionInput" placeholder="Caption for this post..."></textarea>
      <label>Post to</label>
      <div class="chip-row">${platformChipsHtml('card-'+item.export_id)}</div>
      <label>Publish time</label>
      <button type="button" class="sched-trigger" style="width:100%;justify-content:flex-start;" onclick="openCardPicker('${item.export_id}')">
        <span class="cardSchedLabel">🕒 Now (publish immediately)</span>
      </button>
      <button type="button" class="publish-btn" onclick="publishOne('${item.export_id}')">✅ Publish</button>
      <div class="prog-wrap" id="prog-${item.export_id}"></div>
      <p class="status-line" id="status-${item.export_id}"></p>
    </div>
  </div>`;
}

// per-card schedule (single publish) — reuses the same popover, writes into
// the card's own dataset instead of bulk/upload state.
function openCardPicker(exportId){
  closePicker();
  const card = document.getElementById('card-'+exportId);
  const existingIso = card.dataset.scheduledIso || null;
  const anchorDate = existingIso ? new Date(existingIso) : new Date(Date.now() + 60*60*1000);
  _pickerCtx = { prefix: 'card:'+exportId, view: new Date(anchorDate.getFullYear(), anchorDate.getMonth(), 1), selected: new Date(anchorDate) };

  const trigger = card.querySelector('.sched-trigger');
  const overlay = document.createElement('div');
  overlay.className = 'sched-pop-overlay';
  overlay.onclick = closePicker;
  document.body.appendChild(overlay);

  const rect = trigger.getBoundingClientRect();
  const pop = document.createElement('div');
  pop.className = 'sched-pop';
  pop.id = 'schedPop';
  pop.style.top = Math.min(rect.bottom + 8, window.innerHeight - 420) + 'px';
  pop.style.left = Math.min(rect.left, window.innerWidth - 312) + 'px';
  pop.onclick = (e)=>e.stopPropagation();
  pop.innerHTML = `
    <div class="sched-pop-quick" id="pkQuick"></div>
    <div id="pkCal"></div>
    <div class="time-row">
      <input type="number" min="1" max="12" id="pkHH" value="8">
      <span class="colon">:</span>
      <input type="number" min="0" max="59" id="pkMM" value="00">
      <div class="ampm-toggle">
        <button type="button" class="ampm-btn active" data-v="AM" onclick="setAmPm('AM')">AM</button>
        <button type="button" class="ampm-btn" data-v="PM" onclick="setAmPm('PM')">PM</button>
      </div>
    </div>
    <div class="sched-pop-actions">
      <button type="button" class="sched-pop-cancel" onclick="closePicker()">Cancel</button>
      <button type="button" class="sched-pop-confirm" onclick="confirmCardPicker('${exportId}')">Set schedule</button>
    </div>`;
  document.body.appendChild(pop);

  let h24 = anchorDate.getHours();
  const ampm = h24 >= 12 ? 'PM' : 'AM';
  let h12 = h24 % 12; if(h12 === 0) h12 = 12;
  document.getElementById('pkHH').value = h12;
  document.getElementById('pkMM').value = String(anchorDate.getMinutes()).padStart(2,'0');
  setAmPm(ampm);
  renderCardQuickChips(exportId);
  renderCal();
}

function renderCardQuickChips(exportId){
  const now = new Date();
  const opts = [
    { label:'Now', get:()=>null },
    { label:'+1h', get:()=>new Date(now.getTime()+60*60000) },
    { label:'+4h', get:()=>new Date(now.getTime()+4*60*60000) },
    { label:'+6h', get:()=>new Date(now.getTime()+6*60*60000) },
    { label:'Tomorrow 9AM', get:()=>{ const d=new Date(now); d.setDate(d.getDate()+1); d.setHours(9,0,0,0); return d; } },
  ];
  const wrap = document.getElementById('pkQuick');
  wrap.innerHTML = opts.map((o,i)=>`<button type="button" data-i="${i}">${o.label}</button>`).join('');
  wrap.querySelectorAll('button').forEach((btn,i)=>{
    btn.onclick = ()=>{
      const d = opts[i].get();
      if(d === null){ setCardSchedule(exportId, null); closePicker(); return; }
      _pickerCtx.selected = d;
      _pickerCtx.view = new Date(d.getFullYear(), d.getMonth(), 1);
      let h24 = d.getHours(); const ampm = h24>=12?'PM':'AM'; let h12=h24%12; if(h12===0) h12=12;
      document.getElementById('pkHH').value = h12;
      document.getElementById('pkMM').value = String(d.getMinutes()).padStart(2,'0');
      setAmPm(ampm);
      renderCal();
    };
  });
}

function confirmCardPicker(exportId){
  const ctx = _pickerCtx;
  if(!ctx || !ctx.selected){ closePicker(); return; }
  let h12 = parseInt(document.getElementById('pkHH').value) || 12;
  let mm = parseInt(document.getElementById('pkMM').value) || 0;
  const ampm = document.querySelector('.ampm-btn.active').dataset.v;
  h12 = Math.min(Math.max(h12,1),12); mm = Math.min(Math.max(mm,0),59);
  let h24 = h12 % 12; if(ampm === 'PM') h24 += 12;
  const d = new Date(ctx.selected.getFullYear(), ctx.selected.getMonth(), ctx.selected.getDate(), h24, mm, 0, 0);
  setCardSchedule(exportId, d);
  closePicker();
}

function setCardSchedule(exportId, date){
  const card = document.getElementById('card-'+exportId);
  const label = card.querySelector('.cardSchedLabel');
  if(!date){
    delete card.dataset.scheduledIso;
    label.textContent = '🕒 Now (publish immediately)';
  }else{
    card.dataset.scheduledIso = toUTCISO(date);
    label.innerHTML = `🗓️ ${fmtWhen(date)} <span style="color:var(--accent2);font-family:'IBM Plex Mono',monospace;font-size:11px;">(${fmtRelative(date)})</span>`;
  }
}

function escapeAttr(s){
  return String(s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function refreshExports(){
  try{
    const r = await fetch('/api/publish/exports');
    const d = await r.json();
    if(d.platforms) PLATFORMS = d.platforms;
    if(d.platform_meta) PLATFORM_META = d.platform_meta;
    const grid = document.getElementById('exportsGrid');
    const existingIds = new Set([...grid.querySelectorAll('.card')].map(c=>c.dataset.exportId));
    const incomingIds = new Set((d.exports||[]).map(e=>e.export_id));

    // remove cards that are no longer pending (published elsewhere / gone)
    existingIds.forEach(id => { if(!incomingIds.has(id)) document.getElementById('card-'+id)?.remove(); });

    // add new cards
    (d.exports||[]).forEach(item => {
      if(!existingIds.has(item.export_id)){
        grid.insertAdjacentHTML('beforeend', cardHtml(item));
      }
    });

    document.getElementById('exportsEmpty').style.display = (d.exports||[]).length ? 'none' : 'block';

    if(highlightExportId){
      const el = document.getElementById('card-'+highlightExportId);
      if(el){
        el.scrollIntoView({behavior:'smooth', block:'center'});
        el.classList.add('just-highlighted');
        setTimeout(()=>el.classList.remove('just-highlighted'), 3200);
      }
      highlightExportId = null;
    }
  }catch(e){ /* silent - will retry on next poll */ }
}

async function aiFillCard(exportId){
  const card = document.getElementById('card-'+exportId);
  const btn = card.querySelector('.ai-btn');
  const titleInput = card.querySelector('.titleInput');
  const captionInput = card.querySelector('.captionInput');
  btn.disabled = true; btn.textContent = '…';
  try{
    const r = await fetch('/api/publish/ai_generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({source_title: titleInput.value, context_hint: captionInput.value})
    });
    const d = await r.json();
    if(d.title) titleInput.value = d.title;
    if(d.description){
      captionInput.value = d.description + (d.hashtags && d.hashtags.length ? '\n\n' + d.hashtags.map(h=>'#'+h).join(' ') : '');
    }
    refreshPreview('bulk');
  }catch(e){}
  btn.disabled = false; btn.textContent = '✨ AI';
}

function setStatus(exportId, msg, kind){
  const el = document.getElementById('status-'+exportId);
  if(!el) return;
  el.textContent = msg;
  el.className = 'status-line' + (kind ? ' '+kind : '');
}

async function publishOne(exportId){
  const card = document.getElementById('card-'+exportId);
  const btn = card.querySelector('.publish-btn');
  const title = card.querySelector('.titleInput').value;
  const caption = card.querySelector('.captionInput').value;
  const platforms = [...card.querySelectorAll('.chip-row input:checked')].map(c=>c.value);
  const scheduledIso = card.dataset.scheduledIso || null; // already UTC ISO, or null = now
  const uploadId = genUploadId();
  btn.disabled = true;
  setStatus(exportId, '');
  initProgress(uploadId, 'prog-'+exportId, {hasSendLeg:false, title: title || 'Untitled'});
  pollServerProgress(uploadId); // starts watching immediately; shows "Waiting…" till the server registers it
  try{
    const r = await fetch('/api/publish/upload', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({export_id: exportId, title, caption, platforms, scheduled_time: scheduledIso, upload_id: uploadId})
    });
    const d = await r.json();
    stopProgress(uploadId);
    if(d.error){
      updateProgressUI(uploadId, {stage:'error', message: d.error});
      setStatus(exportId, '❌ '+d.error, 'err');
      btn.disabled = false;
      attachExportRetry(exportId, uploadId);
      return;
    }
    updateProgressUI(uploadId, {stage:'done', pct:100, detail:'Uploaded', eta:'', message:''});
    setStatus(exportId, '✅ Queued for ' + fmtWhen(new Date(d.scheduled_time)) + ' (' + fmtRelative(new Date(d.scheduled_time)) + ')', 'ok');
    setTimeout(()=>{ card.remove(); trayRemove(uploadId); }, 1400);
  }catch(e){
    stopProgress(uploadId);
    updateProgressUI(uploadId, {stage:'error', message: e.message});
    setStatus(exportId, '❌ ' + e.message, 'err');
    btn.disabled = false;
    attachExportRetry(exportId, uploadId);
  }
}

function onCardCheck(){
  const chks = [...document.querySelectorAll('.cardChk')];
  const n = chks.filter(c=>c.checked).length;
  document.getElementById('selCount').textContent = n + ' selected';
  document.querySelectorAll('.card').forEach(c=>{
    const chk = c.querySelector('.cardChk');
    c.classList.toggle('selected', chk && chk.checked);
  });
  refreshPreview('bulk');
}

function toggleSelectAll(box){
  document.querySelectorAll('.cardChk').forEach(c=>{ c.checked = box.checked; });
  onCardCheck();
}

async function publishSelected(){
  const selectedCards = [...document.querySelectorAll('.card')].filter(c => c.querySelector('.cardChk')?.checked);
  if(!selectedCards.length){ alert('Tick at least one card first.'); return; }
  const gapMinutes = intervalMinutes('bulk');
  const platforms = [...document.querySelectorAll('#bulkPlatformChips input:checked')].map(c=>c.value);

  const items = selectedCards.map(c => {
    const exportId = c.dataset.exportId;
    const uploadId = genUploadId();
    const itemTitle = c.querySelector('.titleInput').value;
    setStatus(exportId, '');
    initProgress(uploadId, 'prog-'+exportId, {hasSendLeg:false, title: itemTitle || 'Untitled'});
    // items further back in the batch will genuinely sit at "Waiting…"
    // (0%) until the server actually gets to them — that's real, not fake.
    // IMPORTANT: retry is attached HERE, the moment THIS item's own poll
    // reports "error" — not after the whole bulk request finishes. With 4
    // items where #2 fails, its retry button appears right away while #3
    // and #4 are still uploading, instead of everyone waiting for the
    // batch to fully end.
    pollServerProgress(uploadId, (d)=>{
      if(d.stage === 'done'){
        setStatus(exportId, '✅ Uploaded — queued', 'ok');
        setTimeout(()=>{ document.getElementById('card-'+exportId)?.remove(); trayRemove(uploadId); }, 1400);
      }else if(d.stage === 'error'){
        setStatus(exportId, '❌ ' + (d.message || 'Failed'), 'err');
        attachExportRetry(exportId, uploadId);
      }
    });
    return {
      export_id: exportId,
      title: itemTitle,
      caption: c.querySelector('.captionInput').value,
      upload_id: uploadId,
    };
  });

  const btn = document.getElementById('bulkPublishBtn');
  btn.disabled = true; btn.textContent = 'Publishing…';
  try{
    const r = await fetch('/api/publish/upload_bulk_exports', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({items, platforms, gap_minutes: gapMinutes, scheduled_time: bulkState.startIso})
    });
    const d = await r.json();
    // safety net only — normally every failed item's retry button already
    // appeared live via the poll callback above, well before we get here
    (d.errors||[]).forEach(e => attachExportRetry(e.export_id, e.upload_id));
    if(d.errors && d.errors.length) console.log(d.errors);
  }catch(e){
    items.forEach(it => {
      stopProgress(it.upload_id);
      updateProgressUI(it.upload_id, {stage:'error', message: e.message});
      setStatus(it.export_id, '❌ ' + e.message, 'err');
    });
  }
  btn.disabled = false; btn.textContent = '🚀 Publish Selected';
  onCardCheck();
}

// ── New Uploads tab ─────────────────────────────────────────────────────
const dz = document.getElementById('dropzone');
['dragover','dragenter'].forEach(evt => dz.addEventListener(evt, e=>{ e.preventDefault(); dz.classList.add('drag'); }));
['dragleave','drop'].forEach(evt => dz.addEventListener(evt, e=>{ e.preventDefault(); dz.classList.remove('drag'); }));
dz.addEventListener('drop', e => { if(e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); });

function handleFiles(fileList){
  [...fileList].forEach(file => pendingFiles.push({file, title: file.name, caption: '', uid: genUploadId()}));
  renderFileRows();
}

function renderFileRows(){
  const wrap = document.getElementById('fileRows');
  wrap.innerHTML = pendingFiles.map((pf, i) => `
    <div class="file-row" id="filerow-${pf.uid}">
      <div class="fname">📹 ${escapeAttr(pf.file.name)}<br><span class="small-dim">${fmtBytes(pf.file.size)}</span></div>
      <div class="ai-row">
        <input type="text" value="${escapeAttr(pf.title)}" oninput="pendingFiles[${i}].title=this.value; refreshPreview('upload');">
        <button type="button" class="ai-btn" onclick="aiFillUploadRow(${i})">✨ AI</button>
      </div>
      <div>
        <textarea placeholder="Caption..." oninput="pendingFiles[${i}].caption=this.value">${escapeAttr(pf.caption)}</textarea>
        <div class="prog-wrap" id="prog-file-${pf.uid}"></div>
      </div>
    </div>
  `).join('');
  document.getElementById('uploadsBar').style.display = pendingFiles.length ? 'flex' : 'none';
  refreshPreview('upload');
}

async function aiFillUploadRow(i){
  const pf = pendingFiles[i];
  const d = await (await fetch('/api/publish/ai_generate', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({source_title: pf.title, context_hint: pf.caption})
  })).json();
  if(d.title) pf.title = d.title;
  if(d.description) pf.caption = d.description + (d.hashtags && d.hashtags.length ? '\n\n' + d.hashtags.map(h=>'#'+h).join(' ') : '');
  renderFileRows();
}

// Uploads ONE file with real, live progress:
//   1) XHR upload.onprogress -> real browser->server byte progress
//   2) once the browser is done sending, switch to polling the server's
//      real storage-upload progress (see pollServerProgress above)
// Resolves {ok:true/false, pf}. Never throws.
function uploadSingleFile(pf, index){
  return new Promise((resolve)=>{
    initProgress(pf.uid, 'prog-file-'+pf.uid, {hasSendLeg:true, title: pf.title || pf.file.name});
    const xhr = new XMLHttpRequest();
    trackXhrSend(xhr, pf.uid);

    const fd = new FormData();
    fd.append('video_file', pf.file);
    fd.append('upload_id', pf.uid);
    fd.append('title', pf.title || pf.file.name);
    fd.append('caption', pf.caption || '');
    [...document.querySelectorAll('#uploadPlatformChips input:checked')].forEach(c => fd.append('platforms', c.value));
    fd.append('scheduled_time', computeSlot('upload', index).toISOString());

    const finishNeedsResend = (msg)=>{
      // browser never finished delivering the file to the server at all —
      // there's nothing server-side to resume yet, has to be sent again
      updateProgressUI(pf.uid, {stage:'error', message: msg});
      showRetryBtn('prog-file-'+pf.uid, ()=>uploadSingleFile(pf, index));
      resolve({ok:false, pf});
    };
    const finishCanResume = (msg)=>{
      // the file DID make it to the server (we got an HTTP response back);
      // only the storage leg failed — resume from there, don't resend
      updateProgressUI(pf.uid, {stage:'error', message: msg});
      showRetryBtn('prog-file-'+pf.uid, ()=>resumeUpload(pf.uid, 'prog-file-'+pf.uid, (dd)=>{
        if(dd.stage === 'done'){
          updateProgressUI(pf.uid, {stage:'done', pct:100, detail:'Uploaded', eta:'', message:''});
          setTimeout(()=>trayRemove(pf.uid), 1400);
        }
      }, ()=>uploadSingleFile(pf, index)));
      resolve({ok:false, pf});
    };

    xhr.open('POST', '/api/publish/upload_file');
    xhr.onerror = ()=> finishNeedsResend('Network error — check your internet connection.');
    xhr.ontimeout = ()=> finishNeedsResend('Upload timed out.');
    xhr.onload = ()=>{
      let d = {};
      try{ d = JSON.parse(xhr.responseText); }catch(e){}
      if(xhr.status >= 200 && xhr.status < 300 && !d.error){
        // browser is done sending — now watch the real server-side
        // "uploading to storage" + "saving" legs to completion
        pollServerProgress(pf.uid, (dd)=>{
          if(dd.stage === 'done'){
            updateProgressUI(pf.uid, {stage:'done', pct:100, detail:'Uploaded', eta:'', message:''});
            setTimeout(()=>trayRemove(pf.uid), 1400);
            resolve({ok:true, pf});
          }else{
            finishCanResume(dd.message || 'Upload failed on the server.');
          }
        });
      }else{
        // we got a real HTTP response — the file reached the server, it
        // just failed after that, so resume rather than resend
        finishCanResume(d.error || ('Server error (' + xhr.status + ')'));
      }
    };
    xhr.send(fd);
  });
}

async function queueAllUploads(){
  if(!pendingFiles.length) return;
  const btn = document.getElementById('queueAllBtn');
  btn.disabled = true; btn.textContent = 'Uploading…';
  const files = pendingFiles.slice();
  const results = new Array(files.length).fill(null);
  const CONCURRENCY = 3; // a few at once so bulk feels fast, without hammering the connection
  const statusEl = document.getElementById('uploadsStatus');
  const tick = ()=>{
    const doneCount = results.filter(r=>r).length;
    const ok = results.filter(r=>r && r.ok).length;
    const fail = results.filter(r=>r && !r.ok).length;
    statusEl.textContent = `Uploading ${doneCount}/${files.length} (✅ ${ok}  ❌ ${fail})…`;
    statusEl.className = 'status-line';
  };
  tick();
  let next = 0;
  async function worker(){
    while(next < files.length){
      const i = next++;
      results[i] = await uploadSingleFile(files[i], i);
      tick();
    }
  }
  await Promise.all(Array.from({length: Math.min(CONCURRENCY, files.length)}, worker));

  const ok = results.filter(r=>r.ok).length;
  const fail = results.filter(r=>!r.ok).length;
  statusEl.textContent = `✅ Uploaded ${ok}/${files.length}.` + (fail ? ` ❌ ${fail} failed — hit Retry on the red bar(s) above.` : '');
  statusEl.className = 'status-line ' + (fail ? 'err' : 'ok');

  // keep failed files (with their error + retry button) so nothing is
  // silently lost; drop the ones that succeeded
  pendingFiles = files.filter((pf,i)=> !results[i].ok);
  renderFileRows();
  pendingFiles.forEach((pf,i)=>{
    initProgress(pf.uid, 'prog-file-'+pf.uid, {hasSendLeg:true, title: pf.title || pf.file.name});
    updateProgressUI(pf.uid, {stage:'error', pct:0, detail:'—', eta:'—', message:'Upload failed — click Retry to try again.'});
    showRetryBtn('prog-file-'+pf.uid, ()=>resumeUpload(pf.uid, 'prog-file-'+pf.uid, (dd)=>{
      if(dd.stage === 'done'){
        trayRemove(pf.uid);
        pendingFiles = pendingFiles.filter(x=>x.uid !== pf.uid);
        renderFileRows();
      }
    }, ()=>uploadSingleFile(pf, i).then(res=>{
      if(res.ok){
        trayRemove(pf.uid);
        pendingFiles = pendingFiles.filter(x=>x.uid !== pf.uid);
        renderFileRows();
      }
    })));
  });
  document.getElementById('fileInput').value = '';
  btn.disabled = false; btn.textContent = '🚀 Queue All';
}

// ── Listen for "focus this clip" messages from the parent Studio page ───
window.addEventListener('message', (ev) => {
  const msg = ev.data || {};
  if(msg.type === 'publish_focus_export' && msg.export_id){
    highlightExportId = msg.export_id;
    showView('exports');
    refreshExports();
  }
});

loadPlatforms();
refreshExports();
setInterval(refreshExports, 4000);
setInterval(()=>{ refreshPreview('bulk'); refreshPreview('upload'); }, 30000); // keep "in Xh Ym" labels honest
</script>
</body>
</html>
"""


@publish_bp.route("/api/publish/publish")
def publish_page():
    return Response(PUBLISH_PAGE, mimetype="text/html")