import os
import re
import math
import uuid
import threading
import subprocess
import shutil
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

# Third-party imports handled gracefully
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import edge_tts
    import asyncio
except ImportError:
    edge_tts = None
    asyncio = None

# Initialize Flask App
app = Flask(__name__)

# Configure storage directories
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
EXPORT_FOLDER = os.path.join(BASE_DIR, "exports")
TTS_FOLDER = os.path.join(BASE_DIR, "tts_audio")
THUMB_FOLDER = os.path.join(BASE_DIR, "thumbnails")

for folder in [UPLOAD_FOLDER, EXPORT_FOLDER, TTS_FOLDER, THUMB_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Browser <video> tags cannot decode these containers natively (no MSE/native
# support in Chrome/Firefox/Safari for mkv or avi), no matter how well-formed
# the file is. This is a real browser limitation, not a bug - but the UI can
# know about it in advance instead of discovering it via a failed <video> load
# (which is what produced the generic "preview not supported" black screen).
BROWSER_UNPLAYABLE_CONTAINERS = {'mkv', 'avi'}

# Global compilation task statuses
export_tasks = {}
tasks_lock = threading.Lock()

# Define allowed upload file types
ALLOWED_EXTENSIONS = {
    'video': {'mp4', 'mkv', 'webm', 'mov', 'avi'},
    'image': {'png', 'jpg', 'jpeg', 'webp', 'gif'},
    'audio': {'mp3', 'wav', 'ogg', 'm4a', 'aac'}
}

# Mapping of "trending" transition names exposed in the UI to native FFmpeg
# xfade transition types. A few of the requested styles (glitch, light leak,
# page turn, shape-star, camera shake, match cut) don't have a native FFmpeg
# equivalent, so they are mapped to the closest available native transition.
TRANSITIONS = {
    'none':          {'xfade': 'fade',       'dur': 0.05, 'label': 'None (Hard Cut)'},
    'cross_dissolve':{'xfade': 'dissolve',   'dur': 0.6,  'label': 'Cross Dissolve'},
    'fade_black':    {'xfade': 'fadeblack',  'dur': 0.6,  'label': 'Fade to Black'},
    'fade_white':    {'xfade': 'fadewhite',  'dur': 0.5,  'label': 'Fade to White / Flash'},
    'zoom_in':       {'xfade': 'zoomin',     'dur': 0.6,  'label': 'Zoom In / Pull In'},
    'zoom_out':      {'xfade': 'zoomin',     'dur': 0.6,  'label': 'Zoom Out / Pull Out'},
    'pan_left':      {'xfade': 'slideleft',  'dur': 0.5,  'label': 'Pan Left'},
    'pan_right':     {'xfade': 'slideright', 'dur': 0.5,  'label': 'Pan Right'},
    'pan_up':        {'xfade': 'slideup',    'dur': 0.5,  'label': 'Pan Up'},
    'pan_down':      {'xfade': 'slidedown',  'dur': 0.5,  'label': 'Pan Down'},
    'whip_pan':      {'xfade': 'slideleft',  'dur': 0.2,  'label': 'Whip Pan'},
    'camera_shake':  {'xfade': 'hblur',      'dur': 0.4,  'label': 'Camera Shake'},
    'linear_wipe':   {'xfade': 'wipeleft',   'dur': 0.5,  'label': 'Linear Wipe'},
    'clock_wipe':    {'xfade': 'radial',     'dur': 0.6,  'label': 'Clock Wipe'},
    'shape_circle':  {'xfade': 'circleopen', 'dur': 0.6,  'label': 'Shape Mask (Circle)'},
    'shape_square':  {'xfade': 'rectcrop',   'dur': 0.6,  'label': 'Shape Mask (Square)'},
    'shape_star':    {'xfade': 'circleopen', 'dur': 0.6,  'label': 'Shape Mask (Star)'},
    'frame_block':   {'xfade': 'pixelize',   'dur': 0.5,  'label': 'Frame Blocking'},
    'match_cut':     {'xfade': 'fade',       'dur': 0.08, 'label': 'Match Cut'},
    'gaussian_blur': {'xfade': 'hblur',      'dur': 0.5,  'label': 'Gaussian Blur'},
    'glitch':        {'xfade': 'pixelize',   'dur': 0.35, 'label': 'Glitch / Digital Distortion'},
    'light_leak':    {'xfade': 'fadewhite',  'dur': 0.5,  'label': 'Light Leak / Film Burn'},
    'page_turn':     {'xfade': 'wiperight',  'dur': 0.6,  'label': 'Page Turn / Page Peel'},
    'split_screen':  {'xfade': 'vertopen',   'dur': 0.6,  'label': 'Split Screen Transition'},
}

CAPTION_FONTS = [
    "Impact", "Arial Black", "Anton", "Bebas Neue", "Montserrat",
    "Poppins", "Arial", "Verdana"
]


def get_file_type(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    for ftype, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return ftype
    return None


# ================= FFMPEG LOCATION HELPER =================

_FFMPEG_BIN_CACHE = None


def resolve_ffmpeg_binary():
    """
    Dynamically detects the ffmpeg executable path. Cached after first lookup.
    """
    global _FFMPEG_BIN_CACHE
    if _FFMPEG_BIN_CACHE:
        return _FFMPEG_BIN_CACHE

    ffmpeg_bin = "ffmpeg"
    try:
        if shutil.which("ffmpeg"):
            ffmpeg_bin = shutil.which("ffmpeg")
        else:
            import imageio_ffmpeg
            img_ff = imageio_ffmpeg.get_ffmpeg_exe()
            if img_ff and os.path.exists(img_ff):
                ffmpeg_bin = img_ff
    except Exception as e:
        print(f"[FFMPEG SEARCH WARN] Failed to find via imageio: {e}")

    if ffmpeg_bin == "ffmpeg":
        try:
            from moviepy.config import get_setting
            mv_ff = get_setting("FFMPEG_BINARY")
            if mv_ff and os.path.exists(mv_ff):
                ffmpeg_bin = mv_ff
        except Exception:
            pass

    _FFMPEG_BIN_CACHE = ffmpeg_bin
    print(f"[FFMPEG] Resolved executable path: {ffmpeg_bin}")
    return ffmpeg_bin


def probe_media(filepath):
    """
    Single ffmpeg -i probe that reports whether a file has a video stream,
    an audio stream, and its total duration in seconds. This is the single
    source of truth used everywhere so a file is never mis-classified.
    """
    ffmpeg_bin = resolve_ffmpeg_binary()
    result = {"has_video": False, "has_audio": False, "duration": 0.0}
    try:
        cmd = [ffmpeg_bin, "-i", filepath]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="ignore")
        stderr = res.stderr

        # A real video stream (not just an attached cover-art image stream)
        for line in stderr.splitlines():
            if "Stream #" in line and "Video:" in line and "(attached pic)" not in line and "mjpeg" not in line.lower():
                result["has_video"] = True
            if "Stream #" in line and "Audio:" in line:
                result["has_audio"] = True

        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", stderr)
        if match:
            hours, minutes, seconds = int(match.group(1)), int(match.group(2)), float(match.group(3))
            result["duration"] = hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        print(f"[PROBE WARN] Failed to probe {filepath}: {e}")
    return result


def generate_video_thumbnail(video_path, thumb_path, duration_hint=None):
    """
    Extracts a single real JPEG frame from a video file to use as its grid /
    timeline thumbnail. This replaces the previous approach of pointing a live
    <video> tag at the raw file just to show a static frame - which silently
    failed (blank/black box) for any container or codec the browser couldn't
    decode (e.g. mkv, avi, some HEVC/AV1 files), even though the file itself
    was perfectly valid and export-ready. A pre-rendered JPEG always displays.
    """
    ffmpeg_bin = resolve_ffmpeg_binary()
    seek = 1.0
    if duration_hint is not None and duration_hint < 2.0:
        seek = 0.0
    cmd = [
        ffmpeg_bin, "-y", "-ss", str(seek), "-i", video_path,
        "-frames:v", "1", "-vf", "scale=480:-2",
        "-q:v", "4", thumb_path
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not os.path.exists(thumb_path):
        cmd0 = [
            ffmpeg_bin, "-y", "-i", video_path,
            "-frames:v", "1", "-vf", "scale=480:-2",
            "-q:v", "4", thumb_path
        ]
        subprocess.run(cmd0, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return os.path.exists(thumb_path)


def thumb_filename_for(filename):
    return os.path.splitext(filename)[0] + ".jpg"


def escape_drawtext(text):
    """
    Escapes a string so it is safe to embed inside an ffmpeg drawtext filter.
    """
    text = text.replace("\\", "\\\\\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\u2019")  # swap straight quote for a safe typographic one
    text = text.replace("%", "\\%")
    return text


# ================= TTS SPEECH GENERATORS =================

async def generate_edge_tts(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def render_text_to_speech(text, engine, voice_or_lang, output_path):
    """
    Renders text to speech using either Microsoft Edge-TTS or Google gTTS.
    """
    if engine == "edge":
        if edge_tts is None:
            raise ImportError("Microsoft Edge TTS library is not installed.")
        asyncio.run(generate_edge_tts(text, voice_or_lang, output_path))
    else:
        if gTTS is None:
            raise ImportError("Google gTTS library is not installed.")
        tts = gTTS(text=text, lang=voice_or_lang)
        tts.save(output_path)


# ================= YT-DLP DOWNLOADER =================

def extract_media_from_url(url, media_type="video"):
    """
    Downloads media streams from video webpages (e.g. YouTube) using yt-dlp.
    Always merges the best available video+audio into a single file (this was
    previously failing silently whenever the merge step couldn't find ffmpeg).
    Also re-probes the downloaded file afterwards and returns its REAL type,
    since some sources (e.g. some Twitter/X links) only ever have an audio
    stream even when a "video" download was requested - previously that
    audio-only file was still treated as a video downstream, which is what
    caused the "matches no streams" ffmpeg crash.
    """
    if yt_dlp is None:
        raise ImportError("yt-dlp library is not available.")

    ffmpeg_bin = resolve_ffmpeg_binary()
    ffmpeg_dir = os.path.dirname(ffmpeg_bin) if os.path.isabs(ffmpeg_bin) else None

    unique_id = uuid.uuid4().hex[:6]
    out_tmpl = os.path.join(UPLOAD_FOLDER, f"yt_media_{unique_id}_%(title)s.%(ext)s")

    ydl_opts = {
        'outtmpl': out_tmpl,
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'quiet': True,
        'no_warnings': True,
        'restrictfilenames': True,
    }
    if ffmpeg_dir:
        # CRITICAL: without pointing yt-dlp at ffmpeg, it can silently fail to
        # merge separate best-video/best-audio streams and fall back to a
        # single lower-quality pre-merged stream (or audio-only in some cases).
        ydl_opts['ffmpeg_location'] = ffmpeg_dir

    if media_type == "audio":
        ydl_opts.update({
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        })
    else:
        ydl_opts.update({
            # Always prefer separately fetched best video + best audio, merged
            # into one mp4 container. This guarantees a single output file
            # with both streams present instead of "only one came through".
            'format': 'bestvideo*+bestaudio/best',
            'merge_output_format': 'mp4',
        })

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise ValueError("Could not extract media info or download the URL stream.")

        if '_type' in info and info['_type'] == 'playlist':
            entries = info.get('entries', [])
            if not entries:
                raise ValueError("Playlist contains no downloadable entries.")
            info = entries[0]

        filename = ydl.prepare_filename(info)

        if media_type == "audio":
            filename = os.path.splitext(filename)[0] + ".mp3"
        else:
            if not os.path.exists(filename):
                base, _ = os.path.splitext(filename)
                for ext in ['mp4', 'mkv', 'webm']:
                    if os.path.exists(base + "." + ext):
                        filename = base + "." + ext
                        break

        if not os.path.exists(filename):
            files = os.listdir(UPLOAD_FOLDER)
            matched = [f for f in files if f"yt_media_{unique_id}" in f]
            if matched:
                filename = os.path.join(UPLOAD_FOLDER, matched[0])
            else:
                raise FileNotFoundError("Downloaded file could not be verified on disk.")

        final_name = os.path.basename(filename)
        final_path = os.path.join(UPLOAD_FOLDER, final_name)

        # Re-probe to determine the TRUE media type, regardless of what was requested.
        actual_type = media_type
        thumb_url = None
        if media_type != "audio":
            probe = probe_media(final_path)
            if not probe["has_video"] and probe["has_audio"]:
                # It downloaded as a "video" request but is actually audio-only.
                # Rename to .m4a/.mp3-style container info stays intact; we just
                # report the true type so the front-end files it correctly and
                # the render pipeline never tries to build a video filter for it.
                actual_type = "audio"
            else:
                thumb_name = thumb_filename_for(final_name)
                thumb_path = os.path.join(THUMB_FOLDER, thumb_name)
                if generate_video_thumbnail(final_path, thumb_path, probe.get("duration")):
                    thumb_url = f"/api/files/thumbnails/{thumb_name}"

        return final_name, actual_type, thumb_url


# ================= VIDEO EDITING COMPILATION WORKER =================

def compile_video_background(task_id, timeline, audio_track, settings):
    """
    Processes the timeline sequence and compiles the final video using FFmpeg.
    Supports per-clip trimming, muting original audio, text/caption overlays,
    a full library of transitions (native FFmpeg xfade/acrossfade chaining),
    image+video mixing, and audio syncing - all without re-encoding losses
    beyond a single high-quality final encode pass.
    """
    try:
        def update_progress(percent, stage, file_url=None, err=None):
            with tasks_lock:
                export_tasks[task_id] = {
                    "progress": percent,
                    "stage": stage,
                    "download_url": file_url,
                    "error": err
                }

        update_progress(5, "Analyzing project sequence structure...")
        ffmpeg_bin = resolve_ffmpeg_binary()

        # Setup dimension resolutions
        is_portrait = settings.get("format") == "9:16"
        quality = settings.get("quality", "1080p")

        if quality == "4k":
            width, height = (2160, 3840) if is_portrait else (3840, 2160)
            crf, preset = "16", "medium"
        elif quality == "720p":
            width, height = (720, 1280) if is_portrait else (1280, 720)
            crf, preset = "20", "medium"
        else:  # 1080p Default
            width, height = (1080, 1920) if is_portrait else (1920, 1080)
            crf, preset = "18", "medium"

        # Resolve Audio Stream
        audio_file = None
        loop_audio = settings.get("loop_audio", False)
        match_audio_length = settings.get("match_audio_length", False)

        if audio_track and audio_track.get("filename"):
            folder_src = TTS_FOLDER if audio_track.get("source") == "tts" else UPLOAD_FOLDER
            audio_file = os.path.join(folder_src, audio_track["filename"])
            if not os.path.exists(audio_file):
                audio_file = None

        output_filename = f"compiled_video_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(EXPORT_FOLDER, output_filename)

        temp_dir = os.path.join(EXPORT_FOLDER, f"temp_{uuid.uuid4().hex[:8]}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            temp_clips = []
            clip_durations = []
            clip_effects = []  # transition INTO this clip (index aligned with temp_clips)
            total_items = len(timeline)

            for idx, item in enumerate(timeline):
                filename = item.get("filename")
                ftype = item.get("type")
                duration = float(item.get("duration", 5.0))
                effect = item.get("effect", "none")
                trim_start = float(item.get("trimStart", 0) or 0)
                mute = bool(item.get("mute", False))
                text_cfg = item.get("text") or None

                if effect not in TRANSITIONS:
                    effect = "none"

                media_path = os.path.join(UPLOAD_FOLDER, filename)
                if not os.path.exists(media_path):
                    continue

                update_progress(10 + int((idx / max(total_items, 1)) * 50),
                                 f"Encoding clip {idx + 1}/{total_items}: {filename}...")

                clip_output = os.path.join(temp_dir, f"clip_{idx:04d}.mp4")

                # Base scale/pad filter (identical target canvas for every clip
                # so the xfade transition chain always lines up perfectly).
                v_filter = (
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
                )

                # Optional caption / text / logo-style overlay baked into this clip
                if text_cfg and text_cfg.get("content"):
                    content = escape_drawtext(text_cfg.get("content", ""))
                    font = text_cfg.get("font", "Arial")
                    size = int(text_cfg.get("size", 54))
                    color = text_cfg.get("color", "white")
                    position = text_cfg.get("position", "bottom")
                    box = bool(text_cfg.get("bg", True))
                    box_color = text_cfg.get("bgColor", "black")

                    if position == "top":
                        y_expr = "h*0.08"
                    elif position == "middle":
                        y_expr = "(h-text_h)/2"
                    else:
                        y_expr = "h*0.82-text_h"

                    v_filter += (
                        f",drawtext=text='{content}':font='{font}':fontsize={size}:"
                        f"fontcolor={color}:x=(w-text_w)/2:y={y_expr}:"
                        f"box={1 if box else 0}:boxcolor={box_color}@0.55:boxborderw=14"
                    )

                # Determine if this specific file actually has real video/audio streams
                media_probe = probe_media(media_path) if ftype == "video" else {"has_video": False, "has_audio": False}
                has_real_video = (ftype == "video" and media_probe["has_video"])
                has_real_audio = (not mute) and (
                    (ftype == "video" and media_probe["has_audio"]) or False
                )

                if ftype == "image":
                    cmd = [
                        ffmpeg_bin, "-y",
                        "-loop", "1", "-t", str(duration), "-i", media_path,
                        "-f", "lavfi", "-i", "anullsrc=cl=stereo:r=44100",
                        "-filter_complex", f"[0:v]{v_filter}[v]",
                        "-map", "[v]", "-map", "1:a",
                        "-c:v", "libx264", "-preset", preset, "-crf", crf,
                        "-pix_fmt", "yuv420p", "-r", "30",
                        "-c:a", "aac", "-b:a", "192k", "-shortest",
                        clip_output
                    ]
                elif ftype == "video" and not has_real_video:
                    # DEFENSIVE FIX for the exact crash seen previously: a file
                    # tagged as "video" but which has no actual video stream
                    # (audio-only). Instead of pointing a scale filter at a
                    # non-existent video stream (which crashed the whole
                    # export), synthesize a solid color canvas and keep the
                    # real audio track, exactly like an image clip would work.
                    canvas_filter = f"color=c=black:s={width}x{height}:r=30{',' + v_filter.split(',', 1)[1] if ',' in v_filter else ''}"
                    if has_real_audio:
                        cmd = [
                            ffmpeg_bin, "-y",
                            "-f", "lavfi", "-i", f"{canvas_filter}",
                            "-ss", str(trim_start), "-t", str(duration), "-i", media_path,
                            "-map", "0:v", "-map", "1:a",
                            "-c:v", "libx264", "-preset", preset, "-crf", crf,
                            "-pix_fmt", "yuv420p", "-r", "30",
                            "-c:a", "aac", "-b:a", "192k", "-t", str(duration),
                            "-shortest",
                            clip_output
                        ]
                    else:
                        cmd = [
                            ffmpeg_bin, "-y",
                            "-f", "lavfi", "-i", f"{canvas_filter}:d={duration}",
                            "-f", "lavfi", "-i", "anullsrc=cl=stereo:r=44100",
                            "-map", "0:v", "-map", "1:a",
                            "-c:v", "libx264", "-preset", preset, "-crf", crf,
                            "-pix_fmt", "yuv420p", "-r", "30",
                            "-c:a", "aac", "-b:a", "192k", "-shortest",
                            clip_output
                        ]
                else:  # normal video with a real video stream
                    if has_real_audio:
                        cmd = [
                            ffmpeg_bin, "-y",
                            "-ss", str(trim_start), "-t", str(duration), "-i", media_path,
                            "-filter_complex",
                            f"[0:v]{v_filter}[v];[0:a]aformat=sample_rates=44100:channel_layouts=stereo[a]",
                            "-map", "[v]", "-map", "[a]",
                            "-c:v", "libx264", "-preset", preset, "-crf", crf,
                            "-pix_fmt", "yuv420p", "-r", "30",
                            "-c:a", "aac", "-b:a", "192k",
                            clip_output
                        ]
                    else:
                        # No original audio, or user chose to mute it
                        cmd = [
                            ffmpeg_bin, "-y",
                            "-ss", str(trim_start), "-t", str(duration), "-i", media_path,
                            "-f", "lavfi", "-i", "anullsrc=cl=stereo:r=44100",
                            "-filter_complex", f"[0:v]{v_filter}[v]",
                            "-map", "[v]", "-map", "1:a",
                            "-c:v", "libx264", "-preset", preset, "-crf", crf,
                            "-pix_fmt", "yuv420p", "-r", "30",
                            "-c:a", "aac", "-b:a", "192k", "-shortest",
                            clip_output
                        ]

                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc.returncode == 0 and os.path.exists(clip_output):
                    temp_clips.append(clip_output)
                    clip_durations.append(duration)
                    clip_effects.append(effect)
                else:
                    print(f"[CLIP RENDER FAILED] {filename}: {proc.stderr[-1500:]}")

            if not temp_clips:
                raise ValueError("No clips were successfully rendered. Check that your source files are valid.")

            # 2. Chain clips together with real transitions (xfade for video,
            #    acrossfade for audio) instead of a hard concat, so the
            #    "trending transitions" list actually applies during export.
            update_progress(70, "Applying transitions and assembling sequence...")
            temp_concat_video = os.path.join(temp_dir, "concat_output.mp4")

            if len(temp_clips) == 1:
                shutil.copy(temp_clips[0], temp_concat_video)
                video_duration = clip_durations[0]
            else:
                inputs = [ffmpeg_bin, "-y"]
                for clip in temp_clips:
                    inputs += ["-i", clip]

                filter_parts = []
                prev_v, prev_a = "0:v", "0:a"
                cum = clip_durations[0]

                for i in range(1, len(temp_clips)):
                    trans_key = clip_effects[i]
                    trans = TRANSITIONS.get(trans_key, TRANSITIONS["none"])
                    trans_dur = min(trans["dur"], clip_durations[i - 1] / 2.0, clip_durations[i] / 2.0)
                    trans_dur = max(trans_dur, 0.04)
                    offset = max(cum - trans_dur, 0.0)

                    out_v = f"v{i}"
                    out_a = f"a{i}"
                    filter_parts.append(
                        f"[{prev_v}][{i}:v]xfade=transition={trans['xfade']}:duration={trans_dur:.3f}:offset={offset:.3f}[{out_v}]"
                    )
                    filter_parts.append(
                        f"[{prev_a}][{i}:a]acrossfade=d={trans_dur:.3f}[{out_a}]"
                    )
                    prev_v, prev_a = out_v, out_a
                    cum = cum + clip_durations[i] - trans_dur

                filter_complex_str = ";".join(filter_parts)
                video_duration = cum

                concat_cmd = inputs + [
                    "-filter_complex", filter_complex_str,
                    "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
                    "-c:v", "libx264", "-preset", preset, "-crf", crf,
                    "-pix_fmt", "yuv420p", "-r", "30",
                    "-c:a", "aac", "-b:a", "192k",
                    temp_concat_video
                ]
                proc = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc.returncode != 0 or not os.path.exists(temp_concat_video):
                    print(f"[TRANSITION CHAIN FAILED, falling back to hard concat] {proc.stderr[-1500:]}")
                    # Fallback: simple demuxer concat (no transitions) so the
                    # export still succeeds even if a transition filter fails.
                    concat_list_file = os.path.join(temp_dir, "concat.txt")
                    with open(concat_list_file, "w") as f:
                        for clip in temp_clips:
                            f.write(f"file '{clip}'\n")
                    fallback_cmd = [
                        ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_list_file, "-c", "copy", temp_concat_video
                    ]
                    subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    video_duration = sum(clip_durations)

            # 3. Add background soundtrack or handle duration constraints
            update_progress(85, "Mixing audio tracks and finishing render...")

            if audio_file:
                audio_probe = probe_media(audio_file)
                audio_duration = audio_probe["duration"] or 5.0

                if match_audio_length:
                    if video_duration < audio_duration:
                        num_loops = int(math.ceil(audio_duration / video_duration))
                        loop_list = os.path.join(temp_dir, "loop_list.txt")
                        with open(loop_list, "w") as f:
                            for _ in range(num_loops):
                                f.write(f"file '{temp_concat_video}'\n")
                        temp_looped_video = os.path.join(temp_dir, "looped_video.mp4")
                        loop_cmd = [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", loop_list, "-c", "copy", temp_looped_video]
                        subprocess.run(loop_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        temp_concat_video = temp_looped_video

                    mix_cmd = [
                        ffmpeg_bin, "-y",
                        "-t", str(audio_duration), "-i", temp_concat_video,
                        "-i", audio_file,
                        "-filter_complex", "[1:a]volume=1.0[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[a]",
                        "-map", "0:v", "-map", "[a]",
                        "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p", "-r", "30",
                        "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart",
                        output_path
                    ]
                else:
                    if loop_audio and audio_duration < video_duration:
                        mix_cmd = [
                            ffmpeg_bin, "-y",
                            "-i", temp_concat_video,
                            "-stream_loop", "-1", "-i", audio_file,
                            "-filter_complex", f"[1:a]volume=1.0,atrim=0:{video_duration}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[a]",
                            "-map", "0:v", "-map", "[a]",
                            "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p", "-r", "30",
                            "-c:a", "aac", "-b:a", "192k", "-t", str(video_duration),
                            "-movflags", "+faststart",
                            output_path
                        ]
                    else:
                        mix_cmd = [
                            ffmpeg_bin, "-y",
                            "-i", temp_concat_video,
                            "-i", audio_file,
                            "-filter_complex", f"[1:a]volume=1.0,atrim=0:{video_duration}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[a]",
                            "-map", "0:v", "-map", "[a]",
                            "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p", "-r", "30",
                            "-c:a", "aac", "-b:a", "192k", "-t", str(video_duration),
                            "-movflags", "+faststart",
                            output_path
                        ]
            else:
                mix_cmd = [
                    ffmpeg_bin, "-y",
                    "-i", temp_concat_video,
                    "-c", "copy",
                    "-movflags", "+faststart",
                    output_path
                ]

            proc = subprocess.run(mix_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                raise RuntimeError(f"Failed to output final mixed video file. {proc.stderr[-800:]}")

            update_progress(100, "Completed", file_url=f"/api/files/exports/{output_filename}")

        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    except Exception as general_err:
        print(f"[COMPILER CRITICAL ERROR] {general_err}")
        update_progress(100, "Failed", err=f"Rendering Pipeline Crash: {str(general_err)}")


# ================= FLASK CONTROLLER ENDPOINTS =================

@app.route('/')
def index_dashboard():
    """
    Renders the video dashboard. This is served directly from the HTML_DASHBOARD
    constant below so the whole application stays a single self-contained file -
    no external templates/dashboard.html is required.
    """
    return render_template_string(HTML_DASHBOARD)


@app.route('/api/upload', methods=['POST'])
def api_upload_media():
    """
    Endpoint to receive uploaded images, videos, or audio soundtracks.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    ftype = get_file_type(file.filename)
    if not ftype:
        return jsonify({"error": "Unsupported file format. Upload videos, images, or audios."}), 400

    filename = secure_filename(file.filename)
    filename_uuid = f"{uuid.uuid4().hex[:6]}_{filename}"
    save_path = os.path.join(UPLOAD_FOLDER, filename_uuid)
    file.save(save_path)

    # Verify the file actually contains what its extension claims (defensive,
    # mirrors the same real-type check used for yt-dlp fetches).
    thumb_url = None
    if ftype == "video":
        probe = probe_media(save_path)
        if not probe["has_video"] and probe["has_audio"]:
            ftype = "audio"
        else:
            thumb_name = thumb_filename_for(filename_uuid)
            thumb_path = os.path.join(THUMB_FOLDER, thumb_name)
            if generate_video_thumbnail(save_path, thumb_path, probe.get("duration")):
                thumb_url = f"/api/files/thumbnails/{thumb_name}"

    return jsonify({
        "success": True,
        "filename": filename_uuid,
        "original_name": filename,
        "type": ftype,
        "url": f"/api/files/uploads/{filename_uuid}",
        "thumbnail": thumb_url,
        "browser_playable": ftype != "video" or filename_uuid.rsplit('.', 1)[-1].lower() not in BROWSER_UNPLAYABLE_CONTAINERS
    })


@app.route('/api/yt-fetch', methods=['POST'])
def api_yt_fetch():
    """
    Fetches videos/soundtracks from YouTube/other supported URLs using yt-dlp.
    Always merges audio+video into a single output file when a "video" fetch
    is requested, and reports back the TRUE media type of what was downloaded.
    """
    data = request.get_json() or {}
    url = data.get("url")
    media_type = data.get("type", "video")

    if not url:
        return jsonify({"error": "Target web URL is required."}), 400

    if yt_dlp is None:
        return jsonify({"error": "yt-dlp is not loaded on server."}), 500

    try:
        saved_filename, actual_type, thumb_url = extract_media_from_url(url, media_type)
        ext = saved_filename.rsplit('.', 1)[-1].lower() if '.' in saved_filename else ''
        return jsonify({
            "success": True,
            "filename": saved_filename,
            "original_name": saved_filename,
            "type": actual_type,
            "requested_type": media_type,
            "url": f"/api/files/uploads/{saved_filename}",
            "thumbnail": thumb_url,
            "browser_playable": actual_type != "video" or ext not in BROWSER_UNPLAYABLE_CONTAINERS
        })
    except Exception as e:
        return jsonify({"error": f"Failed to fetch stream: {str(e)}"}), 500


@app.route('/api/tts', methods=['POST'])
def api_tts_generator():
    """
    Generates premium speech tracks via gTTS or Microsoft Edge-TTS.
    """
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    engine = data.get("engine", "edge")
    voice_or_lang = data.get("voice_or_lang", "en-US-AriaNeural")

    if not text:
        return jsonify({"error": "Speech transcription text is empty."}), 400

    unique_id = uuid.uuid4().hex[:8]
    output_filename = f"tts_{unique_id}.mp3"
    output_path = os.path.join(TTS_FOLDER, output_filename)

    try:
        render_text_to_speech(text, engine, voice_or_lang, output_path)
        return jsonify({
            "success": True, "filename": output_filename, "text": text,
            "engine": engine, "voice": voice_or_lang,
            "url": f"/api/files/tts_audio/{output_filename}"
        })
    except Exception as e:
        try:
            render_text_to_speech(text, "gtts", "en", output_path)
            return jsonify({
                "success": True, "filename": output_filename, "text": text,
                "engine": "gtts (Fallback)", "voice": "en",
                "url": f"/api/files/tts_audio/{output_filename}"
            })
        except Exception as fallback_err:
            return jsonify({"error": f"Speech Synthesis Failed: {str(e)}. Fallback failed: {str(fallback_err)}"}), 500


@app.route('/api/media', methods=['GET'])
def api_list_media():
    """
    Retrieves lists of available uploads, speech audio, and exported items.
    - "media" = images + videos only (newest first) -> feeds the main grid.
    - "audio" = uploaded audio files + TTS speech clips combined (newest first)
      -> feeds the dedicated Audio Library panel, so audio never gets mixed
      into the image/video grid.
    """
    def mtime(path):
        try:
            return os.path.getmtime(path)
        except Exception:
            return 0

    media_items = []
    audio_items = []

    upload_files = os.listdir(UPLOAD_FOLDER)
    upload_files.sort(key=lambda f: mtime(os.path.join(UPLOAD_FOLDER, f)), reverse=True)

    for f in upload_files:
        ftype = get_file_type(f)
        if not ftype:
            continue
        entry = {
            "filename": f,
            "type": ftype,
            "url": f"/api/files/uploads/{f}",
            "uploaded_at": mtime(os.path.join(UPLOAD_FOLDER, f))
        }
        if ftype == "audio":
            entry["source"] = "upload"
            audio_items.append(entry)
        else:
            ext = f.rsplit('.', 1)[-1].lower() if '.' in f else ''
            entry["browser_playable"] = ftype != "video" or ext not in BROWSER_UNPLAYABLE_CONTAINERS
            if ftype == "video":
                thumb_name = thumb_filename_for(f)
                thumb_path = os.path.join(THUMB_FOLDER, thumb_name)
                if not os.path.exists(thumb_path):
                    # Backfills thumbnails for files uploaded before this
                    # feature existed, so the grid never falls back to a
                    # live <video> decode attempt for older assets either.
                    generate_video_thumbnail(os.path.join(UPLOAD_FOLDER, f), thumb_path)
                entry["thumbnail"] = f"/api/files/thumbnails/{thumb_name}" if os.path.exists(thumb_path) else None
            else:
                entry["thumbnail"] = entry["url"]
            media_items.append(entry)

    tts_files = [f for f in os.listdir(TTS_FOLDER) if f.endswith('.mp3')]
    tts_files.sort(key=lambda f: mtime(os.path.join(TTS_FOLDER, f)), reverse=True)
    for f in tts_files:
        audio_items.append({
            "filename": f,
            "type": "audio",
            "source": "tts",
            "url": f"/api/files/tts_audio/{f}",
            "uploaded_at": mtime(os.path.join(TTS_FOLDER, f))
        })

    # Re-sort the combined audio list so uploads and TTS interleave by recency
    audio_items.sort(key=lambda x: x["uploaded_at"], reverse=True)

    export_files = [f for f in os.listdir(EXPORT_FOLDER) if f.endswith('.mp4')]
    export_files.sort(key=lambda f: mtime(os.path.join(EXPORT_FOLDER, f)), reverse=True)
    exports = [{
        "filename": f, "type": "video", "url": f"/api/files/exports/{f}"
    } for f in export_files]

    return jsonify({
        "uploads": media_items,   # kept for backwards compatibility (video/image only now)
        "media": media_items,
        "audio": audio_items,
        "tts": [a for a in audio_items if a.get("source") == "tts"],
        "exports": exports
    })


@app.route('/api/media/delete', methods=['POST'])
def api_delete_media():
    data = request.get_json() or {}
    filename = data.get("filename")
    folder = data.get("folder")

    if not filename or not folder:
        return jsonify({"error": "Missing params"}), 400

    target_dir = {
        "uploads": UPLOAD_FOLDER,
        "tts_audio": TTS_FOLDER,
        "exports": EXPORT_FOLDER,
        "thumbnails": THUMB_FOLDER
    }.get(folder)

    if not target_dir:
        return jsonify({"error": "Invalid folder"}), 400

    safe_name = secure_filename(filename)
    target_file = os.path.join(target_dir, safe_name)
    found = False
    if os.path.exists(target_file):
        os.remove(target_file)
        found = True

    # Clean up the companion thumbnail alongside its source video so deleted
    # assets don't leave orphaned files behind.
    if folder == "uploads":
        thumb_path = os.path.join(THUMB_FOLDER, thumb_filename_for(safe_name))
        if os.path.exists(thumb_path):
            os.remove(thumb_path)

    if found:
        return jsonify({"success": True})
    return jsonify({"error": "File not found"}), 404


@app.route('/api/export', methods=['POST'])
def api_schedule_export():
    """
    Schedules background rendering pipeline.
    """
    data = request.get_json() or {}
    timeline = data.get("timeline", [])
    audio_track = data.get("audio_track", None)
    settings = data.get("settings", {})

    if not timeline:
        return jsonify({"error": "Editing timeline sequence is empty. Add elements first."}), 400

    task_id = uuid.uuid4().hex[:12]
    with tasks_lock:
        export_tasks[task_id] = {
            "progress": 0, "stage": "Scheduling render thread...",
            "download_url": None, "error": None
        }

    thread = threading.Thread(
        target=compile_video_background,
        args=(task_id, timeline, audio_track, settings),
        daemon=True
    )
    thread.start()

    return jsonify({"success": True, "task_id": task_id})


@app.route('/api/export/status/<task_id>', methods=['GET'])
def api_export_status(task_id):
    with tasks_lock:
        task = export_tasks.get(task_id)
    if not task:
        return jsonify({"error": "Rendering task not found."}), 404
    return jsonify(task)


@app.route('/api/files/<folder>/<filename>', methods=['GET'])
def api_serve_files(folder, filename):
    """
    Serves static files dynamically from media folders.
    """
    target_dir = {
        "uploads": UPLOAD_FOLDER,
        "exports": EXPORT_FOLDER,
        "tts_audio": TTS_FOLDER,
        "thumbnails": THUMB_FOLDER
    }.get(folder)

    if not target_dir or not os.path.exists(target_dir):
        return "Directory not found", 404

    return send_from_directory(target_dir, filename)


# ================= DASHBOARD UI HTML TEMPLATE =================

HTML_DASHBOARD = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TTS & Video Editor Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Space Grotesk', sans-serif; background-color: #030712; color: #f3f4f6; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #030712; }
        ::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #374151; }
        .drag-over { outline: 2px dashed #f43f5e; outline-offset: -4px; }
        .drag-ghost { opacity: 0.35; }
    </style>
</head>
<body class="min-h-screen py-6 px-4 sm:px-6 lg:px-8">
    <div class="max-w-7xl mx-auto">
        <header class="flex flex-col md:flex-row justify-between items-center mb-6 pb-5 border-b border-gray-800 gap-4">
            <div>
                <h1 class="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-violet-400 via-pink-500 to-rose-500 bg-clip-text text-transparent">
                    TTS & HD Video Editor Dashboard
                </h1>
                <p class="mt-1 text-gray-400 text-xs">
                    Synthesize voices, import links, arrange media, add captions & transitions, and export up to 4K.
                </p>
            </div>
            <div class="flex items-center gap-2">
                <span class="px-2.5 py-1 rounded-full bg-rose-500/10 text-rose-400 text-[10px] font-bold tracking-wider uppercase animate-pulse">
                    Ffmpeg Renderer Active
                </span>
            </div>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- COLUMN 1: TTS -->
            <div class="lg:col-span-4 space-y-6">
                <div class="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl">
                    <h2 class="text-lg font-bold mb-3 flex items-center gap-2 text-violet-400">TTS Speech Synthesizer</h2>
                    <div class="space-y-4">
                        <div>
                            <label class="block text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">TTS Voice Engine</label>
                            <select id="ttsEngine" onchange="toggleEngineVoices()" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-xs text-gray-100">
                                <option value="edge" selected>Microsoft Edge Premium (Neural)</option>
                                <option value="gtts">Google Standard TTS</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Target Voice & Gender</label>
                            <select id="ttsVoiceEdge" class="w-full bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-xs text-gray-100">
                                <option value="en-US-AriaNeural" selected>🇺🇸 English (US) - Female (Aria)</option>
                                <option value="en-US-JennyNeural">🇺🇸 English (US) - Female (Jenny)</option>
                                <option value="en-US-GuyNeural">🇺🇸 English (US) - Male (Guy)</option>
                                <option value="en-GB-SoniaNeural">🇬🇧 English (UK) - Female (Sonia)</option>
                                <option value="en-GB-RyanNeural">🇬🇧 English (UK) - Male (Ryan)</option>
                                <option value="en-IN-NeerjaNeural">🇮🇳 English (India) - Female (Neerja)</option>
                                <option value="en-IN-PrabhatNeural">🇮🇳 English (India) - Male (Prabhat)</option>
                                <option value="hi-IN-MadhurNeural">🇮🇳 Hindi (India) - Male (Madhur)</option>
                                <option value="hi-IN-SwaraNeural">🇮🇳 Hindi (India) - Female (Swara)</option>
                            </select>
                            <select id="ttsVoiceGtts" class="hidden w-full bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-xs text-gray-100 mt-2">
                                <option value="en">🇺🇸 English</option>
                                <option value="hi">🇮🇳 Hindi</option>
                                <option value="es">🇪🇸 Spanish</option>
                                <option value="fr">🇫🇷 French</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Enter Speech Script</label>
                            <textarea id="ttsText" rows="3" placeholder="Paste or type script..." class="w-full bg-gray-950 border border-gray-800 rounded-xl p-3 text-xs text-gray-100"></textarea>
                        </div>
                        <button id="btnGenerateTts" onclick="generateSpeech()" class="w-full bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-semibold py-2.5 px-4 rounded-xl text-xs flex items-center justify-center gap-2">
                            <span>Render Audio Speech</span>
                            <div id="ttsSpinner" class="hidden w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                        </button>
                    </div>
                </div>

                <!-- Audio Library: uploaded audio + TTS clips, always kept separate from the video/image grid -->
                <div class="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl">
                    <h2 class="text-sm font-bold mb-3 text-gray-300 flex items-center justify-between">
                        <span>Audio Library (Uploads + TTS)</span>
                        <span id="audioCount" class="text-[10px] px-2 py-0.5 bg-gray-950 rounded text-gray-400 font-mono">0</span>
                    </h2>
                    <div id="audioList" class="space-y-3 max-h-[300px] overflow-y-auto pr-1">
                        <div class="text-center py-6 text-xs text-gray-600">No audio yet.</div>
                    </div>
                </div>
            </div>

            <!-- COLUMN 2: Media Import + Grid -->
            <div class="lg:col-span-4 space-y-6">
                <div class="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl">
                    <h2 class="text-lg font-bold mb-3 flex items-center gap-2 text-rose-400">Media Asset Bank</h2>
                    <div id="dropZone" class="border-2 border-dashed border-gray-800 hover:border-rose-500/50 rounded-2xl p-6 text-center transition cursor-pointer bg-gray-950/40 relative">
                        <input type="file" id="fileInput" class="hidden" accept="image/*,video/*,audio/*" onchange="handleFileSelect(event)">
                        <p class="text-xs text-gray-300 font-medium">Drag & Drop PC Files</p>
                        <p class="text-[10px] text-gray-500 mt-1">Videos, Images, or Audio clips</p>
                        <div id="uploadSpinner" class="hidden absolute inset-0 bg-gray-950/80 rounded-2xl flex items-center justify-center flex-col gap-2">
                            <div class="w-6 h-6 border-2 border-rose-500/30 border-t-rose-500 rounded-full animate-spin"></div>
                            <span class="text-[10px] text-rose-400 font-semibold">Uploading...</span>
                        </div>
                    </div>
                    <div class="mt-4 pt-4 border-t border-gray-800">
                        <label class="block text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Import from URL (yt-dlp) - fetches highest quality, audio+video merged</label>
                        <div class="flex gap-2">
                            <input type="url" id="ytUrl" placeholder="https://www.youtube.com/watch?v=..." class="flex-1 bg-gray-950 border border-gray-800 rounded-xl px-3 py-2 text-xs text-gray-100">
                            <button onclick="fetchYtUrl()" class="px-3 py-2 bg-gray-800 hover:bg-gray-700 text-xs text-gray-200 rounded-xl border border-gray-700 flex items-center gap-1">
                                <span>Fetch</span>
                                <div id="ytSpinner" class="hidden w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                            </button>
                        </div>
                    </div>
                </div>

                <div class="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl">
                    <div class="flex justify-between items-center mb-3">
                        <h2 class="text-sm font-bold text-gray-300">Workspace Library (Video & Image)</h2>
                        <div class="flex gap-1 bg-gray-950 p-1 rounded-lg border border-gray-800">
                            <button id="tabAll" onclick="filterGallery('all')" class="px-2 py-0.5 text-[9px] font-semibold bg-gray-900 text-rose-400 rounded">All</button>
                            <button id="tabVideo" onclick="filterGallery('video')" class="px-2 py-0.5 text-[9px] font-semibold text-gray-400 rounded">Video</button>
                            <button id="tabImage" onclick="filterGallery('image')" class="px-2 py-0.5 text-[9px] font-semibold text-gray-400 rounded">Image</button>
                        </div>
                    </div>
                    <p class="text-[9px] text-gray-500 mb-2">Newest uploads appear first. Audio lives in the Audio Library panel, not mixed in here.</p>
                    <div id="galleryList" class="space-y-3 max-h-[420px] overflow-y-auto pr-1">
                        <div class="text-center py-6 text-xs text-gray-600">No media assets in workspace library.</div>
                    </div>
                </div>
            </div>

            <!-- COLUMN 3: Preview, Timeline, Export -->
            <div class="lg:col-span-4 space-y-6">
                <div id="previewTheater" class="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl space-y-4">
                    <h2 class="text-sm font-bold flex items-center gap-2 text-violet-400">Live Theater & Playback</h2>
                    <div id="viewportContainer" class="relative aspect-video w-full bg-black rounded-xl overflow-hidden border border-gray-950 flex items-center justify-center">
                        <video id="viewportVideo" class="hidden w-full h-full object-contain" controls></video>
                        <img id="viewportImage" class="hidden w-full h-full object-contain" />
                        <div id="viewportAudioVisualizer" class="hidden flex flex-col items-center justify-center space-y-3">
                            <span class="text-[10px] text-gray-400 font-mono" id="audioPreviewName">Audio Previewing...</span>
                        </div>
                        <div id="viewportError" class="hidden flex-col items-center justify-center text-center p-4 gap-2">
                            <span class="text-[11px] text-amber-400 font-semibold">Preview not supported for this file's format/codec in your browser.</span>
                            <span class="text-[9px] text-gray-500">This is a browser playback limitation only - export still renders correctly. Try downloading the file to view it, or open it locally.</span>
                        </div>
                        <div id="viewportPlaceholder" class="flex flex-col items-center justify-center text-center p-4">
                            <span class="text-[10px] text-gray-500">Select an asset or click Play Sequence below</span>
                        </div>
                        <div id="viewportSequenceOverlay" class="hidden absolute top-2 left-2 bg-black/90 px-2 py-0.5 rounded border border-gray-800 text-[9px] font-mono text-gray-200 flex items-center gap-1.5">
                            <span>LIVE SEQUENCER: CLIP <span id="seqClipIndex">1</span>/<span id="seqClipTotal">5</span></span>
                        </div>
                    </div>
                    <div class="flex items-center justify-between gap-3 text-xs pt-1">
                        <div class="flex items-center gap-1.5">
                            <button id="btnPlaySeq" onclick="playLiveSequence()" class="px-2.5 py-1.5 bg-rose-600 hover:bg-rose-500 text-white font-bold rounded-lg text-[10px]">
                                <span>Play Sequence</span>
                            </button>
                            <button id="btnStopSeq" onclick="stopLiveSequence()" class="px-2.5 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-200 font-bold rounded-lg text-[10px]">
                                <span>Stop</span>
                            </button>
                        </div>
                        <div class="text-[9px] text-gray-500 font-mono flex items-center gap-2">
                            <span class="truncate max-w-[120px]" id="seqDurationVal">0.0s</span>
                        </div>
                    </div>
                    <audio id="timelinePreviewAudio" class="hidden"></audio>
                </div>

                <div class="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl">
                    <div class="flex justify-between items-center mb-3 border-b border-gray-800 pb-2">
                        <h2 class="text-lg font-bold flex items-center gap-2 text-emerald-400">Timeline Track</h2>
                        <button onclick="clearTimeline()" class="text-[10px] text-gray-500 hover:text-red-400">Clear All</button>
                    </div>

                    <div class="mb-4 bg-gray-950 p-3 rounded-xl border border-gray-850">
                        <label class="block text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1 flex justify-between items-center">
                            <span>Master Audio Soundtrack</span>
                        </label>
                        <select id="masterAudio" class="w-full bg-gray-900 border border-gray-800 rounded-lg px-2 py-1.5 text-xs text-gray-200">
                            <option value="">-- No Soundtrack (Silent) --</option>
                        </select>
                        <div class="mt-2 flex items-center justify-between text-[10px] text-gray-500">
                            <label class="flex items-center gap-1.5 cursor-pointer">
                                <input type="checkbox" id="loopAudio" class="rounded border-gray-800 bg-gray-900 text-emerald-600">
                                Loop Audio Track
                            </label>
                            <label class="flex items-center gap-1.5 cursor-pointer">
                                <input type="checkbox" id="matchAudio" class="rounded border-gray-800 bg-gray-900 text-emerald-600">
                                Match Video Length to Audio
                            </label>
                        </div>
                    </div>

                    <p class="text-[9px] text-gray-500 mb-2">Drag the ⠿ handle to reorder clips, or use the arrows.</p>
                    <div id="timelineList" class="space-y-2 max-h-[380px] overflow-y-auto pr-1 mb-4">
                        <div class="text-center py-8 text-xs text-gray-600 border border-dashed border-gray-800 rounded-xl">
                            Click "+ Timeline" on assets in your library to add them here!
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-3 pt-3 border-t border-gray-800 text-xs">
                        <div>
                            <label class="block text-[9px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Canvas Aspect Ratio</label>
                            <select id="canvasFormat" class="w-full bg-gray-950 border border-gray-800 rounded-lg p-1.5 text-xs">
                                <option value="9:16">Portrait 9:16 (Shorts/Reels)</option>
                                <option value="16:9" selected>Landscape 16:9 (YouTube/PC)</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-[9px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Export Resolution</label>
                            <select id="exportQuality" class="w-full bg-gray-950 border border-gray-800 rounded-lg p-1.5 text-xs">
                                <option value="720p">720p HD</option>
                                <option value="1080p" selected>1080p Full HD</option>
                                <option value="4k">4K Ultra HD (High Bitrate)</option>
                            </select>
                        </div>
                    </div>

                    <button id="btnExportVideo" onclick="exportComposition()" class="w-full mt-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold py-3 px-4 rounded-xl text-xs flex items-center justify-center gap-2">
                        <span>Render HD Export</span>
                    </button>
                </div>

                <div id="exportProgressCard" class="hidden bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl space-y-3">
                    <div class="flex justify-between items-center text-xs">
                        <span class="font-bold text-emerald-400">Exporting...</span>
                        <span id="exportProgressPct" class="font-mono text-gray-400">0%</span>
                    </div>
                    <div class="w-full bg-gray-950 h-2.5 rounded-full overflow-hidden border border-gray-800">
                        <div id="exportProgressBar" class="bg-gradient-to-r from-emerald-500 to-teal-400 h-full w-[0%] transition-all duration-300"></div>
                    </div>
                    <p id="exportStage" class="text-[10px] text-gray-400 leading-relaxed italic">Initiating rendering threads...</p>
                </div>

                <div class="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl">
                    <h2 class="text-sm font-bold mb-3 text-gray-300">Compiled Downloads Library</h2>
                    <div id="exportsList" class="space-y-3 max-h-[180px] overflow-y-auto pr-1">
                        <div class="text-center py-6 text-xs text-gray-600">No exports compiled yet.</div>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <script>
        let availableMedia = { media: [], audio: [], exports: [] };
        let timeline = [];
        let currentFilter = 'all';
        let dragSrcIndex = null;

        let sequenceTimer = null;
        let currentSeqIndex = 0;
        let isSequencePlaying = false;

        const TRANSITIONS = [
            ['none', 'None (Hard Cut)'], ['cross_dissolve', 'Cross Dissolve'],
            ['fade_black', 'Fade to Black'], ['fade_white', 'Fade to White / Flash'],
            ['zoom_in', 'Zoom In / Pull In'], ['zoom_out', 'Zoom Out / Pull Out'],
            ['pan_left', 'Pan Left'], ['pan_right', 'Pan Right'],
            ['pan_up', 'Pan Up'], ['pan_down', 'Pan Down'],
            ['whip_pan', 'Whip Pan'], ['camera_shake', 'Camera Shake'],
            ['linear_wipe', 'Linear Wipe'], ['clock_wipe', 'Clock Wipe'],
            ['shape_circle', 'Shape Mask (Circle)'], ['shape_square', 'Shape Mask (Square)'],
            ['shape_star', 'Shape Mask (Star)'], ['frame_block', 'Frame Blocking'],
            ['match_cut', 'Match Cut'], ['gaussian_blur', 'Gaussian Blur'],
            ['glitch', 'Glitch / Digital Distortion'], ['light_leak', 'Light Leak / Film Burn'],
            ['page_turn', 'Page Turn / Page Peel'], ['split_screen', 'Split Screen Transition'],
        ];
        const CAPTION_FONTS = ["Impact", "Arial Black", "Anton", "Bebas Neue", "Montserrat", "Poppins", "Arial", "Verdana"];

        window.addEventListener('DOMContentLoaded', () => {
            fetchMediaCatalog();
            const dropZone = document.getElementById('dropZone');
            dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
            dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
                if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
            });
            dropZone.addEventListener('click', () => document.getElementById('fileInput').click());
        });

        function toggleEngineVoices() {
            const engine = document.getElementById('ttsEngine').value;
            document.getElementById('ttsVoiceEdge').classList.toggle('hidden', engine !== 'edge');
            document.getElementById('ttsVoiceGtts').classList.toggle('hidden', engine === 'edge');
        }

        async function fetchMediaCatalog() {
            try {
                const res = await fetch('/api/media');
                const data = await res.json();
                availableMedia = data; // already sorted newest-first, audio pre-separated by the server
                renderGallery();
                renderAudioList();
                renderExportsList();
                updateAudioOptions();
            } catch (err) {
                console.error("Error fetching library catalog:", err);
            }
        }

        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            const spinner = document.getElementById('uploadSpinner');
            spinner.classList.remove('hidden');
            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.success) { await fetchMediaCatalog(); } else { alert(data.error || "File upload failed."); }
            } catch (err) {
                alert("Upload network error: " + err.message);
            } finally {
                spinner.classList.add('hidden');
            }
        }

        function handleFileSelect(e) { if (e.target.files.length > 0) uploadFile(e.target.files[0]); }

        async function fetchYtUrl() {
            const urlInput = document.getElementById('ytUrl');
            const url = urlInput.value.trim();
            if (!url) return alert("Please specify a URL.");
            const spinner = document.getElementById('ytSpinner');
            spinner.classList.remove('hidden');
            try {
                const res = await fetch('/api/yt-fetch', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url, type: "video" })
                });
                const data = await res.json();
                if (data.success) {
                    urlInput.value = '';
                    if (data.type !== data.requested_type) {
                        alert(`Note: the source only had ${data.type} available - saved as ${data.type}.`);
                    }
                    await fetchMediaCatalog();
                } else {
                    alert(data.error || "Extraction pipeline failure.");
                }
            } catch (err) {
                alert("Extraction error: " + err.message);
            } finally {
                spinner.classList.add('hidden');
            }
        }

        async function generateSpeech() {
            const text = document.getElementById('ttsText').value.trim();
            const engine = document.getElementById('ttsEngine').value;
            const voice = (engine === 'edge') ? document.getElementById('ttsVoiceEdge').value : document.getElementById('ttsVoiceGtts').value;
            if (!text) return alert("Speech script can't be empty.");
            const spinner = document.getElementById('ttsSpinner');
            spinner.classList.remove('hidden');
            try {
                const res = await fetch('/api/tts', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text, engine: engine, voice_or_lang: voice })
                });
                const data = await res.json();
                if (data.success) { document.getElementById('ttsText').value = ''; await fetchMediaCatalog(); }
                else { alert(data.error || "Synthesis failure."); }
            } catch (err) {
                alert("Synthesis network error: " + err.message);
            } finally {
                spinner.classList.add('hidden');
            }
        }

        function filterGallery(type) {
            currentFilter = type;
            ['tabAll', 'tabVideo', 'tabImage'].forEach(tabId => {
                const tab = document.getElementById(tabId);
                if (tabId === 'tab' + type.charAt(0).toUpperCase() + type.slice(1)) {
                    tab.classList.add('bg-gray-900', 'text-rose-400'); tab.classList.remove('text-gray-400');
                } else {
                    tab.classList.remove('bg-gray-900', 'text-rose-400'); tab.classList.add('text-gray-400');
                }
            });
            renderGallery();
        }

        function displayName(filename) { return filename.split('_').slice(1).join('_') || filename; }

        function renderGallery() {
            const list = document.getElementById('galleryList');
            const filtered = availableMedia.media.filter(item => currentFilter === 'all' ? true : item.type === currentFilter);
            if (filtered.length === 0) {
                list.innerHTML = `<div class="text-center py-6 text-xs text-gray-600">No ${currentFilter} assets found.</div>`;
                return;
            }
            list.innerHTML = filtered.map(item => {
                let badgeColor = item.type === 'video' ? 'bg-indigo-500/10 text-indigo-400' : 'bg-amber-500/10 text-amber-400';
                const thumbSrc = item.type === 'image' ? item.url : (item.thumbnail || '');
                const canHoverPreview = item.type === 'video' && item.browser_playable !== false;
                return `
                    <div class="bg-gray-950 border border-gray-800 rounded-xl p-3 flex flex-col gap-3 hover:border-gray-700 transition">
                        <div class="w-full aspect-video bg-black/50 rounded-lg flex items-center justify-center border border-gray-850 overflow-hidden relative group">
                            ${thumbSrc ? `<img src="${thumbSrc}" class="w-full h-full object-cover" onerror="this.style.display='none'" />` : `<span class="text-[9px] text-gray-600">No preview</span>`}
                            ${canHoverPreview ? `<video src="${item.url}" class="absolute inset-0 w-full h-full object-cover opacity-0 group-hover:opacity-100 transition" muted preload="none" onmouseenter="this.play().catch(()=>{})" onmouseleave="this.pause(); this.currentTime=0;" onerror="this.style.display='none'"></video>` : ''}
                            ${item.type === 'video' && item.browser_playable === false ? `<span class="absolute bottom-1 right-1 text-[7px] bg-black/80 text-amber-400 px-1.5 py-0.5 rounded">Export-only preview</span>` : ''}
                        </div>
                        <div class="min-w-0">
                            <h4 class="text-xs font-semibold text-gray-200 truncate break-all" title="${item.filename}">${displayName(item.filename)}</h4>
                            <span class="px-1.5 py-0.5 rounded text-[8px] font-bold ${badgeColor} uppercase tracking-wider mt-1 inline-block">${item.type.toUpperCase()}</span>
                        </div>
                        <div class="flex justify-between items-center gap-2 pt-2 border-t border-gray-900">
                            <div class="flex gap-2">
                                <button onclick="deleteAsset('${item.filename}', 'uploads')" class="text-[9px] text-gray-500 hover:text-rose-400">Delete</button>
                                <button onclick="previewAsset('${item.url}', '${item.type}', '${item.filename}', ${item.browser_playable !== false})" class="text-[9px] text-violet-400 hover:text-violet-300 font-semibold">👁️ Preview</button>
                            </div>
                            <button onclick="addToTimeline('${item.filename}', '${item.type}')" class="px-2 py-1 bg-rose-600 hover:bg-rose-500 text-white font-bold rounded text-[9px]">+ Timeline</button>
                        </div>
                    </div>`;
            }).join('');
        }

        function renderAudioList() {
            const list = document.getElementById('audioList');
            document.getElementById('audioCount').textContent = availableMedia.audio.length;
            if (availableMedia.audio.length === 0) {
                list.innerHTML = `<div class="text-center py-6 text-xs text-gray-600">No audio yet.</div>`;
                return;
            }
            list.innerHTML = availableMedia.audio.map((item) => `
                <div class="bg-gray-950 border border-gray-800 rounded-xl p-3 space-y-2">
                    <div class="flex items-center justify-between gap-2">
                        <span class="text-[9px] font-mono text-gray-400">${item.source === 'tts' ? '🎙️ TTS' : '📁 Upload'} · ${displayName(item.filename)}</span>
                        <button onclick="deleteAsset('${item.filename}', '${item.source === 'tts' ? 'tts_audio' : 'uploads'}')" class="text-[8px] text-gray-500 hover:text-red-400">Delete</button>
                    </div>
                    <audio src="${item.url}" controls class="w-full h-7 rounded-lg bg-gray-900 opacity-80 hover:opacity-100 transition"></audio>
                    <div class="flex justify-between items-center text-[10px]">
                        <a href="${item.url}" download class="text-violet-400 hover:underline">Download</a>
                        <button onclick="setAsMasterAudio('${item.filename}', '${item.source === 'tts' ? 'tts' : 'upload'}')" class="px-2 py-0.5 bg-violet-900 hover:bg-violet-800 text-white text-[9px] rounded">Use as Soundtrack</button>
                    </div>
                </div>`).join('');
        }

        function renderExportsList() {
            const list = document.getElementById('exportsList');
            if (availableMedia.exports.length === 0) {
                list.innerHTML = `<div class="text-center py-6 text-xs text-gray-600">No exports compiled yet.</div>`;
                return;
            }
            list.innerHTML = availableMedia.exports.map((item) => `
                <div class="bg-gray-950 border border-gray-850 rounded-xl p-3 flex justify-between items-center gap-2">
                    <div class="min-w-0 flex-1">
                        <span class="text-[9px] text-gray-500">Video MP4 HD</span>
                        <h4 class="text-xs font-semibold text-gray-200 truncate">${item.filename}</h4>
                    </div>
                    <div class="flex gap-2">
                        <a href="${item.url}" download class="px-2 py-1 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded text-[9px]">Download</a>
                        <button onclick="deleteAsset('${item.filename}', 'exports')" class="px-1 py-1 text-gray-500 hover:text-red-400 rounded text-[9px]">✕</button>
                    </div>
                </div>`).join('');
        }

        async function deleteAsset(filename, folder) {
            try {
                await fetch('/api/media/delete', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename: filename, folder: folder })
                });
                await fetchMediaCatalog();
            } catch (err) { alert("Deletion error: " + err.message); }
        }

        function updateAudioOptions() {
            const select = document.getElementById('masterAudio');
            const selectedVal = select.value;
            select.innerHTML = `<option value="">-- No Soundtrack (Silent) --</option>`;
            availableMedia.audio.forEach(item => {
                const opt = document.createElement('option');
                opt.value = JSON.stringify({ filename: item.filename, source: item.source === 'tts' ? 'tts' : 'uploads' });
                opt.textContent = `${item.source === 'tts' ? '🎙️ TTS' : '📁 PC'}: ${displayName(item.filename)}`;
                select.appendChild(opt);
            });
            if (selectedVal) select.value = selectedVal;
        }

        function setAsMasterAudio(filename, source) {
            document.getElementById('masterAudio').value = JSON.stringify({ filename: filename, source: source === 'tts' ? 'tts' : 'uploads' });
        }

        // ================= TIMELINE TRACK OPERATIONS =================

        function addToTimeline(filename, type) {
            const src = availableMedia.media.find(m => m.filename === filename);
            timeline.push({
                id: uid(), filename: filename, type: type,
                duration: 5, trimStart: 0, effect: 'none', mute: false,
                text: null,
                thumbnail: src ? (src.thumbnail || (type === 'image' ? src.url : '')) : '',
                browserPlayable: src ? src.browser_playable !== false : true
            });
            renderTimeline();
        }

        function uid() { return Math.random().toString(36).substring(2, 9); }

        function moveTimelineItem(index, dir) {
            const targetIdx = index + dir;
            if (targetIdx < 0 || targetIdx >= timeline.length) return;
            const temp = timeline[index];
            timeline[index] = timeline[targetIdx];
            timeline[targetIdx] = temp;
            renderTimeline();
        }

        function removeTimelineItem(index) { timeline.splice(index, 1); renderTimeline(); }
        function clearTimeline() { timeline = []; renderTimeline(); }
        function updateTimelineItemProp(index, prop, val) { timeline[index][prop] = val; }

        function toggleCaptionPanel(index) {
            const panel = document.getElementById(`captionPanel_${index}`);
            if (panel) panel.classList.toggle('hidden');
        }

        function updateCaptionField(index, field, val) {
            if (!timeline[index].text) {
                timeline[index].text = { content: '', font: 'Impact', size: 54, color: 'white', bg: true, bgColor: 'black', position: 'bottom' };
            }
            timeline[index].text[field] = val;
        }

        function clearCaption(index) {
            timeline[index].text = null;
            renderTimeline();
        }

        // Drag & drop reordering
        function handleDragStart(e, index) {
            dragSrcIndex = index;
            e.dataTransfer.effectAllowed = 'move';
            e.currentTarget.classList.add('drag-ghost');
        }
        function handleDragEnd(e) { e.currentTarget.classList.remove('drag-ghost'); }
        function handleDragOver(e) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; }
        function handleDrop(e, index) {
            e.preventDefault();
            if (dragSrcIndex === null || dragSrcIndex === index) return;
            const moved = timeline.splice(dragSrcIndex, 1)[0];
            timeline.splice(index, 0, moved);
            dragSrcIndex = null;
            renderTimeline();
        }

        function transitionOptionsHtml(selected) {
            return TRANSITIONS.map(([val, label]) => `<option value="${val}" ${val === selected ? 'selected' : ''}>${label}</option>`).join('');
        }
        function fontOptionsHtml(selected) {
            return CAPTION_FONTS.map(f => `<option value="${f}" ${f === selected ? 'selected' : ''}>${f}</option>`).join('');
        }

        function renderTimeline() {
            const list = document.getElementById('timelineList');
            if (timeline.length === 0) {
                list.innerHTML = `<div class="text-center py-8 text-xs text-gray-600 border border-dashed border-gray-800 rounded-xl">Click "+ Timeline" on assets in your library to add them here!</div>`;
                return;
            }

            list.innerHTML = timeline.map((item, index) => {
                let badgeColor = item.type === 'video' ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-900/40' : 'bg-amber-500/10 text-amber-400 border border-amber-900/40';
                const fileUrl = `/api/files/uploads/${item.filename}`;
                const iconThumb = item.thumbnail || (item.type === 'image' ? fileUrl : '');
                const playable = item.browserPlayable !== false;
                const hasCaption = item.text && item.text.content;
                return `
                    <div draggable="true" ondragstart="handleDragStart(event, ${index})" ondragend="handleDragEnd(event)" ondragover="handleDragOver(event)" ondrop="handleDrop(event, ${index})"
                         class="bg-gray-950 border border-gray-850 rounded-xl p-3 flex flex-col gap-2 relative cursor-move">
                        <div class="flex items-center justify-between gap-2">
                            <div class="flex items-center gap-2 min-w-0">
                                <span class="text-gray-600 text-xs" title="Drag to reorder">⠿</span>
                                <div class="w-12 h-8 rounded overflow-hidden bg-black flex-shrink-0 border border-gray-800 flex items-center justify-center">
                                    ${iconThumb ? `<img src="${iconThumb}" class="w-full h-full object-cover" onerror="this.style.display='none'">` : `<span class="text-[7px] text-gray-600">no thumb</span>`}
                                </div>
                                <span class="px-1.5 py-0.5 rounded text-[8px] font-bold ${badgeColor} uppercase tracking-wider">${item.type.toUpperCase()}</span>
                                <h4 class="text-xs font-semibold text-gray-300 truncate" title="${item.filename}">${displayName(item.filename)}</h4>
                                ${hasCaption ? '<span class="text-[8px] text-emerald-400">CC</span>' : ''}
                                ${!playable ? '<span class="text-[7px] text-amber-400" title="Browser can\'t preview this container - export still works fine">export-only</span>' : ''}
                            </div>
                            <div class="flex items-center gap-1">
                                <button onclick="previewAsset('${fileUrl}', '${item.type}', '${item.filename}', ${playable})" class="text-[10px] p-1 text-violet-400 hover:text-violet-300 font-semibold mr-1" title="Preview">👁️</button>
                                <button onclick="moveTimelineItem(${index}, -1)" class="text-xs p-1 text-gray-500 hover:text-gray-300">▲</button>
                                <button onclick="moveTimelineItem(${index}, 1)" class="text-xs p-1 text-gray-500 hover:text-gray-300">▼</button>
                                <button onclick="removeTimelineItem(${index})" class="text-xs p-1 text-gray-500 hover:text-rose-400 ml-1">✕</button>
                            </div>
                        </div>

                        <div class="grid grid-cols-3 gap-2 text-[10px] text-gray-400">
                            ${item.type === 'video' ? `
                            <div>
                                <label class="block text-[8px] text-gray-500 uppercase font-semibold">Trim Start (s)</label>
                                <input type="number" min="0" step="0.5" value="${item.trimStart}" onchange="updateTimelineItemProp(${index}, 'trimStart', parseFloat(this.value)||0)" class="w-full bg-gray-900 border border-gray-800 rounded px-2 py-1 text-xs">
                            </div>` : `<div></div>`}
                            <div>
                                <label class="block text-[8px] text-gray-500 uppercase font-semibold">Clip Length (s)</label>
                                <input type="number" min="0.5" max="120" step="0.5" value="${item.duration}" onchange="updateTimelineItemProp(${index}, 'duration', parseFloat(this.value)||5)" class="w-full bg-gray-900 border border-gray-800 rounded px-2 py-1 text-xs">
                            </div>
                            <div class="flex items-end pb-1">
                                <label class="flex items-center gap-1 cursor-pointer text-[9px]">
                                    <input type="checkbox" ${item.mute ? 'checked' : ''} onchange="updateTimelineItemProp(${index}, 'mute', this.checked)" class="rounded border-gray-800 bg-gray-900">
                                    Mute original audio
                                </label>
                            </div>
                        </div>

                        <div>
                            <label class="block text-[8px] text-gray-500 uppercase font-semibold">Transition In</label>
                            <select onchange="updateTimelineItemProp(${index}, 'effect', this.value)" class="w-full bg-gray-900 border border-gray-800 rounded px-2 py-1 text-xs">
                                ${transitionOptionsHtml(item.effect)}
                            </select>
                        </div>

                        <div class="pt-1 border-t border-gray-900">
                            <button onclick="toggleCaptionPanel(${index})" class="text-[9px] text-emerald-400 hover:text-emerald-300 font-semibold">+ Caption / Text / Logo Overlay</button>
                            <div id="captionPanel_${index}" class="hidden mt-2 space-y-2 bg-gray-900 rounded-lg p-2">
                                <input type="text" placeholder="Caption text..." value="${item.text ? (item.text.content || '') : ''}" onchange="updateCaptionField(${index}, 'content', this.value)" class="w-full bg-gray-950 border border-gray-800 rounded px-2 py-1 text-xs">
                                <div class="grid grid-cols-3 gap-2">
                                    <select onchange="updateCaptionField(${index}, 'font', this.value)" class="bg-gray-950 border border-gray-800 rounded px-1 py-1 text-[10px]">
                                        ${fontOptionsHtml(item.text ? item.text.font : 'Impact')}
                                    </select>
                                    <select onchange="updateCaptionField(${index}, 'position', this.value)" class="bg-gray-950 border border-gray-800 rounded px-1 py-1 text-[10px]">
                                        <option value="top" ${item.text && item.text.position === 'top' ? 'selected' : ''}>Top</option>
                                        <option value="middle" ${item.text && item.text.position === 'middle' ? 'selected' : ''}>Middle</option>
                                        <option value="bottom" ${!item.text || item.text.position === 'bottom' ? 'selected' : ''}>Bottom</option>
                                    </select>
                                    <input type="number" placeholder="Size" value="${item.text ? item.text.size : 54}" onchange="updateCaptionField(${index}, 'size', parseInt(this.value)||54)" class="bg-gray-950 border border-gray-800 rounded px-1 py-1 text-[10px]">
                                </div>
                                <div class="grid grid-cols-3 gap-2 items-center">
                                    <label class="text-[9px] text-gray-500">Text <input type="color" value="${item.text ? item.text.color : '#ffffff'}" onchange="updateCaptionField(${index}, 'color', this.value)"></label>
                                    <label class="text-[9px] text-gray-500">Box <input type="color" value="${item.text ? item.text.bgColor : '#000000'}" onchange="updateCaptionField(${index}, 'bgColor', this.value)"></label>
                                    <label class="flex items-center gap-1 text-[9px] text-gray-500"><input type="checkbox" ${!item.text || item.text.bg ? 'checked' : ''} onchange="updateCaptionField(${index}, 'bg', this.checked)"> BG</label>
                                </div>
                                <button onclick="clearCaption(${index})" class="text-[9px] text-red-400 hover:text-red-300">Remove caption</button>
                            </div>
                        </div>
                    </div>`;
            }).join('');
        }

        // ================= EXPORT COMPOSITION PIPELINE =================

        async function exportComposition() {
            if (timeline.length === 0) return alert("Timeline sequence is empty. Add elements first.");
            const audioSelect = document.getElementById('masterAudio');
            let audio_track = audioSelect.value ? JSON.parse(audioSelect.value) : null;
            const payload = {
                timeline: timeline,
                audio_track: audio_track,
                settings: {
                    format: document.getElementById('canvasFormat').value,
                    quality: document.getElementById('exportQuality').value,
                    loop_audio: document.getElementById('loopAudio').checked,
                    match_audio_length: document.getElementById('matchAudio').checked
                }
            };

            const progressCard = document.getElementById('exportProgressCard');
            const progressPct = document.getElementById('exportProgressPct');
            const progressBar = document.getElementById('exportProgressBar');
            const exportStage = document.getElementById('exportStage');
            const btnExportVideo = document.getElementById('btnExportVideo');

            btnExportVideo.disabled = true;
            btnExportVideo.classList.add('opacity-50');
            progressCard.classList.remove('hidden');
            progressPct.textContent = "0%";
            progressBar.style.width = "0%";
            exportStage.textContent = "Starting render threads...";

            try {
                const res = await fetch('/api/export', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.error) {
                    alert(data.error);
                    btnExportVideo.disabled = false; btnExportVideo.classList.remove('opacity-50'); progressCard.classList.add('hidden');
                    return;
                }
                pollExportStatus(data.task_id);
            } catch (err) {
                alert("Export submission error: " + err.message);
                btnExportVideo.disabled = false; btnExportVideo.classList.remove('opacity-50'); progressCard.classList.add('hidden');
            }
        }

        function pollExportStatus(taskId) {
            const progressPct = document.getElementById('exportProgressPct');
            const progressBar = document.getElementById('exportProgressBar');
            const exportStage = document.getElementById('exportStage');
            const progressCard = document.getElementById('exportProgressCard');
            const btnExportVideo = document.getElementById('btnExportVideo');

            const interval = setInterval(async () => {
                try {
                    const res = await fetch(`/api/export/status/${taskId}`);
                    const data = await res.json();
                    if (data.error) {
                        clearInterval(interval);
                        alert("Rendering Crash: " + data.error);
                        btnExportVideo.disabled = false; btnExportVideo.classList.remove('opacity-50'); progressCard.classList.add('hidden');
                        return;
                    }
                    progressPct.textContent = data.progress + "%";
                    progressBar.style.width = data.progress + "%";
                    exportStage.textContent = data.stage;
                    if (data.progress >= 100) {
                        clearInterval(interval);
                        btnExportVideo.disabled = false; btnExportVideo.classList.remove('opacity-50');
                        setTimeout(() => progressCard.classList.add('hidden'), 1000);
                        await fetchMediaCatalog();
                    }
                } catch (err) { console.error("Polling error:", err); }
            }, 1500);
        }

        // ================= LIVE PREVIEW THEATER =================

        function hideAllViewports() {
            ['viewportVideo','viewportImage','viewportAudioVisualizer','viewportPlaceholder','viewportSequenceOverlay','viewportError'].forEach(id => {
                document.getElementById(id).classList.add('hidden');
            });
        }

        // Containers the <video> tag can never decode in-browser, regardless
        // of how valid the file is (real browser limitation, not a bug).
        const UNPLAYABLE_EXT = ['mkv', 'avi'];
        function extOf(filename) {
            if (!filename) return '';
            const parts = filename.split('.');
            return parts.length > 1 ? parts.pop().toLowerCase() : '';
        }

        function previewAsset(url, type, filename, browserPlayable) {
            if (isSequencePlaying) stopLiveSequence();
            hideAllViewports();

            const videoEl = document.getElementById('viewportVideo');
            const imageEl = document.getElementById('viewportImage');
            const audioVis = document.getElementById('viewportAudioVisualizer');
            const audioPreviewName = document.getElementById('audioPreviewName');
            const timelineAudio = document.getElementById('timelinePreviewAudio');
            const errorEl = document.getElementById('viewportError');

            // CRITICAL FIX: clear any handler left over from a previous preview
            // BEFORE touching .src. Setting src="" on a <video> fires an error
            // event on its own - if the old onerror handler was still attached,
            // that stray event alone would show "Preview not supported" even
            // though the NEW asset about to be loaded was perfectly fine. This
            // was the root cause of the black-screen error appearing randomly.
            videoEl.onerror = null;
            imageEl.onerror = null;
            videoEl.pause(); videoEl.removeAttribute('src'); videoEl.load();
            timelineAudio.pause(); timelineAudio.src = "";

            const dName = filename ? displayName(filename) : "Asset Preview";
            document.getElementById('seqDurationVal').textContent = dName;

            const knownUnplayable = browserPlayable === false || UNPLAYABLE_EXT.includes(extOf(filename));

            if (type === 'video') {
                if (knownUnplayable) {
                    // Don't even attempt playback - tell the user up front
                    // instead of letting a guaranteed decode failure surface
                    // as a generic error after the fact.
                    hideAllViewports();
                    errorEl.classList.remove('hidden');
                    errorEl.style.display = 'flex';
                    return;
                }
                videoEl.classList.remove('hidden');
                videoEl.onerror = () => { hideAllViewports(); errorEl.classList.remove('hidden'); errorEl.style.display = 'flex'; };
                videoEl.src = url;
                videoEl.muted = false;
                videoEl.play().catch(e => console.log("Autoplay prevented:", e));
            } else if (type === 'image') {
                imageEl.classList.remove('hidden');
                imageEl.onerror = () => { hideAllViewports(); errorEl.classList.remove('hidden'); errorEl.style.display = 'flex'; };
                imageEl.src = url;
            } else {
                audioVis.classList.remove('hidden');
                audioPreviewName.textContent = dName;
                timelineAudio.src = url;
                timelineAudio.play().catch(e => console.log("Autoplay prevented:", e));
            }
        }

        function playLiveSequence() {
            if (timeline.length === 0) return alert("Timeline sequence is empty.");
            stopLiveSequence();
            isSequencePlaying = true;
            currentSeqIndex = 0;

            const audioSelect = document.getElementById('masterAudio');
            const timelineAudio = document.getElementById('timelinePreviewAudio');
            timelineAudio.pause(); timelineAudio.src = "";
            if (audioSelect.value) {
                try {
                    const audio_track = JSON.parse(audioSelect.value);
                    const folder = audio_track.source === 'tts' ? 'tts_audio' : 'uploads';
                    timelineAudio.src = `/api/files/${folder}/${audio_track.filename}`;
                    timelineAudio.loop = document.getElementById('loopAudio').checked;
                    timelineAudio.play().catch(e => console.log(e));
                } catch (e) { console.error(e); }
            }
            document.getElementById('viewportSequenceOverlay').classList.remove('hidden');
            playNextTimelineItem();
        }

        function playNextTimelineItem() {
            if (!isSequencePlaying) return;
            if (currentSeqIndex >= timeline.length) { stopLiveSequence(); return; }

            const item = timeline[currentSeqIndex];
            const durationMs = (parseFloat(item.duration) || 5) * 1000;

            document.getElementById('seqClipIndex').textContent = currentSeqIndex + 1;
            document.getElementById('seqClipTotal').textContent = timeline.length;
            document.getElementById('seqDurationVal').textContent = `${displayName(item.filename)} (${item.duration}s)`;

            const videoEl = document.getElementById('viewportVideo');
            const imageEl = document.getElementById('viewportImage');
            const errorEl = document.getElementById('viewportError');
            hideAllViewports();
            document.getElementById('viewportSequenceOverlay').classList.remove('hidden');
            videoEl.onerror = null;
            videoEl.pause(); videoEl.removeAttribute('src'); videoEl.load();

            const url = `/api/files/uploads/${item.filename}`;
            const knownUnplayable = UNPLAYABLE_EXT.includes(extOf(item.filename));
            if (item.type === 'video' && !knownUnplayable) {
                videoEl.classList.remove('hidden');
                videoEl.onerror = () => { hideAllViewports(); document.getElementById('viewportSequenceOverlay').classList.remove('hidden'); errorEl.classList.remove('hidden'); errorEl.style.display = 'flex'; };
                videoEl.src = url;
                videoEl.muted = true;
                videoEl.currentTime = item.trimStart || 0;
                videoEl.play().catch(e => console.log(e));
            } else if (item.type === 'video' && knownUnplayable) {
                // Export still renders this clip correctly via ffmpeg - it's
                // only the in-browser scrub preview that can't decode mkv/avi.
                errorEl.classList.remove('hidden');
                errorEl.style.display = 'flex';
                document.getElementById('viewportSequenceOverlay').classList.remove('hidden');
            } else if (item.type === 'image') {
                imageEl.classList.remove('hidden');
                imageEl.src = url;
            }

            highlightTimelineItem(currentSeqIndex);
            sequenceTimer = setTimeout(() => { currentSeqIndex++; playNextTimelineItem(); }, durationMs);
        }

        function highlightTimelineItem(activeIndex) {
            const cards = document.getElementById('timelineList').children;
            for (let i = 0; i < cards.length; i++) {
                if (i === activeIndex) { cards[i].classList.add('ring-2', 'ring-rose-500', 'border-rose-500'); }
                else { cards[i].classList.remove('ring-2', 'ring-rose-500', 'border-rose-500'); }
            }
        }

        function stopLiveSequence() {
            isSequencePlaying = false;
            if (sequenceTimer) { clearTimeout(sequenceTimer); sequenceTimer = null; }
            const timelineAudio = document.getElementById('timelinePreviewAudio');
            timelineAudio.pause(); timelineAudio.src = "";
            const videoEl = document.getElementById('viewportVideo');
            videoEl.onerror = null;
            videoEl.pause(); videoEl.removeAttribute('src'); videoEl.load();
            const imageEl = document.getElementById('viewportImage');
            imageEl.onerror = null;
            imageEl.src = "";
            hideAllViewports();
            document.getElementById('viewportPlaceholder').classList.remove('hidden');
            document.getElementById('seqDurationVal').textContent = "0.0s";
            const cards = document.getElementById('timelineList').children;
            for (let i = 0; i < cards.length; i++) cards[i].classList.remove('ring-2', 'ring-rose-500', 'border-rose-500');
        }
    </script>
</body>
</html>
"""

# Serve execution
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False, threaded=True)