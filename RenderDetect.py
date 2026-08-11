#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shorts Studio — fetch a YouTube video, auto/manually cut it into vertical
shorts, LIVE-EDIT each one in your browser (speed, zoom, mute/replace audio,
AI Hindi/English voiceover, text, logo, blur/black/emoji hide-regions,
rect/circle/arrow shapes, color grading) and only write the final file to
disk when you click Save.

One process, one script: a local Flask server + a modern single-page UI
(vanilla HTML/CSS/JS, no build step) that opens in your default browser.

PERFORMANCE NOTES (this version):
  - Clips are cut in PARALLEL (ThreadPoolExecutor) instead of one-by-one,
    since each cut is an independent network-read + ffmpeg job.
  - Proxy cuts use preset=ultrafast (they're just scratch/preview files;
    the real quality knob is the final export preset/CRF you pick).
  - ffmpeg is told to use all CPU threads (-threads 0).
  - Export has a fast "stream copy" path: if you didn't change speed/zoom/
    color/regions/text/logo/audio/resolution/format from defaults, Save
    just remuxes the proxy file instead of re-encoding it — this turns a
    multi-second re-encode into a near-instant copy.

Run:
    pip install flask yt-dlp imageio-ffmpeg edge-tts gTTS
    (optional, only needed for auto smart-reframe / speaker tracking:)
    pip install opencv-python-headless numpy
    (optional on top of that, only needed for multi-speaker active-speaker
    detection via mouth-motion scoring — without it, Smart Reframe still
    works fine with Kalman smoothing + scene-cut detection, just single-face):
    pip install mediapipe
    python shortvideo.py
    (put .mp3 background-music files in an "audio" folder next to this script)
    (smart_reframe_engine.py must sit in the same folder as this script)
"""

import os
import sys
import re
import json
import time
import uuid
import asyncio
import threading
import subprocess
import webbrowser
import tempfile
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, Blueprint, request, jsonify, send_file, Response
try:
    import yt_dlp
    import imageio_ffmpeg
except ImportError:
    print("Run: pip install yt-dlp imageio-ffmpeg flask")
    raise SystemExit(1)

try:
    import edge_tts
    EDGE_TTS_OK = True
except ImportError:
    EDGE_TTS_OK = False

# Advanced Smart-Reframe engine (Kalman filtering, multi-speaker mouth-motion
# scoring, scene-cut segmentation, optical-flow fallback) — lives in its own
# module file (smart_reframe_engine.py, next to this script) so it stays
# independently testable/maintainable. Entirely optional: if the module or
# its own dependencies (opencv-python, numpy, and optionally mediapipe for
# multi-speaker scoring) aren't installed, everything else in the app keeps
# working and face tracking just falls back to the simpler built-in detector
# further down this file.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from smart_reframe_engine import generate_smart_track as _enterprise_generate_track
    from smart_reframe_engine import ENGINE_AVAILABLE as _ENTERPRISE_ENGINE_AVAILABLE
except ImportError:
    _enterprise_generate_track = None
    _ENTERPRISE_ENGINE_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_OK = True
except ImportError:
    GTTS_OK = False

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

from word_captions import get_word_level_captions

def _no_console_kwargs():
    """Prevents a flashing black CMD window on Windows every time ffmpeg is
    shelled out to — keeps things looking clean and modern for users who
    aren't going to understand a terminal popping up mid-task."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}

if getattr(sys, 'frozen', False):
    # Running as a packaged EXE/APK: keep data next to the actual executable,
    # not inside PyInstaller's temporary extraction folder (which gets deleted).
    BASE = Path(sys.executable).resolve().parent
else:
    BASE = Path(__file__).resolve().parent
PROXY_DIR = BASE / "proxy_clips"
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = BASE / "shorts_final"
AUDIO_LIB_DIR = BASE / "audio"          # <-- put your .mp3 background-music files here
for d in (PROXY_DIR, UPLOAD_DIR, OUTPUT_DIR, AUDIO_LIB_DIR):
    d.mkdir(exist_ok=True)

TTS_TEMP_DIR = BASE / "tts_temp"     # generated voiceovers live here — temporary, listed + deletable from the UI, never a "keep this file" deliverable
TTS_TEMP_DIR.mkdir(exist_ok=True)

# Voice catalog. Format: key -> (engine, voice_or_lang, extra)
#   engine "edge" -> voice_or_lang is a Microsoft edge-tts neural voice ID, extra unused
#   engine "gtts"  -> voice_or_lang is the gTTS language code, extra is the gTTS "tld"
#                     (accent variant — co.in/co.uk/com.au/com give different English accents
#                     out of the same Google engine)
#
# HONEST NOTE on voice quality: edge-tts (Microsoft's neural voices, the same
# engine behind "Read Aloud" in Edge/Word) is the most natural-sounding option
# that works fully free/offline-callable here, but it is not — and nothing
# free/offline can currently be — a genuine ElevenLabs-level match. ElevenLabs'
# realism comes from a proprietary emotion-conditioned model behind a paid API;
# edge-tts and gTTS only expose rate/pitch/volume knobs, so "emotion" here is a
# tonal approximation (faster+higher for excited, slower+lower for sad, etc.),
# not true expressive acting. edge-tts is easily the more natural-sounding of
# the two engines below; gTTS is flatter/more robotic but kept as a reliable
# fallback and for its extra English accent variants. Microsoft occasionally
# adds/retires voices — if a voice ID below ever 404s, run
# `edge-tts --list-voices` in your terminal to see what's currently live and
# swap the ID here.
TTS_VOICES = {
    # ---- Hindi (India) — edge-tts neural (only 2 official Hindi neural voices exist) ----
    "hi_male":              ("edge", "hi-IN-MadhurNeural", None),
    "hi_female":             ("edge", "hi-IN-SwaraNeural", None),

    # ---- English (India) — edge-tts neural (only 2 official en-IN neural voices exist) ----
    "en_in_male":            ("edge", "en-IN-PrabhatNeural", None),
    "en_in_female":          ("edge", "en-IN-NeerjaNeural", None),

    # ---- English (US) — edge-tts neural, several distinct real voices ----
    "en_us_male_guy":        ("edge", "en-US-GuyNeural", None),
    "en_us_male_davis":      ("edge", "en-US-DavisNeural", None),
    "en_us_male_andrew":     ("edge", "en-US-AndrewNeural", None),   # newer conversational voice, one of the more natural-sounding US male options
    "en_us_male_brian":      ("edge", "en-US-BrianNeural", None),
    "en_us_female_aria":     ("edge", "en-US-AriaNeural", None),
    "en_us_female_jenny":    ("edge", "en-US-JennyNeural", None),
    "en_us_female_emma":     ("edge", "en-US-EmmaNeural", None),     # newer conversational voice, pairs with Andrew
    "en_us_female_michelle": ("edge", "en-US-MichelleNeural", None),

    # ---- English (UK) — edge-tts neural ----
    "en_gb_male_ryan":       ("edge", "en-GB-RyanNeural", None),
    "en_gb_male_thomas":     ("edge", "en-GB-ThomasNeural", None),
    "en_gb_female_sonia":    ("edge", "en-GB-SoniaNeural", None),
    "en_gb_female_libby":    ("edge", "en-GB-LibbyNeural", None),

    # ---- gTTS fallback — simpler/flatter, but rock-solid and gives a couple
    # of extra English accent variants via `tld` ----
    "gtts_hi":               ("gtts", "hi", "co.in"),
    "gtts_en_in":            ("gtts", "en", "co.in"),
    "gtts_en_us":            ("gtts", "en", "com"),
    "gtts_en_uk":            ("gtts", "en", "co.uk"),
}

TTS_VOICE_LABELS = {
    "hi_male":              "Hindi — Male (Madhur, Neural)",
    "hi_female":             "Hindi — Female (Swara, Neural)",
    "en_in_male":            "English (India) — Male (Prabhat, Neural)",
    "en_in_female":          "English (India) — Female (Neerja, Neural)",
    "en_us_male_guy":        "English (US) — Male (Guy, Neural)",
    "en_us_male_davis":      "English (US) — Male (Davis, Neural)",
    "en_us_male_andrew":     "English (US) — Male (Andrew, Neural)",
    "en_us_male_brian":      "English (US) — Male (Brian, Neural)",
    "en_us_female_aria":     "English (US) — Female (Aria, Neural)",
    "en_us_female_jenny":    "English (US) — Female (Jenny, Neural)",
    "en_us_female_emma":     "English (US) — Female (Emma, Neural)",
    "en_us_female_michelle": "English (US) — Female (Michelle, Neural)",
    "en_gb_male_ryan":       "English (UK) — Male (Ryan, Neural)",
    "en_gb_male_thomas":     "English (UK) — Male (Thomas, Neural)",
    "en_gb_female_sonia":    "English (UK) — Female (Sonia, Neural)",
    "en_gb_female_libby":    "English (UK) — Female (Libby, Neural)",
    "gtts_hi":               "Hindi — Google TTS (basic quality)",
    "gtts_en_in":            "English (India accent) — Google TTS (basic quality)",
    "gtts_en_us":            "English (US accent) — Google TTS (basic quality)",
    "gtts_en_uk":            "English (UK accent) — Google TTS (basic quality)",
}

# In-memory registry of generated voiceovers: fname -> {id,url,voice_key,...}
# Deliberately NOT persisted to disk as a manifest — this is a session-scoped
# "recent generations" list so the user can reuse or delete one; the files
# themselves live in TTS_TEMP_DIR and get removed as soon as they're deleted
# from this list (or you can safely wipe the whole tts_temp/ folder any time).
TTS_LIBRARY = {}

# Emotion presets for AI voiceover. edge-tts genuinely supports rate/volume/pitch
# SSML-style params per request, so those give a real tonal shift. gTTS has no
# such control at all, so for gTTS we simulate the same emotion by post-processing
# the generated mp3 with ffmpeg (tempo + pitch + loudness) - same emotions, same
# dropdown, both engines actually respond differently to match the label.
EMOTION_PRESETS = {
    "neutral":  {"label": "😐 Neutral",   "edge_rate": "+0%",  "edge_volume": "+0%",  "edge_pitch": "+0Hz",
                 "gtts_tempo": 1.00, "gtts_pitch": 1.00, "gtts_vol_db": 0},
    "happy":    {"label": "😄 Happy",     "edge_rate": "+10%", "edge_volume": "+8%",  "edge_pitch": "+35Hz",
                 "gtts_tempo": 1.08, "gtts_pitch": 1.04, "gtts_vol_db": 2},
    "sad":      {"label": "😢 Sad",       "edge_rate": "-14%", "edge_volume": "-6%",  "edge_pitch": "-40Hz",
                 "gtts_tempo": 0.90, "gtts_pitch": 0.94, "gtts_vol_db": -2},
    "angry":    {"label": "😠 Angry",     "edge_rate": "+14%", "edge_volume": "+16%", "edge_pitch": "+15Hz",
                 "gtts_tempo": 1.12, "gtts_pitch": 1.02, "gtts_vol_db": 4},
    "excited":  {"label": "🤩 Excited",   "edge_rate": "+20%", "edge_volume": "+12%", "edge_pitch": "+45Hz",
                 "gtts_tempo": 1.16, "gtts_pitch": 1.06, "gtts_vol_db": 3},
    "calm":     {"label": "😌 Calm",      "edge_rate": "-10%", "edge_volume": "+0%",  "edge_pitch": "-10Hz",
                 "gtts_tempo": 0.93, "gtts_pitch": 0.98, "gtts_vol_db": 0},
    "serious":  {"label": "🧐 Serious",   "edge_rate": "-6%",  "edge_volume": "+0%",  "edge_pitch": "-15Hz",
                 "gtts_tempo": 0.96, "gtts_pitch": 0.96, "gtts_vol_db": 0},
    "whisper":  {"label": "🤫 Whisper",   "edge_rate": "-10%", "edge_volume": "-35%", "edge_pitch": "-5Hz",
                 "gtts_tempo": 0.92, "gtts_pitch": 0.98, "gtts_vol_db": -10},
    "fear":     {"label": "😨 Fearful",   "edge_rate": "+12%", "edge_volume": "-4%",  "edge_pitch": "+22Hz",
                 "gtts_tempo": 1.10, "gtts_pitch": 1.03, "gtts_vol_db": -1},
}

# in-memory background job registry for progressive (live) clip cutting:
# job_id -> {"clips": [...], "done": False, "title": "", "error": None, "total": N}
JOBS = {}

# in-memory registry for word-perfect caption fetch jobs:
# video_id -> {"status": "running"/"done"/"error", "stage": str, "words": [...], "error": str|None}
WORD_CAPTIONS_JOBS = {}

# CapCut-style one-click color/look presets (pure ffmpeg eq/curves/colorchannelmixer combos)
COLOR_PRESETS = {
    "none": {},
    "vivid": {"contrast": 1.15, "saturation": 1.35, "brightness": 0.02},
    "cinematic": {"contrast": 1.2, "saturation": 0.85, "brightness": -0.02, "curves": "vintage"},
    "moody": {"contrast": 1.25, "saturation": 0.7, "brightness": -0.05},
    "vintage_vhs": {"contrast": 0.95, "saturation": 0.8, "brightness": 0.0, "curves": "vintage", "vignette": True},
    "warm_glow": {"contrast": 1.08, "saturation": 1.2, "brightness": 0.03, "colorchannelmixer": "rr=1.1:gg=1.02:bb=0.9"},
    "cool_blue": {"contrast": 1.1, "saturation": 1.05, "brightness": 0.0, "colorchannelmixer": "rr=0.9:gg=1.0:bb=1.15"},
}

# How many clips to cut at once. Network-bound + ffmpeg, so a modest
# worker count helps a lot without saturating CPU/bandwidth.


# MAX_CUT_WORKERS = min(6, max(2, (os.cpu_count() or 4)))

import psutil
_avail_gb = psutil.virtual_memory().available / (1024**3)
MAX_CUT_WORKERS = max(1, min(6, os.cpu_count() or 4, int(_avail_gb // 1.2)))

render_bp = Blueprint("render_bp", __name__)

# in-memory job/clip registry: clip_id -> {path, w, h, duration}
CLIPS = {}


# ───────────────────────────── helpers ─────────────────────────────

def probe(path):
    proc = subprocess.run([FFMPEG, "-i", str(path)], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", **_no_console_kwargs())
    out = proc.stdout
    w = h = None
    dur = None
    m = re.search(r"(\d{2,5})x(\d{2,5})", out)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
    m2 = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out)
    if m2:
        hh, mm, ss = m2.groups()
        dur = int(hh) * 3600 + int(mm) * 60 + float(ss)
    return w, h, dur


COOKIES_FILE = BASE / "cookies.txt"   # export from your browser (see note below) if auto-detect fails
COOKIE_BROWSERS = ["chrome", "edge", "firefox", "brave"]  # tried in this order

# Remembers whichever auth mode last succeeded so repeat resolves (every new
# URL pasted on the main page) skip straight to the method that's known to
# work, instead of re-running the whole slow cookies_file->chrome->edge->
# firefox->brave->default->bypass fallback chain from scratch every time.
# Each browser-cookie attempt can cost 15-30+ seconds (yt-dlp has to copy and
# decrypt that browser's entire cookie DB), so paying that cost once per
# process instead of once per video is the single biggest speed win here.
_RESOLVED_MODE = {"mode": None}


def _ydl_base_opts():
    opts = {
        'quiet': True, 'no_warnings': True, 'noplaylist': True,
        'socket_timeout': 15, 'retries': 2, 'extractor_retries': 1,
        'geo_bypass': True,
    }
    if COOKIES_FILE.exists():
        opts['cookiefile'] = str(COOKIES_FILE)
    return opts


def _auth_attempts():
    """Cached working mode first, then the rest of the fallback chain
    (skipping duplicates) — same pattern as downloader.py's smart caching."""
    attempts = []
    if _RESOLVED_MODE["mode"]:
        attempts.append(_RESOLVED_MODE["mode"])
    rest = []
    if COOKIES_FILE.exists():
        rest.append("cookies_file")
    rest.extend(COOKIE_BROWSERS)
    rest.append("default")
    rest.append("bypass")
    for m in rest:
        if m not in attempts:
            attempts.append(m)
    return attempts


def resolve_stream(url, max_height=None):
    ydl_opts = _ydl_base_opts()
    if max_height and str(max_height).lower() not in ("best", "auto", "0", ""):
        h = int(max_height)
        ydl_opts['format'] = f'bestvideo[height<={h}]+bestaudio/best[height<={h}]/best'
    else:
        ydl_opts['format'] = 'bestvideo+bestaudio/best'

    attempts = _auth_attempts()

    last_err = None
    info = None
    for mode in attempts:
        opts = dict(ydl_opts)
        if mode == "cookies_file":
            opts['cookiefile'] = str(COOKIES_FILE)
        elif mode in COOKIE_BROWSERS:
            opts.pop('cookiefile', None)
            opts['cookiesfrombrowser'] = (mode,)
        elif mode == "default":
            opts.pop('cookiefile', None)
        elif mode == "bypass":
            opts.pop('cookiefile', None)
            opts['extractor_args'] = {
                'youtube': {'player_client': ['ios', 'android', 'mweb']}
            }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            _RESOLVED_MODE["mode"] = mode  # remember what worked
            break
        except Exception as e:
            last_err = e
            info = None
            continue

    if info is None:
        raise last_err or RuntimeError("Could not resolve video")

    video_url = audio_url = None
    video_headers = audio_headers = None
    if info.get('requested_formats'):
        for f in info['requested_formats']:
            if f.get('vcodec') != 'none' and f.get('acodec') == 'none':
                video_url = f['url']
                video_headers = f.get('http_headers')
            elif f.get('acodec') != 'none' and f.get('vcodec') == 'none':
                audio_url = f['url']
                audio_headers = f.get('http_headers')
    if not video_url:
        video_url = info.get('url')
        video_headers = info.get('http_headers')
    if not audio_url:
        audio_url = video_url
        audio_headers = video_headers
    # Fall back to top-level headers if a specific format didn't carry its own —
    # googlevideo URLs are signed to a particular client (see the c= param in
    # the URL) and will 403 if ffmpeg's request doesn't look like that client.
    video_headers = video_headers or info.get('http_headers') or {}
    audio_headers = audio_headers or info.get('http_headers') or {}
    title = info.get('title', 'video')
    safe = re.sub(r'[^\w\-]+', '_', title)[:40]
    return video_url, audio_url, safe, info.get('duration'), video_headers, audio_headers, info.get('description') or '', info.get('tags') or []


def _ffmpeg_header_args(headers):
    """Builds ffmpeg's -headers block from a yt-dlp http_headers dict.
    Without this, googlevideo signed URLs come back 403 because ffmpeg's
    default User-Agent doesn't match the client the URL was signed for."""
    if not headers:
        return []
    lines = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    return ["-headers", lines]


def cut_proxy(video_url, audio_url, start, end, out_path, video_headers=None, audio_headers=None):
    duration = end - start
    vf = "crop=ih*9/16:ih"
    v_hdr = _ffmpeg_header_args(video_headers)
    a_hdr = _ffmpeg_header_args(audio_headers)
    # Proxy clips are throwaway scratch/preview files that get re-encoded
    # again on export — ultrafast + a light crf is plenty and cuts encode
    # time dramatically vs. the previous near-lossless veryfast/crf15.
    if video_url == audio_url:
        cmd = [FFMPEG, "-y"] + v_hdr + ["-ss", str(start), "-i", video_url, "-t", str(duration),
           "-vf", vf,
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
           "-c:a", "aac", "-b:a", "192k",
           "-threads", "2", "-movflags", "+faststart", str(out_path)]
    else:
        cmd = [FFMPEG, "-y"] + v_hdr + ["-ss", str(start), "-i", video_url] + a_hdr + ["-ss", str(start), "-i", audio_url,
           "-t", str(duration),
           "-vf", vf,
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
           "-c:a", "aac", "-b:a", "192k",
           "-threads", "2", "-movflags", "+faststart", str(out_path)]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", **_no_console_kwargs())
    if not (out_path.exists() and out_path.stat().st_size > 1000):
        print("──── FFMPEG CUT FAILED ────")
        print(result.stdout[-2000:])   # last 2000 chars — error usually right at the end
        print("───────────────────────────")
    return out_path.exists() and out_path.stat().st_size > 1000


def _parse_resolution(resolution, src_w, src_h):
    """Resolution dropdown string ('1080x1920' / 'original') ko actual
    target width/height me convert karta hai. 'original' ya bad input
    hone par source dimensions hi wapas kar deta hai."""
    if not resolution or str(resolution).lower() == "original":
        return src_w, src_h
    m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", str(resolution))
    if m:
        return int(m.group(1)), int(m.group(2))
    return src_w, src_h


# ---- Multi-ratio export (Opus-Clip style: one clip -> several aspect
# ratios in one export job) — high-quality fixed presets, no arbitrary
# resolution dropdown involved. All presets below are full 1080-class
# targets so there is no resolution/quality tradeoff between ratios. ----
RATIO_PRESETS = {
    "9:16": (1080, 1920),
    "1:1":  (1080, 1080),
    "16:9": (1920, 1080),
    "4:5":  (1080, 1350),
}

def _resolve_ratio_dims(ratio, src_w, src_h):
    """Ratio label ('9:16' / '1:1' / '16:9' / '4:5'), raw 'WxH', ya 'original'
    ko actual target width/height me convert karta hai. Unknown custom ratio
    ('W:H' jo preset me nahi hai) aane par 1920px long-edge pe scale karta
    hai. Kuch samajh na aaye toh source dims wapas."""
    if not ratio or str(ratio).lower() == "original":
        return src_w, src_h
    key = str(ratio).strip()
    if key in RATIO_PRESETS:
        return RATIO_PRESETS[key]
    m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", key)
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", key)
    if m2:
        rw, rh = int(m2.group(1)), int(m2.group(2))
        if rw >= rh:
            return 1920, int(round(1920 * rh / rw))
        else:
            return int(round(1920 * rw / rh)), 1920
    return src_w, src_h


def fetch_source_segment(source, out_path, pad=3.0):
    """Original stream se hi (proxy ki jagah) ek padded window ko
    LOSSLESSLY remux karta hai (-c copy => koi re-encode nahi, zero
    quality loss). Padding is liye taaki -ss/-c-copy ka keyframe-seek
    thoda idhar-udhar ho toh bhi asli clip ka koi frame miss na ho —
    exact boundary export pass me (jahan already re-encode ho raha hai)
    frame-accurate trim se theek ho jaati hai, wahan cost bhi nahi lagti
    kyunki woh pass already har frame decode/encode kar raha hai.
    Returns (success, actual_clamped_start_used)."""
    video_url = source["video_url"]
    audio_url = source["audio_url"]
    start = max(0.0, source.get("start", 0) - pad)
    end = source.get("end", 0) + pad
    duration = max(0.1, end - start)
    v_hdr = _ffmpeg_header_args(source.get("video_headers"))
    a_hdr = _ffmpeg_header_args(source.get("audio_headers"))

    if video_url == audio_url:
        cmd = [FFMPEG, "-y"] + v_hdr + ["-ss", str(start), "-i", video_url,
               "-t", str(duration), "-c", "copy", "-avoid_negative_ts", "make_zero",
               "-movflags", "+faststart", str(out_path)]
    else:
        cmd = [FFMPEG, "-y"] + v_hdr + ["-ss", str(start), "-i", video_url] + a_hdr + \
              ["-ss", str(start), "-i", audio_url, "-t", str(duration),
               "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
               "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(out_path)]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             encoding="utf-8", errors="replace", **_no_console_kwargs())
    ok = out_path.exists() and out_path.stat().st_size > 1000
    if not ok:
        print("──── LOSSLESS SOURCE FETCH FAILED (falling back to proxy) ────")
        print(result.stdout[-2000:])
        print("────────────────────────────────────────────────────────────")
    return ok, start


def detect_silences(path, noise_db=-30, min_silence_dur=0.5, pad=0.12):
    """Runs ffmpeg's silencedetect audio filter over a clip and returns:
      - silences: list of {start, end} raw silent ranges (seconds)
      - keep_segments: the inverse — the non-silent parts to keep, each
        padded by `pad` seconds on both sides (so words don't get clipped),
        clamped to the clip's actual duration.
      - removed_duration: total seconds that would be cut if all silences
        are applied.

    noise_db: how quiet (in dB) audio has to be to count as "silence".
    min_silence_dur: minimum length (seconds) of a quiet stretch to count
    (avoids treating natural micro-pauses between words as cuts).
    """
    _, _, total_dur = probe(Path(path))
    if not total_dur:
        return {"silences": [], "keep_segments": [], "removed_duration": 0, "error": "Could not read clip duration"}

    cmd = [FFMPEG, "-i", str(path), "-af",
           f"silencedetect=noise={noise_db}dB:d={min_silence_dur}",
           "-f", "null", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           encoding="utf-8", errors="replace", **_no_console_kwargs())
    out = proc.stdout

    # ffmpeg prints these as separate log lines, e.g.:
    #   [silencedetect @ ...] silence_start: 12.34
    #   [silencedetect @ ...] silence_end: 14.10 | silence_duration: 1.76
    starts = [float(m) for m in re.findall(r"silence_start:\s*(-?\d+\.?\d*)", out)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*(-?\d+\.?\d*)", out)]

    silences = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else total_dur  # trailing silence with no logged end = runs to the end
        s = max(0.0, s)
        e = min(total_dur, e)
        if e > s:
            silences.append({"start": round(s, 3), "end": round(e, 3)})

    # Build the inverse (keep segments), padding each side so we don't
    # clip the start/end of a word that sits right at the silence boundary.
    keep_segments = []
    cursor = 0.0
    for sil in silences:
        seg_start = cursor
        seg_end = min(total_dur, sil["start"] + pad)
        if seg_end > seg_start + 0.05:  # ignore near-zero slivers
            keep_segments.append({"start": round(seg_start, 3), "end": round(seg_end, 3)})
        cursor = max(cursor, sil["end"] - pad)
    if cursor < total_dur - 0.05:
        keep_segments.append({"start": round(cursor, 3), "end": round(total_dur, 3)})

    removed = sum(s["end"] - s["start"] for s in silences)
    return {
        "silences": silences,
        "keep_segments": keep_segments,
        "removed_duration": round(removed, 2),
        "clip_duration": round(total_dur, 2),
    }


@render_bp.route("/api/detect_silence", methods=["POST"])
def api_detect_silence():
    """Body: {clip_id, noise_db?, min_silence_dur?, pad?}
    Returns detected silent gaps + suggested keep-segments for that clip's
    proxy file. Detection-only for now — does not modify or trim anything;
    the editor is expected to store the returned keep_segments on the
    clip's edit-state and apply them at export time."""
    data = request.json or {}
    clip_id = data.get("clip_id")
    clip = CLIPS.get(clip_id)
    if not clip:
        return jsonify({"error": "Unknown clip_id"}), 404

    noise_db = float(data.get("noise_db", -30))
    min_silence_dur = float(data.get("min_silence_dur", 0.5))
    pad = float(data.get("pad", 0.12))

    result = detect_silences(clip["path"], noise_db=noise_db,
                              min_silence_dur=min_silence_dur, pad=pad)
    if result.get("error"):
        return jsonify(result), 400
    return jsonify(result)


def apply_silence_cut(path, keep_segments):
    """Physically cuts `keep_segments` out of the video/audio at `path` and
    concatenates them back-to-back, replacing the file IN PLACE — same
    "build to temp file, then atomically replace" pattern already used for
    TTS audio post-processing above (safe: if ffmpeg fails, the original
    file is left untouched).

    Returns the new total duration after cutting.
    """
    path = Path(path)
    n = len(keep_segments)
    if n == 0:
        _, _, dur = probe(path)
        return dur

    filter_parts = []
    v_labels, a_labels = [], []
    for i, seg in enumerate(keep_segments):
        st, en = float(seg["start"]), float(seg["end"])
        filter_parts.append(f"[0:v]trim=start={st}:end={en},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim=start={st}:end={en},asetpts=PTS-STARTPTS[a{i}]")
        v_labels.append(f"[v{i}]")
        a_labels.append(f"[a{i}]")
    concat_inputs = "".join(f"{v_labels[i]}{a_labels[i]}" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[vout][aout]")
    filter_complex = ";".join(filter_parts)

    tmp = path.with_name(path.stem + "_silcut_" + uuid.uuid4().hex[:6] + path.suffix)
    cmd = [FFMPEG, "-y", "-i", str(path),
           "-filter_complex", filter_complex,
           "-map", "[vout]", "-map", "[aout]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "192k",
           "-threads", "0", "-movflags", "+faststart", str(tmp)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             encoding="utf-8", errors="replace", **_no_console_kwargs())

    if not (tmp.exists() and tmp.stat().st_size > 1000):
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        print("──── SILENCE-CUT FFMPEG FAILED ────")
        print(result.stdout[-2000:])
        print("────────────────────────────────────")
        raise RuntimeError("Silence-cut ffmpeg failed — see server console for details")

    tmp.replace(path)  # atomic swap — original is only overwritten once the new file is confirmed good
    _, _, new_dur = probe(path)
    return new_dur


DEADAIR_BACKUP_DIR = PROXY_DIR / "_deadair_backups"
DEADAIR_BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def _deadair_backup_path(clip_id):
    return DEADAIR_BACKUP_DIR / f"{clip_id}_orig.mp4"


@render_bp.route("/api/apply_silence_cut", methods=["POST"])
def api_apply_silence_cut():
    """Body: {clip_id, keep_segments?, noise_db?, min_silence_dur?, pad?}

    If keep_segments is omitted, runs detect_silences() first using the
    given (or default) thresholds — so this endpoint alone gives you the
    full "one click, auto-detect-and-remove" behavior. If the frontend
    already called /api/detect_silence and let the user review/toggle gaps,
    pass the (possibly edited) keep_segments straight through instead.

    This is destructive to the clip's PROXY file only (never the original
    source video) — CLIPS[clip_id]["duration"] is updated afterward so the
    rest of the editor (trim range, export duration, etc.) reflects the new,
    shorter length.

    Before cutting, the current proxy file is backed up so /api/undo_silence_cut
    can restore it later. Only the most recent pre-cut state is kept (applying
    twice in a row overwrites the earlier backup with the state right before
    the second cut).
    """
    data = request.json or {}
    clip_id = data.get("clip_id")
    clip = CLIPS.get(clip_id)
    if not clip:
        return jsonify({"error": "Unknown clip_id"}), 404

    keep_segments = data.get("keep_segments")
    detection = None
    if not keep_segments:
        noise_db = float(data.get("noise_db", -30))
        min_silence_dur = float(data.get("min_silence_dur", 0.5))
        pad = float(data.get("pad", 0.12))
        detection = detect_silences(clip["path"], noise_db=noise_db,
                                     min_silence_dur=min_silence_dur, pad=pad)
        if detection.get("error"):
            return jsonify(detection), 400
        keep_segments = detection["keep_segments"]
        # Nothing worth cutting — don't re-encode for no reason.
        if not detection["silences"]:
            return jsonify({"clip_id": clip_id, "removed_duration": 0,
                             "new_duration": round(detection["clip_duration"], 2),
                             "old_duration": round(detection["clip_duration"], 2),
                             "undo_available": bool(clip.get("dead_air_backup")),
                             "message": "No significant silence detected."})

    old_dur = clip.get("duration")

    # Back up the current (pre-cut) file before we touch anything, so this
    # is always undoable from the editor afterward.
    backup_path = _deadair_backup_path(clip_id)
    try:
        import shutil
        shutil.copy2(clip["path"], backup_path)
    except Exception as e:
        return jsonify({"error": f"Could not create undo backup, aborting cut: {e}"}), 500

    try:
        new_dur = apply_silence_cut(clip["path"], keep_segments)
    except Exception as e:
        backup_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500

    clip["duration"] = new_dur
    clip["dead_air_backup"] = str(backup_path)
    removed = (detection["removed_duration"] if detection
               else (round(old_dur - new_dur, 2) if (old_dur and new_dur) else None))

    return jsonify({
        "clip_id": clip_id,
        "new_duration": round(new_dur, 2) if new_dur else None,
        "old_duration": round(old_dur, 2) if old_dur else None,
        "removed_duration": removed,
        "undo_available": True,
    })


@render_bp.route("/api/undo_silence_cut", methods=["POST"])
def api_undo_silence_cut():
    """Body: {clip_id}
    Restores the clip's proxy file to the state it was in right before the
    most recent /api/apply_silence_cut call, using the backup made at that
    time. Only one undo level is kept (the state just before the last cut) —
    applying again after undoing creates a fresh backup as usual."""
    data = request.json or {}
    clip_id = data.get("clip_id")
    clip = CLIPS.get(clip_id)
    if not clip:
        return jsonify({"error": "Unknown clip_id"}), 404

    backup = clip.get("dead_air_backup")
    if not backup or not Path(backup).exists():
        return jsonify({"error": "No dead-air backup available for this clip — nothing to undo."}), 400

    try:
        Path(backup).replace(clip["path"])
    except Exception as e:
        return jsonify({"error": f"Could not restore backup: {e}"}), 500

    clip["dead_air_backup"] = None
    _, _, new_dur = probe(Path(clip["path"]))
    clip["duration"] = new_dur

    return jsonify({
        "clip_id": clip_id,
        "new_duration": round(new_dur, 2) if new_dur else None,
        "undo_available": False,
    })


# ────────────────────── Auto smart-reframe / face tracking ─────────────────
# Follows the speaker's face instead of a fixed center crop when zoomed in.
# Uses OpenCV's Haar cascade face detector, which ships INSIDE the
# opencv-python package itself (cv2.data.haarcascades) - no internet download
# or extra model file needed, just `pip install opencv-python-headless`.
# This is optional: everything else in the app works fine without it, and
# clips simply fall back to the existing static center/manual-pan crop if
# cv2 isn't installed or no face is found.
def _detect_face_track(path, duration, sample_every=0.5, max_samples=240):
    """Samples frames at a fixed interval, finds the largest face in each
    (largest = whoever's closest to camera = almost always the main
    speaker, not a background face), and returns a list of
    [t, center_x_frac, center_y_frac] for samples where a face was found."""
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "Face tracking needs the 'opencv-python' package, which isn't installed. "
            "Install it with:  pip install opencv-python-headless"
        )
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Could not open this clip for face detection.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    step_frames = max(1, int(round(fps * sample_every)))
    track = []
    frame_idx = 0
    samples_taken = 0
    try:
        while samples_taken < max_samples:
            t = frame_idx / fps
            if t > duration:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            min_size = (max(20, int(frame_w * 0.06)), max(20, int(frame_h * 0.06)))
            faces = detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=min_size)
            if len(faces):
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                cx = (fx + fw / 2) / frame_w
                cy = (fy + fh / 2) / frame_h
                track.append([round(t, 2), round(cx, 4), round(cy, 4)])
            frame_idx += step_frames
            samples_taken += 1
    finally:
        cap.release()
    return track


def _detect_face_track_advanced(path, duration):
    """Runs the standalone EnterpriseSmartReframer engine (true Kalman
    filtering, multi-speaker mouth-motion-based active-speaker switching,
    two-pass scene-cut segmentation, and optical-flow/velocity fault
    tolerance) and returns (track, faces_found, meta). The returned track
    is already fully smoothed by the engine itself — do NOT run it back
    through `_smooth_face_track`, which would just add lag on top of the
    engine's own dead-zone/inertial-pan smoothing.
    Raises ImportError if the engine module or its dependencies
    (opencv-python, numpy) aren't installed, so callers can fall back to
    the simpler built-in detector below."""
    if not _ENTERPRISE_ENGINE_AVAILABLE or _enterprise_generate_track is None:
        raise ImportError(
            "The advanced Smart-Reframe engine needs 'opencv-python' and 'numpy' "
            "(and optionally 'mediapipe' for multi-speaker active-speaker detection). "
            "Install with:  pip install opencv-python-headless numpy mediapipe"
        )
    return _enterprise_generate_track(str(path), duration)


def _smooth_face_track(track, max_jump=0.12, alpha=0.35):
    """Exponential-moving-average smoothing so the crop eases toward a new
    face position instead of jittering/whip-panning every sample, and caps
    any single wild jump (a stray false-positive detected elsewhere in
    frame) to a slow ease instead of a snap."""
    if not track:
        return track
    smoothed = [list(track[0])]
    for t, cx, cy in track[1:]:
        _, pcx, pcy = smoothed[-1]
        if abs(cx - pcx) > max_jump:
            cx = pcx + max_jump * (1 if cx > pcx else -1)
        if abs(cy - pcy) > max_jump:
            cy = pcy + max_jump * (1 if cy > pcy else -1)
        smoothed.append([t, round(pcx + alpha * (cx - pcx), 4), round(pcy + alpha * (cy - pcy), 4)])
    return smoothed


def _facetrack_offset_expr(track, axis, in_dim, eff_dim):
    """Builds an ffmpeg expression (for the crop filter's x or y, using the
    frame timestamp `t`) that linearly interpolates between the tracked
    face-center keyframes, already converted to a clamped pixel crop offset
    in Python (so the runtime expression itself stays simple/robust). axis:
    1 for x-center, 2 for y-center. Falls back to a fixed centered offset
    if there's no track (e.g. no faces were ever found)."""
    centered = max(0.0, (in_dim - eff_dim) / 2)
    if not track:
        return f"{centered:.2f}"

    def offset_for(frac):
        off = frac * in_dim - eff_dim / 2
        return max(0.0, min(in_dim - eff_dim, off))

    pts = sorted(track, key=lambda p: p[0])
    expr = f"{offset_for(pts[-1][axis]):.2f}"
    for i in range(len(pts) - 2, -1, -1):
        t0, t1 = pts[i][0], pts[i + 1][0]
        if t1 <= t0:
            continue
        v0, v1 = offset_for(pts[i][axis]), offset_for(pts[i + 1][axis])
        expr = f"if(lt(t,{t1:.3f}),{v0:.2f}+({v1:.2f}-{v0:.2f})*(t-{t0:.3f})/{(t1 - t0):.4f},{expr})"
    return expr


@render_bp.route("/api/detect_face_track", methods=["POST"])
def api_detect_face_track():
    data = request.json or {}
    clip = CLIPS.get(data.get("clip_id"))
    if not clip:
        return jsonify({"error": "Unknown clip_id"}), 404
    duration = float(clip.get("duration") or 0)
    if duration <= 0:
        _, _, duration = probe(Path(clip["path"]))
        duration = duration or 0

    engine_used = "legacy"
    engine_meta = {}
    track = None
    faces_found = False
    needs_opencv_error = None

    # Try the advanced Enterprise engine first (Kalman filtering, multi-
    # speaker active-speaker switching, scene-cut segmentation, optical-flow
    # fallback). Any failure here — missing deps, a bad/corrupt clip, an
    # engine bug — silently falls through to the simpler legacy detector
    # below instead of failing the request outright.
    try:
        track, faces_found, engine_meta = _detect_face_track_advanced(Path(clip["path"]), duration)
        engine_used = "enterprise"
    except ImportError as e:
        needs_opencv_error = str(e)
        track = None
    except Exception:
        track = None

    if track is None:
        try:
            raw_track = _detect_face_track(Path(clip["path"]), duration)
        except ImportError as e:
            # Neither the advanced engine nor the legacy detector have their
            # dependency installed — surface that clearly to the UI.
            return jsonify({"error": needs_opencv_error or str(e), "needs_opencv": True}), 400
        except Exception as e:
            return jsonify({"error": f"Face detection failed: {e}"}), 400
        faces_found = bool(raw_track)
        track = _smooth_face_track(raw_track) if raw_track else []
        engine_used = "legacy"
        engine_meta = {}

    if not faces_found:
        return jsonify({"track": [], "faces_found": False, "engine": engine_used,
                         "message": "No faces detected in this clip — it'll use your normal fixed/manual-pan crop instead."})
    return jsonify({"track": track, "faces_found": True, "duration": duration, "samples": len(track),
                     "engine": engine_used, "meta": engine_meta})


def compute_auto_ranges(total_dur, clip_len, start_from_one=True):
    ranges = []
    t = 1 if start_from_one else 0
    while t < total_dur - 1:
        e = min(t + clip_len, total_dur)
        if e - t >= 2 or len(ranges) == 0:
            ranges.append((int(t), int(e)))
        t += clip_len
    return ranges


def _resolve_user_file(url):
    """A url can point at /uploaded/<fname> (user uploads), /audio_lib/<fname>
    (the local audio/ folder), or /tts_audio/<fname> (generated voiceovers in
    the temp TTS folder) — resolve to the real path on disk either way."""
    if not url:
        return None
    name = Path(url).name
    if "/audio_lib/" in url:
        return AUDIO_LIB_DIR / name
    if "/tts_audio/" in url:
        return TTS_TEMP_DIR / name
    return UPLOAD_DIR / name


def atempo_chain(speed):
    parts = []
    remaining = speed
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.4f}")
    return ",".join(parts)


# ────────────────────── Auto-fetch subtitles/captions ──────────────────────
# Step 5: pull whatever subtitle/caption tracks the source URL already has
# (manual "subtitles" AND auto-generated "automatic_captions" from yt-dlp)
# so the user doesn't have to type captions by hand — pick a language and
# they land as timed text layers on the timeline.

def _extract_info_for_subs(url):
    """Same auth/cookie fallback chain as resolve_stream(), but we only need
    metadata (subtitle track listing), never picks a format so it's fast."""
    ydl_opts = _ydl_base_opts()
    ydl_opts['skip_download'] = True
    attempts = _auth_attempts()
    last_err = None
    for mode in attempts:
        opts = dict(ydl_opts)
        if mode == "cookies_file":
            opts['cookiefile'] = str(COOKIES_FILE)
        elif mode in COOKIE_BROWSERS:
            opts.pop('cookiefile', None)
            opts['cookiesfrombrowser'] = (mode,)
        elif mode == "default":
            opts.pop('cookiefile', None)
        elif mode == "bypass":
            opts.pop('cookiefile', None)
            opts['extractor_args'] = {'youtube': {'player_client': ['ios', 'android', 'mweb']}}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            _RESOLVED_MODE["mode"] = mode
            return info
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Could not resolve video")


def _pick_track_url(tracks):
    """tracks is yt-dlp's list of {ext, url, name} for one language.
    Prefer vtt/srt (easy to parse) over anything else."""
    if not tracks:
        return None, None
    for want in ("vtt", "srt"):
        for t in tracks:
            if t.get("ext") == want:
                return t.get("url"), want
    t = tracks[0]
    return t.get("url"), t.get("ext")


_TIME_RE = re.compile(r'(\d+):(\d{2}):(\d{2})[.,](\d{1,3})')


def _ts_to_sec(ts):
    m = _TIME_RE.search(ts)
    if not m:
        return None
    h, mi, s, ms = m.groups()
    ms = (ms + "000")[:3]
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000.0


def _parse_subtitle_cues(raw_text):
    """Parses both WebVTT and SRT into a flat list of {start, end, text}
    (seconds). Good enough for our needs — we don't care about styling/
    karaoke word-tags, just plain cue text and its time window."""
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r'\n\s*\n', raw_text.strip())
    cues = []
    tag_re = re.compile(r'<[^>]+>')
    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip() != ""]
        if not lines:
            continue
        arrow_idx = None
        for i, l in enumerate(lines):
            if "-->" in l:
                arrow_idx = i
                break
        if arrow_idx is None:
            continue
        time_line = lines[arrow_idx]
        parts = time_line.split("-->")
        if len(parts) != 2:
            continue
        start = _ts_to_sec(parts[0].strip())
        end = _ts_to_sec(parts[1].strip())
        if start is None or end is None:
            continue
        text_lines = lines[arrow_idx + 1:]
        text = " ".join(text_lines).strip()
        text = tag_re.sub("", text)  # strip <b>, <i>, karaoke <00:00:01.000> tags
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue
        cues.append({"start": round(start, 2), "end": round(end, 2), "text": text})
    return cues


@render_bp.route("/api/subtitles/list", methods=["POST"])
def api_subtitles_list():
    data = request.get_json(force=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        info = _extract_info_for_subs(url)
    except Exception as e:
        return jsonify({"error": f"Could not read video: {e}"}), 400

    langs = []
    seen = set()
    for code, tracks in (info.get("subtitles") or {}).items():
        if code in seen:
            continue
        seen.add(code)
        langs.append({"code": code, "name": code, "auto": False})
    for code, tracks in (info.get("automatic_captions") or {}).items():
        if code in seen:
            continue
        seen.add(code)
        langs.append({"code": code, "name": code, "auto": True})

    if not langs:
        return jsonify({"error": "This video has no subtitles or captions available.", "langs": []})

    # Manual subs first (usually higher quality/accurate), then auto-captions,
    # each group alphabetically.
    langs.sort(key=lambda l: (l["auto"], l["code"]))
    return jsonify({"langs": langs, "title": info.get("title")})


@render_bp.route("/api/subtitles/fetch", methods=["POST"])
def api_subtitles_fetch():
    data = request.get_json(force=True) or {}
    url = (data.get("url") or "").strip()
    code = data.get("code")
    auto = bool(data.get("auto"))
    if not url or not code:
        return jsonify({"error": "Missing url or language code"}), 400
    try:
        info = _extract_info_for_subs(url)
    except Exception as e:
        return jsonify({"error": f"Could not read video: {e}"}), 400

    bucket = (info.get("automatic_captions") if auto else info.get("subtitles")) or {}
    tracks = bucket.get(code)
    track_url, ext = _pick_track_url(tracks)
    if not track_url:
        return jsonify({"error": "That subtitle track is no longer available."}), 400

    try:
        req = urllib.request.Request(track_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return jsonify({"error": f"Could not download subtitle file: {e}"}), 400

    cues = _parse_subtitle_cues(raw)
    if not cues:
        return jsonify({"error": "Subtitle file was empty or in an unsupported format."}), 400

    return jsonify({"cues": cues, "lang": code})


# ───────────────────────────── API: fetch + cut ─────────────────────────────

def _maybe_start_word_captions(url):
    """Video ke liye word-level captions background me fetch karta hai
    (video_id se key hoke), agar already running/done hai to dobara fetch
    nahi karta — same source ke saare clips isi result ko reuse karte hain."""
    m = re.search(r'(?:v=|/shorts/|youtu\.be/)([a-zA-Z0-9_-]{6,})', url)
    video_id = m.group(1) if m else (re.sub(r'[^a-zA-Z0-9_-]', '', url)[-16:] or "unknown")

    existing = WORD_CAPTIONS_JOBS.get(video_id)
    if existing and existing["status"] in ("running", "done"):
        return video_id

    WORD_CAPTIONS_JOBS[video_id] = {"status": "running", "stage": "Fetching captions...", "words": [], "error": None}

    def _worker():
        try:
            words = get_word_level_captions(url, lang="hi")
            WORD_CAPTIONS_JOBS[video_id]["words"] = words
            WORD_CAPTIONS_JOBS[video_id]["status"] = "done"
        except Exception as e:
            WORD_CAPTIONS_JOBS[video_id]["status"] = "error"
            WORD_CAPTIONS_JOBS[video_id]["error"] = str(e)

    threading.Thread(target=_worker, daemon=True).start()
    return video_id


def _run_cut_job(job_id, url, height, mode, clip_len, manual_ranges, start_from_one=True, local_path=None, fetch_word_captions=False):    
    job = JOBS[job_id]
    try:
        video_headers = audio_headers = None
        if local_path:
            # User uploaded a file already on their PC/phone - no yt-dlp,
            # no internet fetch, no external server involved. We just point
            # the existing cut pipeline at the local file on disk.
            description = ''
            tags = []
            video_url = audio_url = str(local_path)
            title = Path(local_path).stem
            _, _, total_dur = probe(Path(local_path))
        else:
            video_url, audio_url, title, total_dur, video_headers, audio_headers, description, tags = resolve_stream(url, max_height=height)
            if fetch_word_captions:
                _maybe_start_word_captions(url)

    except Exception as e:
        job["error"] = f"Could not resolve video: {e}"
        job["done"] = True
        return

    job["title"] = title
    job["description"] = description
    job["tags"] = tags

    if mode == "auto":
        if not total_dur:
            job["error"] = "Could not detect duration for auto mode"
            job["done"] = True
            return
        ranges = compute_auto_ranges(total_dur, clip_len, start_from_one)
    else:
        ranges = manual_ranges

    if not ranges:
        job["error"] = "No clip ranges to cut"
        job["done"] = True
        return

    job["total"] = len(ranges)

    work_jobs = []
    for i, (s, e) in enumerate(ranges, start=1):
        clip_id = uuid.uuid4().hex[:12]
        out_path = PROXY_DIR / f"{clip_id}.mp4"
        work_jobs.append((i, s, e, clip_id, out_path))

    def _do_job(j):
        i, s, e, clip_id, out_path = j
        ok = cut_proxy(video_url, audio_url, s, e, out_path, video_headers, audio_headers)
        return j, ok

    # As each clip finishes cutting it's appended to job["clips"] immediately —
    # the frontend polls and shows each short the moment it's ready, instead
    # of waiting for the whole batch.
    with ThreadPoolExecutor(max_workers=MAX_CUT_WORKERS) as pool:
        futures = [pool.submit(_do_job, j) for j in work_jobs]
        for fut in as_completed(futures):
            (i, s, e, clip_id, out_path), ok = fut.result()
            if ok:
                w, h, dur = probe(out_path)
                CLIPS[clip_id] = {"path": str(out_path), "w": w, "h": h, "duration": dur,
                                   "title": f"{title}_{i:02d}", "start": s, "end": e,
                                   "video_url": video_url, "audio_url": audio_url,
                                   "video_headers": video_headers, "audio_headers": audio_headers,
                                   "page_url": url, "fetch_height": height,
                                   "is_local": bool(local_path)}
                job["clips"].append({"index": i, "clip_id": clip_id, "w": w, "h": h,
                                      "duration": dur, "label": f"Short {i} ({s}s–{e}s)"})
    job["done"] = True


@render_bp.route("/api/fetch_and_cut", methods=["POST"])
def api_fetch_and_cut_start():
    """Starts cutting in a background thread and returns immediately with a
    job_id. Poll /api/cut_status/<job_id> to get each clip the moment it's
    ready (live, one-by-one) instead of waiting for the whole batch."""
    data = request.json

    # user_id is only sent by the hub project (mongo2yt) when it calls this
    # service server-to-server via render_client.py. Someone using RenderDetect
    # directly in the browser (manual URL paste) never sends one, and shouldn't
    # be blocked for it - so default to a shared "local" owner instead of 400ing.
    user_id = (data.get("user_id") or "").strip() or "local"

    url = data.get("url", "").strip()
    local_rel = (data.get("local_path") or "").strip()  # e.g. "/uploaded/myclip.mp4" from /api/upload
    height = str(data.get("quality", "1080"))
    mode = data.get("mode", "auto")
    clip_len = int(data.get("clip_len", 30))
    start_from_one = bool(data.get("start_from_one", True))
    fetch_word_captions = bool(data.get("fetch_word_captions", False))
    manual_ranges = []
    if mode != "auto":
        manual_ranges = [(int(r[0]), int(r[1])) for r in data.get("ranges", [])]

    local_path = None
    if local_rel:
        # Only allow files that /api/upload itself created, under UPLOAD_DIR -
        # never trust an arbitrary filesystem path from the browser.
        fname = Path(local_rel).name
        candidate = UPLOAD_DIR / fname
        if not candidate.exists():
            return jsonify({"error": "Uploaded file not found on server"}), 400
        local_path = candidate
    elif not url:
        return jsonify({"error": "No URL or uploaded file given"}), 400

    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"clips": [], "done": False, "title": "", "description": "", "tags": [], "error": None, "total": 0,"user_id": user_id}
    threading.Thread(target=_run_cut_job,
                      args=(job_id, url, height, mode, clip_len, manual_ranges, start_from_one, local_path, fetch_word_captions),
                      daemon=True).start()
    return jsonify({"job_id": job_id})


@render_bp.route("/api/word_captions/start", methods=["POST"])
def api_word_captions_start():
    data = request.json or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL given"}), 400
    video_id = _maybe_start_word_captions(url)
    return jsonify({"video_id": video_id})


@render_bp.route("/api/word_captions/status/<video_id>")
def api_word_captions_status(video_id):
    job = WORD_CAPTIONS_JOBS.get(video_id)
    if not job:
        return jsonify({"error": "Unknown video_id"}), 404
    if job["status"] == "error":
        return jsonify({"status": "error", "error": job["error"]})
    if job["status"] == "running":
        return jsonify({"status": "running", "stage": job["stage"]})
    return jsonify({"status": "done", "word_count": len(job["words"])})


@render_bp.route("/api/word_captions/apply", methods=["POST"])
def api_word_captions_apply():
    data = request.json or {}
    clip_id = data.get("clip_id")
    video_id = data.get("video_id")
    clip = CLIPS.get(clip_id)
    job = WORD_CAPTIONS_JOBS.get(video_id)
    if not clip:
        return jsonify({"error": "Unknown clip_id"}), 404
    if not job or job["status"] != "done":
        return jsonify({"error": "Word captions not ready yet"}), 400

    clip_start = clip.get("start", 0)
    clip_dur = clip.get("duration")
    layers = []
    for i, w in enumerate(job["words"]):
        s0 = w["start"] - clip_start
        e0 = w["end"] - clip_start
        if e0 <= 0:
            continue
        if clip_dur is not None and s0 >= clip_dur:
            continue
        s0 = max(0, s0)
        e0 = min(clip_dur, e0) if clip_dur is not None else e0
        if e0 <= s0:
            continue
        layers.append({
            "id": f"wcap{int(time.time()*1000)}_{i}",
            "content": w["word"],
            "x": 0.5, "y": 0.85, "size": 18, "color": "#ffffff", "box": False,
            "font": "impact", "style": "meme_classic", "boxColor": "#000000",
            "centerX": True, "enabled": True,
            "start": s0, "end": e0,
            "source": "word_caption",
        })
    return jsonify({"layers": layers})


@render_bp.route("/api/cut_status/<job_id>")
def api_cut_status(job_id):
    """Returns all clips sorted by index, plus done/error flags."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    job_owner = job.get("user_id")
    if job_owner and job_owner != "local":
        # Job was started by the hub for a specific user - only that user
        # (or the hub polling on their behalf) may read its status.
        if job_owner != request.args.get("user_id"):
            return jsonify({"error": "Not authorized"}), 403
    clips = sorted(job["clips"], key=lambda c: c["index"])
    
    
    return jsonify({
        "title": job["title"], "description": job["description"], "tags": job["tags"],
        "error": job["error"], "done": job["done"], "total": job["total"], "clips": clips,
    })


# @render_bp.route("/api/jobs/list")
# def api_jobs_list():
#     return jsonify({
#         "jobs": [{"job_id": jid, "done": j["done"], "error": j["error"]} for jid, j in JOBS.items()]
#     })

@render_bp.route("/api/jobs/list")
def api_jobs_list():
    user_id = request.args.get("user_id", "")
    return jsonify({"jobs": [{"job_id": jid, "done": j["done"], "error": j["error"]}
                              for jid, j in JOBS.items() if j.get("user_id") == user_id]})

@render_bp.route("/audio_lib/<fname>")
def audio_lib_file(fname):
    p = AUDIO_LIB_DIR / fname
    if not p.exists():
        return "Not found", 404
    return send_file(p)


# ---------------------------------------------------------------------------
# Auto-detect audio/video already on the user's own device (PC or Android),
# so people don't have to manually copy files into the audio/ folder.
# ---------------------------------------------------------------------------
import platform
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm")
DEVICE_MEDIA_INDEX = {}  # id -> Path, filled in each time /api/audio_library is scanned


def get_common_media_dirs():
    """Common folders where people already keep music/video, per platform,
    including Android's shared storage path when running as a packaged app."""
    home = Path.home()
    system = platform.system()
    candidates = []
    if system == "Windows":
        candidates += [home / "Music", home / "Downloads", home / "Videos"]
    elif system == "Darwin":
        candidates += [home / "Music", home / "Downloads", home / "Movies"]
    else:
        candidates += [
            Path("/storage/emulated/0/Music"),
            Path("/storage/emulated/0/Download"),
            Path("/storage/emulated/0/Movies"),
            Path("/storage/emulated/0/DCIM"),
            home / "Music", home / "Downloads",
        ]
    return [d for d in candidates if d.exists() and d.is_dir()]


def scan_device_media():
    """Walk common device folders (one level deep) for audio/video files
    and register them so they can be streamed back via /device_media/<id>."""
    found = []
    for folder in get_common_media_dirs():
        try:
            for p in folder.iterdir():
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS + VIDEO_EXTS:
                    fid = uuid.uuid4().hex[:12]
                    DEVICE_MEDIA_INDEX[fid] = p
                    found.append({
                        "id": fid,
                        "name": p.stem,
                        "filename": p.name,
                        "kind": "video" if p.suffix.lower() in VIDEO_EXTS else "audio",
                        "source": "device",
                        "url": f"/device_media/{fid}",
                    })
        except Exception:
            continue
    return found


@render_bp.route("/device_media/<fid>")
def device_media_file(fid):
    p = DEVICE_MEDIA_INDEX.get(fid)
    if not p or not p.exists():
        return "Not found", 404
    return send_file(p)


# ---------------------------------------------------------------------------
# Optional: pull a curated background-music library from a GitHub repo
# (e.g. https://github.com/you/your-audio-repo) instead of bundling mp3s
# inside the packaged app itself. Point manifest_url at a raw JSON file
# shaped like: [{"name": "Chill Beat", "url": "https://raw.githubusercontent.com/.../chill.mp3"}]
# ---------------------------------------------------------------------------
import urllib.request

@render_bp.route("/api/sync_github_audio", methods=["POST"])
def api_sync_github_audio():
    manifest_url = (request.json or {}).get("manifest_url", "").strip()
    if not manifest_url:
        return jsonify({"error": "manifest_url required"}), 400
    try:
        with urllib.request.urlopen(manifest_url, timeout=15) as r:
            manifest = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return jsonify({"error": f"Could not fetch manifest: {e}"}), 400

    added = []
    for item in manifest:
        name = item.get("name") or "track"
        url = item.get("url")
        if not url:
            continue
        safe_name = re.sub(r"[^A-Za-z0-9_\-. ]", "_", name) + ".mp3"
        dest = AUDIO_LIB_DIR / safe_name
        if dest.exists():
            continue
        try:
            urllib.request.urlretrieve(url, dest)
            added.append(safe_name)
        except Exception:
            continue
    return jsonify({"added": added, "total_requested": len(manifest)})


@render_bp.route("/api/audio_library")
def api_audio_library():
    files = []
    for p in sorted(AUDIO_LIB_DIR.glob("*.mp3")):
        files.append({
            "name": p.stem,
            "filename": p.name,
            "source": "bundled",
            "url": f"/audio_lib/{p.name}"
        })
    # merge in whatever audio/video the scan finds already on this device
    files += scan_device_media()
    return jsonify({"files": files})

@render_bp.route("/media/<clip_id>")
def media(clip_id):
    info = CLIPS.get(clip_id)
    if not info:
        return "Not found", 404
    return send_file(info["path"])


@render_bp.route("/uploaded/<fname>")
def uploaded(fname):
    p = UPLOAD_DIR / fname
    if not p.exists():
        return "Not found", 404
    return send_file(p)


@render_bp.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files["file"]
    ext = Path(f.filename).suffix
    fname = uuid.uuid4().hex[:10] + ext
    f.save(UPLOAD_DIR / fname)
    return jsonify({"url": f"/uploaded/{fname}", "filename": fname})


@render_bp.route("/api/tts", methods=["POST"])
def api_tts():
    data = request.json
    voice_key = data.get("voice", "hi_male")
    text = data.get("text", "").strip()
    emotion = data.get("emotion", "neutral")
    if emotion not in EMOTION_PRESETS:
        emotion = "neutral"
    preset = EMOTION_PRESETS[emotion]
    if not text:
        return jsonify({"error": "No text given"}), 400
    engine, voice, tld = TTS_VOICES.get(voice_key, TTS_VOICES["hi_male"])
    fname = f"tts_{uuid.uuid4().hex[:10]}.mp3"
    out_path = TTS_TEMP_DIR / fname

    if engine == "gtts":
        if not GTTS_OK:
            return jsonify({"error": "gTTS not installed. Run: pip install gTTS"}), 400
        try:
            gTTS(text=text, lang=voice, tld=(tld or "com")).save(str(out_path))
            if emotion != "neutral":
                _apply_emotion_to_gtts(out_path, preset)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        if not EDGE_TTS_OK:
            return jsonify({"error": "edge-tts not installed. Run: pip install edge-tts"}), 400

        async def _run():
            comm = edge_tts.Communicate(
                text, voice,
                rate=preset["edge_rate"], volume=preset["edge_volume"], pitch=preset["edge_pitch"]
            )
            await comm.save(str(out_path))

        try:
            asyncio.run(_run())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    entry = {
        "id": fname,
        "url": f"/tts_audio/{fname}",
        "voice_key": voice_key,
        "voice_label": TTS_VOICE_LABELS.get(voice_key, voice_key),
        "emotion": emotion,
        "emotion_label": preset["label"],
        "text": text,
        "text_preview": (text[:80] + "…") if len(text) > 80 else text,
        "created_at": time.time(),
    }
    TTS_LIBRARY[fname] = entry
    return jsonify({"url": entry["url"], "entry": entry})


@render_bp.route("/tts_audio/<fname>")
def tts_audio_file(fname):
    return send_file(str(TTS_TEMP_DIR / fname))


@render_bp.route("/api/tts_library")
def api_tts_library():
    entries = sorted(TTS_LIBRARY.values(), key=lambda e: e["created_at"], reverse=True)
    return jsonify({"entries": entries})


@render_bp.route("/api/tts_library/delete", methods=["POST"])
def api_tts_library_delete():
    data = request.get_json(force=True) or {}
    fname = data.get("id")
    if not fname or fname not in TTS_LIBRARY:
        return jsonify({"error": "Not found"}), 404
    TTS_LIBRARY.pop(fname, None)
    try:
        p = TTS_TEMP_DIR / fname
        if p.exists():
            p.unlink()
    except Exception as e:
        return jsonify({"error": f"Removed from list but couldn't delete file: {e}"}), 200
    return jsonify({"ok": True})


def _apply_emotion_to_gtts(path, preset):
    """gTTS has no built-in emotion/pitch/rate control at all - it's just a
    plain wrapper around Google's translate-TTS endpoint. So to make the same
    emotion dropdown actually do something audible for gTTS voices, we
    post-process the generated mp3 with ffmpeg: atempo shifts speaking speed,
    asetrate+aresample shifts pitch, and a volume filter adjusts loudness -
    the same three knobs edge-tts exposes natively, just applied after the fact."""
    tmp = path.with_suffix(".emo.mp3")
    filt = (f"atempo={preset['gtts_tempo']},"
            f"asetrate=44100*{preset['gtts_pitch']},aresample=44100,"
            f"volume={preset['gtts_vol_db']}dB")
    cmd = [FFMPEG, "-y", "-i", str(path), "-filter:a", filt, str(tmp)]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                   encoding="utf-8", errors="replace", **_no_console_kwargs())
    if tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(path)


# ───────────────────────────── API: export ─────────────────────────────

EXPORT_JOBS = {}

# Caps how many exports actually run their ffmpeg re-encode at once. The
# frontend now fires several exports concurrently (MAX_CONCURRENT_EXPORTS),
# so this is the backend-side safety net that keeps CPU usage sane even if
# more requests land than we want running heavy re-encodes simultaneously.
MAX_EXPORT_WORKERS = min(4, max(2, (os.cpu_count() or 4) // 2))
_EXPORT_SEMAPHORE = threading.Semaphore(MAX_EXPORT_WORKERS)


def run_export_thread(export_id, src, w, h, settings, title, src_duration, source=None):
    def progress_cb(pct):
        if export_id in EXPORT_JOBS:
            EXPORT_JOBS[export_id]["progress"] = pct

    with _EXPORT_SEMAPHORE:
        try:
            results = build_and_run_export(
                src, w, h, settings, title,
                progress_callback=progress_cb,
                src_duration=src_duration,
                source=source
            )
            if export_id in EXPORT_JOBS:
                EXPORT_JOBS[export_id].update({
                    "status": "done",
                    "progress": 100,
                    "results": [
                        {"ratio": r["ratio"], "path": str(r["path"]), "url": f"/api/download/{r['path'].name}"}
                        for r in results
                    ],
                    # purane frontend code ke liye backward-compat — pehla result
                    "path": str(results[0]["path"]) if results else None,
                    "url": f"/api/download/{results[0]['path'].name}" if results else None,
                })
        except Exception as e:
            if export_id in EXPORT_JOBS:
                EXPORT_JOBS[export_id].update({
                    "status": "failed",
                    "error": str(e)
                })


@render_bp.route("/api/export", methods=["POST"])
def api_export():
    data = request.json
    clip_id = data.get("clip_id")
    settings = data.get("settings", {})
    info = CLIPS.get(clip_id)
    if not info:
        return jsonify({"error": "Unknown clip"}), 400

    src = Path(info["path"])
    w, h = info["w"] or 720, info["h"] or 1280
    src_duration = info.get("duration") or 30.0

    # Original-stream info — export ab isse fresh render karega, proxy se
    # nahi, agar available ho (local uploads is liye proxy hi use karte
    # rahenge, unke liye video_url wahi local path hai).
    source = {
        "video_url": info.get("video_url"),
        "audio_url": info.get("audio_url"),
        "video_headers": info.get("video_headers"),
        "audio_headers": info.get("audio_headers"),
        "page_url": info.get("page_url"),
        "fetch_height": info.get("fetch_height"),
        "is_local": info.get("is_local", False),
        "start": info.get("start", 0),
        "end": info.get("end", 0),
    }

    export_id = uuid.uuid4().hex[:12]
    EXPORT_JOBS[export_id] = {
        "status": "processing",
        "progress": 0,
        "path": None,
        "url": None,
        "error": None,
        "clip_id": clip_id,
        "title": info.get("title"),
    }

    threading.Thread(
        target=run_export_thread,
        args=(export_id, src, w, h, settings, info["title"], src_duration, source),
        daemon=True
    ).start()

    return jsonify({"export_id": export_id})


@render_bp.route("/api/export_status/<export_id>")
def api_export_status(export_id):
    job = EXPORT_JOBS.get(export_id)
    if not job:
        return jsonify({"error": "Unknown export job"}), 404
    return jsonify(job)


@render_bp.route("/api/download/<fname>")
def api_download(fname):
    p = OUTPUT_DIR / fname
    if not p.exists():
        return "Not found", 404
    return send_file(p, as_attachment=True)

@render_bp.route("/api/cleanup_clip", methods=["POST"])
def api_cleanup_clip():
    fname = request.json.get("fname", "")
    p = PROXY_DIR / Path(fname).name
    if p.exists():
        p.unlink()
    return jsonify({"ok": True})

# ═══════════════════ Downloader (separate file, imported here) ═══════════════════
# The general "download any video" feature lives entirely in downloader.py —
# it registers itself as a Flask Blueprint, so it runs in this SAME process
# and on this SAME port (no second server, nothing extra to start).
from downloader import downloader_bp, init_downloader
from video_editor_final import editor_bp, init_editor
from publish_module import publish_bp, init_publish

def init_render(main_app):
    """Called from project2/app.py — RenderDetect + downloader + editor +
    publish sab isi function se register hote hain."""
    
    init_downloader(BASE, FFMPEG)
    main_app.register_blueprint(downloader_bp)
    init_editor(BASE, FFMPEG)
    main_app.register_blueprint(editor_bp)
    # Publish Studio needs read-only access to the same in-memory EXPORT_JOBS
    # / CLIPS registries this file owns — passed as callables (not the dicts
    # themselves) so it always sees the live, current state.
    init_publish(BASE, get_export_jobs=lambda: EXPORT_JOBS, get_clips=lambda: CLIPS)
    main_app.register_blueprint(publish_bp)
    main_app.register_blueprint(render_bp)


def _is_trivial_export(s, src_ext):
    """True if the requested settings make no actual change to the video —
    in that case we can stream-copy instead of re-encoding (near-instant)."""
    ratios = s.get("ratios")
    if ratios and len(ratios) > 1:
        return False   # multi-ratio me har ratio ka apna crop chahiye — kabhi trivial nahi
    if float(s.get("speed", 1.0)) != 1.0:
        return False
    if float(s.get("zoom", 1.0)) != 1.0:
        return False
    if abs(float(s.get("contrast", 1.0)) - 1.0) > 1e-3:
        return False
    if abs(float(s.get("saturation", 1.0)) - 1.0) > 1e-3:
        return False
    if abs(float(s.get("brightness", 0.0))) > 1e-3:
        return False
    if bool(s.get("sharpen", False)):
        return False
    if bool(s.get("enhance", False)):
        return False
    if bool(s.get("hflip", False)):
        return False
    if s.get("rotate", "0") != "0":
        return False
    if s.get("regions"):
        return False
    if s.get("text") and s["text"].get("content"):
        return False
    if s.get("texts"):
        return False
    if s.get("logo") and s["logo"].get("url"):
        return False
    if s.get("color_preset", "none") != "none":
        return False
    if abs(float(s.get("pan_x", 0.0))) > 1e-3 or abs(float(s.get("pan_y", 0.0))) > 1e-3:
        return False
    audio_mode = s.get("audio_mode", "original")
    if audio_mode not in ("original",):
        return False
    single_ratio = (ratios[0] if ratios else s.get("resolution", "1080x1920"))
    if single_ratio not in (None, "original"):
        return False
    fmt = s.get("format", "mp4")
    if fmt != src_ext.lstrip("."):
        return False
    return True


def fast_copy_export(src, fmt, title, source=None):
    """Just remux to the output dir — no re-encode (zero quality loss).
    Agar `source` (original stream) available hai aur local upload nahi hai,
    toh seedha wahan se lossless remux karta hai — taaki 'no edits' export
    bhi genuinely lossless ho, proxy ke crf23 bits ko copy karne ke bajaye."""
    out_name = f"{title}_final_{uuid.uuid4().hex[:6]}.{fmt}"
    out_path = OUTPUT_DIR / out_name

    if source and source.get("video_url") and not source.get("is_local"):
        tmp_src = OUTPUT_DIR / f"_src_{uuid.uuid4().hex[:10]}.mp4"
        ok, clamped_start = fetch_source_segment(source, tmp_src)
        if ok:
            trim_offset = source.get("start", 0) - clamped_start
            duration = source.get("end", 0) - source.get("start", 0)
            cmd2 = [FFMPEG, "-y", "-ss", f"{trim_offset:.3f}", "-i", str(tmp_src),
                    "-t", f"{duration:.3f}", "-c", "copy", "-movflags", "+faststart", str(out_path)]
            subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           encoding="utf-8", errors="replace", **_no_console_kwargs())
            tmp_src.unlink(missing_ok=True)
            if out_path.exists() and out_path.stat().st_size > 1000:
                return out_path
            # fetch/trim fail ho gaya (expired URL waghera) -> neeche proxy copy pe fallback

    cmd = [FFMPEG, "-y", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(out_path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", **_no_console_kwargs())
    if not out_path.exists() or out_path.stat().st_size < 1000:
        raise RuntimeError("FFmpeg copy failed:\n" + result.stdout[-1500:])
    return out_path


# ────────────────────── Caption fonts + style presets (Step 3) ──────────────
# Only fonts that are realistically already on a normal Windows/Mac/Linux box
# are offered here - each entry lists candidate file paths across OSes, tried
# in order; the first one that actually exists on this machine wins. If none
# of a family's candidates exist, we silently fall back to whatever generic
# system font build_and_run_export already found (so drawtext never crashes).
FONT_CHOICES = {
    "default":    {"label": "Default (System)",       "candidates": []},
    "arial":      {"label": "Arial",                  "candidates": [
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]},
    "arial_bold": {"label": "Arial Bold",              "candidates": [
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]},
    "impact":     {"label": "Impact (Meme style)",     "candidates": [
        "C:\\Windows\\Fonts\\impact.ttf",
        "/usr/share/fonts/truetype/wine/impact.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
    ]},
    "comic_sans": {"label": "Comic Sans MS",            "candidates": [
        "C:\\Windows\\Fonts\\comic.ttf",
        "C:\\Windows\\Fonts\\comicbd.ttf",
        "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    ]},
    "times":      {"label": "Times New Roman",          "candidates": [
        "C:\\Windows\\Fonts\\times.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]},
    "georgia":    {"label": "Georgia",                  "candidates": [
        "C:\\Windows\\Fonts\\georgia.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ]},
    "verdana":    {"label": "Verdana",                  "candidates": [
        "C:\\Windows\\Fonts\\verdana.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]},
    "courier":    {"label": "Courier New (Typewriter)", "candidates": [
        "C:\\Windows\\Fonts\\cour.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]},
    "trebuchet":  {"label": "Trebuchet MS",              "candidates": [
        "C:\\Windows\\Fonts\\trebuc.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]},
}

# Trending caption-style presets. Each maps to the extra drawtext filter
# fragment that gets appended after fontsize/fontcolor/x/y. box_color (a hex
# string from the UI's color picker) is only actually used by the presets
# that have a background/glow; the others ignore it.
STYLE_PRESETS = {
    "classic_box":   "Classic Box",
    "bold_outline":  "Bold Outline",
    "neon_glow":     "Neon Glow",
    "shadow_pop":    "Drop Shadow",
    "highlight_bar": "Highlight Bar (solid)",
    "outline_only":  "Outline Only",
    "minimal":       "Minimal",
    "meme_classic":  "Meme Classic",
    "gold_glow":     "Gold Glow",
    "gradient_bar":  "Trendy Bar",
}

_FONT_PATH_CACHE = {}


def _resolve_font_path(font_key, fallback_path):
    """Looks up (and caches) the actual font file on disk for a chosen family.
    Falls back to whatever generic system font was already found elsewhere if
    this family isn't installed on the current machine."""
    font_key = font_key or "default"
    if font_key in _FONT_PATH_CACHE:
        cached = _FONT_PATH_CACHE[font_key]
        return cached if cached else fallback_path
    entry = FONT_CHOICES.get(font_key, FONT_CHOICES["default"])
    for p in entry["candidates"]:
        if os.path.exists(p):
            _FONT_PATH_CACHE[font_key] = p
            return p
    _FONT_PATH_CACHE[font_key] = None
    return fallback_path


def _sec_to_ass_time(sec):
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _build_word_captions_ass(word_texts, out_w, out_h, speed=1.0):
    """Saare word-caption words ke liye EK hi .ass subtitle file banata hai —
    style (font/size/color/bold/outline/background) sirf EK jagah define hoti
    hai, har word sirf ek chhota Dialogue line hai. Isse filter_complex me
    sirf ek chhota 'subtitles=...' filter reh jaata hai, chahe words 500 ho ya
    5 — command-line length ka koi issue nahi aata."""
    first = word_texts[0]
    color = (first.get("color") or "#ffffff").lstrip("#")
    if len(color) != 6:
        color = "ffffff"
    r, g, b = color[0:2], color[2:4], color[4:6]
    primary = f"&H00{b}{g}{r}"          # ASS colors BGR order me hote hain
    fontsize = int(first.get("size", 52))
    font_name = "Impact" if first.get("font") == "impact" else "Arial"
    x = int(float(first.get("x", 0.5)) * out_w)
    y = int(float(first.get("y", 0.78)) * out_h)

    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {out_w}\nPlayResY: {out_h}\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: WordCap,{font_name},{fontsize},{primary},&H000000FF,&H00000000,&H00000000,"
        "1,0,0,0,100,100,0,0,1,3,0,5,10,10,10,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = []
    for t in word_texts:
        if not t.get("content"):
            continue
        start = _sec_to_ass_time(float(t.get("start", 0)) / speed)
        end = _sec_to_ass_time(float(t.get("end", 0)) / speed)
        text = t["content"].replace("\\", "\\\\").replace("\n", "\\N").replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{start},{end},WordCap,,0,0,0,,{{\\pos({x},{y})}}{text}")

    ass_path = os.path.join(tempfile.gettempdir(), f"wordcap_{uuid.uuid4().hex[:8]}.ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")
    return ass_path


def _build_style_opts(style_key, box_color_hex):
    """Extra drawtext filter fragment (box/border/shadow/glow) for a caption
    style preset. box_color_hex is the raw '#rrggbb' (or None) from the UI."""
    bc = (box_color_hex or "000000").lstrip("#")
    if len(bc) != 6:
        bc = "000000"
    if style_key == "bold_outline":
        return ":borderw=4:bordercolor=black"
    if style_key == "neon_glow":
        return f":borderw=6:bordercolor=0x{bc}@0.85"
    if style_key == "shadow_pop":
        return ":shadowx=3:shadowy=3:shadowcolor=black@0.8"
    if style_key == "highlight_bar":
        return f":box=1:boxcolor=0x{bc}@0.95:boxborderw=10"
    if style_key == "outline_only":
        return ":borderw=3:bordercolor=black@0.9"
    if style_key == "minimal":
        return ":shadowx=1:shadowy=1:shadowcolor=black@0.5"
    if style_key == "meme_classic":
        return ":borderw=6:bordercolor=black:fontcolor=white"
    if style_key == "gold_glow":
        return ":borderw=3:bordercolor=0xFFD700@0.9:shadowx=2:shadowy=2:shadowcolor=black@0.6"
    if style_key == "gradient_bar":
        return f":box=1:boxcolor=0x{bc}@0.95:boxborderw=10:borderw=2:bordercolor=white@0.7"
    # "classic_box" and any unknown/missing key: same look as the original
    # hardcoded box, but honors a custom color if the user picked one.
    return f":box=1:boxcolor=0x{bc}@0.45:boxborderw=14"


# ────────────────────── Hardware-accelerated export (growth feature) ───────
# Uses the GPU's built-in H.264 encoder when one is available (NVIDIA NVENC,
# Intel QuickSync, AMD AMF, or Apple VideoToolbox) instead of always encoding
# on the CPU with libx264 - this is dramatically faster on a machine that has
# a GPU, with zero setup from the user. Detected once per run and cached.
# Just being *listed* by ffmpeg doesn't guarantee a working GPU is actually
# present, so build_and_run_export always verifies the output file afterward
# and silently retries with plain CPU libx264 if the hardware attempt failed.
_HW_ENCODER_CACHE = {"done": False, "codec": None, "kind": None}


def _detect_hw_encoder():
    if _HW_ENCODER_CACHE["done"]:
        return _HW_ENCODER_CACHE["codec"], _HW_ENCODER_CACHE["kind"]
    _HW_ENCODER_CACHE["done"] = True
    try:
        proc = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", **_no_console_kwargs())
        out = proc.stdout or ""
    except Exception:
        return None, None
    # Preference order: NVIDIA > Intel QuickSync > AMD > Apple VideoToolbox.
    for codec, kind in [("h264_nvenc", "nvenc"), ("h264_qsv", "qsv"),
                         ("h264_amf", "amf"), ("h264_videotoolbox", "videotoolbox")]:
        if codec in out:
            _HW_ENCODER_CACHE["codec"] = codec
            _HW_ENCODER_CACHE["kind"] = kind
            return codec, kind
    return None, None


def _hw_video_args(codec, kind, preset, crf):
    """Maps our x264-style preset/crf onto each hardware encoder's own knobs
    (they don't share libx264's preset/crf vocabulary)."""
    fast_presets = {"ultrafast", "superfast", "veryfast", "faster", "fast"}
    slow_presets = {"slow", "slower", "veryslow"}
    speed_bucket = "fast" if preset in fast_presets else ("slow" if preset in slow_presets else "medium")
    if kind == "nvenc":
        return ["-c:v", codec, "-preset", speed_bucket, "-rc", "vbr",
                "-cq", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p"]
    if kind == "qsv":
        return ["-c:v", codec, "-preset", speed_bucket, "-global_quality", str(crf), "-pix_fmt", "nv12"]
    if kind == "amf":
        return ["-c:v", codec, "-quality", speed_bucket, "-rc", "cqp",
                "-qp_i", str(crf), "-qp_p", str(crf), "-pix_fmt", "yuv420p"]
    if kind == "videotoolbox":
        return ["-c:v", codec, "-q:v", str(max(1, min(100, 100 - crf * 3))), "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", "0"]


def _run_export_attempt(base_cmd, vcodec_args, tail_cmd, progress_callback, target_duration, out_path):
    """Runs one full encode attempt (base_cmd + this attempt's video-codec args
    + tail_cmd) and reports back whether it actually produced a valid file -
    same live-progress-parsing behavior as before, just reusable so hardware
    and CPU attempts share one code path instead of duplicating it."""
    cmd = base_cmd + vcodec_args + tail_cmd
    if progress_callback is None:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 encoding="utf-8", errors="replace", **_no_console_kwargs())
        ok = out_path.exists() and out_path.stat().st_size >= 1000
        return ok, (None if ok else result.stdout[-1500:])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             encoding="utf-8", errors="replace", bufsize=1, universal_newlines=True)
    full_output = []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        full_output.append(line)
        m = re.search(r"time=(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", line)
        if m and target_duration > 0:
            hh, mm, ss = m.groups()
            cur_secs = int(hh) * 3600 + int(mm) * 60 + float(ss)
            pct = min(99, int((cur_secs / target_duration) * 100))
            progress_callback(pct)
    proc.wait()
    ok = proc.returncode == 0 and out_path.exists() and out_path.stat().st_size >= 1000
    stdout_tail = "".join(full_output[-100:]) if full_output else ""
    return ok, (None if ok else f"exit code {proc.returncode}:\n{stdout_tail[-1500:]}")


def build_and_run_export(src, w, h, s, title, progress_callback=None, src_duration=30.0, source=None):
    """Ek clip ke ek ya zyada export-ratios (9:16 / 1:1 / 16:9 / 4:5 ...)
    banata hai. Source (agar remote hai) sirf EK BAAR fetch hota hai,
    saare ratios usi local segment se bante hain — extra network load
    kabhi nahi. Har ratio apna independent crop+scale+encode pass leta hai
    (sequential — CPU-bound kaam hai, parallel se sirf thrash hoga), isliye
    koi ratio doosre se quality "udhaar" nahi leta.
    Returns: list of {"ratio": <label>, "path": <Path>}.
    """
    fmt = s.get("format", "mp4")
    ratios = s.get("ratios") or [s.get("resolution", "1080x1920")]

    # ---- fast path: sirf ek ratio, "original", aur kuch edit nahi hua ----
    if len(ratios) == 1 and ratios[0] in (None, "original") and _is_trivial_export(s, src.suffix):
        return [{"ratio": "original", "path": fast_copy_export(src, fmt, title, source=source)}]

    # ---- source ko SIRF EK BAAR fetch karo (lossless), sab ratios isi ko
    # reuse karenge — yehi is poore design ka core efficiency point hai ----
    trim_offset = 0.0
    tmp_src = None
    if source and source.get("video_url") and not source.get("is_local"):
        tmp_src = OUTPUT_DIR / f"_src_{uuid.uuid4().hex[:10]}.mp4"
        ok, clamped_start = fetch_source_segment(source, tmp_src)
        if ok:
            src = tmp_src
            trim_offset = source.get("start", 0) - clamped_start
            pw, ph, pdur = probe(src)
            if pw and ph:
                w, h = pw, ph
            if pdur:
                src_duration = max(0.1, pdur - trim_offset)
        else:
            tmp_src = None   # fetch fail -> proxy `src` pe hi fallback, cleanup me skip karo

    results = []
    n = len(ratios)
    try:
        for i, ratio in enumerate(ratios):
            out_w, out_h = _resolve_ratio_dims(ratio, w, h)
            safe_ratio = str(ratio or "original").replace(":", "x").replace("/", "x")

            def sub_progress(pct, i=i, n=n):
                if progress_callback:
                    progress_callback(int((i * 100 + pct) / n))

            out_path = _encode_export_variant(
                src, w, h, s, f"{title}_{safe_ratio}",
                trim_offset, src_duration, out_w, out_h,
                progress_callback=sub_progress
            )
            results.append({"ratio": ratio or "original", "path": out_path})
    finally:
        if tmp_src is not None:
            try:
                tmp_src.unlink(missing_ok=True)
            except Exception:
                pass

    return results


def _encode_export_variant(src, w, h, s, title, trim_offset, src_duration, out_w, out_h, progress_callback=None):
    """Ek hi ratio ke liye poora crop+filter+encode pass. `out_w`/`out_h`
    orchestrator (build_and_run_export) se aate hain — is function ko khud
    resolution/ratio decide nahi karna, na hi source dobara fetch karna
    (wo sab ek hi baar orchestrator me ho chuka hota hai)."""
    fmt = s.get("format", "mp4")
    speed = float(s.get("speed", 1.0))
    zoom = float(s.get("zoom", 1.0))
    pan_x = float(s.get("pan_x", 0.0))   # -1..1, manual drag-to-pan inside the zoomed crop
    pan_y = float(s.get("pan_y", 0.0))
    mute = bool(s.get("mute", False))
    contrast = float(s.get("contrast", 1.0))
    saturation = float(s.get("saturation", 1.0))
    brightness = float(s.get("brightness", 0.0))
    sharpen = bool(s.get("sharpen", False))
    enhance = bool(s.get("enhance", False))  # extra detail/denoise pass, keeps quality from breaking on upscale
    hflip = bool(s.get("hflip", False))
    rotate = s.get("rotate", "0")
    crf = int(s.get("crf", 18))
    preset = s.get("preset", "medium")
    regions = s.get("regions", [])
    # texts: prefer the new multi-text array; fall back to the legacy single `text` dict
    texts = s.get("texts") or ([s["text"]] if s.get("text") and s["text"].get("content") else [])
    logo = s.get("logo", None)
    audio_mode = s.get("audio_mode", "original")  # original | mute | replace | tts
    audio_file_url = s.get("audio_file_url")
    tts_url = s.get("tts_url")
    tts_mix = bool(s.get("tts_mix", False))
    preset_name = s.get("color_preset", "none")
    look = COLOR_PRESETS.get(preset_name, {})
    # Note: We do NOT multiply or offset contrast, saturation, or brightness by look values here,
    # because the client-side sliders already contain the preset's exact values.
    # We still use 'look' below to apply advanced look-specific filters like curves, vignette, and mixers.

    if enhance:
        contrast = contrast * 1.04
        saturation = saturation * 1.08

    ext_codec = {"mp4": ("mp4", "libx264", "aac"), "mov": ("mov", "libx264", "aac"),
                 "webm": ("webm", "libvpx-vp9", "libopus")}[fmt]
    ext, vcodec, acodec = ext_codec

    eff_w = w / zoom if zoom > 1.0 else w
    eff_h = h / zoom if zoom > 1.0 else h
    # manual pan moves the crop window inside the available slack instead of
    # always centering it — clamped so we never crop outside the source frame
    max_off_x = max(0.0, (w - eff_w) / 2)
    max_off_y = max(0.0, (h - eff_h) / 2)
    crop_x_off = max_off_x * (1 - pan_x)   # pan_x: -1 = full left, 0 = center, 1 = full right
    crop_y_off = max_off_y * (1 - pan_y)

    # zoomed region (eff_w x eff_h) ka aspect target (out_w:out_h) se match
    # na ho toh (jaise 16:9 source ko 9:16 short bana rahe ho) crop box ko
    # scale se PEHLE hi sahi aspect me shrink karo — warna scale filter
    # image ko stretch/squash kar dega.
    target_ar = out_w / out_h
    src_ar = eff_w / eff_h
    if src_ar > target_ar:
        crop_h = eff_h
        crop_w = eff_h * target_ar
    else:
        crop_w = eff_w
        crop_h = eff_w / target_ar
    crop_x_off += (eff_w - crop_w) / 2
    crop_y_off += (eff_h - crop_h) / 2

    scale_x = out_w / crop_w
    scale_y = out_h / crop_h

    # Auto smart-reframe: if a face track was captured for this clip, follow
    # it instead of the static/manual-pan offset above - this is what
    # actually makes the crop "chase" the speaker over time.
    face_track = s.get("face_track")
    use_face_track = bool(face_track) and zoom > 1.0

    vf = []
    if use_face_track:
        x_off_expr = _facetrack_offset_expr(face_track, 1, w, crop_w)
        y_off_expr = _facetrack_offset_expr(face_track, 2, h, crop_h)
        vf.append(f"crop={int(crop_w)}:{int(crop_h)}:{x_off_expr}:{y_off_expr}")
    else:
        vf.append(f"crop={int(crop_w)}:{int(crop_h)}:{int(crop_x_off)}:{int(crop_y_off)}")

    # scale ab HAMESHA lagega — zoom ho ya na ho, resolution setting ab
    # kabhi ignore nahi hogi
    vf.append(f"scale={out_w}:{out_h}:flags=lanczos+accurate_rnd")

    # Rotate & Mirror Video (Horizontal Flip)
    if hflip:
        vf.append("hflip")
    if rotate == "90":
        vf.append("transpose=1")
    elif rotate == "180":
        vf.append("transpose=2,transpose=2")
    elif rotate == "270":
        vf.append("transpose=2")

    # Dynamic sharpening on zoom to maintain clarity and avoid pixelation
    if zoom > 1.0:
        zoom_sharp = min(1.2, 0.4 * (zoom - 1.0))
        if zoom_sharp > 0.05:
            vf.append(f"unsharp=5:5:{zoom_sharp:.2f}:5:5:0.0")

    if enhance:
        # Professional-grade 4K-upscale enhancement:
        # 1. Advanced 3D denoising to clear compression artifacts from source/proxy
        vf.append("hqdn3d=2.0:2.0:6.0:6.0")
        # 2. Stronger unsharp mask for pristine edge clarity without haloing
        vf.append("unsharp=5:5:1.5:5:5:0.8")

    eq_parts = []
    if abs(contrast - 1.0) > 1e-3:
        eq_parts.append(f"contrast={contrast:.3f}")
    if abs(saturation - 1.0) > 1e-3:
        eq_parts.append(f"saturation={saturation:.3f}")
    if abs(brightness) > 1e-3:
        eq_parts.append(f"brightness={brightness:.3f}")
    if eq_parts:
        vf.append("eq=" + ":".join(eq_parts))
    if look.get("curves"):
        vf.append(f"curves=preset={look['curves']}")
    if look.get("colorchannelmixer"):
        vf.append(f"colorchannelmixer={look['colorchannelmixer']}")
    if s.get("vignette") or look.get("vignette"):
        vf.append("vignette=PI/4")
    if s.get("film_grain"):
        vf.append("noise=alls=3:allf=t")
    if sharpen:
        vf.append("unsharp=5:5:0.6:5:5:0.0")
    if abs(speed - 1.0) > 1e-3:
        vf.append(f"setpts=PTS/{speed:.4f}")

    cmd = [FFMPEG, "-y", "-i", str(src)]
    if trim_offset > 0.01:
        # src ab padded lossless segment hai (source se seedha) — isko yahan
        # exact clip boundary pe trim karo. Output-seeking hai (post -i),
        # isliye frame-accurate hai, aur cost bhi nahi lagti kyunki har frame
        # already decode/filter/encode ho hi raha hai is pass me.
        cmd += ["-ss", f"{trim_offset:.3f}", "-t", f"{src_duration:.3f}"]
    extra_audio_idx = None
    input_count = 1

    if audio_mode == "tts" and tts_url:
        local_tts = _resolve_user_file(tts_url)
        cmd += ["-i", str(local_tts)]
        extra_audio_idx = input_count
        input_count += 1
    elif audio_mode == "replace" and audio_file_url:
        local_audio = _resolve_user_file(audio_file_url)
        # Infinite looping of replaced audio to handle video length larger than audio length
        cmd += ["-stream_loop", "-1", "-i", str(local_audio)]
        extra_audio_idx = input_count
        input_count += 1

    logo_idx = None
    if logo and logo.get("url"):
        local_logo = _resolve_user_file(logo["url"])
        cmd += ["-i", str(local_logo)]
        logo_idx = input_count
        input_count += 1

    emoji_inputs = []
    for r in regions:
        if r.get("kind") == "emoji" and r.get("emoji_url"):
            local_emoji = UPLOAD_DIR / Path(r["emoji_url"]).name
            cmd += ["-i", str(local_emoji)]
            emoji_inputs.append((r, input_count))
            input_count += 1

    fc = [f"[0:v]{','.join(vf) if vf else 'null'}[vbase]"]
    cur = "vbase"
    idx = 0
    emoji_map = {id(r): i for r, i in emoji_inputs}
    for r in regions:
        idx += 1
        rx = r["x"] * w - crop_x_off
        ry = r["y"] * h - crop_y_off
        rw = r["w"] * w
        rh = r["h"] * h
        x = int(rx * scale_x); y = int(ry * scale_y)
        bw = max(2, int(rw * scale_x)); bh = max(2, int(rh * scale_y))
        kind = r.get("kind")
        nxt = f"v{idx}"
        if kind == "blur":
            fc.append(f"[{cur}]split[{nxt}m][{nxt}c];[{nxt}c]crop={bw}:{bh}:{x}:{y},boxblur=20:2[{nxt}b];"
                       f"[{nxt}m][{nxt}b]overlay={x}:{y}[{nxt}]")
            cur = nxt
        elif kind == "black":
            fc.append(f"[{cur}]drawbox=x={x}:y={y}:w={bw}:h={bh}:color=black@1.0:t=fill[{nxt}]")
            cur = nxt
        elif kind == "emoji":
            iidx = emoji_map.get(id(r))
            if iidx is not None:
                fc.append(f"[{iidx}]scale={bw}:{bh}[{nxt}e]")
                fc.append(f"[{cur}][{nxt}e]overlay={x}:{y}[{nxt}]")
                cur = nxt
        elif kind in ("rect", "circle"):
            rgb = (r.get("color") or "#ff3b30").lstrip("#")
            fc.append(f"[{cur}]drawbox=x={x}:y={y}:w={bw}:h={bh}:color={rgb}@1.0:t=4[{nxt}]")
            cur = nxt
        elif kind == "arrow":
            rgb = (r.get("color") or "#ff3b30").lstrip("#")
            fc.append(f"[{cur}]drawbox=x={x}:y={y}:w={bw}:h=4:color={rgb}@1.0:t=fill[{nxt}]")
            cur = nxt

    if logo_idx is not None:
        lw = float(logo.get("w", 0.18)) * out_w
        lop = float(logo.get("opacity", 1.0))
        lx = float(logo.get("x", 0.78)) * out_w
        ly = float(logo.get("y", 0.04)) * out_h
        fc.append(f"[{logo_idx}]scale={int(lw)}:-2,format=rgba,colorchannelmixer=aa={lop:.2f}[logo]")
        fc.append(f"[{cur}][logo]overlay={int(lx)}:{int(ly)}[vlogo]")
        cur = "vlogo"

    # Try to find system font to prevent drawtext crashes on headless environments
    sys_font = None
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf"
    ]:
        if os.path.exists(p):
            sys_font = p
            break
    if not sys_font:
        try:
            for p in Path("/usr/share/fonts").glob("**/*.ttf"):
                sys_font = str(p)
                break
        except Exception:
            pass

    stage_w = float(s.get("stage_w", 360.0))
    stage_h = float(s.get("stage_h", 640.0))
    if stage_h <= 0: stage_h = 640.0

    word_caption_texts = [t for t in texts if t and t.get("content") and t.get("source") == "word_caption"]
    other_texts = [t for t in texts if not (t and t.get("content") and t.get("source") == "word_caption")]

    for t in other_texts:
        if not t or not t.get("content"):
            continue
        esc = t["content"].replace("'", "'\\''")
        # Word-by-word karaoke captions center themselves horizontally (each
        # word is a different width, so a fixed x wouldn't stay centered) -
        # ffmpeg's drawtext can do this itself via the text_w expression.
        x_expr = "(w-text_w)/2" if t.get("center_x") else str(int(float(t.get("x", 0.1)) * out_w))
        ty = int(float(t.get("y", 0.8)) * out_h)
        orig_size = int(t.get("size", 56))
        # Scale size proportionally based on stage height to match preview exactly
        size = int(orig_size * (out_h / stage_h))
        color = (t.get("color") or "#ffffff").lstrip("#")
        style_key = t.get("style")
        if style_key and style_key in STYLE_PRESETS:
            box = _build_style_opts(style_key, t.get("box_color"))
        elif style_key:
            # Unknown style key (e.g. corrupted/old data) - safest fallback.
            box = _build_style_opts("classic_box", t.get("box_color"))
        else:
            # No style field at all -> legacy layer (pre-Step-3 save or an
            # auto-fetched caption). Honor its old plain boolean "box" flag.
            box = ":box=1:boxcolor=black@0.45:boxborderw=14" if t.get("box", True) else ""
        font_path = _resolve_font_path(t.get("font"), sys_font)
        if font_path:
            escaped_font = font_path.replace("\\", "/").replace(":", "\\:")
            font_opt = f":fontfile='{escaped_font}'"
        else:
            font_opt = ""
        nxt = "vtext" + uuid.uuid4().hex[:6]
        # Auto-fetched captions (Step 5) carry start/end (seconds, already
        # clip-local) so they only show during their own time window;
        # manually-added text layers have no start/end and stay always-on.
        enable_opt = ""
        if t.get("start") is not None and t.get("end") is not None:
            enable_opt = f":enable='between(t,{float(t['start'])/speed:.3f},{float(t['end'])/speed:.3f})'"
        fc.append(f"[{cur}]drawtext=text='{esc}':fontsize={size}:fontcolor='#{color}':x={x_expr}:y={ty}{box}{font_opt}{enable_opt}[{nxt}]")
        cur = nxt

    if word_caption_texts:
        ass_path = _build_word_captions_ass(word_caption_texts, out_w, out_h, speed)
        ass_escaped = ass_path.replace("\\", "/").replace(":", "\\:")
        nxt = "vwordcap" + uuid.uuid4().hex[:6]
        fc.append(f"[{cur}]subtitles=filename='{ass_escaped}'[{nxt}]")
        cur = nxt

    filter_complex = ";".join(fc)
    cmd += ["-filter_complex", filter_complex, "-map", f"[{cur}]"]

    pitch_af = None
    if s.get("audio_pitch"):
        # We can shift pitch slightly (e.g. 1.025x which is about +40 cents, completely natural for humans but breaks audio hashing algorithms)
        pitch_af = f"asetrate=44100*1.025,atempo={1/1.025:.4f}"

    if audio_mode == "mute":
        cmd += ["-an"]
    elif extra_audio_idx is not None:
        if audio_mode == "tts" and tts_mix:
            a0 = "[0:a]volume=0.25"
            if abs(speed - 1.0) > 1e-3:
                a0 += "," + atempo_chain(speed)   # keep original track in sync with sped-up video
            if pitch_af:
                a0 += "," + pitch_af
            a0 += "[a0]"
            cmd += ["-filter_complex:a",
                    f"{a0};[{extra_audio_idx}:a]volume=1.0[a1];[a0][a1]amix=inputs=2:duration=longest[aout]"]
            cmd += ["-map", "[aout]"]
        else:
            # replaced file / TTS-only audio is an independent track — it must
            # play at its own normal speed regardless of video speed changes
            cmd += ["-map", f"{extra_audio_idx}:a"]
            if pitch_af:
                cmd += ["-af", pitch_af]
    else:
        cmd += ["-map", "0:a?"]
        af_parts = []
        if pitch_af:
            af_parts.append(pitch_af)
        if abs(speed - 1.0) > 1e-3:
            af_parts.append(atempo_chain(speed))
        if af_parts:
            cmd += ["-af", ",".join(af_parts)]

    out_name = f"{title}_final_{uuid.uuid4().hex[:6]}.{ext}"
    out_path = OUTPUT_DIR / out_name

    base_cmd = cmd  # everything built so far: input(s), filters, maps
    audio_tail = ["-c:a", acodec, "-b:a", "160k"] if audio_mode != "mute" else []
    target_duration = src_duration / speed if speed > 0 else src_duration
    tail_cmd = audio_tail + ["-t", f"{target_duration:.3f}", "-movflags", "+faststart", str(out_path)]

    # GPU acceleration only applies to h264 (mp4/mov) - vp9/webm always CPU.
    # Defaults on; the Export tab lets the user turn it off if they'd rather
    # always use the CPU encoder (e.g. for exact quality parity across runs).
    want_hw = bool(s.get("hw_accel", True)) and vcodec == "libx264"
    hw_codec, hw_kind = _detect_hw_encoder() if want_hw else (None, None)

    attempts = []
    if hw_codec:
        attempts.append(_hw_video_args(hw_codec, hw_kind, preset, crf))
    attempts.append(["-c:v", vcodec, "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", "0"])

    ok, last_err = False, None
    for vcodec_args in attempts:
        if out_path.exists():
            try: out_path.unlink()
            except Exception: pass
        ok, last_err = _run_export_attempt(base_cmd, vcodec_args, tail_cmd, progress_callback, target_duration, out_path)
        if ok:
            break
    if not ok:
        raise RuntimeError(f"FFmpeg failed after {len(attempts)} attempt(s):\n{last_err}")

    return out_path


# ───────────────────────────── Frontend ─────────────────────────────

@render_bp.route("/render")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AutoShortAi</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
:root{
  --bg:#0b0d12; --panel:#13151c; --panel2:#191c25; --border:#262a36;
  --text:#eef0f6; --dim:#8a90a4; --accent:#6e5bff; --accent2:#22d3c4;
  --grad: linear-gradient(135deg,#6e5bff,#22d3c4);
  --btn-text:#0a0a0f;
  --danger:#ff5a6e; --warn:#ffb84d; --radius:16px;
}
/* ── Theme presets — switch instantly via the 🎨 menu top-right ───────── */
html[data-theme="light"]{
  --bg:#f4f5f9; --panel:#ffffff; --panel2:#eef0f6; --border:#dde1ea;
  --text:#171923; --dim:#5b6472; --accent:#5b46e0; --accent2:#0891b2;
  --grad: linear-gradient(135deg,#5b46e0,#0891b2);
  --btn-text:#ffffff; --danger:#e0324a; --warn:#c47a06;
}
html[data-theme="midnight-blue"]{
  --bg:#050a16; --panel:#0c1526; --panel2:#12213a; --border:#1f3358;
  --text:#e8f1ff; --dim:#7f93b8; --accent:#3b82f6; --accent2:#22d3ee;
  --grad: linear-gradient(135deg,#3b82f6,#22d3ee);
  --btn-text:#04101f;
}
html[data-theme="forest"]{
  --bg:#0a130d; --panel:#0f1e14; --panel2:#16301f; --border:#234a30;
  --text:#eafbea; --dim:#86b399; --accent:#22c55e; --accent2:#a3e635;
  --grad: linear-gradient(135deg,#16a34a,#a3e635);
  --btn-text:#06170a;
}
html[data-theme="sunset"]{
  --bg:#1a0e12; --panel:#241419; --panel2:#331d27; --border:#4a2635;
  --text:#fdf1ee; --dim:#cb9aa3; --accent:#fb7185; --accent2:#f59e0b;
  --grad: linear-gradient(135deg,#fb7185,#f59e0b);
  --btn-text:#1a0e12;
}
html[data-theme="ocean"]{
  --bg:#061319; --panel:#0b1f27; --panel2:#123039; --border:#1e4a56;
  --text:#e6fbff; --dim:#7fb8c4; --accent:#06b6d4; --accent2:#6366f1;
  --grad: linear-gradient(135deg,#06b6d4,#6366f1);
  --btn-text:#031015;
}
html[data-theme="rose"]{
  --bg:#170f14; --panel:#221720; --panel2:#2e1e2b; --border:#4a2f45;
  --text:#fdeef5; --dim:#c79ab0; --accent:#ec4899; --accent2:#f472b6;
  --grad: linear-gradient(135deg,#ec4899,#f472b6);
  --btn-text:#170f14;
}
html[data-theme="amber"]{
  --bg:#160f05; --panel:#221708; --panel2:#33230d; --border:#4d3416;
  --text:#fdf4e3; --dim:#c9a877; --accent:#f59e0b; --accent2:#eab308;
  --grad: linear-gradient(135deg,#f59e0b,#eab308);
  --btn-text:#160f05;
}
html[data-theme="slate"]{
  --bg:#101215; --panel:#181b1f; --panel2:#20242a; --border:#333941;
  --text:#eceef1; --dim:#9aa2ad; --accent:#64748b; --accent2:#94a3b8;
  --grad: linear-gradient(135deg,#475569,#94a3b8);
  --btn-text:#0b0d0f;
}
html[data-theme="cyberpunk"]{
  --bg:#0a0014; --panel:#150124; --panel2:#210235; --border:#4a0a6b;
  --text:#f5e6ff; --dim:#b083d6; --accent:#e91e9c; --accent2:#00f0ff;
  --grad: linear-gradient(135deg,#e91e9c,#00f0ff);
  --btn-text:#0a0014;
}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;}
.topbar{display:flex;align-items:center;gap:14px;padding:18px 28px;border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,var(--panel),var(--bg));}
.logo{font-weight:800;font-size:20px;background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent;}
.sub{color:var(--dim);font-size:13px;}
.nav-tabs{display:flex; gap:8px; margin-left:24px;}
.nav-tab-btn{
  background:var(--panel2); border:1px solid var(--border); color:var(--text); border-radius:10px;
  padding:9px 16px; font-size:13px; font-weight:700; cursor:pointer; transition:.15s;
}
.nav-tab-btn:hover{border-color:var(--accent);}
.nav-tab-btn.active{background:var(--grad); color:var(--btn-text); border-color:transparent;}
@media (max-width:760px){
  .nav-tabs{margin-left:0; order:3; width:100%; margin-top:10px;}
}

/* ── Downloader feature ─────────────────────────────────────────────── */
.dl-info-row{display:flex; gap:22px; flex-wrap:wrap; align-items:flex-start;}
.dl-thumb-wrap{flex:0 0 auto; width:320px; max-width:100%;}
.dl-thumb-main{
  width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:12px;
  border:1px solid var(--border); background:var(--panel2);
}
.dl-thumb-strip{display:flex; gap:8px; margin-top:8px;}
.dl-thumb-strip img{
  width:64px; height:40px; object-fit:cover; border-radius:6px; cursor:pointer;
  border:1px solid var(--border); opacity:.75; transition:.15s;
}
.dl-thumb-strip img:hover{opacity:1; border-color:var(--accent);}
.dl-meta{flex:1; min-width:260px;}
.dl-title{margin:0 0 10px 0; font-size:19px; font-weight:800; color:var(--text); line-height:1.35;}
.dl-meta-badges{display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px;}
.dl-meta-badges span{
  background:var(--panel2); border:1px solid var(--border); color:var(--text);
  padding:5px 10px; border-radius:8px; font-size:12px; font-weight:700;
}
.dl-copy-row{display:flex; gap:8px;}
.dl-copy-row input{
  flex:1; padding:10px 12px; border-radius:10px; border:1px solid var(--border);
  background:var(--panel2); color:var(--text); font-size:12px;
}
.dl-section-title{margin:22px 0 10px 0; font-size:15px; font-weight:800; color:var(--text);}
.dl-formats-table-wrap{overflow-x:auto; border:1px solid var(--border); border-radius:12px;}
.dl-formats-table{width:100%; border-collapse:collapse; min-width:520px;}
.dl-formats-table th{
  text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.4px;
  color:var(--dim); padding:10px 12px; background:var(--panel2); border-bottom:1px solid var(--border);
}
.dl-formats-table td{padding:10px 12px; font-size:13px; color:var(--text); border-bottom:1px solid var(--border);}
.dl-formats-table tr:last-child td{border-bottom:none;}
.dl-formats-table tr:hover td{background:rgba(255,255,255,0.03);}
.dl-fmt-btn{
  background:var(--grad); color:var(--btn-text); border:none; border-radius:8px;
  padding:7px 14px; font-size:12px; font-weight:800; cursor:pointer; white-space:nowrap;
}
.dl-fmt-btn:hover{filter:brightness(1.08);}
.dl-progress-stats{
  display:grid; grid-template-columns:repeat(auto-fit, minmax(130px,1fr)); gap:10px; margin-bottom:14px;
}
.dl-stat{
  background:var(--panel2); border:1px solid var(--border); border-radius:10px; padding:10px 12px;
  display:flex; flex-direction:column; gap:4px;
}
.dl-stat-label{font-size:10px; text-transform:uppercase; letter-spacing:.4px; color:var(--dim);}
.dl-stat-val{font-size:16px; font-weight:800; color:var(--text);}
@media (max-width:600px){
  .dl-thumb-wrap{width:100%;}
}

.dl-inline-status{
  margin-top:12px; padding:10px 14px; border-radius:10px; font-size:13px; font-weight:600;
  background:var(--panel2); border:1px solid var(--border); color:var(--text);
}
.dl-title-row{display:flex; align-items:flex-start; gap:8px;}
.dl-copy-icon-btn{
  background:var(--panel2); border:1px solid var(--border); color:var(--text);
  border-radius:8px; width:30px; height:30px; flex:0 0 auto; cursor:pointer; font-size:13px;
  display:flex; align-items:center; justify-content:center; transition:.15s;
}
.dl-copy-icon-btn:hover{border-color:var(--accent); filter:brightness(1.1);}
.dl-details-grid{
  display:grid; grid-template-columns:repeat(auto-fill, minmax(190px,1fr)); gap:10px; margin-bottom:6px;
}
.dl-detail-item{
  background:var(--panel2); border:1px solid var(--border); border-radius:10px; padding:9px 12px;
  display:flex; flex-direction:column; gap:3px; position:relative;
}
.dl-detail-label{font-size:10px; text-transform:uppercase; letter-spacing:.4px; color:var(--dim);}
.dl-detail-val{font-size:13.5px; font-weight:700; color:var(--text); word-break:break-word; padding-right:22px;}
.dl-detail-item .dl-copy-icon-btn{
  position:absolute; top:6px; right:6px; width:22px; height:22px; font-size:11px; background:transparent; border:none;
}
.dl-desc-wrap{margin:16px 0; border:1px solid var(--border); border-radius:10px; overflow:hidden;}
.dl-desc-head{
  display:flex; justify-content:space-between; align-items:center; padding:8px 12px;
  background:var(--panel2); font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.4px; color:var(--dim);
}
.dl-desc-text{
  padding:12px 14px; font-size:13px; line-height:1.6; color:var(--text); white-space:pre-wrap;
}
.dl-desc-text.collapsed{max-height:70px; overflow:hidden; mask-image:linear-gradient(to bottom, black 60%, transparent 100%);}
.dl-chips-wrap{margin-bottom:14px;}
.dl-chips-label{font-size:11px; text-transform:uppercase; letter-spacing:.4px; color:var(--dim); display:block; margin-bottom:6px;}
.dl-chips{display:flex; flex-wrap:wrap; gap:6px;}
.dl-chips span{
  background:var(--panel2); border:1px solid var(--border); color:var(--text);
  padding:4px 10px; border-radius:999px; font-size:11.5px; font-weight:600;
}

/* ── Modern download progress: step tracker instead of a raw scrolling log ── */
.dl-stepper{display:flex; align-items:flex-start; margin-bottom:18px; gap:0;}
.dl-step{display:flex; flex-direction:column; align-items:center; gap:6px; flex:0 0 auto; width:80px;}
.dl-step-ico{
  width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  background:var(--panel2); border:2px solid var(--border); font-size:16px; opacity:.5; transition:.2s;
}
.dl-step-label{font-size:10.5px; font-weight:700; color:var(--dim); text-align:center;}
.dl-step.active .dl-step-ico{border-color:var(--accent); opacity:1; box-shadow:0 0 0 4px rgba(110,91,255,0.15); animation:dlpulse 1.4s ease-in-out infinite;}
.dl-step.active .dl-step-label{color:var(--text);}
.dl-step.complete .dl-step-ico{background:var(--grad); border-color:transparent; opacity:1;}
.dl-step.complete .dl-step-label{color:var(--text);}
@keyframes dlpulse{ 0%,100%{box-shadow:0 0 0 4px rgba(110,91,255,0.15);} 50%{box-shadow:0 0 0 8px rgba(110,91,255,0.05);} }
.dl-step-line{flex:1 1 auto; height:2px; background:var(--border); margin-top:19px; min-width:16px;}
.dl-step-line.complete{background:var(--accent, #6e5bff);}
.dl-status-line{text-align:center; font-size:13px; font-weight:600; color:var(--dim); margin-top:4px;}
.dl-error-banner{
  margin-top:14px; padding:12px 14px; border-radius:10px; background:rgba(220,53,69,0.12);
  border:1px solid rgba(220,53,69,0.4); color:#ff6b7a; font-size:13px; font-weight:600;
}
@media (max-width:600px){
  .dl-step{width:60px;}
  .dl-step-label{font-size:9px;}
}

.theme-picker{margin-left:auto;position:relative;}
.theme-btn{background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:10px;
  padding:9px 16px;font-size:13px;cursor:pointer;font-weight:600;}
.theme-btn:hover{border-color:var(--accent);}
.theme-menu{position:absolute;top:calc(100% + 8px);right:0;background:var(--panel);border:1px solid var(--border);
  border-radius:12px;padding:6px;min-width:180px;max-height:360px;overflow-y:auto;box-shadow:0 12px 32px rgba(0,0,0,0.45);z-index:1000;}
.theme-menu.hidden{display:none;}
.theme-opt{padding:10px 12px;border-radius:8px;font-size:13px;color:var(--text);cursor:pointer;display:flex;align-items:center;gap:8px;}
.theme-opt:hover{background:var(--panel2);}
.theme-opt.active{background:var(--panel2);font-weight:700;}
.theme-dot{width:14px;height:14px;border-radius:50%;flex-shrink:0;}
.wrap{max-width:1500px;margin:0 auto;padding:24px;}
.card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-bottom:20px;}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;}
label{display:block;font-size:12px;color:var(--dim);margin-bottom:6px;}
input[type=text],input[type=number],textarea,select{
  background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:10px;
  padding:10px 12px;font-size:14px;width:100%;}
textarea{resize:vertical;}
button{cursor:pointer;border:none;border-radius:10px;padding:10px 18px;font-size:14px;font-weight:600;
  background:var(--panel2);color:var(--text);border:1px solid var(--border);transition:.15s;}
button:hover{border-color:var(--accent);}
.btn-grad{background:var(--grad);color:#0a0a0f;border:none;}
.btn-grad:hover{filter:brightness(1.08);}
.pill{display:inline-flex;gap:6px;align-items:center;background:var(--panel2);border:1px solid var(--border);
  border-radius:999px;padding:6px 12px;font-size:12px;color:var(--dim);}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:14px;margin-top:14px;}
.clip-card{background:var(--panel2);border:1px solid var(--border);border-radius:14px;overflow:hidden;cursor:pointer;
  transition:.15s;}
.clip-card:hover{border-color:var(--accent);transform:translateY(-2px);}
.clip-card video{width:100%;display:block;aspect-ratio:9/16;object-fit:cover;background:#000;}
.clip-card .lbl{padding:8px 10px;font-size:12px;color:var(--dim);}
.editor{display:none;gap:20px;}
.editor.active{display:grid;grid-template-columns:380px 1fr;}
.stage-col{display:flex;flex-direction:column;align-items:center;gap:12px;}
.stage{position:relative;background:#000;border-radius:14px;overflow:hidden;width:100%;max-width:380px;aspect-ratio:9/16;touch-action:none;}
.stage video{display:block;width:100%;height:100%;object-fit:cover;}
.overlay-layer{position:absolute;inset:0;}
.ov-region{position:absolute;border:2px dashed var(--accent2);box-sizing:border-box;cursor:move;}
.ov-region.kind-black{background:#000;border-style:solid;border-color:#ff5a6e;}
.ov-region.kind-blur{backdrop-filter:blur(12px);border-color:#67b7f0;}
.ov-region.kind-emoji{border:none;background-size:contain;background-repeat:no-repeat;background-position:center;}
.ov-region.kind-rect{background:transparent;border-style:solid;border-width:3px;}
.ov-region.kind-circle{background:transparent;border-style:solid;border-width:3px;border-radius:50%;}
.ov-region.kind-arrow{background:transparent;border:none;}
.ov-region.kind-arrow::after{content:'➜';position:absolute;right:-6px;top:50%;transform:translateY(-50%);font-size:20px;}
.ov-region .del{position:absolute;top:-10px;right:-10px;width:20px;height:20px;border-radius:50%;background:var(--danger);
  color:#fff;font-size:12px;display:flex;align-items:center;justify-content:center;cursor:pointer;}
.ov-text{position:absolute;cursor:move;font-weight:700;white-space:nowrap;padding:4px 8px;border-radius:6px;touch-action:none;}
.ov-logo{position:absolute;cursor:move;touch-action:none;}
.ov-region{touch-action:none;}
.resize-handle{position:absolute;right:-8px;bottom:-8px;width:16px;height:16px;border-radius:4px;
  background:var(--accent2);border:2px solid #fff;cursor:nwse-resize;touch-action:none;z-index:5;}
.tabs{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap;}
.tab{padding:8px 14px;border-radius:999px;font-size:12px;background:var(--panel2);border:1px solid var(--border);
  color:var(--dim);cursor:pointer;}
.tab.active{background:var(--grad);color:#0a0a0f;border:none;}
.tabpanel{display:none;}
.tabpanel.active{display:block;}
.slider-row{margin-bottom:14px;}
.slider-row .val{float:right;color:var(--accent2);font-size:12px;}
input[type=range]{width:100%;accent-color:var(--accent);}
.tool-row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;}
.tool-btn{padding:7px 10px;font-size:12px;}
.tool-btn.active{background:var(--grad);color:#0a0a0f;border:none;}
.log{background:#05060a;border:1px solid var(--border);border-radius:10px;padding:10px;font-family:monospace;
  font-size:12px;color:#9fe6c9;height:90px;overflow:auto;margin-top:10px;}
.flex-between{display:flex;justify-content:space-between;align-items:center;}
.hidden{display:none !important;}
.badge{font-size:11px;padding:2px 8px;border-radius:999px;background:rgba(110,91,255,.18);color:#b4a8ff;}
.checkrow{display:flex;align-items:center;gap:8px;margin:8px 0;font-size:13px;}
.right-col{flex:1;}
.export-actions{display:flex;gap:10px;margin-top:16px;}
a.dl-link{color:var(--accent2);}
.audio-card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:8px;font-size:12px;}
.audio-card .name{color:var(--text);font-weight:600;margin-bottom:6px;word-break:break-word;}
.audio-card .arow{display:flex;gap:6px;}
.audio-card button{padding:5px 8px;font-size:11px;flex:1;}
.audio-card button.sel{background:var(--grad);color:#0a0a0f;border:none;}
.pan-hint{font-size:11px;color:var(--dim);background:rgba(110,91,255,.12);border:1px solid var(--border);
  border-radius:8px;padding:6px 10px;margin-top:6px;}
.text-row{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:10px;margin-bottom:10px;}
.text-row .top{display:flex;gap:8px;align-items:center;}
.text-row .top input[type=text]{flex:1;}
.text-row .del{cursor:pointer;color:var(--danger);font-size:12px;padding:6px 10px;}
.progress-container {
  background: var(--panel2);
  border: 1px solid var(--border);
  border-radius: 10px;
  height: 24px;
  width: 100%;
  overflow: hidden;
  position: relative;
  margin-top: 14px;
}
.progress-fill {
  background: var(--grad);
  height: 100%;
  width: 0%;
  transition: width 0.2s ease-out;
}
.progress-text {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  color: #fff;
  text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Custom styles for auto-exporting overlays and buttons in clip-grid cards */
.clip-preview-wrap {
  position: relative;
  width: 100%;
  aspect-ratio: 9/16;
  background: #000;
  overflow: hidden;
}
.clip-preview-wrap video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.clip-status-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.3s ease;
  z-index: 5;
}
.clip-status-overlay.active {
  opacity: 1;
  pointer-events: auto;
}
.clip-status-overlay.done {
  background: rgba(16, 28, 23, 0.55);
  opacity: 1;
}
.clip-card:hover .clip-status-overlay.done {
  opacity: 0;
}
.clip-status-overlay.failed {
  background: rgba(30, 15, 15, 0.7);
  opacity: 1;
}
.clip-status-overlay.done .spinner {
  display: none;
}
.clip-status-overlay.failed .spinner {
  display: none;
}
.clip-status-overlay .spinner {
  width: 24px;
  height: 24px;
  border: 3px solid rgba(255,255,255,0.2);
  border-top-color: var(--accent2);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
.clip-status-overlay .status-text {
  font-size: 11px;
  font-weight: 700;
  color: #fff;
  text-align: center;
  padding: 0 8px;
  text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}
.clip-info {
  padding: 10px;
}
.clip-actions {
  display: flex;
  gap: 6px;
  margin-top: 6px;
}
.clip-card-btn {
  flex: 1;
  font-size: 11px;
  padding: 6px 10px;
  text-align: center;
  border-radius: 6px;
  text-decoration: none;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: none;
  background: var(--panel);
  color: var(--text);
  box-sizing: border-box;
}
.clip-card-btn.edit-btn {
  background: var(--panel2);
  color: var(--text);
  border: 1px solid var(--border);
}
.clip-card-btn.edit-btn:hover {
  border-color: var(--accent);
}
.clip-card-btn.dl-btn {
  background: var(--accent2);
  color: #0c0d12;
  border: none;
}
.clip-card-btn.dl-btn:hover {
  filter: brightness(1.12);
}

/* ── Hero URL / Upload section ───────────────────────────────────────── */
.hero-row{display:flex;gap:18px;align-items:stretch;flex-wrap:wrap;}
.hero-box{
  flex:1 1 320px; min-width:280px; position:relative;
  background:linear-gradient(160deg, var(--panel2), var(--panel));
  border:1.5px solid var(--border); border-radius:18px;
  padding:22px 24px; display:flex; flex-direction:column; gap:12px;
  transition:.2s; cursor:default;
}
.hero-box:hover, .hero-box:focus-within{
  border-color:var(--accent);
  box-shadow:0 0 0 3px rgba(110,91,255,0.15), 0 8px 24px rgba(0,0,0,0.35);
}
.hero-box-upload{cursor:pointer;}
.hero-box-upload:hover{border-color:var(--accent2); box-shadow:0 0 0 3px rgba(34,211,196,0.15), 0 8px 24px rgba(0,0,0,0.35);}
.hero-icon{
  width:44px; height:44px; border-radius:12px; background:var(--grad);
  display:flex; align-items:center; justify-content:center; font-size:20px;
  box-shadow:0 4px 14px rgba(110,91,255,0.35);
}
.hero-text label{font-size:15px; font-weight:700; color:var(--text); margin:0;}
.hero-sub{font-size:12px; color:var(--dim); line-height:1.4;}
.hero-input{
  width:100%; box-sizing:border-box; padding:16px 18px; font-size:15px;
  background:var(--bg); border:1.5px solid var(--border); border-radius:12px;
  color:var(--text); margin-top:auto;
}
.hero-input:focus{outline:none; border-color:var(--accent);}
.hero-upload-btn{
  width:100%; padding:14px 18px; font-size:14px; font-weight:700; margin-top:auto;
  background:var(--grad); color:#0a0a0f; border:none; border-radius:12px;
  box-shadow:0 4px 14px rgba(34,211,196,0.25);
}
.hero-upload-btn:hover{filter:brightness(1.08);}
.hero-divider{
  align-self:center; color:var(--dim); font-size:12px; font-weight:700;
  letter-spacing:1px;
}
@media (max-width:700px){
  .hero-row{flex-direction:column;}
  .hero-divider{padding:2px 0;}
}

/* ── Settings bar + big Fetch & Cut CTA ──────────────────────────────── */
.settings-bar{
  display:flex; gap:16px; flex-wrap:wrap; align-items:flex-end;
  margin-top:18px; padding:16px 18px;
  background:var(--panel2); border:1px solid var(--border); border-radius:14px;
}
.settings-field{min-width:160px;}
.settings-field label{font-size:11px; text-transform:uppercase; letter-spacing:.5px;}
.settings-field select, .settings-field input[type=number], .settings-field textarea{
  padding:13px 14px; font-size:14px; font-weight:600;
}
.settings-check{display:flex; flex-direction:column;}
.automod-check{
  display:flex; align-items:center; gap:8px; margin:0; height:48px; padding:0 6px;
}
.automod-check span{font-weight:700; color:#ff9f0a; font-size:14px; text-shadow:0 0 10px rgba(255,159,10,0.2); white-space:nowrap;}
.automod-check input[type=checkbox]{width:18px; height:18px;}
.fetch-cta{
  display:block; width:100%; margin-top:16px; padding:20px; font-size:18px; font-weight:800;
  background:var(--grad); color:var(--btn-text); border:none; border-radius:14px;
  box-shadow:0 10px 28px rgba(110,91,255,0.4); cursor:pointer; letter-spacing:.3px;
  transition:.15s;
}
.fetch-cta:hover{filter:brightness(1.08); transform:translateY(-2px); box-shadow:0 14px 34px rgba(110,91,255,0.5);}
.fetch-cta:active{transform:translateY(0);}
.fetch-cta:disabled{transform:none;}
.btn-spinner{
  display:inline-block; width:18px; height:18px; vertical-align:middle;
  border:3px solid rgba(255,255,255,0.35); border-top-color:#fff; border-radius:50%;
  animation:btnspin .7s linear infinite;
}
@keyframes btnspin{ to{ transform:rotate(360deg); } }

</style>
</head>
<body>
<div class="topbar">
  <div class="logo">⚡ AutoShortAi</div>
  <div class="sub">Fetch → Cut → Live Edit → Save, all in one place</div>
  <div class="nav-tabs">
    <button type="button" class="nav-tab-btn active" id="studioTabBtn" onclick="showStudioView()">🎬 Studio</button>
    <button type="button" class="nav-tab-btn" id="downloaderTabBtn" onclick="showDownloaderView()">⬇️ Downloader</button>
    <button type="button" class="nav-tab-btn" id="editorTabBtn" onclick="showEditorView()">🪄 Auto Edit</button>
    <button type="button" class="nav-tab-btn" id="publishTabBtn" onclick="showPublishView()">📤 Publish</button>
  </div>
  <div class="theme-picker">
    <button type="button" class="theme-btn" onclick="toggleThemeMenu()">🎨 Theme</button>
    <div id="themeMenu" class="theme-menu hidden">
      <div class="theme-opt" data-t="dark-violet" onclick="setTheme('dark-violet')"><span class="theme-dot" style="background:linear-gradient(135deg,#6e5bff,#22d3c4)"></span>Dark Violet</div>
      <div class="theme-opt" data-t="light" onclick="setTheme('light')"><span class="theme-dot" style="background:linear-gradient(135deg,#5b46e0,#0891b2)"></span>Light</div>
      <div class="theme-opt" data-t="midnight-blue" onclick="setTheme('midnight-blue')"><span class="theme-dot" style="background:linear-gradient(135deg,#3b82f6,#22d3ee)"></span>Midnight Blue</div>
      <div class="theme-opt" data-t="forest" onclick="setTheme('forest')"><span class="theme-dot" style="background:linear-gradient(135deg,#16a34a,#a3e635)"></span>Forest</div>
      <div class="theme-opt" data-t="sunset" onclick="setTheme('sunset')"><span class="theme-dot" style="background:linear-gradient(135deg,#fb7185,#f59e0b)"></span>Sunset</div>
      <div class="theme-opt" data-t="ocean" onclick="setTheme('ocean')"><span class="theme-dot" style="background:linear-gradient(135deg,#06b6d4,#6366f1)"></span>Ocean</div>
      <div class="theme-opt" data-t="rose" onclick="setTheme('rose')"><span class="theme-dot" style="background:linear-gradient(135deg,#ec4899,#f472b6)"></span>Rose</div>
      <div class="theme-opt" data-t="amber" onclick="setTheme('amber')"><span class="theme-dot" style="background:linear-gradient(135deg,#f59e0b,#eab308)"></span>Amber</div>
      <div class="theme-opt" data-t="slate" onclick="setTheme('slate')"><span class="theme-dot" style="background:linear-gradient(135deg,#475569,#94a3b8)"></span>Slate</div>
      <div class="theme-opt" data-t="cyberpunk" onclick="setTheme('cyberpunk')"><span class="theme-dot" style="background:linear-gradient(135deg,#e91e9c,#00f0ff)"></span>Cyberpunk</div>
    </div>
  </div>
</div>
<div class="wrap" id="studioView">

  <div class="card" id="fetchCard">
    <div class="hero-row">
      <div class="hero-box" id="urlHeroBox">
        <div class="hero-icon">🔗</div>
        <div class="hero-text">
          <label style="margin-bottom:4px;">Paste Video URL</label>
          <span class="hero-sub">YouTube, Instagram, TikTok, Facebook, X/Twitter, Vimeo &amp; 1000+ more sites</span>
        </div>
        <input type="text" id="ytUrl" class="hero-input" placeholder="https://...">
      </div>

      <div class="hero-divider">OR</div>

      <div class="hero-box hero-box-upload" id="uploadHeroBox" onclick="document.getElementById('deviceVideoInput').click()">
        <div class="hero-icon">📁</div>
        <div class="hero-text">
          <label style="margin-bottom:4px;">Upload from PC / Device</label>
          <span class="hero-sub">Already have the video? Use it directly - stays on this device, no external server</span>
        </div>
        <input type="file" id="deviceVideoInput" accept="video/*" style="display:none" onchange="uploadDeviceVideo(this.files[0])">
        <button type="button" class="hero-upload-btn" onclick="event.stopPropagation();document.getElementById('deviceVideoInput').click()">Choose File</button>
      </div>
    </div>

    <div class="settings-bar">
      <div class="settings-field">
        <label>Quality</label>
        <select id="quality">
          <option value="best" selected>🚀 Highest Quality (Auto)</option>
          <option value="2160">4K (2160p)</option>
          <option value="1440">2K (1440p)</option>
          <option value="1080">1080p Full HD</option>
          <option value="720">720p HD</option>
        </select>
      </div>
      <div class="settings-field">
        <label>Mode</label>
        <select id="mode" onchange="toggleMode()"><option value="auto">Auto-split</option><option value="manual">Manual ranges</option><option value="hooks">🔥 AI Hook/Viral Detect</option></select>
      </div>
      <div id="autoLenWrap" class="settings-field">
        <label>Clip length (s)</label>
        <input type="number" id="clipLen" value="10">
      </div>
      <div class="settings-field settings-check">
        <label>&nbsp;</label>
        <div class="checkrow automod-check">
          <input type="checkbox" id="autoMode" checked>
          <span title="Automatically cut, apply settings, replace audio, and export all clips">🔥 Automod (Auto Export)</span>
        </div>
      </div>
      <div id="manualWrap" class="hidden settings-field" style="flex:2;min-width:240px;">
        <label>Ranges (start-end, one per line, seconds)</label>
        <textarea id="ranges" rows="2" placeholder="0-30
30-60"></textarea>
      </div>
      <div id="hooksWrap" class="hidden settings-field" style="display:flex; gap:10px;">
        <div><label>Clips</label><input type="number" id="hookNumClips" value="5" min="1" max="15" style="width:70px"></div>
        <div><label>Min len(s)</label><input type="number" id="hookMinLen" value="15" min="5" style="width:70px"></div>
        <div><label>Max len(s)</label><input type="number" id="hookMaxLen" value="60" min="10" style="width:70px"></div>
      </div>
    </div>
    <button class="fetch-cta" onclick="startFetchOrHooks()">🔻 &nbsp;Fetch &amp; Cut Video</button>
    
    <!-- YouTube Automation & Automod Settings Panel -->
    <div style="margin-top: 15px; border-top: 1px solid var(--border); padding-top: 15px;">
      <h4 style="margin: 0 0 12px 0; color: #ff9f0a; display: flex; align-items: center; gap: 8px; font-size: 14px; text-shadow: 0 0 10px rgba(255,159,10,0.2);">
        <span>🔥</span> YouTube Automation &amp; Automod Settings
      </h4>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 8px;">
        <div>
          <label style="font-size: 11px; text-transform: uppercase; color: var(--dim); display: block; margin-bottom: 4px;">Watermark Text</label>
          <input type="text" id="automodWatermark" value="FondPeace.com" placeholder="e.g. FondPeace.com">
        </div>
        <div>
          <label style="font-size: 11px; text-transform: uppercase; color: var(--dim); display: block; margin-bottom: 4px;">Logo Font Size</label>
          <input type="number" id="automodFontSize" value="15" min="12" max="100">
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center;">
          <div class="checkrow" style="margin: 0;">
            <input type="checkbox" id="automodColorVary" checked>
            <span style="font-weight: 600;">🎨 Vary color presets per clip</span>
          </div>
          <span style="font-size: 10px; color: var(--dim); margin-left: 20px; display: block; margin-top: 2px;">Bypass YouTube duplicate matched content</span>
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center;">
          <div class="checkrow" style="margin: 0;">
            <input type="checkbox" id="automodMirror" checked>
            <span style="font-weight: 600;">🪞 Apply mirror flip (Horizontal)</span>
          </div>
          <span style="font-size: 10px; color: var(--dim); margin-left: 20px; display: block; margin-top: 2px;">Alternate flip to bypass identification</span>
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center;">
          <div class="checkrow" style="margin: 0;">
            <input type="checkbox" id="automodAudioReplace" checked>
            <span style="font-weight: 600;">🎵 Replace background audio</span>
          </div>
          <span style="font-size: 10px; color: var(--dim); margin-left: 20px; display: block; margin-top: 2px;">Round-robin background tracks from audio/</span>
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center;">
          <div class="checkrow" style="margin: 0;">
            <input type="checkbox" id="automodStartFromOne" checked>
            <span style="font-weight: 600;">⏱️ Cut precisely from Second 1</span>
          </div>
          <span style="font-size: 10px; color: var(--dim); margin-left: 20px; display: block; margin-top: 2px;">Cuts into clean 10s blocks (1-11, 11-21)</span>
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center;">
          <div class="checkrow" style="margin: 0;">
            <input type="checkbox" id="automodFilmGrain" checked>
            <span style="font-weight: 600;">🎞️ Anti-Copyright Film Grain / Noise</span>
          </div>
          <span style="font-size: 10px; color: var(--dim); margin-left: 20px; display: block; margin-top: 2px;">Adds micro-noise to scramble pixel-matching hashes</span>
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center;">
          <div class="checkrow" style="margin: 0;">
            <input type="checkbox" id="automodAudioPitch" checked>
            <span style="font-weight: 600;">🎙️ Anti-Copyright Audio Pitch Tuning</span>
          </div>
          <span style="font-size: 10px; color: var(--dim); margin-left: 20px; display: block; margin-top: 2px;">Micro-shift audio frequency to bypass acoustic matching</span>
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center;">
          <div class="checkrow" style="margin: 0;">
            <input type="checkbox" id="automodVignette" checked>
            <span style="font-weight: 600;">📐 Dark Vignette Border Overlay</span>
          </div>
          <span style="font-size: 10px; color: var(--dim); margin-left: 20px; display: block; margin-top: 2px;">Adds corner shading to break visual fingerprinting</span>
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center;">
          <div class="checkrow" style="margin: 0;">
            <input type="checkbox" id="automodDeadAir">
            <span style="font-weight: 600;">🔇 Auto-remove Dead Air on cut</span>
          </div>
          <span style="font-size: 10px; color: var(--dim); margin-left: 20px; display: block; margin-top: 2px;">Off by default. When on, each short is silence-cut right after it's cut, before export — you can still Undo it per-clip in Speed/Zoom or Export tab.</span>
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center;">
          <div class="checkrow" style="margin: 0;">
            <input type="checkbox" id="automodFaceTrack">
            <span style="font-weight: 600;">🎯 Auto-apply Smart Reframe on cut</span>
          </div>
          <span style="font-size: 10px; color: var(--dim); margin-left: 20px; display: block; margin-top: 2px;">Off by default. When on, every short is scanned for the active speaker (Kalman-smoothed, multi-speaker aware) right after it's cut and "Follow face" is switched on automatically — still fully toggleable/removable per-clip in Speed/Zoom.</span>
        </div>
        <div style="display: flex; flex-direction: column; justify-content: center;">
          <div class="checkrow" style="margin: 0;">
            <input type="checkbox" id="automodWordCaptions">
            <span style="font-weight: 600;">🎤 Auto-fetch word-perfect captions on cut</span>
          </div>
          <span style="font-size: 10px; color: var(--dim); margin-left: 20px; display: block; margin-top: 2px;">Off by default (optional/removable). On hone par, source video fetch/cut hote hi background me real per-word timestamps bhi fetch ho jaate hain — Text tab me ek click me import kar sakte ho.</span>
        </div>
      </div>
    </div>

    <div class="log" id="fetchLog"></div>
  </div>

  <div class="card hidden" id="hooksCard">
    <h3 style="margin:0 0 10px 0">🔥 Detected Hooks</h3>
    <div id="hooksStatusLine" class="sub"></div>
    <div id="hooksResults" class="grid" style="grid-template-columns:1fr;"></div>
  </div>

  <div class="card hidden" id="clipsCard">
    <div class="flex-between" style="align-items: center; gap: 12px; flex-wrap: wrap;">
      <h3 style="margin:0">Your Shorts</h3>
      <div style="display:flex; gap:10px; align-items:center;">
        <button class="btn-grad" id="exportAllBtn" onclick="applyToAllAndExport()" style="padding: 6px 12px; font-size:12px; margin:0; cursor:pointer;">🚀 Apply current settings to All &amp; Export Batch</button>
        <span class="badge" id="clipCount"></span>
      </div>
    </div>
    <div class="grid" id="clipGrid"></div>
  </div>

  <!-- Polished Exported Video Downloader & Batch Manager Panel -->
  <div class="card hidden" id="downloadsCard" style="margin-top: 20px;">
    <div class="flex-between" style="align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 15px;">
      <h3 style="margin:0; display:flex; align-items:center; gap:8px;">
        <span style="color:var(--yellow)">📦</span> Exported Downloads Manager
      </h3>
      <div style="display:flex; gap:10px; align-items:center;">
        <button class="btn-grad" id="downloadSelectedBtn" onclick="downloadSelectedVideos()" style="padding: 6px 12px; font-size:12px; margin:0; cursor:pointer;">📥 Download Selected</button>
        <button class="btn-grad" id="selectAllExportsBtn" onclick="toggleSelectAllExports()" style="padding: 6px 12px; font-size:12px; margin:0; cursor:pointer; background:#434348; color:#fff;">✅ Toggle All</button>
      </div>
    </div>
    
    <div style="overflow-x: auto;">
      <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 13px;">
        <thead>
          <tr style="border-bottom: 2px solid var(--border); color: var(--dim);">
            <th style="padding: 8px; width: 40px;">Select</th>
            <th style="padding: 8px;">Video Title</th>
            <th style="padding: 8px;">Specifications</th>
            <th style="padding: 8px; width: 150px; text-align: right;">Actions</th>
          </tr>
        </thead>
        <tbody id="exportedVideosList">
          <tr id="emptyExportRow">
            <td colspan="4" style="padding: 20px; text-align: center; color: var(--dim);">No videos successfully exported yet. Click "Export Video" on a card or run a batch export!</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="card editor" id="editorCard">
    <div class="stage-col">
      <div class="stage" id="stage">
        <video id="player" loop playsinline></video>
        <div class="overlay-layer" id="overlayLayer"></div>
      </div>
      <div class="row" style="justify-content:center">
        <button onclick="togglePlay()" id="playBtn">▶ Play</button>
        <button onclick="closeEditor()">✕ Close</button>
      </div>
    </div>

    <div class="right-col">
      <div class="tabs">
        <div class="tab active" data-t="speed">Speed/Zoom</div>
        <div class="tab" data-t="audio">Audio/Voice</div>
        <div class="tab" data-t="text">Text/Logo</div>
        <div class="tab" data-t="shapes">Blur/Shapes</div>
        <div class="tab" data-t="color">Color</div>
        <div class="tab" data-t="export">Export</div>
      </div>

      <div class="tabpanel active" id="panel-speed">
        <div class="slider-row"><label>Speed <span class="val" id="speedVal">0.75x</span></label>
          <input type="range" id="speed" min="0.25" max="4" step="0.05" value="0.75" oninput="onSpeed()"></div>
        <div class="slider-row"><label>Zoom <span class="val" id="zoomVal">1.20x</span></label>
          <input type="range" id="zoom" min="1" max="3" step="0.05" value="1.20" oninput="onZoom()"></div>
        <p class="sub">Zoom crops toward center then re-scales to your export size with high-quality (lanczos) scaling — never stretched or pixelated.</p>
        <div class="pan-hint">🖱 When zoomed in (&gt;1.00x), <b>drag directly on the video preview</b> to move/pan which part stays in frame.</div>

        <hr style="border-color:var(--border);margin:14px 0">
        <label>🎯 Smart Reframe (Enterprise face &amp; speaker tracking)</label>
        <p class="sub">Detects and follows the active speaker automatically instead of a fixed center crop or manual pan — only kicks in while zoomed in above 1.00x. Uses Kalman-filtered motion smoothing, scene-cut-aware clean jump-cuts, and (when multiple people are on screen) switches the anchor to whoever's actively talking. Needs <code>opencv-python-headless</code> installed, with <code>mediapipe</code> optional for multi-speaker detection — falls back automatically to a simpler single-face detector if those aren't available.</p>
        <div class="row" style="gap:8px;flex-wrap:wrap;">
          <button type="button" id="faceTrackDetectBtn" onclick="detectFaceTrack()">🔍 Detect &amp; track</button>
          <div class="checkrow" style="margin:0"><input type="checkbox" id="faceTrackEnabled" disabled onchange="onFaceTrackToggle()"> Follow speaker</div>
          <button type="button" id="faceTrackRemoveBtn" onclick="removeFaceTrack()" disabled>🗑️ Remove tracking</button>
        </div>
        <p class="sub" id="faceTrackStatus"></p>

        <hr style="border-color:var(--border);margin:14px 0">
        <label>🔇 Dead-air removal</label>
        <p class="sub">Detects quiet/silent gaps in this clip's audio. Nothing is cut until you click Apply, and you can Undo the cut afterward — this never happens automatically unless you turn on "Auto-remove Dead Air" in the automation settings.</p>
        <div class="row" style="gap:8px;">
          <button type="button" id="deadAirPreviewBtn" onclick="previewDeadAir()">🔍 Preview Gaps</button>
          <button type="button" id="deadAirApplyBtn" onclick="applyDeadAir()" disabled>✂️ Apply Cut</button>
          <button type="button" id="deadAirUndoBtn" onclick="undoDeadAir()" disabled>↩️ Undo</button>
        </div>
        <p class="sub" id="deadAirStatus"></p>
      </div>

      <div class="tabpanel" id="panel-audio">
        <div class="checkrow"><input type="radio" name="amode" value="original" checked onchange="onAudioMode()"> Keep original audio</div>
        <div class="checkrow"><input type="radio" name="amode" value="mute" onchange="onAudioMode()"> Mute</div>
        <div class="checkrow"><input type="radio" name="amode" value="replace" onchange="onAudioMode()"> Replace with file</div>
        <div class="checkrow"><input type="radio" name="amode" value="tts" onchange="onAudioMode()"> AI Voiceover</div>
        <div class="checkrow"><input type="radio" name="amode" value="record" onchange="onAudioMode()"> 🎤 Record my own voice</div>

        <div id="replaceWrap" class="hidden">
          <label>🎵 Pick from your audio library</label>
          <div style="display:flex; gap:8px; margin:6px 0;">
            <button type="button" onclick="rescanAudioLibrary()">📱 Rescan device for audio/video</button>
            <button type="button" onclick="syncGithubAudio()">🔗 Sync library from GitHub</button>
          </div>
          <div class="grid" id="audioLibGrid" style="grid-template-columns:repeat(auto-fill,minmax(140px,1fr));"></div>
          <p class="sub">Shows files from the <code>audio/</code> folder, PLUS music/video already on this device (Music, Downloads, etc). Play each to check the fit, then hit Add to use it.</p>
          <hr style="border-color:var(--border);margin:14px 0">
          <label>...or upload your own audio file</label>
          <input type="file" id="audioFile" accept="audio/*" onchange="uploadAudio()">
          <div id="chosenAudioRow" class="hidden" style="margin-top:8px">
            <span class="badge">Selected: <span id="chosenAudioName"></span></span>
          </div>
        </div>

        <div id="recordWrap" class="hidden">
          <label>🎤 Record your own voice</label>
          <p class="sub">Records from this device's microphone. Works on PC and Android browsers alike.</p>
          <div class="row" style="align-items:center; margin-top:8px;">
            <button type="button" id="recBtn" class="btn-grad" onclick="toggleRecording()">🔴 Start Recording</button>
            <span id="recTimer" style="font-weight:700; color:var(--accent2); font-size:14px;"></span>
          </div>
          <audio id="recPreview" controls class="hidden" style="width:100%;margin-top:10px"></audio>
          <div id="recChosenRow" class="hidden" style="margin-top:8px">
            <span class="badge">✅ Recording ready - will be used as this clip's audio</span>
            <button type="button" onclick="discardRecording()" style="margin-left:8px">🗑 Discard</button>
          </div>
        </div>

        <div id="ttsWrap" class="hidden">
          <label>Voice</label>
          <select id="ttsVoice">
            <optgroup label="Hindi (India) — Neural">
              <option value="hi_male">Male — Madhur</option>
              <option value="hi_female">Female — Swara</option>
            </optgroup>
            <optgroup label="English (India) — Neural">
              <option value="en_in_male">Male — Prabhat</option>
              <option value="en_in_female">Female — Neerja</option>
            </optgroup>
            <optgroup label="English (US) — Neural">
              <option value="en_us_male_andrew">Male — Andrew (most natural/conversational)</option>
              <option value="en_us_female_emma">Female — Emma (most natural/conversational)</option>
              <option value="en_us_male_guy">Male — Guy</option>
              <option value="en_us_male_davis">Male — Davis</option>
              <option value="en_us_male_brian">Male — Brian</option>
              <option value="en_us_female_aria">Female — Aria</option>
              <option value="en_us_female_jenny">Female — Jenny</option>
              <option value="en_us_female_michelle">Female — Michelle</option>
            </optgroup>
            <optgroup label="English (UK) — Neural">
              <option value="en_gb_male_ryan">Male — Ryan</option>
              <option value="en_gb_male_thomas">Male — Thomas</option>
              <option value="en_gb_female_sonia">Female — Sonia</option>
              <option value="en_gb_female_libby">Female — Libby</option>
            </optgroup>
            <optgroup label="Basic quality (Google TTS)">
              <option value="gtts_hi">Hindi — Google TTS</option>
              <option value="gtts_en_in">English (India accent) — Google TTS</option>
              <option value="gtts_en_us">English (US accent) — Google TTS</option>
              <option value="gtts_en_uk">English (UK accent) — Google TTS</option>
            </optgroup>
          </select>
          <p class="sub">"Neural" voices (edge-tts) sound noticeably more human than the "Basic quality" (Google TTS) ones — start there. No free/offline engine fully matches paid tools like ElevenLabs, but Andrew/Emma and the Hindi/India voices above are the most natural-sounding options available here.</p>
          <label style="margin-top:10px">Emotion</label>
          <select id="ttsEmotion">
            <option value="neutral">😐 Neutral</option>
            <option value="happy">😄 Happy</option>
            <option value="sad">😢 Sad</option>
            <option value="angry">😠 Angry</option>
            <option value="excited">🤩 Excited</option>
            <option value="calm">😌 Calm</option>
            <option value="serious">🧐 Serious</option>
            <option value="whisper">🤫 Whisper</option>
            <option value="fear">😨 Fearful</option>
          </select>
          <label style="margin-top:10px">Script</label>
          <textarea id="ttsText" rows="4" placeholder="Yahan apna script likhiye..."></textarea>
          <div class="row" style="margin-top:8px">
            <button onclick="generateTTS()">🎙 Generate &amp; Preview</button>
            <div class="checkrow"><input type="checkbox" id="ttsMix"> Mix with lowered original audio</div>
          </div>
          <audio id="ttsAudio" controls class="hidden" style="width:100%;margin-top:8px"></audio>

          <hr style="border-color:var(--border);margin:14px 0">
          <div class="flex-between"><label style="margin:0">🗂 Your generated voices</label>
            <button type="button" onclick="loadTtsLibrary()">↻ Refresh</button></div>
          <p class="sub">Temporary — lives only for this session. Reuse any of these as this clip's voiceover, or delete the ones you don't need.</p>
          <div id="ttsLibraryList"></div>
        </div>
      </div>

      <div class="tabpanel" id="panel-text">
        <div class="flex-between"><label style="margin:0">Text layers</label>
          <button onclick="addTextLayer()">+ Add text</button></div>
        <div id="textLayersWrap"></div>
        <p class="sub">Drag any text directly on the preview to position it. Add as many as you need (titles, captions, callouts).</p>
        <hr style="border-color:var(--border);margin:14px 0">
        <div class="flex-between"><label style="margin:0">🔤 Auto-fetch captions from source video</label>
          <button id="fetchCapBtn" onclick="fetchCaptionLanguages()">🔤 Auto-fetch Captions</button></div>
        <p class="sub">Pulls the subtitle/caption track this video already has (YouTube etc.) and drops it in as timed captions — no typing needed.</p>
        <div id="capLangWrap" class="hidden" style="margin-top:8px">
          <label>Language / track</label>
          <select id="capLangSelect"></select>
          <label style="margin-top:8px">Caption style</label>
          <select id="capStyleSelect">
            <option value="sentence">Full sentence (classic)</option>
            <option value="karaoke">Word-by-word (karaoke pop) 🔥</option>
          </select>
          <div class="row" style="margin-top:8px">
            <button onclick="importCaptions()">⬇ Import as captions</button>
            <button onclick="document.getElementById('capLangWrap').classList.add('hidden')">Cancel</button>
          </div>
        </div>
        <div id="capStatus" class="sub" style="display:none"></div>
        <hr style="border-color:var(--border);margin:14px 0">
        <div class="flex-between"><label style="margin:0">🎤 Word-Perfect Captions (real timestamps)</label>
          <button id="wordCapBtn" onclick="fetchWordCaptions()">🎤 Fetch Word Timestamps</button></div>
        <p class="sub">YouTube ke internal per-word timing se EXACT karaoke captions banata hai (guess nahi, real data). Source video ke liye ek baar fetch hota hai, phir sab clips isi ko reuse karte hain.</p>
        <div id="wordCapStatus" class="sub"></div>
        <div id="wordCapImportRow" class="hidden" style="margin-top:8px">
          <button onclick="importWordPerfectCaptions()">🎯 Import Word-Perfect Karaoke (is clip me)</button>
        </div>
        <hr style="border-color:var(--border);margin:14px 0">
        <div class="flex-between"><label style="margin:0">Logo / watermark image</label>
          <div class="checkrow" style="margin:0"><input type="checkbox" id="logoEnabled" checked onchange="renderLogo()"> Show on video</div></div>
        <input type="file" id="logoFile" accept="image/*" onchange="uploadLogo()">
        <div class="row" style="margin-top:8px">
          <div style="flex:1"><label>Width % <span class="val" id="logoWVal">18</span></label>
            <input type="range" id="logoW" min="5" max="50" value="18" oninput="renderLogo()"></div>
          <div style="flex:1"><label>Opacity % <span class="val" id="logoOVal">100</span></label>
            <input type="range" id="logoO" min="10" max="100" value="100" oninput="renderLogo()"></div>
        </div>
        <p class="sub">Drag the logo on the preview to position it.</p>
      </div>

      <div class="tabpanel" id="panel-shapes">
        <p class="sub">Pick a tool, then drag on the preview to draw. Blur/Black/Emoji hide something. Rect/Circle/Arrow highlight something.</p>
        <div class="tool-row">
          <button class="tool-btn active" data-k="blur" onclick="setTool('blur')">Blur</button>
          <button class="tool-btn" data-k="black" onclick="setTool('black')">Black</button>
          <button class="tool-btn" data-k="emoji" onclick="setTool('emoji')">Emoji/Sticker</button>
          <button class="tool-btn" data-k="rect" onclick="setTool('rect')">Rectangle</button>
          <button class="tool-btn" data-k="circle" onclick="setTool('circle')">Circle</button>
          <button class="tool-btn" data-k="arrow" onclick="setTool('arrow')">Arrow</button>
        </div>
        <div id="emojiUploadWrap" class="hidden">
          <label>Emoji/sticker image</label>
          <input type="file" id="emojiFile" accept="image/*" onchange="uploadEmoji()">
          <label style="margin-top:10px">...or tap a quick emoji (no upload needed)</label>
          <div id="emojiQuickPick" class="tool-row"></div>
          <p class="sub" id="emojiPickStatus"></p>
        </div>
        <div class="row">
          <div style="width:90px"><label>Shape color</label><input type="color" id="shapeColor" value="#ff3b30"></div>
          <button onclick="clearRegions()">Clear all regions</button>
        </div>
      </div>

      <div class="tabpanel" id="panel-color">
        <label>CapCut-style one-click look</label>
        <select id="colorPreset" onchange="onColorPresetChange()">
          <option value="none">None</option>
          <option value="vivid">Vivid Pop</option>
          <option value="cinematic">Cinematic</option>
          <option value="moody">Moody</option>
          <option value="vintage_vhs">Vintage VHS</option>
          <option value="warm_glow">Warm Glow</option>
          <option value="cool_blue">Cool Blue</option>
        </select>
        <div class="slider-row" style="margin-top:14px"><label>Contrast <span class="val" id="contrastVal">1.12</span></label>
          <input type="range" id="contrast" min="0.5" max="2" step="0.01" value="1.12" oninput="currentPresetName='none'; document.getElementById('colorPreset').value='none'; onColor()"></div>
        <div class="slider-row"><label>Saturation <span class="val" id="satVal">1.25</span></label>
          <input type="range" id="saturation" min="0" max="2" step="0.01" value="1.25" oninput="currentPresetName='none'; document.getElementById('colorPreset').value='none'; onColor()"></div>
        <div class="slider-row"><label>Brightness <span class="val" id="brightVal">0.02</span></label>
          <input type="range" id="brightness" min="-0.3" max="0.3" step="0.01" value="0.02" oninput="currentPresetName='none'; document.getElementById('colorPreset').value='none'; onColor()"></div>
        <div class="checkrow"><input type="checkbox" id="sharpen" onchange="onColor()" checked> Sharpen (clarity)</div>
        <div class="checkrow"><input type="checkbox" id="enhance" onchange="onColor()" checked> HD Enhance (denoise + detail pass — keeps quality, stops pixelation on zoom/upscale)</div>
        



        <div style="border-top:1px solid var(--border); margin:14px 0; padding-top:14px;">
          <h4 style="margin:0 0 10px 0; font-size:13px; color:var(--text);">🔄 Transformations</h4>
          <div class="row" style="margin-top:6px; gap:10px;">
            <div style="flex:1">
              <label>Rotate</label>
              <select id="rotate" onchange="onTransform()">
                <option value="0">0° (Normal)</option>
                <option value="90">90° Clockwise</option>
                <option value="180">180°</option>
                <option value="270">270°</option>
              </select>
            </div>
            <div style="flex:1; display:flex; align-items:flex-end;">
              <div class="checkrow" style="margin:0; padding-bottom:8px;"><input type="checkbox" id="hflip" onchange="onTransform()" checked> Mirror (Horizontal Flip)</div>
            </div>
          </div>
        </div>
        
        <button onclick="resetColor()" style="margin-top:12px;">Reset</button>
      </div>

      <div class="tabpanel" id="panel-export">
        <label>Export Ratios (ek ya zyada select karo — sab ek hi export job me bante hain)</label>
        <div id="ratioChecks" style="display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 10px;">
          <label class="checkrow" style="margin:0;"><input type="checkbox" class="ratioBox" value="9:16" checked onchange="updateStageAspectRatio()"> 9:16 Shorts</label>
          <label class="checkrow" style="margin:0;"><input type="checkbox" class="ratioBox" value="1:1" onchange="updateStageAspectRatio()"> 1:1 Square</label>
          <label class="checkrow" style="margin:0;"><input type="checkbox" class="ratioBox" value="16:9" onchange="updateStageAspectRatio()"> 16:9 Landscape</label>
          <label class="checkrow" style="margin:0;"><input type="checkbox" class="ratioBox" value="4:5" onchange="updateStageAspectRatio()"> 4:5 Portrait</label>
        </div>
        <div class="row" style="margin-top:10px">
          <div style="flex:1"><label>Format</label><select id="format"><option value="mp4">MP4 (H.264/AAC)</option>
            <option value="mov">MOV</option><option value="webm">WEBM (VP9/Opus)</option></select></div>
          <div style="flex:1"><label>Quality (CRF)</label><input type="number" id="crf" value="18" min="12" max="30"></div>
          <div style="flex:1"><label>Speed preset</label><select id="preset"><option value="ultrafast">ultrafast</option>
            <option value="fast" selected>fast</option><option value="medium">medium</option>
            <option value="slow">slow</option></select></div>
        </div>
        <p class="sub">If you don't change anything else (speed, zoom, color, regions, text, logo, audio), Save just copies the file instantly instead of re-encoding.</p>
        <div class="checkrow" style="margin-top:8px">
          <input type="checkbox" id="hwAccel" checked>
          <span title="Uses your GPU's built-in H.264 encoder (NVIDIA/Intel/AMD/Apple) when available - much faster than CPU-only encoding. Automatically falls back to the CPU encoder if no compatible GPU is found or the GPU attempt fails.">⚡ Use GPU acceleration when available</span>
        </div>

        <hr style="border-color:var(--border);margin:14px 0">
        <label>🔇 Dead-air removal</label>
        <p class="sub">Same silence-cut tool as the Speed/Zoom tab, handy right before export. Apply is manual and always Undo-able.</p>
        <div class="row" style="gap:8px;">
          <button type="button" id="deadAirApplyBtn2" onclick="applyDeadAir()">✂️ Apply Cut</button>
          <button type="button" id="deadAirUndoBtn2" onclick="undoDeadAir()" disabled>↩️ Undo</button>
        </div>
        <p class="sub" id="deadAirStatus2"></p>

        <div class="export-actions">
          <button class="btn-grad" id="saveBtn" onclick="saveVideo()">💾 Save Final Video</button>
        </div>
        <div class="progress-container hidden" id="exportProgressWrap">
          <div class="progress-fill" id="exportProgressFill"></div>
          <div class="progress-text" id="exportProgressText">0%</div>
        </div>
        <div class="log" id="exportLog"></div>
      </div>
    </div>
  </div>

</div>

<div class="wrap hidden" id="downloaderView">
  <div class="card" id="dlFetchCard">
    <div class="hero-row">
      <div class="hero-box" id="dlUrlHeroBox" style="width:100%;">
        <div class="hero-icon">⬇️</div>
        <div class="hero-text">
          <label style="margin-bottom:4px;">Paste Video URL to Download</label>
          <span class="hero-sub">YouTube, Instagram, TikTok, Facebook, X/Twitter, Vimeo &amp; 1000+ more sites</span>
        </div>
        <input type="text" id="dlUrl" class="hero-input" placeholder="https://...">
      </div>
    </div>
    <button class="fetch-cta" id="dlFetchBtn" onclick="fetchDownloadInfo()">🔎 &nbsp;Fetch Video Details</button>
    <div class="dl-inline-status hidden" id="dlLog"></div>
  </div>

  <div class="card hidden" id="dlInfoCard">
    <div class="dl-info-row">
      <div class="dl-thumb-wrap">
        <img id="dlThumbMain" class="dl-thumb-main" src="" alt="thumbnail">
        <div class="dl-thumb-strip" id="dlThumbStrip"></div>
      </div>
      <div class="dl-meta">
        <div class="dl-title-row">
          <h3 id="dlTitle" class="dl-title"></h3>
          <button class="dl-copy-icon-btn" title="Copy title" onclick="copyDlValue(document.getElementById('dlTitle').textContent, this)">📋</button>
        </div>
        <div class="dl-meta-badges" id="dlMetaBadges"></div>
        <div class="dl-copy-row">
          <input type="text" id="dlPageUrl" readonly>
          <button class="clip-card-btn" onclick="copyDlText('dlPageUrl')">📋 Copy Link</button>
        </div>
      </div>
    </div>

    <h4 class="dl-section-title">Full Video Details</h4>
    <div class="dl-details-grid" id="dlDetailsGrid"></div>

    <div class="dl-desc-wrap" id="dlDescWrap" style="display:none;">
      <div class="dl-desc-head">
        <span>Description</span>
        <div style="display:flex; gap:6px;">
          <button class="dl-copy-icon-btn" title="Copy description" id="dlDescCopyBtn">📋</button>
          <button class="dl-copy-icon-btn" title="Expand / collapse" id="dlDescToggleBtn">⬇️</button>
        </div>
      </div>
      <div class="dl-desc-text collapsed" id="dlDescText"></div>
    </div>

    <div class="dl-chips-wrap" id="dlCategoriesWrap" style="display:none;">
      <span class="dl-chips-label">Categories</span>
      <div class="dl-chips" id="dlCategoriesChips"></div>
    </div>
    <div class="dl-chips-wrap" id="dlTagsWrap" style="display:none;">
      <span class="dl-chips-label">Tags</span>
      <div class="dl-chips" id="dlTagsChips"></div>
    </div>

    <h4 class="dl-section-title">Available Formats &amp; Quality</h4>
    <div class="dl-formats-table-wrap">
      <table class="dl-formats-table" id="dlFormatsTable">
        <thead>
          <tr>
            <th>Type</th><th>Quality</th><th>Format</th><th>Codec</th><th>Size</th><th></th>
          </tr>
        </thead>
        <tbody id="dlFormatsBody"></tbody>
      </table>
    </div>
  </div>

  <div class="card hidden" id="dlProgressCard">
    <h4 class="dl-section-title" style="margin-top:0;" id="dlProgressTitle">Downloading…</h4>

    <div class="dl-stepper" id="dlStepper">
      <div class="dl-step" data-step="connect"><div class="dl-step-ico">🔗</div><div class="dl-step-label">Connecting</div></div>
      <div class="dl-step-line" data-line="1"></div>
      <div class="dl-step" data-step="download"><div class="dl-step-ico">⬇️</div><div class="dl-step-label">Downloading</div></div>
      <div class="dl-step-line" data-line="2"></div>
      <div class="dl-step" data-step="merge"><div class="dl-step-ico">🔀</div><div class="dl-step-label">Merging</div></div>
      <div class="dl-step-line" data-line="3"></div>
      <div class="dl-step" data-step="done"><div class="dl-step-ico">✅</div><div class="dl-step-label">Ready</div></div>
    </div>

    <div class="dl-progress-stats">
      <div class="dl-stat"><span class="dl-stat-label">Progress</span><span class="dl-stat-val" id="dlPercentVal">0%</span></div>
      <div class="dl-stat"><span class="dl-stat-label">Downloaded</span><span class="dl-stat-val" id="dlSizeVal">0 MB / 0 MB</span></div>
      <div class="dl-stat"><span class="dl-stat-label">Speed</span><span class="dl-stat-val" id="dlSpeedVal">– MB/s</span></div>
      <div class="dl-stat"><span class="dl-stat-label">Time Left</span><span class="dl-stat-val" id="dlEtaVal">–</span></div>
    </div>
    <div class="progress-container" id="dlProgressWrap">
      <div class="progress-fill" id="dlProgressFill" style="width:0%"></div>
      <div class="progress-text" id="dlProgressText">0%</div>
    </div>
    <div class="dl-status-line" id="dlStatusLine">Getting things ready…</div>
    <div class="dl-error-banner hidden" id="dlErrorBanner"></div>
  </div>
</div>

<div class="wrap hidden" id="editorView" style="padding:0; max-width:none;">
  <!-- Auto Edit lives on THIS page now (no more window.open / new tab). It's the
       same Flask app/port serving /api/editor/editor, so an <iframe> to that
       same-origin URL shares localStorage automatically — the editor picks up
       whatever theme is active here, no separate host/port to mis-type. The
       src is only set the first time this tab is opened, so nothing in the
       editor runs until the user actually wants it. -->
  <iframe id="editorFrame" title="Auto Edit"
    style="display:block; width:100%; height:calc(100vh - 90px); border:none; background:var(--bg);">
  </iframe>
</div>

<div class="wrap hidden" id="publishView" style="padding:0; max-width:none;">
  <!-- Publish Studio: same-origin iframe to /api/publish/publish, same pattern
       as the Auto Edit tab above. Lazy-loaded — nothing here runs until the
       user actually opens this tab. -->
  <iframe id="publishFrame" title="Publish Studio"
    style="display:block; width:100%; height:calc(100vh - 90px); border:none; background:var(--bg);">
  </iframe>
</div>

<script>
// ── Multi-ratio export helper (Opus-Clip style) ────────────────────────────
function getSelectedRatios() {
  const boxes = document.querySelectorAll('.ratioBox:checked');
  const vals = Array.from(boxes).map(b => b.value);
  return vals.length ? vals : ['9:16'];   // kuch bhi check na ho toh default
}
// ── Theme system ─────────────────────────────────────────────────────────
(function(){
  const saved = localStorage.getItem('shortsStudioTheme') || 'dark-violet';
  if(saved !== 'dark-violet') document.documentElement.setAttribute('data-theme', saved);
})();
function toggleThemeMenu(){
  document.getElementById('themeMenu').classList.toggle('hidden');
  markActiveTheme();
}
function setTheme(name){
  if(name === 'dark-violet'){
    document.documentElement.removeAttribute('data-theme'); // it's the default :root palette
  } else {
    document.documentElement.setAttribute('data-theme', name);
  }
  localStorage.setItem('shortsStudioTheme', name);
  document.getElementById('themeMenu').classList.add('hidden');
}
function markActiveTheme(){
  const current = localStorage.getItem('shortsStudioTheme') || 'dark-violet';
  document.querySelectorAll('.theme-opt').forEach(el=>{
    el.classList.toggle('active', el.getAttribute('data-t') === current);
  });
}
document.addEventListener('click', function(e){
  const picker = document.querySelector('.theme-picker');
  const menu = document.getElementById('themeMenu');
  if(picker && menu && !picker.contains(e.target)) menu.classList.add('hidden');
});

let currentClipId = null, ttsUrl = null, audioFileUrl = null, logoUrl = null;
let regions = [], logoState = {x:0.78,y:0.04};
let textLayers = []; // {id, content, x, y, size, color, box}
let tool = 'blur';
let panX = 0, panY = 0; // manual drag-to-pan, -1..1 each axis, only used when zoom>1
let clipSettingsMap = {};

let audioLibraryFiles = null;
let exportQueue = [];
let exportQueueActive = false;
let currentPresetName = 'none';
let allClips = [];
let lastSourceUrl = '';       // URL of the last fetched source video (for caption auto-fetch)
let fetchedCaptionLangs = []; // languages returned by the last /api/subtitles/list call

const JS_COLOR_PRESETS = {
  none: {},
  vivid: {contrast: 1.15, saturation: 1.35, brightness: 0.02},
  cinematic: {contrast: 1.2, saturation: 0.85, brightness: -0.02},
  moody: {contrast: 1.25, saturation: 0.7, brightness: -0.05},
  vintage_vhs: {contrast: 0.95, saturation: 0.8, brightness: 0.0},
  warm_glow: {contrast: 1.08, saturation: 1.2, brightness: 0.03},
  cool_blue: {contrast: 1.1, saturation: 1.05, brightness: 0.0}
};

function toggleMode(){
  const m = document.getElementById('mode').value;
  document.getElementById('autoLenWrap').classList.toggle('hidden', m!=='auto');
  document.getElementById('manualWrap').classList.toggle('hidden', m!=='manual');
  document.getElementById('hooksWrap').classList.toggle('hidden', m!=='hooks');
}

// Hook detection: chalta hai jab mode="hooks" ho. Job start karta hai,
// live status/progress dikhata hai, aur result milte hi unhi timestamps ko
// "manual" mode ke ranges textarea me fill karke NORMAL fetchAndCut() flow
// ko trigger kar deta hai — cutting-pipeline bilkul waisi hi chalti hai
// jaisi manual mode me chalti hai, kuch alag nahi.
async function detectHooksAndFillRanges(){
  const url = document.getElementById('ytUrl').value.trim();
  if(!url){ alert('Pehle YouTube URL daalo'); return false; }

  document.getElementById('hooksCard').classList.remove('hidden');
  document.getElementById('hooksResults').innerHTML = '';
  document.getElementById('hooksStatusLine').innerText = '🚀 Hook-detection job shuru ho raha hai...';

  const payload = {
    url,
    num_clips: parseInt(document.getElementById('hookNumClips').value || '5'),
    min_len: parseInt(document.getElementById('hookMinLen').value || '15'),
    max_len: parseInt(document.getElementById('hookMaxLen').value || '60'),
  };

  let res, data;
  try{
    res = await fetch('/hooks/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    data = await res.json();
  } catch(e){
    document.getElementById('hooksStatusLine').innerText = '❌ Network error hook-job start karte waqt';
    return false;
  }
  if(data.error){
    document.getElementById('hooksStatusLine').innerText = '❌ ' + data.error;
    return false;
  }

  const jobId = data.job_id;
  const stageLabels = {
    starting: 'Shuru ho raha hai...',
    fetching_captions: '📝 Captions fetch ho rahi hain...',
    stage1_analysis: '🤔 Stage 1 — candidates dhoondh rahe hain...',
    stage2_selection: '🎯 Stage 2 — best clips select ho rahe hain...',
  };

  while(true){
    await new Promise(r=>setTimeout(r, 1500));
    let st, sd;
    try{
      st = await fetch(`/hooks/job/${jobId}`);
      sd = await st.json();
    } catch(e){ continue; }
    if(sd.error){
      document.getElementById('hooksStatusLine').innerText = '❌ ' + sd.error;
      return false;
    }
    if(sd.status === 'running'){
      document.getElementById('hooksStatusLine').innerText = stageLabels[sd.stage] || ('⏳ ' + sd.stage);
      continue;
    }
    if(sd.status === 'error'){
      document.getElementById('hooksStatusLine').innerText = '❌ ' + sd.error;
      return false;
    }
    if(sd.status === 'done'){
      const moments = sd.moments || [];
      document.getElementById('hooksStatusLine').innerText = `✅ "${sd.title}" — ${moments.length} hook(s) mile, ab inhi ko cut kiya ja raha hai...`;
      document.getElementById('hooksResults').innerHTML = moments.map(m => `
        <div class="clip-card" style="padding:12px; cursor:default;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="badge">${m.start} – ${m.end}</span>
            <span class="badge" style="background:rgba(34,211,196,.18); color:#7bf0e0;">${m.virality_score}/100</span>
          </div>
          <div style="margin-top:8px; font-size:13px;">${m.dialogue}</div>
          <div class="sub" style="margin-top:6px; font-style:italic;">${m.structure_check || ''}</div>
          <div style="margin-top:8px; display:flex; gap:6px; flex-wrap:wrap; font-size:11px;">
            <span class="badge">${m.hook_type}</span>
            <span class="badge">Standalone ${m.standalone_clarity}/10</span>
            <span class="badge">Hook ${m.hook_strength}/10</span>
            <span class="badge">Emotion ${m.emotional_intensity}/10</span>
            <span class="badge">Quotable ${m.quotability}/10</span>
          </div>
        </div>
      `).join('');

      // Ranges ko manual-mode textarea me fill karo, mode ko manual set karo
      // taaki neeche ka EXISTING fetchAndCut() flow bilkul waisa hi chale.
      const ranges = moments.map(m => `${Math.floor(m.start_sec)}-${Math.ceil(m.end_sec)}`).join('\n');
      document.getElementById('ranges').value = ranges;
      document.getElementById('mode').value = 'manual';
      toggleMode();
      return true;
    }
  }
}

function log(el, msg){ const l=document.getElementById(el); l.innerHTML += msg+"<br>"; l.scrollTop=l.scrollHeight; }

function addClipCard(c){
  document.getElementById('clipsCard').classList.remove('hidden');
  const grid = document.getElementById('clipGrid');
  
  // Ensure we don't duplicate card elements if called twice for some reason
  let div = document.getElementById('card_' + c.clip_id);
  if (div) {
    return;
  }
  
  // Track all clips for batch operations
  if (!allClips.some(x => x.clip_id === c.clip_id)) {
    allClips.push(c);
  }
  document.getElementById('clipCount').innerText = allClips.length + ' clips';

  // Initialize clipSettingsMap for this clip
  if (!clipSettingsMap[c.clip_id]) {
    // 1. Color variation
    let selectedPreset = 'none';
    let contrastVal = 1.12;
    let saturationVal = 1.25;
    let brightnessVal = 0.02;
    
    const colorsChecked = document.getElementById('automodColorVary') ? document.getElementById('automodColorVary').checked : true;
    if (colorsChecked) {
      const presets = ['vivid', 'cinematic', 'moody', 'warm_glow', 'cool_blue'];
      selectedPreset = presets[Math.floor(Math.random() * presets.length)];
      const vals = JS_COLOR_PRESETS[selectedPreset];
      contrastVal = vals.contrast;
      saturationVal = vals.saturation;
      brightnessVal = vals.brightness;
    }
    
    // 2. Watermark / Logo text
    const watermarkText = document.getElementById('automodWatermark') ? document.getElementById('automodWatermark').value : 'FondPeace.com';
    const watermarkSize = document.getElementById('automodFontSize') ? parseInt(document.getElementById('automodFontSize').value) : 36;
    
    const defaultTexts = [];
    if (watermarkText) {
      defaultTexts.push({
        id: 'txt_watermark',
        content: watermarkText,
        x: 0.55, // Center right x-coordinate
        y: 0.45, // Center y-coordinate
        size: watermarkSize,
        color: '#ffffff',
        box: false,
        enabled: true
      });
    }
    
    // 3. Mirroring
    const mirrorChecked = document.getElementById('automodMirror') ? document.getElementById('automodMirror').checked : true;
    // Alternate mirroring or always on? Let's make it so if mirror is checked, we alternate or apply it
    // Let's make it alternate: odd indexed clips are mirrored, even are normal, to make them non-similar
    const isMirrored = mirrorChecked ? (c.index % 2 === 1) : false;
    
    // 4. Background Audio selection round-robin
    const replaceAudioChecked = document.getElementById('automodAudioReplace') ? document.getElementById('automodAudioReplace').checked : true;
    let chosenAudioUrl = null;
    if (replaceAudioChecked && audioLibraryFiles && audioLibraryFiles.length > 0) {
      chosenAudioUrl = audioLibraryFiles[(c.index - 1) % audioLibraryFiles.length].url;
    }

    const grainChecked = document.getElementById('automodFilmGrain') ? document.getElementById('automodFilmGrain').checked : true;
    const pitchChecked = document.getElementById('automodAudioPitch') ? document.getElementById('automodAudioPitch').checked : true;
    const vignetteChecked = document.getElementById('automodVignette') ? document.getElementById('automodVignette').checked : true;
    
    clipSettingsMap[c.clip_id] = {
      speed: 0.75,
      zoom: 1.20,
      pan_x: 0,
      pan_y: 0,
      contrast: contrastVal,
      saturation: saturationVal,
      brightness: brightnessVal,
      sharpen: true,
      enhance: true,
      film_grain: grainChecked,
      audio_pitch: pitchChecked,
      vignette: vignetteChecked,
      color_preset: selectedPreset,
      ratios: getSelectedRatios(),
      format: 'mp4',
      crf: 18,
      preset: 'fast',
      hw_accel: true,
      face_track: null,
      regions: [],
      audio_mode: chosenAudioUrl ? 'replace' : 'original',
      audio_file_url: chosenAudioUrl,
      tts_url: null,
      tts_mix: false,
      mute: false,
      texts: defaultTexts,
      logo: null,
      rotate: '0',
      hflip: isMirrored,
      stage_w: 360,
      stage_h: 640
    };
  }

  const s = clipSettingsMap[c.clip_id];

  div = document.createElement('div');
  div.className = 'clip-card';
  div.id = 'card_' + c.clip_id;
  div.dataset.index = c.index;
  
  // Set direct CSS mirror transform on card video if active
  const mirrorStyle = s.hflip ? 'transform: scaleX(-1);' : '';
  
  div.innerHTML = `
    <div class="clip-preview-wrap">
      <video src="/media/${c.clip_id}" muted style="${mirrorStyle}" loop></video>
      <div class="clip-status-overlay" id="status_overlay_${c.clip_id}">
        <div class="spinner"></div>
        <div class="status-text" id="status_txt_${c.clip_id}">Waiting to export...</div>
      </div>
    </div>
    <div class="clip-info" style="padding: 10px;">
      <div class="lbl" style="padding:0; margin-bottom:4px; font-weight:700; color:var(--text); font-size:13px;">${c.label}</div>
      
      <!-- Automated Settings Badges -->
      <div id="badges_${c.clip_id}"></div>
      
      <!-- Actions Grid -->
      <div class="clip-actions" id="actions_${c.clip_id}" style="margin-top: 10px; display: flex; flex-direction: column; gap: 6px;">
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 6px;">
          <button class="clip-card-btn edit-btn" style="padding: 6px 4px; font-size:11px;" onclick="event.stopPropagation(); openEditor('${c.clip_id}')">⚙️ Edit Settings</button>
          <button class="clip-card-btn mirror-btn" style="padding: 6px 4px; font-size:11px; background:#434348; color:#fff;" onclick="event.stopPropagation(); toggleCardMirror('${c.clip_id}')">🪞 Toggle Mirror</button>
        </div>
        <button class="clip-card-btn export-single-btn" style="width:100%; padding: 6px 4px; font-size:11px; background:var(--grad); color:#000;" onclick="event.stopPropagation(); exportSingleClip('${c.clip_id}')">🚀 Export Video</button>
      </div>
    </div>
  `;
  
  // Custom video play/pause on click
  const videoEl = div.querySelector('video');
  videoEl.onclick = (e) => {
    e.stopPropagation();
    if (videoEl.paused) {
      document.querySelectorAll('.clip-card video').forEach(v => {
        if (v !== videoEl) v.pause();
      });
      videoEl.play();
    } else {
      videoEl.pause();
    }
  };
  
  div.onclick = () => openEditor(c.clip_id);
  grid.appendChild(div);

  // Dynamic DOM sorting: keep the clips strictly sequential (Short 1, Short 2, Short 3...)
  const cards = Array.from(grid.children);
  cards.sort((a, b) => parseInt(a.dataset.index || 0) - parseInt(b.dataset.index || 0));
  cards.forEach(card => grid.appendChild(card));
  
  // Render settings badges inside the card
  updateClipCardBadge(c.clip_id);
}

async function uploadDeviceVideo(file){
  if(!file) return;
  setFetchBtnLoading(true, 'Uploading your video…');
  log('fetchLog', '⏳ Uploading your video from this device (stays local, no external server)...');
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/api/upload', {method:'POST', body: fd});
  const data = await res.json();
  if(!data.url){ log('fetchLog', '❌ Upload failed'); setFetchBtnLoading(false); return; }
  document.getElementById('ytUrl').value = ''; // visually clear the URL field, we're using the upload instead
  await fetchAndCut(data.url);
}

// Toggles the big "Fetch & Cut" button between its normal state and a
// disabled, spinning, "working on it" state — so the user always has a
// clear, immediate visual signal that something is happening the moment
// they add a URL/device video, without needing to click anything else.
function setFetchBtnLoading(isLoading, label){
  const btn = document.querySelector('.fetch-cta');
  if(!btn) return;
  if(isLoading){
    if(!btn.dataset.origHtml) btn.dataset.origHtml = btn.innerHTML;
    btn.disabled = true;
    btn.style.opacity = '0.85';
    btn.style.cursor = 'wait';
    btn.innerHTML = `<span class="btn-spinner"></span>&nbsp;${label || 'Working…'}`;
  } else {
    btn.disabled = false;
    btn.style.opacity = '';
    btn.style.cursor = '';
    if(btn.dataset.origHtml) btn.innerHTML = btn.dataset.origHtml;
  }
}


async function startFetchOrHooks(){
  const mode = document.getElementById('mode').value;
  if(mode === 'hooks'){
    const ok = await detectHooksAndFillRanges();
    if(!ok) return;   // hook-detection fail hui to cut mat karo
  }
  fetchAndCut();   // ab ye bilkul normal auto/manual flow hai, unchanged
}


async function fetchAndCut(localPath){
  const url = document.getElementById('ytUrl').value.trim();
  if(!localPath && !url){ alert('Paste a video URL first, or use "Upload from PC/Device"'); return; }
  // Remembered so the "Auto-fetch Captions" button in the editor knows which
  // source video's subtitle tracks to pull from — local uploads have no URL,
  // so there's simply nothing to auto-fetch for those clips.
  lastSourceUrl = localPath ? '' : url;
  
  setFetchBtnLoading(true, localPath ? 'Preparing uploaded video…' : 'Fetching video…');

  // Preload audio library upfront so card generation can use tracks round-robin immediately
  if (!audioLibraryFiles) {
    try {
      const audioRes = await fetch('/api/audio_library');
      const audioData = await audioRes.json();
      audioLibraryFiles = audioData.files || [];
    } catch(e) {
      console.warn("Could not preload audio library upfront", e);
    }
  }

  const mode = document.getElementById('mode').value;
  const startFromOneVal = document.getElementById('automodStartFromOne') ? document.getElementById('automodStartFromOne').checked : true;
  const payload = { 
    url, 
    local_path: localPath || '',
    quality: document.getElementById('quality').value, 
    mode,
    clip_len: document.getElementById('clipLen').value,
    start_from_one: startFromOneVal,
    fetch_word_captions: document.getElementById('automodWordCaptions') ? document.getElementById('automodWordCaptions').checked : false
  };
  if(mode === 'manual'){
    payload.ranges = document.getElementById('ranges').value.trim().split('\n').map(l=>l.split('-').map(Number)).filter(r=>r.length===2);
  }
  document.getElementById('clipGrid').innerHTML = '';
  document.getElementById('clipsCard').classList.add('hidden');
  
  // Reset all state for new batch
  allClips = [];
  exportQueue = [];
  exportQueueActive = false;
  const t0 = performance.now();
  log('fetchLog', localPath ? '⏳ Preparing your uploaded video...' : '⏳ Resolving video...');
  let res, data;
  try{
    res = await fetch('/api/fetch_and_cut', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    data = await res.json();
  } catch(e){
    log('fetchLog', '❌ Network error while starting fetch'); setFetchBtnLoading(false); return;
  }
  if(data.error){ log('fetchLog', '❌ '+data.error); setFetchBtnLoading(false); return; }
  const jobId = data.job_id;
  const processedClipIds = new Set();
  let done = false, total = 0;
  setFetchBtnLoading(true, 'Cutting shorts…');
  while(!done){
    await new Promise(r=>setTimeout(r, 700));
    const st = await fetch(`/api/cut_status/${jobId}`);
    const sd = await st.json();
    if(sd.error){ log('fetchLog', '❌ '+sd.error); setFetchBtnLoading(false); return; }
    total = sd.total;
    sd.clips.forEach(c=>{
      if (!processedClipIds.has(c.clip_id)) {
        processedClipIds.add(c.clip_id);
        addClipCard(c); // each short appears the moment it's ready, no waiting for the rest
        log('fetchLog', `✅ Short ${c.index}/${total||'?'} ready — "${c.label}"`);
        setFetchBtnLoading(true, `Cutting… ${processedClipIds.size}/${total||'?'} ready`);
        // Add to automatic queue for background export if Automod is enabled!
        const queueForExport = () => {
          if (document.getElementById('autoMode') && document.getElementById('autoMode').checked) {
            exportQueue.push(c);
            pumpExportQueue();
          }
        };
        // Opt-in only (checkboxes default to OFF): silently remove dead air
        // and/or run Smart Reframe speaker tracking right after this clip is
        // cut, before it's queued for export. Both remain fully
        // toggleable/undoable per-clip afterward from the Speed/Zoom tab.
        const autoDeadAirOn = document.getElementById('automodDeadAir') && document.getElementById('automodDeadAir').checked;
        const autoFaceTrackOn = document.getElementById('automodFaceTrack') && document.getElementById('automodFaceTrack').checked;
        if (autoDeadAirOn || autoFaceTrackOn) {
          const overlayTxt = document.getElementById('status_txt_' + c.clip_id);
          (async () => {
            if (autoDeadAirOn) {
              if (overlayTxt) overlayTxt.textContent = 'Removing dead air…';
              await autoApplyDeadAirSilently(c.clip_id);
            }
            if (autoFaceTrackOn) {
              if (overlayTxt) overlayTxt.textContent = 'Tracking speaker…';
              await autoApplyFaceTrackSilently(c.clip_id);
            }
          })().finally(queueForExport);
        } else {
          queueForExport();
        }
      }
    });
    done = sd.done;
  }
  const secs = ((performance.now()-t0)/1000).toFixed(1);
  log('fetchLog', `🎉 All ${total} short(s) from "${jobId}" ready in ${secs}s — click any to edit & save.`);
  loadAudioLibrary();
  setFetchBtnLoading(false);
}

// Pressing Enter in the URL field acts exactly like clicking "Fetch & Cut Video" —
// works the same on PC, tablet, or phone keyboards.
document.addEventListener('DOMContentLoaded', () => {
  const ytUrlInput = document.getElementById('ytUrl');
  if(ytUrlInput){
    ytUrlInput.addEventListener('keydown', (e) => {
      if(e.key === 'Enter'){
        e.preventDefault();
        fetchAndCut();
      }
    });
  }
});


function openEditor(clipId){
  currentClipId = clipId;
  const s = clipSettingsMap[clipId] || {
    speed: 0.75, zoom: 1.20, pan_x: 0, pan_y: 0,
    contrast: 1.12, saturation: 1.25, brightness: 0.02,
    sharpen: true, enhance: true, color_preset: 'none',
    ratios: ['9:16'], format: 'mp4', crf: 18, preset: 'fast', hw_accel: true,
    regions: [], audio_mode: 'original', audio_file_url: null,
    tts_url: null, tts_mix: false, texts: [], logo: null, rotate: '0', hflip: true
  };
  
  regions = s.regions || [];
  ttsUrl = s.tts_url;
  audioFileUrl = s.audio_file_url;
  logoUrl = s.logo ? s.logo.url : null;
  panX = s.pan_x || 0;
  panY = s.pan_y || 0;

  // Restore whichever voiceover preview matches the saved mode, so reopening
  // a clip's settings doesn't lose the "what am I currently using" context.
  const ttsAudioEl = document.getElementById('ttsAudio');
  if (ttsUrl) { ttsAudioEl.src = ttsUrl; ttsAudioEl.classList.remove('hidden'); }
  else { ttsAudioEl.classList.add('hidden'); ttsAudioEl.removeAttribute('src'); }
  const recPreviewEl = document.getElementById('recPreview');
  if (s.audio_mode === 'record' && audioFileUrl) {
    recPreviewEl.src = audioFileUrl; recPreviewEl.classList.remove('hidden');
    document.getElementById('recChosenRow').classList.remove('hidden');
  } else {
    recPreviewEl.classList.add('hidden'); recPreviewEl.removeAttribute('src');
    document.getElementById('recChosenRow').classList.add('hidden');
  }
  
  // Clone text layers
  textLayers = s.texts ? s.texts.map(t => ({
    id: t.id || 'txt'+Date.now()+'_'+Math.random(),
    content: t.content,
    x: t.x,
    y: t.y,
    size: t.size,
    color: t.color,
    box: t.box,
    font: t.font || 'default',
    style: t.style || 'classic_box',
    boxColor: t.boxColor || t.box_color || '#000000',
    centerX: t.centerX || t.center_x || false,
    enabled: t.enabled !== false,
    start: t.start,
    end: t.end,
    source: t.source
  })) : [];
  
  document.getElementById('overlayLayer').innerHTML='';
  document.getElementById('textLayersWrap').innerHTML='';
  
  // word_caption layers ko ek hi grouped row me dikhao (100 alag rows
  // ki jagah), stage par har word apne asli time par individually dikhta rahega
  const wcapGroup = textLayers.filter(l => l.source === 'word_caption');
  const otherLayers = textLayers.filter(l => l.source !== 'word_caption');
  otherLayers.forEach(l => renderTextPanelRow(l));
  if (wcapGroup.length) renderWordCaptionGroupRow(wcapGroup);
  textLayers.forEach(l => renderTextOnStage(l));
  
  if (audioFileUrl) {
    document.getElementById('chosenAudioRow').classList.remove('hidden');
    document.getElementById('chosenAudioName').innerText = audioFileUrl.split('/').pop();
  } else {
    document.getElementById('chosenAudioRow').classList.add('hidden');
  }
  
  document.getElementById('colorPreset').value = s.color_preset || 'none';
  document.getElementById('enhance').checked = s.enhance;
  document.getElementById('sharpen').checked = s.sharpen;
  
  if (s.logo) {
    document.getElementById('logoEnabled').checked = true;
    document.getElementById('logoW').value = Math.round(s.logo.w * 100);
    document.getElementById('logoO').value = Math.round(s.logo.opacity * 100);
    logoUrl = s.logo.url;
    logoState = { x: s.logo.x, y: s.logo.y };
    renderLogo();
  } else {
    document.getElementById('logoEnabled').checked = false;
    const logoImg = document.getElementById('ovLogo');
    if (logoImg) logoImg.style.display = 'none';
  }
  
  document.getElementById('editorCard').classList.add('active');
  const player = document.getElementById('player');
  player.src = '/media/'+clipId;
  player.style.objectPosition = '50% 50%';
  player.muted = false;
  
  // Select matching audio mode radio
  const amodeRadios = document.querySelectorAll('input[name=amode]');
  amodeRadios.forEach(r => {
    r.checked = (r.value === s.audio_mode);
  });
  onAudioMode();
  
  // Set values and trigger callbacks
  document.getElementById('contrast').value = s.contrast;
  document.getElementById('saturation').value = s.saturation;
  document.getElementById('brightness').value = s.brightness;
  onColor();
  
  document.getElementById('speed').value = s.speed;
  onSpeed();
  
  document.getElementById('zoom').value = s.zoom;
  onZoom();
  
  document.getElementById('rotate').value = s.rotate;
  document.getElementById('hflip').checked = s.hflip;
  onTransform();
  
  updateDeadAirUI(clipId);
  updateFaceTrackUI(clipId);
  
  window.scrollTo({top: document.getElementById('editorCard').offsetTop-20, behavior:'smooth'});
}
function closeEditor(){ document.getElementById('editorCard').classList.remove('active'); document.getElementById('player').pause(); }

function togglePlay(){
  const p = document.getElementById('player');
  if(p.paused){ p.play(); document.getElementById('playBtn').innerText='⏸ Pause'; }
  else { p.pause(); document.getElementById('playBtn').innerText='▶ Play'; }
}

document.querySelectorAll('.tab').forEach(t=>{
  t.onclick = ()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tabpanel').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('panel-'+t.dataset.t).classList.add('active');
  };
});

function onSpeed(){
  const v = parseFloat(document.getElementById('speed').value);
  document.getElementById('speedVal').innerText = v.toFixed(2)+'x';
  document.getElementById('player').playbackRate = v;
  saveCurrentSettingsToMap();
}
function onZoom(){
  const v = parseFloat(document.getElementById('zoom').value);
  document.getElementById('zoomVal').innerText = v.toFixed(2)+'x';
  applyZoomPan(v);
  saveCurrentSettingsToMap();
}

// ---- Dead-air (silence) removal: manual Preview / Apply / Undo ----
// Nothing here ever runs automatically unless the "Auto-remove Dead Air"
// checkbox in the automation settings panel is turned on (off by default).
let deadAirPreviewMap = {};   // clip_id -> {keep_segments, removed_duration, silences} from last Preview
let deadAirBackupState = {};  // clip_id -> true once a backup exists server-side (i.e. Undo is possible)

let faceTrackMap = {};        // clip_id -> {track, faces_found} from last Detect
let faceTrackEnabledMap = {}; // clip_id -> whether "Follow face" is turned on for that clip

// Restores the Detect/Follow-face UI state for whichever clip is open —
// called from openEditor() so switching clips doesn't carry over stale state.
function updateFaceTrackUI(clipId){
  const statusEl = document.getElementById('faceTrackStatus');
  const checkbox = document.getElementById('faceTrackEnabled');
  const removeBtn = document.getElementById('faceTrackRemoveBtn');
  const found = faceTrackMap[clipId];
  if(found && found.faces_found){
    checkbox.disabled = false;
    checkbox.checked = !!faceTrackEnabledMap[clipId];
    if(removeBtn) removeBtn.disabled = false;
    const engineTag = found.engine === 'enterprise' ? 'Enterprise engine — Kalman-smoothed, speaker-aware' : 'basic detector';
    statusEl.textContent = faceTrackEnabledMap[clipId]
      ? `✅ Following the detected speaker across ${found.track.length} tracked point(s) (${engineTag}). Zoom in above 1.00x to see it kick in.`
      : `🔍 Speaker found (${found.track.length} tracked point(s), ${engineTag}) — tick "Follow speaker" to use it.`;
  } else {
    checkbox.disabled = true;
    checkbox.checked = false;
    if(removeBtn) removeBtn.disabled = true;
    statusEl.textContent = found ? '⚠️ No face detected in this clip — using your normal fixed/manual-pan crop.' : '';
  }
}

// Fully removes any Smart Reframe tracking data/state for the currently
// open clip — unticks "Follow speaker", clears the cached track, and wipes
// face_track from that clip's saved settings so export goes back to the
// normal fixed/manual-pan crop. Doesn't touch other clips.
function removeFaceTrack(){
  if(!currentClipId) return;
  const clipId = currentClipId;
  faceTrackMap[clipId] = null;
  faceTrackEnabledMap[clipId] = false;
  if(clipSettingsMap[clipId]) clipSettingsMap[clipId].face_track = null;
  updateFaceTrackUI(clipId);
  updateClipCardBadge(clipId);
  if (typeof saveCurrentSettingsToMap === 'function') saveCurrentSettingsToMap();
}

// Silent variant of detectFaceTrack() used by the "Auto-apply Smart Reframe
// on cut" automation setting: runs right after a short is cut, before it's
// queued for export, with no button/status-text interaction required.
async function autoApplyFaceTrackSilently(clipId){
  try{
    const res = await fetch('/api/detect_face_track', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({clip_id: clipId})
    });
    const data = await res.json();
    if(!data.error && data.faces_found){
      faceTrackMap[clipId] = data;
      faceTrackEnabledMap[clipId] = true;
      if(clipSettingsMap[clipId]){
        clipSettingsMap[clipId].face_track = data.track;
        // Smart Reframe only has visible effect once zoomed in past 1.00x —
        // nudge the default zoom up a touch so the automation actually shows.
        if(!clipSettingsMap[clipId].zoom || clipSettingsMap[clipId].zoom <= 1.0){
          clipSettingsMap[clipId].zoom = 1.20;
        }
      }
    } else {
      faceTrackMap[clipId] = (!data.error) ? data : null;
      faceTrackEnabledMap[clipId] = false;
    }
    updateClipCardBadge(clipId);
    if(currentClipId === clipId) updateFaceTrackUI(clipId);
  } catch(e){
    console.warn('Auto Smart Reframe failed for', clipId, e);
  }
}

async function detectFaceTrack(){
  if(!currentClipId) return;
  const clipId = currentClipId;
  const btn = document.getElementById('faceTrackDetectBtn');
  const statusEl = document.getElementById('faceTrackStatus');
  btn.disabled = true;
  statusEl.textContent = '⏳ Scanning frames for a face (this can take a few seconds)…';
  try{
    const res = await fetch('/api/detect_face_track', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({clip_id: clipId})
    });
    const data = await res.json();
    btn.disabled = false;
    if(data.error){
      statusEl.textContent = '❌ ' + data.error + (data.needs_opencv ? ' (this feature is optional — everything else still works fine without it)' : '');
      faceTrackMap[clipId] = null;
      updateFaceTrackUI(clipId);
      return;
    }
    faceTrackMap[clipId] = data;
    if(!data.faces_found) faceTrackEnabledMap[clipId] = false;
    updateFaceTrackUI(clipId);
    if(zoom <= 1.001 && data.faces_found){
      statusEl.textContent += ' Tip: zoom in above 1.00x for the face-follow crop to actually take effect.';
    }
  } catch(e){
    btn.disabled = false;
    statusEl.textContent = '❌ Network error while detecting the face.';
  }
}

function onFaceTrackToggle(){
  if(!currentClipId) return;
  faceTrackEnabledMap[currentClipId] = document.getElementById('faceTrackEnabled').checked;
  updateFaceTrackUI(currentClipId);
  if (typeof saveCurrentSettingsToMap === 'function') saveCurrentSettingsToMap();
}


function _deadAirStatusEls(){
  return [document.getElementById('deadAirStatus'), document.getElementById('deadAirStatus2')].filter(Boolean);
}
function _deadAirSetStatus(msg){
  _deadAirStatusEls().forEach(el => el.textContent = msg);
}
function _deadAirSetButtons({previewDisabled, applyDisabled, undoDisabled}){
  const previewBtn = document.getElementById('deadAirPreviewBtn');
  if(previewBtn && previewDisabled !== undefined) previewBtn.disabled = previewDisabled;
  [document.getElementById('deadAirApplyBtn'), document.getElementById('deadAirApplyBtn2')].forEach(b=>{
    if(b && applyDisabled !== undefined) b.disabled = applyDisabled;
  });
  [document.getElementById('deadAirUndoBtn'), document.getElementById('deadAirUndoBtn2')].forEach(b=>{
    if(b && undoDisabled !== undefined) b.disabled = undoDisabled;
  });
}

// Restores the Preview/Apply/Undo button state for whichever clip is open —
// called from openEditor() so switching clips doesn't carry over stale state.
function updateDeadAirUI(clipId){
  _deadAirSetStatus('');
  _deadAirSetButtons({
    previewDisabled: false,
    applyDisabled: !deadAirPreviewMap[clipId],   // Apply enabled once a Preview has been run
    undoDisabled: !deadAirBackupState[clipId]     // Undo enabled once a cut has actually been applied
  });
}

function _reloadPlayerAndDuration(clipId, newDuration){
  const player = document.getElementById('player');
  if(player && currentClipId === clipId){
    const wasPaused = player.paused;
    player.src = '/media/' + clipId + '?t=' + Date.now();
    player.load();
    if(!wasPaused) player.play().catch(()=>{});
  }
  const clip = allClips.find(c => c.clip_id === clipId);
  if(clip && newDuration) clip.duration = newDuration;
  const cardVideo = document.querySelector(`#card_${clipId} video`);
  if(cardVideo){ cardVideo.src = '/media/' + clipId + '?t=' + Date.now(); }
}

async function previewDeadAir(){
  if(!currentClipId) return;
  const clipId = currentClipId;
  _deadAirSetButtons({previewDisabled: true});
  _deadAirSetStatus('⏳ Scanning for silent gaps…');
  try{
    const res = await fetch('/api/detect_silence', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({clip_id: clipId})
    });
    const data = await res.json();
    if(data.error){ _deadAirSetStatus('❌ ' + data.error); _deadAirSetButtons({previewDisabled:false}); return; }
    if(!data.silences || data.silences.length === 0){
      deadAirPreviewMap[clipId] = null;
      _deadAirSetStatus('ℹ️ No significant silence found in this clip.');
      _deadAirSetButtons({previewDisabled:false, applyDisabled:true});
      return;
    }
    deadAirPreviewMap[clipId] = data;
    _deadAirSetStatus(`🔍 Found ${data.silences.length} silent gap(s) totalling ${data.removed_duration}s. Click Apply Cut to actually remove them.`);
    _deadAirSetButtons({previewDisabled:false, applyDisabled:false});
  } catch(e){
    _deadAirSetStatus('❌ Network error while scanning for silence.');
    _deadAirSetButtons({previewDisabled:false});
  }
}

async function applyDeadAir(){
  if(!currentClipId) return;
  const clipId = currentClipId;
  const preview = deadAirPreviewMap[clipId];
  _deadAirSetButtons({previewDisabled:true, applyDisabled:true, undoDisabled:true});
  _deadAirSetStatus('⏳ Cutting out silence (re-encoding, may take a few seconds)…');
  try{
    const body = {clip_id: clipId};
    if(preview && preview.keep_segments) body.keep_segments = preview.keep_segments;
    const res = await fetch('/api/apply_silence_cut', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
    const data = await res.json();
    if(data.error){
      _deadAirSetStatus('❌ ' + data.error);
      updateDeadAirUI(clipId);
      return;
    }
    if(data.removed_duration && data.removed_duration > 0){
      _deadAirSetStatus(`✅ Removed ${data.removed_duration}s of dead air — clip is now ${data.new_duration}s (was ${data.old_duration}s). You can Undo this if needed.`);
    } else {
      _deadAirSetStatus('ℹ️ ' + (data.message || 'No significant silence found in this clip.'));
    }
    deadAirBackupState[clipId] = !!data.undo_available;
    deadAirPreviewMap[clipId] = null; // that preview has now been applied (or was empty); re-preview for another pass
    _reloadPlayerAndDuration(clipId, data.new_duration);
    updateClipCardBadge(clipId);
  } catch(e){
    _deadAirSetStatus('❌ Network error while removing dead air.');
  }
  updateDeadAirUI(clipId);
}

async function undoDeadAir(){
  if(!currentClipId) return;
  const clipId = currentClipId;
  _deadAirSetButtons({previewDisabled:true, applyDisabled:true, undoDisabled:true});
  _deadAirSetStatus('⏳ Undoing dead-air cut, restoring the previous version…');
  try{
    const res = await fetch('/api/undo_silence_cut', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({clip_id: clipId})
    });
    const data = await res.json();
    if(data.error){
      _deadAirSetStatus('❌ ' + data.error);
      updateDeadAirUI(clipId);
      return;
    }
    deadAirBackupState[clipId] = false;
    deadAirPreviewMap[clipId] = null;
    _deadAirSetStatus(`↩️ Undone — clip restored to ${data.new_duration}s (before the dead-air cut).`);
    _reloadPlayerAndDuration(clipId, data.new_duration);
    updateClipCardBadge(clipId);
  } catch(e){
    _deadAirSetStatus('❌ Network error while undoing.');
  }
  updateDeadAirUI(clipId);
}

// Used by the "Auto-remove Dead Air" automation toggle: runs Apply silently
// right after a clip is cut, before it goes into the export queue. Returns
// a Promise so the caller can wait for it before queuing export.
async function autoApplyDeadAirSilently(clipId){
  try{
    const res = await fetch('/api/apply_silence_cut', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({clip_id: clipId})
    });
    const data = await res.json();
    if(!data.error){
      deadAirBackupState[clipId] = !!data.undo_available;
      if(data.new_duration){
        const clip = allClips.find(c => c.clip_id === clipId);
        if(clip) clip.duration = data.new_duration;
      }
      const cardVideo = document.querySelector(`#card_${clipId} video`);
      if(cardVideo){ cardVideo.src = '/media/' + clipId + '?t=' + Date.now(); }
      updateClipCardBadge(clipId);
    }
  } catch(e){
    console.warn('Auto dead-air removal failed for', clipId, e);
  }
}
function updatePreviewTransforms() {
  const v = parseFloat(document.getElementById('zoom').value || 1.0);
  const rotate = document.getElementById('rotate').value || '0';
  const hflip = document.getElementById('hflip').checked;
  const player = document.getElementById('player');
  
  const tx = panX * 50 * (v-1)/Math.max(v,1.001);
  const ty = panY * 50 * (v-1)/Math.max(v,1.001);
  
  let transformStr = `scale(${v}) translate(${tx}%, ${ty}%) `;
  if (hflip) transformStr += "scaleX(-1) ";
  if (rotate !== "0") transformStr += `rotate(${rotate}deg) `;
  
  player.style.transform = transformStr.trim();
}
function applyZoomPan(v){
  updatePreviewTransforms();
}
(function initPanDrag(){
  const player = document.getElementById('player');
  player.addEventListener('mousedown', (e)=>{
    const v = parseFloat(document.getElementById('zoom').value);
    if(v <= 1.001) return; // nothing to pan when not zoomed
    e.preventDefault(); e.stopPropagation();
    const stage = document.getElementById('stage');
    const rect = stage.getBoundingClientRect();
    const startX = e.clientX, startY = e.clientY, startPanX = panX, startPanY = panY;
    function move(ev){
      panX = Math.max(-1, Math.min(1, startPanX - (ev.clientX-startX)/rect.width*2));
      panY = Math.max(-1, Math.min(1, startPanY - (ev.clientY-startY)/rect.height*2));
      applyZoomPan(parseFloat(document.getElementById('zoom').value));
    }
    function up(){ document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); }
    document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
  });
})();
function onColor(){
  const c = document.getElementById('contrast').value, s = document.getElementById('saturation').value, b = document.getElementById('brightness').value;
  document.getElementById('contrastVal').innerText = parseFloat(c).toFixed(2);
  document.getElementById('satVal').innerText = parseFloat(s).toFixed(2);
  document.getElementById('brightVal').innerText = parseFloat(b).toFixed(2);
  const brightPct = 1 + parseFloat(b);
  document.getElementById('player').style.filter = `contrast(${c}) saturate(${s}) brightness(${brightPct})`;
  saveCurrentSettingsToMap();
}
function resetColor(){
  document.getElementById('contrast').value=1.12; document.getElementById('saturation').value=1.25;
  document.getElementById('brightness').value=0.02; document.getElementById('sharpen').checked=true;
  document.getElementById('enhance').checked=true;
  document.getElementById('colorPreset').value='none';
  onColor();
}

function onAudioMode(){
  const mode = document.querySelector('input[name=amode]:checked').value;
  document.getElementById('replaceWrap').classList.toggle('hidden', mode!=='replace');
  document.getElementById('ttsWrap').classList.toggle('hidden', mode!=='tts');
  document.getElementById('recordWrap').classList.toggle('hidden', mode!=='record');
  document.getElementById('player').muted = (mode==='mute');
  if(mode === 'replace') loadAudioLibrary();
  if(mode === 'tts') loadTtsLibrary();
  saveCurrentSettingsToMap();
}

let audioLibCache = null;
async function loadAudioLibrary(){
  const grid = document.getElementById('audioLibGrid');
  if(audioLibCache){ renderAudioLibrary(); return; }
  const res = await fetch('/api/audio_library');
  const data = await res.json();
  audioLibCache = data.files || [];
  renderAudioLibrary();
}

async function rescanAudioLibrary(){
  audioLibCache = null; // force a fresh fetch, which re-scans device folders too
  await loadAudioLibrary();
}

async function syncGithubAudio(){
  const manifestUrl = prompt(
    "Paste the raw GitHub URL of your audio manifest JSON\n" +
    "(a file listing [{name,url}, ...] pointing at your .mp3 files on GitHub):"
  );
  if(!manifestUrl) return;
  const res = await fetch('/api/sync_github_audio', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({manifest_url: manifestUrl})
  });
  const data = await res.json();
  if(data.error){ alert("Sync failed: " + data.error); return; }
  alert(`Added ${data.added.length} new track(s) from GitHub.`);
  await rescanAudioLibrary();
}

function renderAudioLibrary(){
  const grid = document.getElementById('audioLibGrid'); grid.innerHTML='';
  if(!audioLibCache.length){
    grid.innerHTML = '<p class="sub">No .mp3 files found — drop some into the <code>audio/</code> folder next to the script and reopen this panel.</p>';
    return;
  }
  audioLibCache.forEach(f=>{
    const card = document.createElement('div'); card.className='audio-card';
    card.innerHTML = `<div class="name">🎵 ${f.name}</div>
      <audio id="prev_${f.filename}" src="${f.url}" style="display:none"></audio>
      <div class="arow">
        <button onclick="previewAudio('${f.filename}')">▶ Play</button>
        <button id="sel_${f.filename}" onclick="selectLibAudio('${f.url}','${f.name}','${f.filename}')">+ Add</button>
      </div>`;
    grid.appendChild(card);
  });
}
let currentPreview = null;
function previewAudio(fname){
  if(currentPreview && currentPreview !== fname){
    const prev = document.getElementById('prev_'+currentPreview);
    if(prev){ prev.pause(); prev.currentTime = 0; }
  }
  const a = document.getElementById('prev_'+fname);
  if(a.paused){ a.play(); currentPreview = fname; } else { a.pause(); }
}
function selectLibAudio(url, name, fname){
  audioFileUrl = url;
  document.querySelectorAll('.audio-card button.sel').forEach(b=>b.classList.remove('sel'));
  document.getElementById('sel_'+fname).classList.add('sel');
  document.getElementById('chosenAudioRow').classList.remove('hidden');
  document.getElementById('chosenAudioName').innerText = name;
  saveCurrentSettingsToMap();
}

async function uploadAudio(){
  const f = document.getElementById('audioFile').files[0]; if(!f) return;
  const fd = new FormData(); fd.append('file', f);
  const res = await fetch('/api/upload', {method:'POST', body: fd});
  const data = await res.json(); audioFileUrl = data.url;
  document.querySelectorAll('.audio-card button.sel').forEach(b=>b.classList.remove('sel'));
  document.getElementById('chosenAudioRow').classList.remove('hidden');
  document.getElementById('chosenAudioName').innerText = f.name + ' (uploaded)';
  saveCurrentSettingsToMap();
}
async function uploadLogo(){
  const f = document.getElementById('logoFile').files[0]; if(!f) return;
  const fd = new FormData(); fd.append('file', f);
  const res = await fetch('/api/upload', {method:'POST', body: fd});
  const data = await res.json(); logoUrl = data.url;
  renderLogo();
}
let pendingEmojiUrl = null;
async function uploadEmoji(){
  const f = document.getElementById('emojiFile').files[0]; if(!f) return;
  const fd = new FormData(); fd.append('file', f);
  const res = await fetch('/api/upload', {method:'POST', body: fd});
  const data = await res.json(); pendingEmojiUrl = data.url;
  document.getElementById('emojiPickStatus').textContent = '✅ Using your uploaded image.';
}
// Step 7: quick native-emoji picker. We don't send the raw unicode character
// to ffmpeg - drawtext usually can't render color emoji glyphs (most system
// fonts have no color bitmap support), so it would likely show as a blank
// box in the exported video. Instead we draw the emoji onto a canvas here
// (using the browser's own emoji font, which reliably renders in color),
// turn that into a real PNG, and upload it through the exact same endpoint
// an uploaded sticker image would use - so it's just a normal emoji_url
// region under the hood and renders correctly in the final export.
const QUICK_EMOJIS = ['😀','😂','😍','😎','🔥','❤️','👍','🎉','😢','😱','💯','⭐','👀','🤔','😴','🙌'];
(function buildEmojiQuickPick(){
  const wrap = document.getElementById('emojiQuickPick');
  if(!wrap) return;
  QUICK_EMOJIS.forEach(ch=>{
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'tool-btn'; btn.style.fontSize = '20px';
    btn.textContent = ch;
    btn.onclick = ()=>pickEmojiChar(ch, btn);
    wrap.appendChild(btn);
  });
})();
async function pickEmojiChar(ch, btnEl){
  const statusEl = document.getElementById('emojiPickStatus');
  statusEl.textContent = '⏳ Preparing emoji…';
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, size, size);
  ctx.font = (size*0.8) + 'px "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(ch, size/2, size/2 + size*0.05);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
  if(!blob){ statusEl.textContent = '❌ Could not render that emoji on this device.'; return; }
  const fd = new FormData();
  fd.append('file', blob, 'emoji.png');
  try{
    const res = await fetch('/api/upload', {method:'POST', body: fd});
    const data = await res.json();
    pendingEmojiUrl = data.url;
    document.querySelectorAll('#emojiQuickPick .tool-btn').forEach(b=>b.classList.remove('active'));
    if(btnEl) btnEl.classList.add('active');
    statusEl.textContent = `✅ ${ch} ready — now drag on the preview to place it.`;
  }catch(err){
    statusEl.textContent = '❌ Network error preparing that emoji.';
  }
}

async function generateTTS(){
  const voice = document.getElementById('ttsVoice').value;
  const emotion = document.getElementById('ttsEmotion').value;
  const text = document.getElementById('ttsText').value.trim();
  if(!text){ alert('Type a script first'); return; }
  const res = await fetch('/api/tts', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({voice, text, emotion})});
  const data = await res.json();
  if(data.error){ alert(data.error); return; }
  ttsUrl = data.url;
  const a = document.getElementById('ttsAudio'); a.src = data.url; a.classList.remove('hidden'); a.play();
  saveCurrentSettingsToMap();
  loadTtsLibrary(); // new generation shows up at the top of "Your generated voices"
}

let ttsLibraryCache = [];
async function loadTtsLibrary(){
  let res, data;
  try{
    res = await fetch('/api/tts_library');
    data = await res.json();
  } catch(e){ return; }
  ttsLibraryCache = data.entries || [];
  renderTtsLibrary();
}

function renderTtsLibrary(){
  const wrap = document.getElementById('ttsLibraryList');
  if(!wrap) return;
  wrap.innerHTML = '';
  if(!ttsLibraryCache.length){
    wrap.innerHTML = '<p class="sub">No voiceovers generated yet this session.</p>';
    return;
  }
  ttsLibraryCache.forEach(e => {
    const row = document.createElement('div');
    row.className = 'audio-card';
    row.id = 'ttsLib_' + e.id;
    const isActive = (ttsUrl === e.url);
    const safePreview = String(e.text_preview||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    row.innerHTML = `
      <div class="name">🎙 ${e.voice_label || e.voice_key} · ${e.emotion_label || e.emotion}${isActive ? ' <span class="badge">Using this</span>' : ''}</div>
      <div class="sub" style="margin:2px 0 6px">"${safePreview}"</div>
      <audio controls src="${e.url}" style="width:100%"></audio>
      <div class="arow" style="margin-top:6px">
        <button onclick="useTtsLibraryEntry('${e.id}')">${isActive ? '✓ In use' : '✔ Use this'}</button>
        <button onclick="deleteTtsLibraryEntry('${e.id}')">🗑 Delete</button>
      </div>`;
    wrap.appendChild(row);
  });
}

function useTtsLibraryEntry(id){
  const e = ttsLibraryCache.find(x => x.id === id);
  if(!e) return;
  ttsUrl = e.url;
  const a = document.getElementById('ttsAudio'); a.src = e.url; a.classList.remove('hidden');
  document.getElementById('ttsVoice').value = e.voice_key;
  document.getElementById('ttsEmotion').value = e.emotion;
  document.getElementById('ttsText').value = e.text;
  saveCurrentSettingsToMap();
  renderTtsLibrary(); // refresh "Using this" badge
}

async function deleteTtsLibraryEntry(id){
  const e = ttsLibraryCache.find(x => x.id === id);
  if(!confirm('Delete this generated voice? This removes the temp audio file too.')) return;
  try{
    await fetch('/api/tts_library/delete', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({id})});
  } catch(err){ /* still refresh below, list will reflect real backend state */ }
  if(e && ttsUrl === e.url){
    // The clip's current voiceover was just deleted — fall back to original audio so
    // export doesn't point at a now-missing file.
    ttsUrl = null;
    document.getElementById('ttsAudio').classList.add('hidden');
    document.getElementById('ttsAudio').removeAttribute('src');
    saveCurrentSettingsToMap();
  }
  await loadTtsLibrary();
}

let mediaRecorder = null, recChunks = [], recStream = null, recTimerInt = null, recSeconds = 0;
async function toggleRecording(){
  const btn = document.getElementById('recBtn');
  if(mediaRecorder && mediaRecorder.state === 'recording'){
    mediaRecorder.stop();
    return;
  }
  try{
    recStream = await navigator.mediaDevices.getUserMedia({audio:true});
  }catch(e){
    alert('Could not access microphone: ' + e.message);
    return;
  }
  recChunks = [];
  mediaRecorder = new MediaRecorder(recStream);
  mediaRecorder.ondataavailable = e => { if(e.data.size>0) recChunks.push(e.data); };
  mediaRecorder.onstop = async () => {
    clearInterval(recTimerInt);
    document.getElementById('recTimer').textContent = '';
    recStream.getTracks().forEach(t=>t.stop());
    btn.textContent = '🔴 Start Recording';
    const blob = new Blob(recChunks, {type:'audio/webm'});
    const preview = document.getElementById('recPreview');
    preview.src = URL.createObjectURL(blob);
    preview.classList.remove('hidden');
    // Upload so it plugs into the exact same audio_file_url pipeline as any
    // other "replace audio" file - no separate backend handling needed.
    const fd = new FormData();
    fd.append('file', blob, 'recording.webm');
    const res = await fetch('/api/upload', {method:'POST', body: fd});
    const data = await res.json();
    if(data.url){
      audioFileUrl = data.url;
      document.getElementById('recChosenRow').classList.remove('hidden');
      saveCurrentSettingsToMap();
    }
  };
  mediaRecorder.start();
  btn.textContent = '⏹ Stop Recording';
  recSeconds = 0;
  document.getElementById('recTimer').textContent = '0:00';
  recTimerInt = setInterval(()=>{
    recSeconds++;
    const m = Math.floor(recSeconds/60), s = recSeconds%60;
    document.getElementById('recTimer').textContent = m+':'+String(s).padStart(2,'0');
  }, 1000);
}

function discardRecording(){
  audioFileUrl = null;
  document.getElementById('recPreview').classList.add('hidden');
  document.getElementById('recPreview').removeAttribute('src');
  document.getElementById('recChosenRow').classList.add('hidden');
  saveCurrentSettingsToMap();
}

// Step 3: font family + caption style catalogs. Keys must match the backend's
// FONT_CHOICES / STYLE_PRESETS dicts exactly - only the label text is repeated
// here for the dropdowns (kept in the frontend since the sets are small and
// static, no need for a round-trip /api/text_styles call).
const FONT_OPTIONS = [
  {key:'default',    label:'Default (System)'},
  {key:'arial',      label:'Arial'},
  {key:'arial_bold', label:'Arial Bold'},
  {key:'impact',     label:'Impact (Meme style)'},
  {key:'comic_sans', label:'Comic Sans MS'},
  {key:'times',      label:'Times New Roman'},
  {key:'georgia',    label:'Georgia'},
  {key:'verdana',    label:'Verdana'},
  {key:'courier',    label:'Courier New (Typewriter)'},
  {key:'trebuchet',  label:'Trebuchet MS'},
];
const STYLE_OPTIONS = [
  {key:'classic_box',   label:'Classic Box'},
  {key:'bold_outline',  label:'Bold Outline'},
  {key:'neon_glow',     label:'Neon Glow'},
  {key:'shadow_pop',    label:'Drop Shadow'},
  {key:'highlight_bar', label:'Highlight Bar (solid)'},
  {key:'outline_only',  label:'Outline Only'},
  {key:'minimal',       label:'Minimal'},
  {key:'meme_classic',  label:'Meme Classic 🔥'},
  {key:'gold_glow',     label:'Gold Glow ✨'},
  {key:'gradient_bar',  label:'Trendy Bar 🎨'},
];

// CSS font-family stacks used only for the live on-stage preview (the real
// rendered file uses the backend's actual .ttf file lookup, this is just so
// the preview looks close to what Save will produce).
const FONT_CSS_MAP = {
  default:'Arial, Helvetica, sans-serif', arial:'Arial, Helvetica, sans-serif',
  arial_bold:'Arial, Helvetica, sans-serif', impact:'Impact, Haettenschweiler, sans-serif',
  comic_sans:'"Comic Sans MS", cursive', times:'"Times New Roman", Times, serif',
  georgia:'Georgia, serif', verdana:'Verdana, sans-serif',
  courier:'"Courier New", Courier, monospace', trebuchet:'"Trebuchet MS", sans-serif',
};

function addTextLayer(){
  const layer = {id: 'txt'+Date.now(), content: 'Your text', x: 0.1, y: 0.75 - textLayers.length*0.08,
                 size: 36, color: '#ffffff', box: true, font: 'default', style: 'classic_box',
                 boxColor: '#000000', enabled: true};
  textLayers.push(layer);
  renderTextPanelRow(layer);
  renderTextOnStage(layer);
}
function renderTextPanelRow(layer){
  const wrap = document.getElementById('textLayersWrap');
  const row = document.createElement('div'); row.className='text-row'; row.id='row_'+layer.id;
  const timingBadge = (layer.start !== undefined && layer.start !== null)
    ? `<span class="badge" title="Only shows during this window">⏱ ${fmtDuration(layer.start)}–${fmtDuration(layer.end)}</span>` : '';
  const safeContent = String(layer.content || '').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const curFont = layer.font || 'default';
  const curStyle = layer.style || 'classic_box';
  row.innerHTML = `
    <div class="top">
      <input type="checkbox" ${layer.enabled?'checked':''} title="Show on video" onchange="updateTextLayer('${layer.id}','enabled',this.checked)">
      <input type="text" value="${safeContent}" oninput="updateTextLayer('${layer.id}','content',this.value)">
      ${timingBadge}
      <span class="del" onclick="removeTextLayer('${layer.id}')">✕</span>
    </div>
    <div class="row" style="margin-top:8px">
      <div style="flex:1"><label>Size <span class="val">${layer.size}</span></label>
        <input type="range" min="14" max="90" value="${layer.size}" oninput="updateTextLayer('${layer.id}','size',this.value,true)"></div>
      <div style="width:70px"><label>Color</label><input type="color" value="${layer.color}" oninput="updateTextLayer('${layer.id}','color',this.value)"></div>
    </div>
    <div class="row" style="margin-top:8px">
      <div style="flex:1"><label>Font</label>
        <select onchange="updateTextLayer('${layer.id}','font',this.value)">
          ${FONT_OPTIONS.map(f=>`<option value="${f.key}" ${curFont===f.key?'selected':''}>${f.label}</option>`).join('')}
        </select></div>
      <div style="flex:1"><label>Caption style</label>
        <select onchange="updateTextLayer('${layer.id}','style',this.value)">
          ${STYLE_OPTIONS.map(s=>`<option value="${s.key}" ${curStyle===s.key?'selected':''}>${s.label}</option>`).join('')}
        </select></div>
      <div style="width:70px"><label>Box/Glow</label>
        <input type="color" value="${layer.boxColor||'#000000'}" oninput="updateTextLayer('${layer.id}','boxColor',this.value)"></div>
    </div>`;
  wrap.appendChild(row);
}


function renderWordCaptionGroupRow(group){
  const wrap = document.getElementById('textLayersWrap');
  const row = document.createElement('div'); row.className='text-row'; row.id='row_wcapGroup';
  const first = group[0];
  const sentence = group.map(l=>l.content).join(' ');
  const safeSentence = String(sentence).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const curFont = first.font || 'impact';
  const curStyle = first.style || 'meme_classic';
  const allOn = group.every(l => l.enabled);
  row.innerHTML = `
    <div class="top">
      <input type="checkbox" ${allOn?'checked':''} title="Sab word captions on/off" onchange="updateWordCaptionGroup('enabled',this.checked)">
      <span style="flex:1; opacity:0.85; font-size:13px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${safeSentence}">🎤 ${group.length} word captions: "${safeSentence}"</span>
      <span class="del" title="Remove all word captions" onclick="removeWordCaptionGroup()">✕</span>
    </div>
    <div class="row" style="margin-top:8px">
      <div style="flex:1"><label>Size <span class="val">${first.size}</span></label>
        <input type="range" min="14" max="90" value="${first.size}" oninput="updateWordCaptionGroup('size',this.value,true)"></div>
      <div style="width:70px"><label>Color</label><input type="color" value="${first.color}" oninput="updateWordCaptionGroup('color',this.value)"></div>
    </div>
    <div class="row" style="margin-top:8px">
      <div style="flex:1"><label>Font</label>
        <select onchange="updateWordCaptionGroup('font',this.value)">
          ${FONT_OPTIONS.map(f=>`<option value="${f.key}" ${curFont===f.key?'selected':''}>${f.label}</option>`).join('')}
        </select></div>
      <div style="flex:1"><label>Caption style</label>
        <select onchange="updateWordCaptionGroup('style',this.value)">
          ${STYLE_OPTIONS.map(s=>`<option value="${s.key}" ${curStyle===s.key?'selected':''}>${s.label}</option>`).join('')}
        </select></div>
      <div style="width:70px"><label>Box/Glow</label>
        <input type="color" value="${first.boxColor||'#000000'}" oninput="updateWordCaptionGroup('boxColor',this.value)"></div>
    </div>`;
  wrap.appendChild(row);
}
function updateWordCaptionGroup(key, val, isNum){
  const v = isNum ? parseInt(val) : val;
  textLayers.forEach(l => {
    if (l.source !== 'word_caption') return;
    l[key] = v;
    renderTextOnStage(l);
  });
}
function removeWordCaptionGroup(){
  textLayers = textLayers.filter(l => l.source !== 'word_caption');
  document.querySelectorAll('[data-source="word_caption"]').forEach(el => el.remove());
  const row = document.getElementById('row_wcapGroup'); if(row) row.remove();
}


function syncTextSizeUI(layer){
  const row = document.getElementById('row_'+layer.id);
  if(!row) return;
  const slider = row.querySelector('input[type=range]');
  const val = row.querySelector('.val');
  if(slider) slider.value = layer.size;
  if(val) val.textContent = layer.size;
}
function updateTextLayer(id, key, val, isNum){
  const layer = textLayers.find(l=>l.id===id); if(!layer) return;
  layer[key] = isNum ? parseInt(val) : val;
  renderTextOnStage(layer);
}
function removeTextLayer(id){
  textLayers = textLayers.filter(l=>l.id!==id);
  const row = document.getElementById('row_'+id); if(row) row.remove();
  const el = document.getElementById('stg_'+id); if(el) el.remove();
}

function setCapStatus(msg, isError){
  const el = document.getElementById('capStatus');
  el.style.display = '';
  el.textContent = msg;
  el.style.color = isError ? '#ff6b7a' : '';
}

async function fetchCaptionLanguages(){
  if(!lastSourceUrl){
    setCapStatus('❌ No source URL for this clip (it was uploaded from your device, not fetched from a link).', true);
    return;
  }
  document.getElementById('capLangWrap').classList.add('hidden');
  setCapStatus('⏳ Checking what subtitles/captions this video has…', false);
  let res, data;
  try{
    res = await fetch('/api/subtitles/list', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url: lastSourceUrl})});
    data = await res.json();
  } catch(e){
    setCapStatus('❌ Network error while checking captions.', true);
    return;
  }
  if(data.error && (!data.langs || !data.langs.length)){
    setCapStatus('❌ ' + data.error, true);
    return;
  }
  fetchedCaptionLangs = data.langs || [];
  const sel = document.getElementById('capLangSelect');
  sel.innerHTML = '';
  fetchedCaptionLangs.forEach((l, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = l.name + (l.auto ? ' (auto-generated)' : ' (uploaded subtitles)');
    sel.appendChild(opt);
  });
  document.getElementById('capLangWrap').classList.remove('hidden');
  setCapStatus(`✅ Found ${fetchedCaptionLangs.length} caption track(s) — pick one and import.`, false);
}

let currentWordCapVideoId = null;

function setWordCapStatus(msg, isError){
  const el = document.getElementById('wordCapStatus');
  if(!el) return;
  el.textContent = msg;
  el.style.color = isError ? '#ff6b7a' : '';
}

async function fetchWordCaptions(){
  if(!lastSourceUrl){ setWordCapStatus('❌ Source URL nahi hai (local upload me ye kaam nahi karta).', true); return; }
  setWordCapStatus('⏳ Word-level captions fetch ho rahi hain...', false);
  document.getElementById('wordCapImportRow').classList.add('hidden');
  let res, data;
  try{
    res = await fetch('/api/word_captions/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url: lastSourceUrl})});
    data = await res.json();
  } catch(e){ setWordCapStatus('❌ Network error', true); return; }
  if(data.error){ setWordCapStatus('❌ ' + data.error, true); return; }
  currentWordCapVideoId = data.video_id;
  pollWordCaptions(data.video_id);
}

async function pollWordCaptions(videoId){
  while(true){
    await new Promise(r=>setTimeout(r, 1200));
    let sd;
    try{
      const res = await fetch(`/api/word_captions/status/${videoId}`);
      sd = await res.json();
    } catch(e){ continue; }
    if(sd.error){ setWordCapStatus('❌ ' + sd.error, true); return; }
    if(sd.status === 'running'){ setWordCapStatus('⏳ ' + (sd.stage || 'processing...'), false); continue; }
    if(sd.status === 'error'){ setWordCapStatus('❌ ' + sd.error, true); return; }
    if(sd.status === 'done'){
      setWordCapStatus(`✅ ${sd.word_count} words mile — ab is clip me import kar sakte ho.`, false);
      document.getElementById('wordCapImportRow').classList.remove('hidden');
      return;
    }
  }
}

async function importWordPerfectCaptions(){
  if(!currentClipId){ setWordCapStatus('❌ Pehle koi clip kholo.', true); return; }
  setWordCapStatus('⏳ Import ho raha hai...', false);
  try{
    const res = await fetch('/api/word_captions/apply', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({clip_id: currentClipId, video_id: currentWordCapVideoId})
    });

    textLayers = textLayers.filter(l => l.source !== 'word_caption');
    document.querySelectorAll('[data-source="word_caption"]').forEach(el => el.remove());


    const data = await res.json();
    if(data.error){ setWordCapStatus('❌ ' + data.error, true); return; }
    (data.layers || []).forEach(l => {
      textLayers.push(l);
      renderTextOnStage(l);
    });
    const wcapGroup = textLayers.filter(l => l.source === 'word_caption');
    const existingRow = document.getElementById('row_wcapGroup');
    if (existingRow) existingRow.remove();
    if (wcapGroup.length) renderWordCaptionGroupRow(wcapGroup);
    setWordCapStatus(`✅ ${data.layers.length} word(s) real-timed caption ki tarah add ho gaye.`, false);
    if (typeof saveCurrentSettingsToMap === 'function') saveCurrentSettingsToMap();
  } catch(e){ setWordCapStatus('❌ Network error', true); }
}

async function importCaptions(){
  const idx = parseInt(document.getElementById('capLangSelect').value, 10);
  const lang = fetchedCaptionLangs[idx];
  if(!lang) return;
  setCapStatus('⏳ Downloading & lining up captions…', false);
  let res, data;
  try{
    res = await fetch('/api/subtitles/fetch', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url: lastSourceUrl, code: lang.code, auto: lang.auto})});
    data = await res.json();
  } catch(e){
    setCapStatus('❌ Network error while downloading captions.', true);
    return;
  }
  if(data.error){
    setCapStatus('❌ ' + data.error, true);
    return;
  }

  // Cues come back timed against the FULL source video. Each clip here is a
  // trimmed slice of that source (allClips[].start/.end, in seconds), so we
  // keep only the cues that fall inside this clip's slice and shift them to
  // clip-local time (0 = this clip's first frame).
  const clip = allClips.find(c => c.clip_id === currentClipId);
  const clipStart = clip ? (clip.start || 0) : 0;
  const clipDur = clip ? clip.duration : null;

  let added = 0;
  const karaokeMode = document.getElementById('capStyleSelect').value === 'karaoke';
  (data.cues || []).forEach(cue => {
    const localStart = cue.start - clipStart;
    const localEnd = cue.end - clipStart;
    if (localEnd <= 0) return;                       // ends before this clip starts
    if (clipDur !== null && localStart >= clipDur) return; // starts after this clip ends
    const s0 = Math.max(0, localStart);
    const e0 = clipDur !== null ? Math.min(clipDur, localEnd) : localEnd;
    if (e0 <= s0) return;

    if (!karaokeMode) {
      const layer = {
        id: 'cap' + Date.now() + '_' + added,
        content: cue.text,
        x: 0.08, y: 0.82, size: 34, color: '#ffffff', box: true,
        font: 'default', style: 'classic_box', boxColor: '#000000', centerX: false, enabled: true,
        start: s0, end: e0
      };
      textLayers.push(layer);
      renderTextPanelRow(layer);
      added++;
      return;
    }

    // Karaoke mode: split this cue's text into words and hand each word a
    // time slice proportional to its length (a reasonable stand-in for real
    // speech timing since we don't have word-level ASR timestamps here) -
    // then give every word its own always-centered layer, timed so exactly
    // one word is showing on screen at any instant ("one big word pops up
    // at a time" style, popular on TikTok/Reels).
    const words = cue.text.split(/\s+/).filter(Boolean);
    if (words.length === 0) return;
    const weights = words.map(w => w.length + 1);
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    const cueDur = e0 - s0;
    let cursor = s0;
    words.forEach((word, wi) => {
      const share = (weights[wi] / totalWeight) * cueDur;
      const wStart = cursor;
      const wEnd = (wi === words.length - 1) ? e0 : Math.min(e0, cursor + share);
      cursor = wEnd;
      if (wEnd <= wStart) return;
      const layer = {
        id: 'cap' + Date.now() + '_' + added,
        content: word,
        x: 0.5, y: 0.78, size: 52, color: '#ffffff', box: false,
        font: 'impact', style: 'shadow_pop', boxColor: '#000000', centerX: true, enabled: true,
        start: wStart, end: wEnd
      };
      textLayers.push(layer);
      renderTextPanelRow(layer);
      added++;
    });
  });
  document.getElementById('capLangWrap').classList.add('hidden');
  setCapStatus(added ? `✅ Imported ${added} ${karaokeMode ? 'word' : 'caption line'}(s) as timed text layers.` : '⚠️ None of the fetched captions fall inside this clip\'s trimmed range.', added === 0);
}
// CSS approximation of each backend STYLE_PRESETS entry, used only so the
// on-stage preview looks close to what the exported file will actually show
// (the real look comes from ffmpeg's drawtext box/border/shadow options).
function applyCaptionStylePreview(el, layer){
  const style = layer.style || 'classic_box';
  const boxColor = layer.boxColor || '#000000';
  el.style.background = 'transparent';
  el.style.textShadow = 'none';
  el.style.webkitTextStroke = '0px transparent';
  el.style.padding = '2px 8px';
  el.style.borderRadius = '2px';
  if (style === 'classic_box') {
    el.style.background = boxColor + '73'; // ~45% alpha as hex suffix
  } else if (style === 'bold_outline') {
    el.style.webkitTextStroke = '2px black';
  } else if (style === 'neon_glow') {
    el.style.textShadow = `0 0 6px ${boxColor}, 0 0 14px ${boxColor}`;
  } else if (style === 'shadow_pop') {
    el.style.textShadow = '2px 2px 4px rgba(0,0,0,.8)';
  } else if (style === 'highlight_bar') {
    el.style.background = boxColor;
  } else if (style === 'outline_only') {
    el.style.webkitTextStroke = '1.5px black';
  } else if (style === 'minimal') {
    el.style.textShadow = '1px 1px 2px rgba(0,0,0,.5)';
  } else if (layer.box === false) {
    // legacy layer with no style field and box explicitly off
    el.style.background = 'transparent';
  } else {
    // legacy layer with no style field, default old look
    el.style.background = 'rgba(0,0,0,.45)';
  }
}

function renderTextOnStage(layer){
  let el = document.getElementById('stg_'+layer.id);
  if(!el){
    el = document.createElement('div'); el.className='ov-text'; el.id='stg_'+layer.id;
    makeDraggable(el, layer, {
      resizable: true, kind: 'text', lockX: !!layer.centerX,
      onResize: (newSize)=>{ el.style.fontSize = newSize+'px'; syncTextSizeUI(layer); },
      onChange: ()=>{ if (typeof saveCurrentSettingsToMap === 'function') saveCurrentSettingsToMap(); }
    });
    document.getElementById('overlayLayer').appendChild(el);
  }
  el.innerText = layer.content;
  el.style.fontSize = layer.size+'px';
  el.style.color = layer.color;
  el.style.fontFamily = FONT_CSS_MAP[layer.font || 'default'];
  el.style.fontWeight = (layer.font === 'arial_bold' || layer.font === 'impact') ? '700' : '400';
  applyCaptionStylePreview(el, layer);
  if(layer.centerX){
    el.style.left = '50%'; el.style.transform = 'translateX(-50%)';
  } else {
    el.style.left = (layer.x*100)+'%'; el.style.transform = '';
  }
  el.style.top = (layer.y*100)+'%';
  if (layer.start !== undefined && layer.start !== null) {
    // Timed caption layer: only visible while playback is inside its window.
    const t = document.getElementById('player').currentTime || 0;
    const within = t >= layer.start && t <= layer.end;
    el.style.display = (layer.enabled && within) ? '' : 'none';
  } else {
    el.style.display = layer.enabled ? '' : 'none';
  }
}

function renderLogo(){
  if(!logoUrl) return;
  let el = document.getElementById('ovLogo');
  if(!el){
    el = document.createElement('img'); el.className='ov-logo'; el.id='ovLogo';
    makeDraggable(el, logoState, {
      resizable: true, kind: 'logo',
      onResize: (newW)=>{
        const pct = Math.round(newW*100);
        document.getElementById('logoW').value = pct;
        document.getElementById('logoWVal').innerText = pct;
        el.style.width = pct+'%';
      },
      onChange: ()=>{ if (typeof saveCurrentSettingsToMap === 'function') saveCurrentSettingsToMap(); }
    });
    document.getElementById('overlayLayer').appendChild(el);
  }
  el.src = logoUrl;
  el.style.width = document.getElementById('logoW').value+'%';
  el.style.opacity = document.getElementById('logoO').value/100;
  el.style.left = (logoState.x*100)+'%';
  el.style.top = (logoState.y*100)+'%';
  el.style.display = document.getElementById('logoEnabled').checked ? '' : 'none';
  document.getElementById('logoWVal').innerText = document.getElementById('logoW').value;
  document.getElementById('logoOVal').innerText = document.getElementById('logoO').value;
}

// Step 4/7 fix: the old version only listened for mouse events, so dragging
// text/logo/shapes did nothing on Android/touch devices. Pointer Events cover
// mouse + touch + pen with one code path. opts is optional:
//   opts.resizable -> adds a drag-corner resize handle
//   opts.kind       -> 'text' (drives state.size) | 'logo'/'region' (drives state.w/[state.h])
//   opts.onResize(w,h) -> called live while resizing so the caller can update its own element style
//   opts.onChange()    -> called once when a drag or resize finishes (e.g. to persist settings)
function makeDraggable(el, state, opts){
  opts = opts || {};
  const stage = document.getElementById('stage');
  let dragging = false, startPX = 0, startPY = 0, startX = 0, startY = 0;

  el.addEventListener('pointerdown', (e)=>{
    if (e.target.closest('.del') || e.target.closest('.resize-handle')) return;
    e.preventDefault();
    try{ el.setPointerCapture(e.pointerId); }catch(err){}
    dragging = true;
    const rect = stage.getBoundingClientRect();
    startPX = e.clientX; startPY = e.clientY;
    startX = state.x; startY = state.y;
    el._dragRect = rect;
  });
  el.addEventListener('pointermove', (e)=>{
    if(!dragging) return;
    const rect = el._dragRect || stage.getBoundingClientRect();
    let x = startX + (e.clientX - startPX)/rect.width;
    let y = startY + (e.clientY - startPY)/rect.height;
    x = Math.max(0, Math.min(0.95, x)); y = Math.max(0, Math.min(0.95, y));
    state.y = y;
    el.style.top = (y*100)+'%';
    if(opts.lockX){
      el.style.left = '50%'; el.style.transform = 'translateX(-50%)';
    } else {
      state.x = x;
      el.style.left = (x*100)+'%';
    }
  });
  function endDrag(e){
    if(!dragging) return;
    dragging = false;
    try{ el.releasePointerCapture(e.pointerId); }catch(err){}
    if(opts.onChange) opts.onChange();
  }
  el.addEventListener('pointerup', endDrag);
  el.addEventListener('pointercancel', endDrag);

  if(opts.resizable){
    const handle = document.createElement('div');
    handle.className = 'resize-handle';
    let resizing = false, startSize = 0, startW = 0, startH = 0, rect2 = null;
    handle.addEventListener('pointerdown', (e)=>{
      e.preventDefault(); e.stopPropagation();
      try{ handle.setPointerCapture(e.pointerId); }catch(err){}
      resizing = true;
      rect2 = stage.getBoundingClientRect();
      startPX = e.clientX; startPY = e.clientY;
      startSize = state.size || 0; startW = state.w || 0; startH = state.h || 0;
    });
    handle.addEventListener('pointermove', (e)=>{
      if(!resizing) return;
      const dx = (e.clientX - startPX)/rect2.width;
      const dy = (e.clientY - startPY)/rect2.height;
      if(opts.kind === 'text'){
        const newSize = Math.max(14, Math.min(90, Math.round(startSize + dx*rect2.width*0.3)));
        state.size = newSize;
        if(opts.onResize) opts.onResize(newSize);
      } else if(opts.kind === 'logo'){
        const newW = Math.max(0.05, Math.min(0.6, startW + dx));
        if(opts.onResize) opts.onResize(newW);
      } else { // 'region' (rect/circle/arrow/blur/black/emoji shapes) - resize both dimensions
        const newW = Math.max(0.03, Math.min(0.95, startW + dx));
        const newH = Math.max(0.03, Math.min(0.95, startH + dy));
        state.w = newW; state.h = newH;
        el.style.width = (newW*100)+'%'; el.style.height = (newH*100)+'%';
        if(opts.onResize) opts.onResize(newW, newH);
      }
    });
    function endResize(e){
      if(!resizing) return;
      resizing = false;
      try{ handle.releasePointerCapture(e.pointerId); }catch(err){}
      if(opts.onChange) opts.onChange();
    }
    handle.addEventListener('pointerup', endResize);
    handle.addEventListener('pointercancel', endResize);
    el.appendChild(handle);
  }
}

function setTool(k){
  tool = k;
  document.querySelectorAll('.tool-btn').forEach(b=>b.classList.toggle('active', b.dataset.k===k));
  document.getElementById('emojiUploadWrap').classList.toggle('hidden', k!=='emoji');
}

(function initDrawing(){
  const stage = document.getElementById('stage');
  let start = null, liveEl = null, drawing = false;
  stage.addEventListener('pointerdown', (e)=>{
    if(e.target.closest('.ov-region') || e.target.closest('.ov-text') || e.target.closest('.ov-logo')) return;
    e.preventDefault();
    try{ stage.setPointerCapture(e.pointerId); }catch(err){}
    drawing = true;
    const rect = stage.getBoundingClientRect();
    start = {x:(e.clientX-rect.left)/rect.width, y:(e.clientY-rect.top)/rect.height};
    liveEl = document.createElement('div'); liveEl.className='ov-region kind-'+tool;
    liveEl.style.border = '2px dashed #6e5bff';
    document.getElementById('overlayLayer').appendChild(liveEl);
  });
  stage.addEventListener('pointermove', (e)=>{
    if(!drawing || !start) return;
    const rect = stage.getBoundingClientRect();
    let x = (e.clientX-rect.left)/rect.width, y = (e.clientY-rect.top)/rect.height;
    const x0=Math.min(start.x,x), y0=Math.min(start.y,y), w=Math.abs(x-start.x), h=Math.abs(y-start.y);
    liveEl.style.left=(x0*100)+'%'; liveEl.style.top=(y0*100)+'%';
    liveEl.style.width=(w*100)+'%'; liveEl.style.height=(h*100)+'%';
  });
  function finishDraw(e){
    if(!drawing || !start){ drawing = false; return; }
    drawing = false;
    try{ stage.releasePointerCapture(e.pointerId); }catch(err){}
    const rect = stage.getBoundingClientRect();
    let x = (e.clientX-rect.left)/rect.width, y = (e.clientY-rect.top)/rect.height;
    const x0=Math.min(start.x,x), y0=Math.min(start.y,y), w=Math.abs(x-start.x), h=Math.abs(y-start.y);
    start = null;
    if(w<0.02 || h<0.02){ liveEl.remove(); return; }
    if(tool === 'emoji' && !pendingEmojiUrl){
      alert('Upload an emoji/sticker image or tap a quick emoji below first.'); liveEl.remove(); return;
    }
    const color = document.getElementById('shapeColor').value;
    const region = {kind: tool, x:x0, y:y0, w, h, color, emoji_url: tool==='emoji'?pendingEmojiUrl:null};
    regions.push(region);
    liveEl.classList.add('kind-'+tool);
    liveEl.style.border = '';
    if(tool==='blur' || tool==='black'){
      liveEl.classList.add('kind-'+tool);
    }
    if(tool==='emoji'){ liveEl.style.backgroundImage = `url(${pendingEmojiUrl})`; }
    if(tool==='rect' || tool==='circle' || tool==='arrow'){ liveEl.style.borderColor = color; }
    const del = document.createElement('div'); del.className='del'; del.innerText='✕';
    del.onclick = (ev)=>{ ev.stopPropagation(); liveEl.remove(); regions = regions.filter(r=>r!==region); };
    liveEl.appendChild(del);
    // Step 7 fix: shapes could previously only be deleted, never repositioned
    // or resized once drawn. Wire the same drag+resize behavior text/logo use.
    makeDraggable(liveEl, region, {
      resizable: true, kind: 'region',
      onChange: ()=>{ if (typeof saveCurrentSettingsToMap === 'function') saveCurrentSettingsToMap(); }
    });
  }
  stage.addEventListener('pointerup', finishDraw);
  stage.addEventListener('pointercancel', finishDraw);
})();

// Hook aspect ratio change events and transformation on load
// (each .ratioBox checkbox already calls updateStageAspectRatio() via its own
// inline onchange handler in the HTML, so no extra listener needed here)
document.getElementById('player').addEventListener('loadedmetadata', updateStageAspectRatio);
document.getElementById('player').addEventListener('timeupdate', () => {
  const t = document.getElementById('player').currentTime;
  textLayers.forEach(l => {
    if (l.start === undefined || l.start === null) return; // untimed layers: leave as-is
    const el = document.getElementById('stg_' + l.id);
    if (!el) return;
    const within = t >= l.start && t <= l.end;
    el.style.display = (l.enabled && within) ? '' : 'none';
  });
});

function clearRegions(){
  regions = [];
  document.querySelectorAll('.ov-region').forEach(e=>e.remove());
}

async function saveVideo(){
  if(!currentClipId){ return; }
  const saveBtn = document.getElementById('saveBtn');
  if(saveBtn) saveBtn.disabled = true;

  const stage = document.getElementById('stage');
  const stageRect = stage.getBoundingClientRect();

  const amode = document.querySelector('input[name=amode]:checked').value;
  const settings = {
    speed: parseFloat(document.getElementById('speed').value),
    zoom: parseFloat(document.getElementById('zoom').value),
    pan_x: panX, pan_y: panY,
    contrast: parseFloat(document.getElementById('contrast').value),
    saturation: parseFloat(document.getElementById('saturation').value),
    brightness: parseFloat(document.getElementById('brightness').value),
    sharpen: document.getElementById('sharpen').checked,
    enhance: document.getElementById('enhance').checked,
    color_preset: document.getElementById('colorPreset').value,
    ratios: getSelectedRatios(),
    format: document.getElementById('format').value,
    crf: parseInt(document.getElementById('crf').value),
    preset: document.getElementById('preset').value,
    hw_accel: document.getElementById('hwAccel').checked,
    face_track: (faceTrackEnabledMap[currentClipId] && faceTrackMap[currentClipId] && faceTrackMap[currentClipId].faces_found) ? faceTrackMap[currentClipId].track : null,
    regions: regions,
    audio_mode: (amode === 'record' ? 'replace' : amode),
    audio_file_url: audioFileUrl,
    tts_url: ttsUrl,
    tts_mix: document.getElementById('ttsMix').checked,
    mute: amode === 'mute',
    texts: textLayers.filter(l=>l.enabled).map(l=>({content:l.content, x:l.x, y:l.y, size:l.size, color:l.color, box:l.box, font:l.font, style:l.style, box_color:l.boxColor, center_x:l.centerX, start:l.start, end:l.end, source:l.source})),
    logo: (logoUrl && document.getElementById('logoEnabled').checked) ? {
      url: logoUrl, x: logoState.x, y: logoState.y,
      w: document.getElementById('logoW').value/100,
      opacity: document.getElementById('logoO').value/100
    } : null,
    rotate: document.getElementById('rotate').value,
    hflip: document.getElementById('hflip').checked,
    stage_w: stageRect.width || 360,
    stage_h: stageRect.height || 640
  };
  const t0 = performance.now();
  
  const pWrap = document.getElementById('exportProgressWrap');
  const pFill = document.getElementById('exportProgressFill');
  const pText = document.getElementById('exportProgressText');
  
  if(pWrap) pWrap.classList.remove('hidden');
  if(pFill) pFill.style.width = '0%';
  if(pText) pText.innerText = '0%';

  log('exportLog', '⏳ Exporting final video...');
  
  try {
    const res = await fetch('/api/export', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({clip_id: currentClipId, settings})
    });
    const data = await res.json();
    if(data.error){
      log('exportLog', '❌ '+data.error);
      if(saveBtn) saveBtn.disabled = false;
      if(pWrap) pWrap.classList.add('hidden');
      return;
    }
    
    const exportId = data.export_id;
    let done = false;
    while(!done){
      await new Promise(r=>setTimeout(r, 1000));
      const statusRes = await fetch(`/api/export_status/${exportId}`);
      const sd = await statusRes.json();
      if(sd.error){
        log('exportLog', '❌ '+sd.error);
        if(saveBtn) saveBtn.disabled = false;
        if(pWrap) pWrap.classList.add('hidden');
        return;
      }
      
      const progress = sd.progress || 0;
      if(pFill) pFill.style.width = progress + '%';
      if(pText) pText.innerText = progress + '%';
      
      if(sd.status === 'done'){
        done = true;
        if(pFill) pFill.style.width = '100%';
        if(pText) pText.innerText = '100%';
        const secs = ((performance.now()-t0)/1000).toFixed(1);
        const results = sd.results && sd.results.length ? sd.results : [{ratio:'original', url: sd.url}];
        const links = results.map(r =>
          `<a class="dl-link" href="javascript:void(0)" onclick="triggerInlineDownload('${r.url}', '${r.url.split('/').pop()}')">${r.ratio} ⬇</a>`
        ).join(' &nbsp;|&nbsp; ');
        log('exportLog', `✅ Saved in ${secs}s (${results.length} ratio${results.length>1?'s':''})! ${links} — <a class="dl-link" href="javascript:void(0)" onclick="publishExportedClip('${exportId}')">📤 Publish</a>`);
        if(saveBtn) saveBtn.disabled = false;
      } else if(sd.status === 'failed'){
        done = true;
        log('exportLog', '❌ Export failed: ' + sd.error);
        if(saveBtn) saveBtn.disabled = false;
        if(pWrap) pWrap.classList.add('hidden');
      }
    }
  } catch(e) {
    log('exportLog', '❌ Connection error: ' + e.message);
    if(saveBtn) saveBtn.disabled = false;
    if(pWrap) pWrap.classList.add('hidden');
  }
}

function onColorPresetChange() {
  const preset = document.getElementById('colorPreset').value;
  currentPresetName = preset;
  if(preset && preset !== 'none' && JS_COLOR_PRESETS[preset]){
    const p = JS_COLOR_PRESETS[preset];
    document.getElementById('contrast').value = p.contrast;
    document.getElementById('saturation').value = p.saturation;
    document.getElementById('brightness').value = p.brightness;
  }
  onColor();
}

async function applyToAllAndExport() {
  if (allClips.length === 0) {
    alert("No clips available to export!");
    return;
  }
  
  log('fetchLog', `🚀 Triggering batch export of all ${allClips.length} clips with current settings...`);
  
  allClips.forEach(c => {
    const statusOverlay = document.getElementById(`status_overlay_${c.clip_id}`);
    const statusTxt = document.getElementById(`status_txt_${c.clip_id}`);
    if (statusOverlay) {
      statusOverlay.className = 'clip-status-overlay active';
    }
    if (statusTxt) {
      statusTxt.innerText = 'Waiting in queue...';
    }
    const actions = document.getElementById(`actions_${c.clip_id}`);
    if (actions) {
      actions.innerHTML = `
        <button class="clip-card-btn edit-btn" onclick="event.stopPropagation(); openEditor('${c.clip_id}')">⚙️ Edit</button>
      `;
    }
  });

  exportQueue = [...allClips];
  pumpExportQueue();
}

function getAutoSettingsForClip(clipId, index, audioFiles, stageWidth, stageHeight) {
  let speed = 0.75;
  let zoom = 1.20;
  let contrast = 1.12;
  let saturation = 1.25;
  let brightness = 0.02;
  let sharpen = true;
  let enhance = true;
  let color_preset = 'none';
  let ratios = ['9:16'];
  let format = 'mp4';
  let crf = 18;
  let preset = 'fast';
  let rotate = '0';
  let hflip = true;
  let amode = 'original';
  let defaultAudioUrl = null;
  let ttsMix = false;
  let texts = [];
  let logo = null;
  let stage_w = stageWidth || 360;
  let stage_h = stageHeight || 640;
  let film_grain = true;
  let audio_pitch = true;
  let vignette = true;

  const speedEl = document.getElementById('speed');
  if (speedEl) {
    speed = parseFloat(speedEl.value);
    zoom = parseFloat(document.getElementById('zoom').value);
    contrast = parseFloat(document.getElementById('contrast').value);
    saturation = parseFloat(document.getElementById('saturation').value);
    brightness = parseFloat(document.getElementById('brightness').value);
    sharpen = document.getElementById('sharpen').checked;
    enhance = document.getElementById('enhance').checked;
    color_preset = currentPresetName;
    ratios = getSelectedRatios();
    format = document.getElementById('format').value;
    crf = parseInt(document.getElementById('crf').value);
    preset = document.getElementById('preset').value;
    rotate = document.getElementById('rotate').value;
    hflip = document.getElementById('hflip').checked;
    const amodeEl = document.querySelector('input[name=amode]:checked');
    if (amodeEl) amode = amodeEl.value;
    defaultAudioUrl = audioFileUrl;
    ttsMix = document.getElementById('ttsMix').checked;
    texts = textLayers.filter(l=>l.enabled).map(l=>({content:l.content, x:l.x, y:l.y, size:l.size, color:l.color, box:l.box, font:l.font, style:l.style, box_color:l.boxColor, center_x:l.centerX, start:l.start, end:l.end, source:l.source}));
    logo = (logoUrl && document.getElementById('logoEnabled').checked) ? {
      url: logoUrl, x: logoState.x, y: logoState.y,
      w: document.getElementById('logoW').value/100,
      opacity: document.getElementById('logoO').value/100
    } : null;
    const stage = document.getElementById('stage');
    if (stage) {
      const stageRect = stage.getBoundingClientRect();
      stage_w = stageRect.width || 360;
      stage_h = stageRect.height || 640;
    }
    film_grain = document.getElementById('automodFilmGrain') ? document.getElementById('automodFilmGrain').checked : true;
    audio_pitch = document.getElementById('automodAudioPitch') ? document.getElementById('automodAudioPitch').checked : true;
    vignette = document.getElementById('automodVignette') ? document.getElementById('automodVignette').checked : true;
  }

  const settings = {
    speed,
    zoom,
    pan_x: panX,
    pan_y: panY,
    contrast,
    saturation,
    brightness,
    sharpen,
    enhance,
    color_preset,
    ratios,
    format,
    crf,
    preset,
    regions: regions,
    audio_mode: (amode === 'record' ? 'replace' : amode),
    audio_file_url: defaultAudioUrl,
    tts_url: ttsUrl,
    tts_mix: ttsMix,
    mute: amode === 'mute',
    texts,
    logo,
    rotate,
    hflip,
    stage_w,
    stage_h,
    film_grain,
    audio_pitch,
    vignette,
    clip_index: index
  };

  // Automatically replace audio file round-robin line-by-line if replaced file mode is active or Automod is checked
  const isAutoMode = document.getElementById('autoMode') && document.getElementById('autoMode').checked;
  if ((amode === 'replace' || isAutoMode) && audioFiles && audioFiles.length > 0) {
    const chosenAudio = audioFiles[index % audioFiles.length];
    settings.audio_mode = 'replace';
    settings.audio_file_url = chosenAudio.url;
  }

  return settings;
}

// How many exports run at once. Each export is a real ffmpeg re-encode
// (speed/zoom/color/watermark/noise/vignette/audio-pitch/etc filters), so
// this is CPU-bound — matches the backend's own concurrency cap.
const MAX_CONCURRENT_EXPORTS = 4;
let activeExportWorkers = 0;

function pumpExportQueue() {
  while (activeExportWorkers < MAX_CONCURRENT_EXPORTS && exportQueue.length > 0) {
    const c = exportQueue.shift();
    activeExportWorkers++;
    exportQueueActive = true;
    runOneExport(c).finally(() => {
      activeExportWorkers--;
      if (exportQueue.length === 0 && activeExportWorkers === 0) {
        exportQueueActive = false;
      }
      pumpExportQueue();
    });
  }
}

async function runOneExport(c) {
  const index = c.index - 1; // 0-based

  const statusOverlay = document.getElementById(`status_overlay_${c.clip_id}`);
  const statusTxt = document.getElementById(`status_txt_${c.clip_id}`);
  
  if (statusTxt) statusTxt.innerText = '⚙️ Exporting...';
  if (statusOverlay) statusOverlay.classList.add('active');
  
  try {
    if (!audioLibraryFiles) {
      const res = await fetch('/api/audio_library');
      const data = await res.json();
      audioLibraryFiles = data.files || [];
    }
    
    // Prefer card-specific settings if they exist, otherwise fallback to global/auto settings
    let settings = clipSettingsMap[c.clip_id];
    if (!settings) {
      settings = getAutoSettingsForClip(c.clip_id, index, audioLibraryFiles, 360, 640);
    } else {
      settings.clip_index = index;
    }
    
    const res = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clip_id: c.clip_id, settings })
    });
    const data = await res.json();
    if (data.error) {
      throw new Error(data.error);
    }
    
    const exportId = data.export_id;
    let done = false;
    while (!done) {
      await new Promise(r => setTimeout(r, 1200));
      const statusRes = await fetch(`/api/export_status/${exportId}`);
      const sd = await statusRes.json();
      if (sd.error) {
        throw new Error(sd.error);
      }
      
      const progress = sd.progress || 0;
      if (statusTxt) statusTxt.innerText = `⚙️ Exporting ${progress}%`;
      
      if (sd.status === 'done') {
        done = true;
        if (statusTxt) statusTxt.innerText = '✅ Exported';
        if (statusOverlay) {
          statusOverlay.classList.remove('active');
          statusOverlay.classList.add('done');
        }
        const results = sd.results && sd.results.length ? sd.results : [{ratio:'original', url: sd.url}];
        const primary = results[0];

        const videoEl = document.querySelector(`#card_${c.clip_id} video`);
        if (videoEl) {
          videoEl.src = primary.url;
          videoEl.controls = true;
          videoEl.muted = false;
        }
        const actions = document.getElementById(`actions_${c.clip_id}`);
        if (actions) {
          const dlButtons = results.map(r =>
            `<button class="clip-card-btn dl-btn" onclick="event.stopPropagation(); triggerInlineDownload('${r.url}', '${r.url.split('/').pop()}')">📥 ${r.ratio}</button>`
          ).join('');
          actions.innerHTML = `
            <button class="clip-card-btn edit-btn" onclick="event.stopPropagation(); openEditor('${c.clip_id}')">⚙️ Edit</button>
            ${dlButtons}
            <button class="clip-card-btn dl-btn" onclick="event.stopPropagation(); publishExportedClip('${exportId}')">📤 Publish</button>
          `;
        }
        // Har ratio Downloads Manager me apni alag entry ke saath
        results.forEach(r => addExportToDownloadsList(c.clip_id, `${c.label} (${r.ratio})`, r.url, settings));
      } else if (sd.status === 'failed') {
        done = true;
        throw new Error(sd.error);
      }
    }
  } catch (e) {
    console.error(e);
    if (statusTxt) statusTxt.innerText = '❌ Failed';
    if (statusOverlay) {
      statusOverlay.className = 'clip-status-overlay failed';
    }
  }
}


function updateStageAspectRatio() {
  const res = getSelectedRatios()[0];   // preview sirf pehle-selected ratio dikhata hai; sabhi selected ratios export me sahi hi banenge
  const stage = document.getElementById('stage');
  const player = document.getElementById('player');
  if (res === '9:16') {
    stage.style.aspectRatio = '9/16';
  } else if (res === '1:1') {
    stage.style.aspectRatio = '1/1';
  } else if (res === '16:9') {
    stage.style.aspectRatio = '16/9';
  } else if (res === '4:5') {
    stage.style.aspectRatio = '4/5';
  } else if (res === 'original') {
    if (player.videoWidth && player.videoHeight) {
      stage.style.aspectRatio = `${player.videoWidth}/${player.videoHeight}`;
    } else {
      stage.style.aspectRatio = '9/16';
    }
  }
}

function onTransform() {
  updatePreviewTransforms();
  saveCurrentSettingsToMap();
}

function saveCurrentSettingsToMap() {
  if (!currentClipId) return;
  const amode = document.querySelector('input[name=amode]:checked') ? document.querySelector('input[name=amode]:checked').value : 'original';
  const stage = document.getElementById('stage');
  const stageRect = stage ? stage.getBoundingClientRect() : {width: 360, height: 640};
  
  // Clean text layers to be formatted correctly
  const cleanTexts = textLayers.filter(l => l.enabled).map(l => ({
    id: l.id,
    content: l.content,
    x: l.x,
    y: l.y,
    size: l.size,
    color: l.color,
    box: l.box,
    font: l.font,
    style: l.style,
    boxColor: l.boxColor,
    centerX: l.centerX,
    enabled: l.enabled !== false,
    start: l.start,
    end: l.end,
    source: l.source
  }));

  clipSettingsMap[currentClipId] = {
    speed: parseFloat(document.getElementById('speed').value),
    zoom: parseFloat(document.getElementById('zoom').value),
    pan_x: panX,
    pan_y: panY,
    contrast: parseFloat(document.getElementById('contrast').value),
    saturation: parseFloat(document.getElementById('saturation').value),
    brightness: parseFloat(document.getElementById('brightness').value),
    sharpen: document.getElementById('sharpen').checked,
    enhance: document.getElementById('enhance').checked,
    color_preset: document.getElementById('colorPreset').value,
    ratios: getSelectedRatios(),
    format: document.getElementById('format').value,
    crf: parseInt(document.getElementById('crf').value),
    preset: document.getElementById('preset').value,
    hw_accel: document.getElementById('hwAccel').checked,
    face_track: (faceTrackEnabledMap[currentClipId] && faceTrackMap[currentClipId] && faceTrackMap[currentClipId].faces_found) ? faceTrackMap[currentClipId].track : null,
    regions: regions,
    audio_mode: (amode === 'record' ? 'replace' : amode),
    audio_file_url: audioFileUrl,
    tts_url: ttsUrl,
    tts_mix: document.getElementById('ttsMix').checked,
    mute: amode === 'mute',
    texts: cleanTexts,
    logo: (logoUrl && document.getElementById('logoEnabled').checked) ? {
      url: logoUrl, x: logoState.x, y: logoState.y,
      w: document.getElementById('logoW').value/100,
      opacity: document.getElementById('logoO').value/100
    } : null,
    rotate: document.getElementById('rotate').value,
    hflip: document.getElementById('hflip').checked,
    stage_w: stageRect.width || 360,
    stage_h: stageRect.height || 640
  };
  
  // Update card visual representation live!
  updateClipCardBadge(currentClipId);
  
  // Set card video scale transform live based on hflip
  const cardVid = document.querySelector(`#card_${currentClipId} video`);
  if (cardVid) {
    if (document.getElementById('hflip').checked) {
      cardVid.style.transform = 'scaleX(-1)';
    } else {
      cardVid.style.transform = '';
    }
  }
}

function updateClipCardBadge(clipId) {
  const s = clipSettingsMap[clipId];
  if (!s) return;
  const badgeWrap = document.getElementById(`badges_${clipId}`);
  if (badgeWrap) {
    let presetName = s.color_preset || 'none';
    presetName = presetName.charAt(0).toUpperCase() + presetName.slice(1);
    
    let audioName = s.audio_file_url ? s.audio_file_url.split('/').pop() : 'None';
    if (audioName.length > 18) audioName = audioName.slice(0, 15) + '...';
    
    badgeWrap.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:4px; margin-top:6px; font-size:11px; color:#a0a0a0; background:rgba(255,255,255,0.03); padding:6px; border-radius:6px; border:1px solid rgba(255,255,255,0.05);">
        <div style="display:flex; justify-content:space-between;"><span>⚡ Speed:</span> <strong style="color:#ffc107">${s.speed}x</strong></div>
        <div style="display:flex; justify-content:space-between;"><span>🔍 Zoom:</span> <strong style="color:#17a2b8">${s.zoom}x</strong></div>
        <div style="display:flex; justify-content:space-between;"><span>🎨 Color:</span> <strong style="color:#20c997">${presetName}</strong></div>
        <div style="display:flex; justify-content:space-between;"><span>🔤 Watermark:</span> <strong style="color:#e83e8c">"FondPeace.com"</strong></div>
        <div style="display:flex; justify-content:space-between;"><span>🪞 Mirror:</span> <strong style="color:#fd7e14">${s.hflip ? 'Yes' : 'No'}</strong></div>
        <div style="display:flex; justify-content:space-between; align-items:center; gap:4px;">
          <span>🎵 Audio:</span>
          <span style="color:#6f42c1; max-width:110px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:bold;" title="${s.audio_file_url ? s.audio_file_url.split('/').pop() : ''}">${audioName}</span>
        </div>
        ${deadAirBackupState[clipId] ? `<div style="display:flex; justify-content:space-between;"><span>🔇 Dead air:</span> <strong style="color:#0dcaf0">Removed (undoable)</strong></div>` : ''}
      </div>
    `;
  }
}

function toggleCardMirror(clipId) {
  const s = clipSettingsMap[clipId];
  if (!s) return;
  s.hflip = !s.hflip;
  updateClipCardBadge(clipId);
  
  // Also apply CSS transform to the card's video element live so the user sees the mirroring immediately!
  const videoEl = document.querySelector(`#card_${clipId} video`);
  if (videoEl) {
    if (s.hflip) {
      videoEl.style.transform = 'scaleX(-1)';
    } else {
      videoEl.style.transform = '';
    }
  }
  
  // If the editor is open with this clip, update the editor's mirror checkbox too
  if (currentClipId === clipId) {
    document.getElementById('hflip').checked = s.hflip;
    onTransform();
  }
}



async function exportSingleClip(clipId) {
  const c = allClips.find(x => x.clip_id === clipId);
  if (!c) return;
  
  if (exportQueue.some(x => x.clip_id === clipId)) {
    alert("This clip is already in the export queue!");
    return;
  }
  
  const statusOverlay = document.getElementById(`status_overlay_${clipId}`);
  const statusTxt = document.getElementById(`status_txt_${clipId}`);
  if (statusTxt) statusTxt.innerText = '⚙️ Exporting...';
  if (statusOverlay) statusOverlay.classList.add('active');
  
  exportQueue.push(c);
  pumpExportQueue();
}

// ──────────────────────── Exported Downloads & Batch Manager ────────────────────────
let exportedVideos = []; // array of {clipId, label, url, settings, filename}
let allSelectedExports = true;

function addExportToDownloadsList(clipId, label, url, settings) {
  document.getElementById('downloadsCard').classList.remove('hidden');
  const emptyRow = document.getElementById('emptyExportRow');
  if (emptyRow) {
    emptyRow.remove();
  }
  
  // Prevent duplicate items in list
  if (exportedVideos.some(v => v.clipId === clipId)) {
    const idx = exportedVideos.findIndex(v => v.clipId === clipId);
    exportedVideos[idx] = { clipId, label, url, settings, filename: url.split('/').pop() };
    renderExportedVideosTable();
    return;
  }
  
  exportedVideos.push({
    clipId,
    label,
    url,
    settings,
    filename: url.split('/').pop()
  });
  
  renderExportedVideosTable();
}

function renderExportedVideosTable() {
  const tbody = document.getElementById('exportedVideosList');
  if (exportedVideos.length === 0) {
    tbody.innerHTML = `
      <tr id="emptyExportRow">
        <td colspan="4" style="padding: 20px; text-align: center; color: var(--dim);">No videos successfully exported yet. Click "Export Video" on a card or run a batch export!</td>
      </tr>
    `;
    return;
  }
  
  tbody.innerHTML = '';
  exportedVideos.forEach(v => {
    let presetName = v.settings.color_preset || 'none';
    presetName = presetName.charAt(0).toUpperCase() + presetName.slice(1);
    
    let audioName = v.settings.audio_file_url ? v.settings.audio_file_url.split('/').pop() : 'Original';
    if (audioName.length > 20) audioName = audioName.slice(0, 17) + '...';
    
    const row = document.createElement('tr');
    row.style.borderBottom = '1px solid var(--border)';
    row.style.background = 'rgba(255,255,255,0.01)';
    row.innerHTML = `
      <td style="padding: 10px 8px; vertical-align: middle;">
        <input type="checkbox" class="export-download-chk" checked data-url="${v.url}" data-fname="${v.filename}" style="width:16px; height:16px; cursor:pointer;">
      </td>
      <td style="padding: 10px 8px; font-weight: 600; color: #fff; vertical-align: middle;">
        <div style="display:flex; align-items:center; gap:8px;">
          <span>🎬</span>
          <div>
            <div style="font-size:13px; color:#fff;">${v.label}</div>
            <div style="font-size:11px; color:var(--dim); font-weight:normal;">File: ${v.filename}</div>
          </div>
        </div>
      </td>
      <td style="padding: 10px 8px; vertical-align: middle;">
        <div style="display:flex; flex-wrap:wrap; gap:6px; font-size:11px;">
          <span style="background:rgba(255,193,7,0.15); color:#ffc107; padding:2px 6px; border-radius:4px; font-weight:bold;">⚡ ${v.settings.speed}x</span>
          <span style="background:rgba(23,162,184,0.15); color:#17a2b8; padding:2px 6px; border-radius:4px; font-weight:bold;">🔍 ${v.settings.zoom}x</span>
          <span style="background:rgba(32,201,151,0.15); color:#20c997; padding:2px 6px; border-radius:4px; font-weight:bold;">🎨 ${presetName}</span>
          <span style="background:rgba(253,126,20,0.15); color:#fd7e14; padding:2px 6px; border-radius:4px; font-weight:bold;">🪞 Mirror: ${v.settings.hflip ? 'Yes' : 'No'}</span>
          <span style="background:rgba(111,66,193,0.15); color:#6f42c1; padding:2px 6px; border-radius:4px; font-weight:bold; max-width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${v.settings.audio_file_url ? v.settings.audio_file_url.split('/').pop() : ''}">🎵 ${audioName}</span>
        </div>
      </td>
      <td style="padding: 10px 8px; text-align: right; vertical-align: middle;">
        <div style="display:flex; justify-content:flex-end; gap:6px;">
          <button class="clip-card-btn" style="padding: 4px 8px; font-size:11px; background:var(--grad); color:#000; font-weight:bold; cursor:pointer;" onclick="triggerInlineDownload('${v.url}', '${v.filename}')">📥 Download</button>
          <button class="clip-card-btn" style="padding: 4px 8px; font-size:11px; background:#434348; color:#fff; cursor:pointer;" onclick="playExportedVideo('${v.url}')">▶ Play</button>
        </div>
      </td>
    `;
    tbody.appendChild(row);
  });
}

function triggerInlineDownload(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  a.setAttribute('download', filename || '');
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function playExportedVideo(url) {
  const modal = document.createElement('div');
  modal.style.position = 'fixed';
  modal.style.top = '0';
  modal.style.left = '0';
  modal.style.width = '100vw';
  modal.style.height = '100vh';
  modal.style.background = 'rgba(0,0,0,0.85)';
  modal.style.zIndex = '99999';
  modal.style.display = 'flex';
  modal.style.flexDirection = 'column';
  modal.style.alignItems = 'center';
  modal.style.justifyContent = 'center';
  modal.style.gap = '15px';
  modal.id = 'temp_video_modal';
  
  modal.innerHTML = `
    <div style="position:relative; width:90%; max-width:400px; aspect-ratio:9/16; background:#000; border-radius:12px; overflow:hidden; border:2px solid var(--border); box-shadow:0 0 30px rgba(0,0,0,0.5);">
      <video src="${url}" controls autoplay loop style="width:100%; height:100%; object-fit:contain;"></video>
      <button onclick="document.getElementById('temp_video_modal').remove()" style="position:absolute; top:12px; right:12px; width:32px; height:32px; border-radius:50%; background:rgba(0,0,0,0.7); color:#fff; border:1px solid rgba(255,255,255,0.2); cursor:pointer; font-size:16px; font-weight:bold; display:flex; align-items:center; justify-content:center; transition:0.2s;">✕</button>
    </div>
    <div style="display:flex; gap:10px;">
      <button class="btn-grad" onclick="triggerInlineDownload('${url}', '${url.split('/').pop()}'); document.getElementById('temp_video_modal').remove();" style="padding:8px 16px; font-size:13px; color:#000; font-weight:bold; border-radius:6px; cursor:pointer;">📥 Download Video</button>
      <button onclick="document.getElementById('temp_video_modal').remove()" style="padding:8px 16px; background:#333; color:#fff; border:1px solid #444; border-radius:6px; cursor:pointer; font-size:13px;">Close Preview</button>
    </div>
  `;
  document.body.appendChild(modal);
}

function toggleSelectAllExports() {
  allSelectedExports = !allSelectedExports;
  document.querySelectorAll('.export-download-chk').forEach(chk => {
    chk.checked = allSelectedExports;
  });
}

async function downloadSelectedVideos() {
  const selectedCheckboxes = document.querySelectorAll('.export-download-chk:checked');
  if (selectedCheckboxes.length === 0) {
    alert("Please select at least one exported video to download.");
    return;
  }
  for (let i = 0; i < selectedCheckboxes.length; i++) {
    const url = selectedCheckboxes[i].dataset.url;
    const fname = selectedCheckboxes[i].dataset.fname;
    
    triggerInlineDownload(url, fname);
    
    // Wait slightly between downloads to avoid browser block
    await new Promise(r => setTimeout(r, 1000));
  }
}

// ──────────────────────── Downloader (Top-right feature) ────────────────────────
// A fully separate video downloader, in the same page/port as the Shorts
// Studio. Paste any link, see full details (thumbnails, title, uploader,
// duration, every available quality/format with its size), copy the link,
// and download with a live percent / MB / speed / ETA readout — no need to
// go anywhere else.

function hideAllViews(){
  document.getElementById('studioView').classList.add('hidden');
  document.getElementById('downloaderView').classList.add('hidden');
  document.getElementById('editorView').classList.add('hidden');
  document.getElementById('publishView').classList.add('hidden');
  document.getElementById('studioTabBtn').classList.remove('active');
  document.getElementById('downloaderTabBtn').classList.remove('active');
  document.getElementById('editorTabBtn').classList.remove('active');
  document.getElementById('publishTabBtn').classList.remove('active');
}
function showStudioView(){
  hideAllViews();
  document.getElementById('studioView').classList.remove('hidden');
  document.getElementById('studioTabBtn').classList.add('active');
}
function showDownloaderView(){
  hideAllViews();
  document.getElementById('downloaderView').classList.remove('hidden');
  document.getElementById('downloaderTabBtn').classList.add('active');
}

function showEditorView(){
  hideAllViews();
  document.getElementById('editorView').classList.remove('hidden');
  document.getElementById('editorTabBtn').classList.add('active');
  const frame = document.getElementById('editorFrame');
  // Lazy-load: only point the iframe at the editor the first time this tab
  // is opened, so it doesn't do any work while the user is on Studio/Downloader.
  if(!frame.dataset.loaded){
    frame.src = '/api/editor/editor';
    frame.dataset.loaded = '1';
  }
}

function showPublishView(){
  hideAllViews();
  document.getElementById('publishView').classList.remove('hidden');
  document.getElementById('publishTabBtn').classList.add('active');
  const frame = document.getElementById('publishFrame');
  // Lazy-load, same pattern as the editor iframe above.
  if(!frame.dataset.loaded){
    frame.src = '/api/publish/publish';
    frame.dataset.loaded = '1';
  }
}

// Jump straight to the Publish tab and focus one specific just-exported
// clip's card there (called from the "📤 Publish this clip" link that
// appears in the export log once an export finishes — see saveVideo()).
function publishExportedClip(exportId){
  showPublishView();
  const frame = document.getElementById('publishFrame');
  const send = () => frame.contentWindow.postMessage({type:'publish_focus_export', export_id: exportId}, '*');
  if(frame.dataset.loaded === '1' && frame.dataset.ready === '1'){
    send();
  } else {
    // First time the iframe is opened it needs a moment to load+init before
    // it's listening for postMessage — retry briefly instead of racing it.
    let tries = 0;
    const iv = setInterval(() => {
      tries++;
      try { send(); } catch(e) {}
      if(tries > 25) clearInterval(iv);
    }, 200);
    frame.addEventListener('load', () => { frame.dataset.ready = '1'; }, {once:true});
  }
}

function setDlBtnLoading(isLoading, label){
  const btn = document.getElementById('dlFetchBtn');
  if(!btn) return;
  if(isLoading){
    if(!btn.dataset.origHtml) btn.dataset.origHtml = btn.innerHTML;
    btn.disabled = true;
    btn.style.opacity = '0.85';
    btn.style.cursor = 'wait';
    btn.innerHTML = `<span class="btn-spinner"></span>&nbsp;${label || 'Working…'}`;
  } else {
    btn.disabled = false;
    btn.style.opacity = '';
    btn.style.cursor = '';
    if(btn.dataset.origHtml) btn.innerHTML = btn.dataset.origHtml;
  }
}

function fmtDuration(sec){
  if(!sec && sec !== 0) return '–';
  sec = Math.round(sec);
  const h = Math.floor(sec/3600), m = Math.floor((sec%3600)/60), s = sec%60;
  return h>0 ? `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}` : `${m}:${String(s).padStart(2,'0')}`;
}

function setDlInlineStatus(msg, isError){
  const el = document.getElementById('dlLog');
  el.classList.remove('hidden');
  el.textContent = msg;
  el.style.color = isError ? '#ff6b7a' : '';
  el.style.borderColor = isError ? 'rgba(220,53,69,0.4)' : '';
}

async function fetchDownloadInfo(){
  const url = document.getElementById('dlUrl').value.trim();
  if(!url){ alert('Paste a video URL first'); return; }
  document.getElementById('dlInfoCard').classList.add('hidden');
  document.getElementById('dlProgressCard').classList.add('hidden');
  setDlBtnLoading(true, 'Fetching details…');
  setDlInlineStatus('⏳ Resolving video & fetching every detail…', false);
  let res, data;
  try{
    res = await fetch('/api/downloader/info', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url})});
    data = await res.json();
  } catch(e){
    setDlInlineStatus('❌ Network error', true); setDlBtnLoading(false); return;
  }
  setDlBtnLoading(false);
  if(data.error){ setDlInlineStatus('❌ '+data.error, true); return; }
  setDlInlineStatus(`✅ Found "${data.title}" — ${(data.formats||[]).length} format(s) available.`, false);
  renderDownloadInfo(data, url);
}

function copyDlValue(text, btnEl){
  if(!text) return;
  navigator.clipboard && navigator.clipboard.writeText(text).catch(()=>{});
  if(btnEl){
    const orig = btnEl.textContent;
    btnEl.textContent = '✅';
    setTimeout(()=>{ btnEl.textContent = orig; }, 1200);
  }
}

function addDetailItem(grid, label, value){
  if(value === null || value === undefined || value === '') return;
  const item = document.createElement('div');
  item.className = 'dl-detail-item';
  item.innerHTML = `
    <span class="dl-detail-label">${label}</span>
    <span class="dl-detail-val">${value}</span>
    <button class="dl-copy-icon-btn" title="Copy" onclick="copyDlValue(${JSON.stringify(String(value)).replace(/"/g,'&quot;')}, this)">📋</button>
  `;
  grid.appendChild(item);
}

function renderDownloadInfo(data, originalUrl){
  document.getElementById('dlInfoCard').classList.remove('hidden');
  document.getElementById('dlTitle').textContent = data.title || 'Untitled video';
  document.getElementById('dlThumbMain').src = data.thumbnail || '';
  document.getElementById('dlPageUrl').value = data.webpage_url || originalUrl;

  // thumbnail preview strip in multiple sizes/formats
  const strip = document.getElementById('dlThumbStrip');
  strip.innerHTML = '';
  (data.thumbnails||[]).forEach(t=>{
    const img = document.createElement('img');
    img.src = t.url;
    img.title = t.width && t.height ? `${t.width}x${t.height}` : 'thumbnail';
    img.onclick = () => { document.getElementById('dlThumbMain').src = t.url; };
    strip.appendChild(img);
  });

  // quick-glance badges
  const badges = document.getElementById('dlMetaBadges');
  badges.innerHTML = '';
  const badgeVals = [];
  if(data.uploader) badgeVals.push(`👤 ${data.uploader}`);
  if(data.duration_str) badgeVals.push(`⏱ ${data.duration_str}`);
  if(data.view_count_str) badgeVals.push(`👁 ${data.view_count_str} views`);
  if(data.extractor) badgeVals.push(`🌐 ${data.extractor}`);
  if(data.is_live) badgeVals.push(`🔴 LIVE`);
  badgeVals.forEach(v=>{ const s=document.createElement('span'); s.textContent=v; badges.appendChild(s); });

  // full details grid — everything meaningful yt-dlp gives us, not just a few picks
  const grid = document.getElementById('dlDetailsGrid');
  grid.innerHTML = '';
  addDetailItem(grid, 'Uploader / Channel', data.uploader);
  addDetailItem(grid, 'Subscribers', data.channel_follower_count_str);
  addDetailItem(grid, 'Upload Date', data.upload_date_str);
  addDetailItem(grid, 'Duration', data.duration_str);
  addDetailItem(grid, 'Views', data.view_count_str);
  addDetailItem(grid, 'Likes', data.like_count_str);
  addDetailItem(grid, 'Comments', data.comment_count_str);
  addDetailItem(grid, 'Best Resolution', data.resolution);
  addDetailItem(grid, 'Largest File Size', data.best_filesize_str);
  addDetailItem(grid, 'Total Formats', data.format_count);
  addDetailItem(grid, 'Language', data.language);
  addDetailItem(grid, 'Age Limit', data.age_limit ? data.age_limit+'+' : (data.age_limit === 0 ? 'None' : null));
  addDetailItem(grid, 'Availability', data.availability);
  addDetailItem(grid, 'License', data.license);
  addDetailItem(grid, 'Subtitles', data.subtitle_langs && data.subtitle_langs.length ? `${data.subtitle_langs.length} language(s)` : null);
  addDetailItem(grid, 'Auto Captions', data.auto_caption_langs_count ? `${data.auto_caption_langs_count} language(s)` : null);
  addDetailItem(grid, 'Chapters', data.chapters_count || null);
  addDetailItem(grid, 'Source Site', data.extractor);

  // description
  const descWrap = document.getElementById('dlDescWrap');
  const descText = document.getElementById('dlDescText');
  if(data.description){
    descWrap.style.display = '';
    descText.textContent = data.description;
    descText.classList.add('collapsed');
    document.getElementById('dlDescToggleBtn').textContent = '⬇️';
    document.getElementById('dlDescToggleBtn').onclick = () => {
      descText.classList.toggle('collapsed');
      document.getElementById('dlDescToggleBtn').textContent = descText.classList.contains('collapsed') ? '⬇️' : '⬆️';
    };
    document.getElementById('dlDescCopyBtn').onclick = (e) => copyDlValue(data.description, e.currentTarget);
  } else {
    descWrap.style.display = 'none';
  }

  // categories & tags chips
  const catWrap = document.getElementById('dlCategoriesWrap');
  const catChips = document.getElementById('dlCategoriesChips');
  catChips.innerHTML = '';
  if(data.categories && data.categories.length){
    catWrap.style.display = '';
    data.categories.forEach(c=>{ const s=document.createElement('span'); s.textContent=c; catChips.appendChild(s); });
  } else { catWrap.style.display = 'none'; }

  const tagWrap = document.getElementById('dlTagsWrap');
  const tagChips = document.getElementById('dlTagsChips');
  tagChips.innerHTML = '';
  if(data.tags && data.tags.length){
    tagWrap.style.display = '';
    data.tags.forEach(t=>{ const s=document.createElement('span'); s.textContent=t; tagChips.appendChild(s); });
    if(data.tags_more > 0){ const s=document.createElement('span'); s.textContent = `+${data.tags_more} more`; tagChips.appendChild(s); }
  } else { tagWrap.style.display = 'none'; }

  // formats table, each row with its own copy button too
  const tbody = document.getElementById('dlFormatsBody');
  tbody.innerHTML = '';
  (data.formats||[]).forEach(f=>{
    const codec = [f.vcodec, f.acodec].filter(Boolean).join(' / ') || '–';
    const summary = `${f.label||''} ${f.ext?.toUpperCase()||''} — ${f.filesize_str}`.trim();
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${f.kind}</td>
      <td><strong>${f.label||'—'}</strong></td>
      <td style="text-transform:uppercase;">${f.ext||'–'}</td>
      <td style="font-size:11.5px; color:var(--dim);">${codec}</td>
      <td>${f.filesize_str}</td>
      <td style="text-align:right; white-space:nowrap;">
        <button class="dl-copy-icon-btn" title="Copy quality info" onclick="copyDlValue(${JSON.stringify(summary).replace(/"/g,'&quot;')}, this)">📋</button>
        <button class="dl-fmt-btn" onclick="startDownload('${(originalUrl+"").replace(/'/g,"\\'")}','${f.format_id}')">⬇ Download</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function copyDlText(elId){
  const el = document.getElementById(elId);
  el.select();
  el.setSelectionRange(0, 99999);
  navigator.clipboard && navigator.clipboard.writeText(el.value).catch(()=>{ document.execCommand('copy'); });
  setDlInlineStatus('📋 Link copied to clipboard.', false);
}

// Moves the little step-tracker (Connecting → Downloading → Merging → Ready)
// forward based on the job's current stage — a clean, modern replacement
// for a raw scrolling console/cmd-style log that most users find confusing.
const DL_STAGE_ORDER = ['connect', 'download', 'merge', 'done'];
function setDlStepperStage(stage){
  const idx = DL_STAGE_ORDER.indexOf(stage);
  document.querySelectorAll('#dlStepper .dl-step').forEach(el=>{
    const stepIdx = DL_STAGE_ORDER.indexOf(el.dataset.step);
    el.classList.remove('active','complete');
    if(stepIdx < idx) el.classList.add('complete');
    else if(stepIdx === idx) el.classList.add('active');
  });
  document.querySelectorAll('#dlStepper .dl-step-line').forEach(el=>{
    const lineIdx = parseInt(el.dataset.line, 10); // line 1 sits between step0/step1, etc.
    el.classList.toggle('complete', lineIdx <= idx);
  });
}

async function startDownload(url, formatId){
  document.getElementById('dlProgressCard').classList.remove('hidden');
  document.getElementById('dlProgressCard').scrollIntoView({behavior:'smooth', block:'nearest'});
  document.getElementById('dlErrorBanner').classList.add('hidden');
  document.getElementById('dlPercentVal').textContent = '0%';
  document.getElementById('dlSizeVal').textContent = '0 MB / 0 MB';
  document.getElementById('dlSpeedVal').textContent = '– MB/s';
  document.getElementById('dlEtaVal').textContent = '–';
  document.getElementById('dlProgressFill').style.width = '0%';
  document.getElementById('dlProgressText').textContent = '0%';
  document.getElementById('dlStatusLine').textContent = 'Connecting…';
  setDlStepperStage('connect');

  let res, data;
  try{
    res = await fetch('/api/downloader/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url, format_id: formatId})});
    data = await res.json();
  } catch(e){
    showDlError('Network error starting download'); return;
  }
  if(data.error){ showDlError(data.error); return; }
  const dlId = data.dl_id;

  let done = false;
  while(!done){
    await new Promise(r=>setTimeout(r, 600));
    let pd;
    try{
      const st = await fetch(`/api/downloader/progress/${dlId}`);
      pd = await st.json();
    } catch(e){ continue; }
    if(pd.error){ showDlError(pd.error); return; }

    setDlStepperStage(pd.stage || 'connect');

    const pct = (pd.percent != null) ? pd.percent : 0;
    document.getElementById('dlPercentVal').textContent = pct + '%';
    document.getElementById('dlProgressFill').style.width = pct + '%';
    document.getElementById('dlProgressText').textContent = pct + '%';
    document.getElementById('dlSizeVal').textContent = `${pd.downloaded_str||'0 B'} / ${pd.total_str||'~unknown'}`;
    document.getElementById('dlSpeedVal').textContent = pd.speed_str || '– MB/s';
    document.getElementById('dlEtaVal').textContent = (pd.eta != null) ? fmtDuration(pd.eta) + ' left' : '–';

    if(pd.status === 'downloading'){
      document.getElementById('dlStatusLine').textContent = 'Downloading your video…';
    } else if(pd.status === 'processing'){
      document.getElementById('dlStatusLine').textContent = 'Merging audio & video…';
    }

    if(pd.status === 'done'){
      done = true;
      setDlStepperStage('done');
      document.getElementById('dlPercentVal').textContent = '100%';
      document.getElementById('dlProgressFill').style.width = '100%';
      document.getElementById('dlProgressText').textContent = '100%';
      document.getElementById('dlStatusLine').textContent = '🎉 Done! Saving to your device…';
      triggerInlineDownload(`/api/downloader/file/${dlId}`, pd.filename);
    } else if(pd.status === 'error'){
      done = true;
      showDlError(pd.error || 'Download failed');
    }
  }
}

function showDlError(msg){
  const banner = document.getElementById('dlErrorBanner');
  banner.textContent = '❌ ' + msg;
  banner.classList.remove('hidden');
  document.getElementById('dlStatusLine').textContent = '';
}

// Enter key in the downloader URL field works exactly like clicking "Fetch Video Details"
document.addEventListener('DOMContentLoaded', () => {
  const dlUrlInput = document.getElementById('dlUrl');
  if(dlUrlInput){
    dlUrlInput.addEventListener('keydown', (e) => {
      if(e.key === 'Enter'){
        e.preventDefault();
        fetchDownloadInfo();
      }
    });
  }
});
</script>
</body>
</html>
"""


def upgrade_yt_dlp_silently():
    import sys
    import subprocess
    try:
        print("[Auto-Update] Checking and upgrading yt-dlp to the latest version to prevent bot-detection issues...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[Auto-Update] yt-dlp has been updated to the latest secure version.")
    except Exception as e:
        print(f"[Auto-Update Warning] Could not auto-upgrade yt-dlp: {e}")


def main():
    # Upgrade yt-dlp silently on startup in a separate thread so it doesn't block Flask booting
    threading.Thread(target=upgrade_yt_dlp_silently, daemon=True).start()
    # threading.Thread(target=preload_ai_models, daemon=True).start()

    print("[RenderDetect] running as a module inside main app on port 5000")

def _cleanup_old_proxy_files():
    import time
    while True:
        time.sleep(3600)
        cutoff = time.time() - (2 * 3600)
        for f in PROXY_DIR.glob("*.mp4"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)

threading.Thread(target=_cleanup_old_proxy_files, daemon=True).start()


if __name__ == "__main__":
    main()