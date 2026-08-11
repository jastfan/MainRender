#!/usr/bin/env python3
# app_hooks.py — 2-stage Hook Detector, Flask version, with a LIVE key-pool
# dashboard + async jobs (NO caching, NO blocking).
#
# CHANGE FROM PICHLI VERSION (jo tumne bataya wo problem thi):
#   - Disk cache (7-din wala) HATA diya — koi purana result silently reuse
#     nahi hota.
#   - "Dusri request wait karti hai pehli ke liye" wala dedup pura HATA
#     diya — ab HAR /analyze call apna khud ka background job turant shuru
#     karta hai, kisi ke result ka wait nahi karta.
#   - Page refresh/reload karo ya nayi URL daalo — turant naya job chalta
#     hai, blocking nahi.
#   - Jo job chal raha hai (ya khatam ho chuka hai) wo apne VIDEO TITLE ke
#     naam se "Jobs" panel me dikhta rehta hai (in-memory, jab tak server
#     chal raha hai) — taaki tum dekh sako "X video abhi process ho raha
#     hai" ya "Y video ka result ready hai", bina kisi naye job ko roke.
#
# Quota-protection jo pehle se thi wahi rakhi hai (kyunki wo asli
# 429/403 problem solve karti hai, aur blocking nahi karti):
#   - Multi-key pool + per-key RPM self-throttle (request bhejne se PEHLE
#     hi wait karta hai agar limit paas hai — isliye ek saath kai jobs
#     chalein bhi to keys aapas me safely share hoti hain)
#   - 403 par model fallback list
#   - Live status panel + activity log (/status) — dikhata hai abhi konsi
#     key/model kaam kar rahi hai, kitna cooldown bacha hai
#
# ⚠️ Yaad rakho: Gemini free-tier limits PER-PROJECT hote hain, per-key
# nahi. Alag Google account/project ki keys hi asli fayda deti hain.
#
# pip install flask google-genai pydantic youtube_transcript_api

import re
import os
import json
import time
import uuid
import threading
import itertools
import urllib.request
from collections import deque
from pathlib import Path
import yt_dlp

from flask import Flask,Blueprint, request, render_template_string, jsonify
from pydantic import BaseModel, Field
from google import genai
from google.genai import errors as genai_errors
from youtube_transcript_api import YouTubeTranscriptApi

from config import Config


# ── Keys ab env-variable se aati hain, code mein hardcode nahi ─────────
# Render pe GEMINI_API_KEYS = "key1,key2,key3" (comma se alag, spaces optional)
_raw_keys = os.environ.get("GEMINI_API_KEYS", "")
API_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
print(API_KEYS)





STAGE1_MODEL_CANDIDATES = [
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
]
STAGE2_MODEL_CANDIDATES = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# Apni AI Studio dashboard me exact RPM check karke yahan update kar lena.
RPM_LIMITS = {
    "gemini-flash-lite-latest": 12,
    "gemini-2.5-flash-lite": 12,
    "gemini-2.0-flash-lite": 12,
    "gemini-flash-latest": 8,
    "gemini-2.5-flash": 8,
    "gemini-2.0-flash": 8,
}
DEFAULT_RPM = 8


# ── Activity log: "abhi kya ho raha hai" wala live feed ───────────────────
_activity = deque(maxlen=80)
_activity_lock = threading.Lock()


def log_activity(msg):
    with _activity_lock:
        _activity.append({"t": time.strftime("%H:%M:%S"), "msg": msg})
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def get_activity():
    with _activity_lock:
        return list(_activity)[::-1]


# ── Multi-key pool: per-key cooldown + RPM self-throttle + status ─────────
class KeyPool:
    def __init__(self, keys):
        if not keys or keys[0].startswith("KEY_"):
            raise RuntimeError("API_KEYS list me apni asli keys daalo (app_hooks.py ke top par).")
        self.keys = keys
        self.lock = threading.Lock()
        self._cycle = itertools.cycle(keys)

        self.cooldown_until = {k: 0.0 for k in keys}
        self.cooldown_reason = {k: None for k in keys}
        self.status = {k: "idle" for k in keys}
        self.total_calls = {k: 0 for k in keys}
        self.call_log = {k: {} for k in keys}

    def _rpm_wait(self, key, model):
        limit = RPM_LIMITS.get(model, DEFAULT_RPM)
        dq = self.call_log[key].setdefault(model, deque())
        now = time.time()
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= limit:
            return max(0.0, 60 - (now - dq[0]) + 0.5)
        return 0.0

    def _record_call(self, key, model):
        self.call_log[key].setdefault(model, deque()).append(time.time())
        self.total_calls[key] += 1

    def _best_key(self, model):
        with self.lock:
            now = time.time()
            for _ in range(len(self.keys)):
                k = next(self._cycle)
                if self.cooldown_until[k] <= now and self._rpm_wait(k, model) == 0:
                    return k, 0.0
            best_key, best_wait = None, None
            for k in self.keys:
                wait = max(0.0, self.cooldown_until[k] - now, self._rpm_wait(k, model))
                if best_wait is None or wait < best_wait:
                    best_key, best_wait = k, wait
            return best_key, best_wait

    def _mark_cooldown(self, key, seconds, reason):
        with self.lock:
            self.cooldown_until[key] = time.time() + seconds
            self.cooldown_reason[key] = reason

    def key_label(self, key):
        return f"...{key[-6:]}"

    def snapshot(self):
        now = time.time()
        rows = []
        for i, k in enumerate(self.keys, 1):
            cd_left = max(0.0, self.cooldown_until[k] - now)
            calls_60s = {m: len(dq) for m, dq in self.call_log[k].items()}
            rpm_usage = {
                m: f"{len(dq)}/{RPM_LIMITS.get(m, DEFAULT_RPM)} used"
                for m, dq in self.call_log[k].items()
            }
            if cd_left > 0:
                state = "cooldown"
            elif self.status[k] == "busy":
                state = "busy"
            else:
                state = "idle"
            rows.append({
                "label": f"Key #{i} ({self.key_label(k)})",
                "state": state,
                "cooldown_seconds_left": round(cd_left, 1),
                "cooldown_reason": self.cooldown_reason[k],
                "total_calls": self.total_calls[k],
                "calls_last_60s": calls_60s,
                "rpm_usage": rpm_usage,
            })
        return rows

    def call(self, model_candidates, fn, max_attempts=8):
        last_err = None
        model_idx = 0
        for attempt in range(max_attempts):
            model = model_candidates[min(model_idx, len(model_candidates) - 1)]
            key, wait = self._best_key(model)
            if wait > 0:
                log_activity(f"⏳ Saari keys busy/cooldown/RPM-limit me hain — {wait:.0f}s wait ({model})...")
                time.sleep(wait)
                key, _ = self._best_key(model)

            self.status[key] = "busy"
            log_activity(f"▶️  {self.key_label(key)} se call ja rahi hai — model={model}")
            client = genai.Client(api_key=key)
            try:
                self._record_call(key, model)
                result = fn(client, model)
                self.status[key] = "idle"
                log_activity(f"✅ {self.key_label(key)} ({model}) — response mil gaya")
                return result
            except genai_errors.APIError as e:
                self.status[key] = "idle"
                last_err = e
                code = getattr(e, "code", None)
                if code == 429:
                    cooldown = 60
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", str(e), re.IGNORECASE)
                    if m:
                        cooldown = float(m.group(1)) + 2
                    self._mark_cooldown(key, cooldown, f"429 rate-limit ({model})")
                    log_activity(f"🔁 {self.key_label(key)} rate-limited ({cooldown:.0f}s cooldown) — dusri key try karenge...")
                    continue
                if code == 403:
                    self._mark_cooldown(key, 15, f"403 model unavailable ({model})")
                    model_idx += 1
                    log_activity(f"⚠️  {self.key_label(key)} par '{model}' 403 (unavailable) — agla model candidate try kar rahe hain...")
                    continue
                if code == 503:
                    log_activity("⚠️  Google server busy (503) — 5s baad retry...")
                    time.sleep(5)
                    continue
                log_activity(f"❌ {self.key_label(key)} par na-sambhalne wali error: {e}")
                raise
        raise last_err


pool = KeyPool(API_KEYS)


# ── Jobs: har request apna khud ka background job hai, koi blocking nahi ──
JOBS = {}          # job_id -> dict
_jobs_lock = threading.Lock()
MAX_JOBS_KEPT = 50  # purane jobs list se hata do taaki memory na bhare


def _new_job(video_id, title, url, num_clips, min_len, max_len):
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "video_id": video_id,
        "title": title,
        "url": url,
        "num_clips": num_clips,
        "min_len": min_len,
        "max_len": max_len,
        "status": "running",       # running | done | error
        "moments": None,
        "error": None,
        "created_at": time.time(),
        "finished_at": None,
    }
    with _jobs_lock:
        JOBS[job_id] = job
        if len(JOBS) > MAX_JOBS_KEPT:
            oldest = sorted(JOBS.values(), key=lambda j: j["created_at"])[0]
            JOBS.pop(oldest["id"], None)
    return job


def _finish_job(job_id, moments=None, error=None):
    with _jobs_lock:
        job = JOBS.get(job_id)
        if not job:
            return
        job["finished_at"] = time.time()
        if error:
            job["status"] = "error"
            job["error"] = str(error)
        else:
            job["status"] = "done"
            job["moments"] = moments


def get_video_title(url, video_id):
    """YouTube ka public oEmbed endpoint — koi API key nahi chahiye. Fail ho
    to bas video_id dikha do, kaam nahi rukega."""
    try:
        oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
        with urllib.request.urlopen(oembed_url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("title") or video_id
    except Exception:
        return video_id


# ── Schema ───────────────────────────────────────────────────────────────
class HookMoment(BaseModel):
    start: str = Field(description="Timestamp MM:SS, must match transcript exactly")
    end: str = Field(description="Timestamp MM:SS, must match transcript exactly")
    dialogue: str = Field(description="The exact key line(s) from the transcript at this moment")
    hook_type: str = Field(description="One of: tonal_shift, direct_address, hidden_knowledge, pattern_interrupt, controversial_claim, shocking_stat, story_climax, punchline, emotional_peak, strong_question, before_after, myth_bust")
    structure_check: str = Field(description="One sentence: what's the Hook, what's the Build, what's the Payoff")
    standalone_clarity: int = Field(description="1-10: understandable with ZERO prior context")
    hook_strength: int = Field(description="1-10: strength of first 3 seconds")
    emotional_intensity: int = Field(description="1-10: strength of emotional pull")
    quotability: int = Field(description="1-10: likely to be quoted/commented on")


# ── Transcript helpers ──────────────────────────────────────────────────
def extract_video_id(url):
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
    return match.group(1) if match else None


# ── Fallback captions source: yt-dlp + browser-cookie chain (same pattern
# RenderDetect.py already uses) — sirf tabhi chalta hai jab youtube_transcript_api
# (default, fast) fail ho jaaye (jaise IP-block error). Optional local
# cookies.txt bhi support karta hai (word_captions2.py jaisa).
COOKIES_FILE = Path(__file__).resolve().parent / "cookies.txt"   # optional — export from browser if needed
COOKIE_BROWSERS = ["chrome", "edge", "firefox", "brave"]
_RESOLVED_MODE = {"mode": None}   # jo mode pehle kaam kar gaya, agli baar seedha wahi try hota hai


def _ydl_base_opts():
    opts = {'quiet': True, 'no_warnings': True, 'noplaylist': True, 'skip_download': True,
            'socket_timeout': 15, 'retries': 2, 'extractor_retries': 1, 'geo_bypass': True,
            'ignoreerrors': 'only_download'}
    if COOKIES_FILE.exists():
        opts['cookiefile'] = str(COOKIES_FILE)
    return opts


def _auth_attempts():
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


def _parse_vtt_cues(raw_text):
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r'\n\s*\n', raw_text.strip())
    tag_re = re.compile(r'<[^>]+>')
    time_re = re.compile(r'(\d+):(\d{2}):(\d{2})[.,](\d{1,3})')

    def ts_to_sec(ts):
        m = time_re.search(ts)
        if not m:
            return None
        h, mi, s, ms = m.groups()
        ms = (ms + "000")[:3]
        return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000.0

    cues = []
    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip()]
        arrow_idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if arrow_idx is None:
            continue
        parts = lines[arrow_idx].split("-->")
        if len(parts) != 2:
            continue
        start, end = ts_to_sec(parts[0].strip()), ts_to_sec(parts[1].strip())
        if start is None or end is None:
            continue
        text = " ".join(lines[arrow_idx + 1:]).strip()
        text = tag_re.sub("", text)
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            cues.append({"start": round(start, 2), "end": round(end, 2), "text": text})
    return cues


def get_cues_via_ytdlp(video_id, url):
    """yt-dlp fallback chain: cached-working-mode -> cookies_file -> browser
    cookies (chrome/edge/firefox/brave) -> default -> bypass (ios/android
    client). Same battle-tested chain RenderDetect.py uses."""
    ydl_opts = _ydl_base_opts()
    attempts = _auth_attempts()
    last_err = None
    info = None
    for mode in attempts:
        opts = dict(ydl_opts)
        t0 = time.time()
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
        log_activity(f"🍪 yt-dlp fallback: '{mode}' try kar rahe hain...")
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            if info:
                _RESOLVED_MODE["mode"] = mode
                log_activity(f"✅ yt-dlp fallback: '{mode}' kaam kar gaya ({time.time()-t0:.1f}s)")
                break
        except Exception as e:
            last_err = e
            info = None
            log_activity(f"⚠️  yt-dlp fallback: '{mode}' fail ({time.time()-t0:.1f}s) — {e}")
            continue

    if not info:
        raise RuntimeError(f"yt-dlp fallback bhi fail ho gaya: {last_err}")

    bucket = info.get('subtitles') or {}
    if not bucket:
        bucket = info.get('automatic_captions') or {}
    if not bucket:
        raise RuntimeError("Is video me koi captions/subtitles nahi hain (manual ya auto, dono nahi mile).")

    lang = "en" if "en" in bucket else next(iter(bucket))
    tracks = bucket[lang]
    track_url = None
    for t in tracks:
        if t.get('ext') in ('vtt', 'srt'):
            track_url = t['url']
            break
    if not track_url and tracks:
        track_url = tracks[0]['url']
    if not track_url:
        raise RuntimeError("Caption track ka URL resolve nahi hua.")

    req = urllib.request.Request(track_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    cues = _parse_vtt_cues(raw)
    if not cues:
        raise RuntimeError("Caption file khaali/unparsable nikla.")
    return cues


def get_cues(video_id, url=None):
    """DEFAULT: pehle youtube_transcript_api (fast, simple, no cookies).
    Fail ho (IP-block/RequestBlocked/koi bhi wajah) to khud-ba-khud yt-dlp +
    browser-cookie fallback chain try karta hai — koi manual switch nahi
    chahiye. Dono steps ka timing activity-log me dikhta hai."""
    t0 = time.time()
    try:
        log_activity("📝 Transcript source: youtube_transcript_api (default, fast path)...")
        yt_api = YouTubeTranscriptApi()
        raw_data = yt_api.fetch(video_id, languages=['hi', 'en', 'en-IN'])
        cues = []
        for seg in raw_data:
            start = float(seg.start)
            dur = float(getattr(seg, "duration", 2.0) or 2.0)
            cues.append({"start": start, "end": start + dur, "text": seg.text})
        if not cues:
            raise ValueError("Khaali transcript mila")
        log_activity(f"✅ youtube_transcript_api se mil gaya ({time.time()-t0:.1f}s)")
        return cues
    except Exception as e:
        log_activity(f"⚠️  youtube_transcript_api fail ({time.time()-t0:.1f}s): {e} — yt-dlp fallback try kar rahe hain...")
        t1 = time.time()
        cues = get_cues_via_ytdlp(video_id, url or f"https://www.youtube.com/watch?v={video_id}")
        log_activity(f"✅ yt-dlp fallback se mil gaya (+{time.time()-t1:.1f}s)")
        return cues


def cues_to_transcript_text(cues):
    lines = []
    for c in cues:
        m, s = divmod(int(c["start"]), 60)
        lines.append(f"[{m:02d}:{s:02d}] {c['text']}")
    return "\n".join(lines)


def mmss_to_sec(ts):
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0


def snap_to_cue(t, cues, prefer="start"):
    if not cues:
        return t
    best = min(cues, key=lambda c: abs((c["start"] if prefer == "start" else c["end"]) - t))
    return best["start"] if prefer == "start" else best["end"]


# ── Stage 1: free-form candidate discovery ─────────────────────────────
STAGE1_PROMPT = """You are an expert short-form video editor with the same eye for
viral moments as professional AI clipping tools (Opus Clip, ClipAnything). Think
step by step, out loud, in prose (not JSON yet).

TRANSCRIPT (timestamped):
{transcript}

Do this analysis:
1. Summarize the video's overall topic/arc in 2-3 sentences.
2. Identify EVERY moment that could work as a short-form hook — over-generate,
   find at least {overgenerate} candidates even if some are weak. For each,
   note: timestamp range, hook type, whether it has a clear Hook->Build->Payoff arc.
3. For each candidate, critically note ONE weakness.
4. Flag which candidates overlap or repeat the same point.
"""


def stage1_analysis(transcript, num_clips):
    prompt = STAGE1_PROMPT.format(transcript=transcript, overgenerate=num_clips * 3)

    def _call(client, model):
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.5, "max_output_tokens": 2048},
        )

    return pool.call(STAGE1_MODEL_CANDIDATES, _call).text


# ── Stage 2: structured selection ──────────────────────────────────────
STAGE2_PROMPT = """You previously did this analysis of a video transcript:

--- YOUR ANALYSIS ---
{analysis}
--- END ANALYSIS ---

Original timestamped transcript (for exact wording/timestamps):
{transcript}

Now select the FINAL {num_clips} best clips, applying these hard filters:
- REJECT any candidate without a real Hook->Build->Payoff arc.
- REJECT any candidate that isn't understandable with ZERO prior context.
- REJECT redundant candidates that overlap or repeat a stronger candidate.
- The final {num_clips} clips must be spread across DIFFERENT parts of the video.
- Each clip should be roughly {min_len}-{max_len} seconds.
- start/end MUST be real timestamps from the transcript.
- Clips must not overlap each other.
Score each surviving clip honestly (1-10 each), don't inflate scores.
"""


def stage2_select(analysis, transcript, num_clips, min_len, max_len):
    prompt = STAGE2_PROMPT.format(
        analysis=analysis, transcript=transcript,
        num_clips=num_clips, min_len=min_len, max_len=max_len,
    )

    def _call(client, model):
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[HookMoment],
                "temperature": 0.15,
            },
        )

    return json.loads(pool.call(STAGE2_MODEL_CANDIDATES, _call).text)


# ── Deterministic scoring + overlap removal ────────────────────────────
WEIGHTS = {"standalone_clarity": 0.35, "hook_strength": 0.35,
           "emotional_intensity": 0.20, "quotability": 0.10}


def compute_virality_score(m):
    raw = sum(m[k] * w for k, w in WEIGHTS.items())
    return round(raw * 10)


def postprocess(moments, cues, total_duration):
    cleaned = []
    for m in moments:
        start = snap_to_cue(mmss_to_sec(m["start"]), cues, "start")
        end = snap_to_cue(mmss_to_sec(m["end"]), cues, "end")
        if total_duration:
            end = min(end, total_duration)
        if end - start < 5:
            continue
        m["start_sec"] = round(start, 2)
        m["end_sec"] = round(end, 2)
        m["virality_score"] = compute_virality_score(m)
        cleaned.append(m)

    cleaned.sort(key=lambda m: m["virality_score"], reverse=True)
    final = []
    for m in cleaned:
        overlaps = any(not (m["end_sec"] <= f["start_sec"] or m["start_sec"] >= f["end_sec"]) for f in final)
        if not overlaps:
            final.append(m)
    final.sort(key=lambda m: m["start_sec"])
    return final


# ── Background worker: ye ek job ke liye poora pipeline chalata hai ───────
def _run_job(job_id, video_id, title, url, num_clips, min_len, max_len):
        
    try:
        log_activity(f"🚀 [{title}] Transcript fetch ho rahi hai...")
        cues = get_cues(video_id, url)
        if not cues:
            raise ValueError("Transcript nahi mili")
        transcript = cues_to_transcript_text(cues)
        total_duration = cues[-1]["end"]

        log_activity(f"🚀 [{title}] Stage 1 (candidate discovery) shuru...")
        analysis = stage1_analysis(transcript, num_clips)

        log_activity(f"🚀 [{title}] Stage 2 (final selection) shuru...")
        raw_moments = stage2_select(analysis, transcript, num_clips, min_len, max_len)

        final_moments = postprocess(raw_moments, cues, total_duration)
        _finish_job(job_id, moments=final_moments)
        log_activity(f"🏁 [{title}] {len(final_moments)} final clips ready.")
    except Exception as e:
        _finish_job(job_id, error=e)
        log_activity(f"❌ [{title}] job fail ho gaya: {e}")


def start_analyze_job(url, num_clips=5, min_len=15, max_len=60):
    """Har call apna NAYA job turant start karta hai — kisi purani/chalti
    request ka wait nahi karta, koi cache check nahi karta."""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    title = get_video_title(url, video_id)
    job = _new_job(video_id, title, url, num_clips, min_len, max_len)

    t = threading.Thread(
        target=_run_job,
        args=(job["id"], video_id, title, url, num_clips, min_len, max_len),
        daemon=True,
    )
    t.start()
    return job


# ── Flask app ────────────────────────────────────────────────────────────
hooks_bp = Blueprint("hooks_bp", __name__, url_prefix="/hooks")

PAGE = """
<!DOCTYPE html>
<html lang="hi">
<head>
<meta charset="UTF-8">
<title>Hook Detector</title>
<style>
  *{box-sizing:border-box;}
  body{background:#0b0d12;color:#eef0f6;font-family:'Segoe UI',sans-serif;padding:24px;margin:0;}
  .wrap{max-width:1200px;margin:0 auto;display:grid;grid-template-columns:1.4fr 1fr;gap:22px;}
  @media (max-width:900px){.wrap{grid-template-columns:1fr;}}
  h1{background:linear-gradient(135deg,#6e5bff,#22d3c4);-webkit-background-clip:text;color:transparent;margin-bottom:4px;}
  h2{font-size:16px;color:#8a90a4;text-transform:uppercase;letter-spacing:.05em;margin:0 0 10px;}
  input{width:60%;padding:12px;border-radius:8px;border:1px solid #262a36;background:#191c25;color:#fff;}
  input[type=number]{width:70px;}
  button{padding:12px 20px;border-radius:8px;border:none;background:linear-gradient(135deg,#6e5bff,#22d3c4);color:#0a0a0f;font-weight:700;cursor:pointer;}
  .card{background:#13151c;border:1px solid #262a36;border-radius:14px;padding:16px;margin-top:14px;}
  .ts{color:#22d3c4;font-weight:700;}
  .score{float:right;background:linear-gradient(135deg,#6e5bff,#22d3c4);color:#0a0a0f;padding:4px 12px;border-radius:999px;font-size:13px;font-weight:800;}
  .dim{color:#8a90a4;font-size:13px;margin-top:6px;}
  .subscores{display:flex;gap:10px;margin-top:10px;flex-wrap:wrap;}
  .subscores span{background:#191c25;border:1px solid #262a36;padding:3px 9px;border-radius:8px;font-size:11px;color:#8a90a4;}
  .htype{display:inline-block;background:#191c25;color:#22d3c4;padding:2px 8px;border-radius:6px;font-size:11px;margin-top:6px;}
  #status{margin-top:12px;color:#8a90a4;}
  .panel{background:#13151c;border:1px solid #262a36;border-radius:14px;padding:16px;margin-bottom:16px;}
  .keyrow{border:1px solid #262a36;border-radius:10px;padding:10px 12px;margin-bottom:10px;}
  .badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:800;text-transform:uppercase;}
  .badge.idle{background:#16351f;color:#3ecf6a;}
  .badge.busy{background:#2a2a4a;color:#8b8bff;}
  .badge.cooldown{background:#3a2416;color:#ff9d4d;}
  .badge.running{background:#2a2a4a;color:#8b8bff;}
  .badge.done{background:#16351f;color:#3ecf6a;}
  .badge.error{background:#3a1616;color:#ff5d5d;}
  .keymeta{color:#8a90a4;font-size:12px;margin-top:6px;}
  .log{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#8a90a4;max-height:220px;overflow-y:auto;}
  .log div{padding:3px 0;border-bottom:1px solid #1c1f29;}
  .log .t{color:#22d3c4;margin-right:6px;}
  .opts{margin-top:10px;color:#8a90a4;font-size:13px;}
  .jobrow{border:1px solid #262a36;border-radius:10px;padding:10px 12px;margin-bottom:8px;cursor:pointer;}
  .jobrow:hover{border-color:#6e5bff;}
  .jobtitle{font-weight:700;font-size:13px;}
</style>
</head>
<body>
<div class="wrap">
  <div>
    <h1>🔥 YouTube Hook Detector</h1>
    <form id="f">
      <input id="url" placeholder="Paste YouTube URL" required>
      <button type="submit">Analyze (naya job)</button>
      <div class="opts">
        Clips: <input type="number" id="numClips" value="5" min="1" max="15">
        Min len(s): <input type="number" id="minLen" value="15" min="5">
        Max len(s): <input type="number" id="maxLen" value="60" min="10">
      </div>
    </form>
    <div id="status"></div>
    <div id="results"></div>
  </div>

  <div>
    <div class="panel">
      <h2>🗂️ Jobs (title ke naam se)</h2>
      <div id="jobs">Loading...</div>
    </div>
    <div class="panel">
      <h2>🔑 Live Key Pool Status</h2>
      <div id="keys">Loading...</div>
    </div>
    <div class="panel">
      <h2>📜 Activity Log</h2>
      <div class="log" id="log">Loading...</div>
    </div>
  </div>
</div>

<script>
let watchedJobId = null;

function showResults(job) {
  if (job.status === 'running') {
    document.getElementById('status').innerText = `⏳ "${job.title}" abhi process ho raha hai...`;
    document.getElementById('results').innerHTML = '';
    return;
  }
  if (job.status === 'error') {
    document.getElementById('status').innerText = `❌ "${job.title}": ${job.error}`;
    document.getElementById('results').innerHTML = '';
    return;
  }
  document.getElementById('status').innerText = `✅ "${job.title}" — ${job.moments.length} final moments`;
  document.getElementById('results').innerHTML = job.moments.map(m => `
    <div class="card">
      <span class="score">${m.virality_score}/100</span>
      <span class="ts">${m.start} – ${m.end}</span>
      <div class="htype">${m.hook_type}</div>
      <div style="margin-top:8px">${m.dialogue}</div>
      <div class="dim" style="font-style:italic">Structure: ${m.structure_check}</div>
      <div class="subscores">
        <span>Standalone: ${m.standalone_clarity}/10</span>
        <span>Hook: ${m.hook_strength}/10</span>
        <span>Emotion: ${m.emotional_intensity}/10</span>
        <span>Quotable: ${m.quotability}/10</span>
      </div>
    </div>
  `).join('');
}

document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const url = document.getElementById('url').value;
  const num_clips = parseInt(document.getElementById('numClips').value || '5');
  const min_len = parseInt(document.getElementById('minLen').value || '15');
  const max_len = parseInt(document.getElementById('maxLen').value || '60');
  document.getElementById('status').innerText = '🚀 Naya job shuru ho gaya, background me chal raha hai...';
  document.getElementById('results').innerHTML = '';
  try {
    // Ye call turant lautta hai — kisi purani request ka wait nahi karta.
    const res = await fetch('/analyze', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({url, num_clips, min_len, max_len})});
    const data = await res.json();
    if (data.error) { document.getElementById('status').innerText = '❌ ' + data.error; return; }
    watchedJobId = data.job_id;
  } catch(err) {
    document.getElementById('status').innerText = '❌ Network error: ' + err.message;
  }
};

function badgeClass(state) {
  return 'badge ' + state;
}
function stateLabel(state) {
  if (state === 'idle') return 'Ready';
  if (state === 'busy') return 'Running now';
  if (state === 'running') return 'Running';
  if (state === 'done') return 'Done';
  if (state === 'error') return 'Error';
  return 'Cooldown';
}

async function pollStatus() {
  try {
    const res = await fetch('/status');
    const data = await res.json();

    document.getElementById('keys').innerHTML = data.keys.map(k => `
      <div class="keyrow">
        <b>${k.label}</b>
        <span class="${badgeClass(k.state)}" style="float:right">${stateLabel(k.state)}${k.state==='cooldown' ? ' · '+k.cooldown_seconds_left+'s' : ''}</span>
        <div class="keymeta">
          ${k.cooldown_reason ? 'Wajah: ' + k.cooldown_reason + '<br>' : ''}
          Total calls (session): ${k.total_calls}<br>
          ${Object.entries(k.rpm_usage).map(([model, usage]) => `${model}: ${usage}`).join('<br>') || 'Abhi tak koi call nahi'}
        </div>
      </div>
    `).join('') || '<div class="keymeta">Koi key configured nahi.</div>';

    document.getElementById('log').innerHTML = data.activity.map(a =>
      `<div><span class="t">${a.t}</span>${a.msg}</div>`
    ).join('') || '<div class="keymeta">Abhi tak koi activity nahi.</div>';
  } catch (e) {}
}

async function pollJobs() {
  try {
    const res = await fetch('/jobs');
    const data = await res.json();
    document.getElementById('jobs').innerHTML = data.jobs.map(j => `
      <div class="jobrow" onclick="watchedJobId='${j.id}'; loadWatchedJob();">
        <span class="jobtitle">${j.title}</span>
        <span class="${badgeClass(j.status)}" style="float:right">${stateLabel(j.status)}</span>
      </div>
    `).join('') || '<div class="keymeta">Abhi tak koi job nahi. Ek URL daal ke Analyze dabao.</div>';

    // Agar koi job track nahi ho raha, sabse naya running/latest job dikha do
    if (!watchedJobId && data.jobs.length) {
      watchedJobId = data.jobs[0].id;
    }
    loadWatchedJob();
  } catch (e) {}
}

async function loadWatchedJob() {
  if (!watchedJobId) return;
  try {
    const res = await fetch('/job/' + watchedJobId);
    if (!res.ok) return;
    const job = await res.json();
    showResults(job);
  } catch (e) {}
}

pollStatus();
pollJobs();
setInterval(pollStatus, 2000);
setInterval(pollJobs, 2000);
</script>
</body>
</html>
"""


@hooks_bp.route("/")
def index():
    return render_template_string(PAGE)


@hooks_bp.route("/status")
def status():
    return jsonify({"keys": pool.snapshot(), "activity": get_activity()})


@hooks_bp.route("/jobs")
def jobs():
    with _jobs_lock:
        all_jobs = sorted(JOBS.values(), key=lambda j: j["created_at"], reverse=True)
        out = [{"id": j["id"], "title": j["title"], "status": j["status"]} for j in all_jobs]
    return jsonify({"jobs": out})


@hooks_bp.route("/job/<job_id>")
def job_detail(job_id):
    with _jobs_lock:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(dict(job))


@hooks_bp.route("/analyze", methods=["POST"])
def analyze():
    body = request.json or {}
    url = body.get("url", "").strip()
    num_clips = int(body.get("num_clips", 5))
    min_len = int(body.get("min_len", 15))
    max_len = int(body.get("max_len", 60))
    if not url:
        return jsonify({"error": "No URL given"}), 400
    try:
        # Yahan koi wait/block nahi — job turant background me shuru hoke
        # job_id turant wapas mil jaata hai.
        job = start_analyze_job(url, num_clips=num_clips, min_len=min_len, max_len=max_len)
        return jsonify({"job_id": job["id"], "title": job["title"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def init_hooks(main_app):
    main_app.register_blueprint(hooks_bp)