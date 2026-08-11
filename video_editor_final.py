#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoShortAi — Video Editor module
──────────────────────────────────
Adds AI auto-reframe (face tracking), trending animated captions,

hook-cut / manual clip reordering, and one-click multi-aspect-ratio
export — all producing a SINGLE muxed video+audio file per ratio.
 
Wire it into the main app with:
 
    from video_editor import editor_bp, init_editor
    init_editor(BASE, FFMPEG)
    app.register_blueprint(editor_bp)
 
Routes exposed (all under /api/editor/...):
    GET  /api/editor/editor             -> serves the standalone editor.html UI (open this in a tab)
    POST /api/editor/upload             -> upload a local file, returns a source_path to use below
    POST /api/editor/analyze/start       -> kicks off a background analyze job (face-track + real captions + AI viral-clip finder + hook/silence detection)
    GET  /api/editor/analyze/progress/<job_id> -> live analyze progress + log console + final result
    GET  /api/editor/styles             -> list of caption style presets
    GET  /api/editor/ratios             -> list of export aspect-ratio presets
    GET  /api/editor/features           -> which advanced features are available (installed libs)
    POST /api/editor/render             -> kicks off a background render job
    GET  /api/editor/progress/<job_id>  -> live progress
    GET  /api/editor/file/<job_id>/<ratio> -> serves one finished ratio's file
    POST /api/editor/keyframe/add       -> add a manual crop keyframe to a job's override track
    POST /api/editor/keyframe/remove    -> remove a manual crop keyframe
    POST /api/editor/segment/add        -> add a hook-cut / reorder segment
    POST /api/editor/segment/remove     -> remove a hook-cut / reorder segment
 
ADVANCED EXTRAS (on top of the original spec):
    • fill_mode="blur_pad"   -> reframe by fitting the WHOLE frame + blurred
      background bars instead of hard-cropping (no content ever cut off).
    • jump_cut feature       -> auto-detects and removes silence/dead-air,
      the other half of "trending" short-form editing besides hook-cut.
    • hook_cut + jump_cut COMBINED -> if both are on, the loudest window is
      used as the cold-open "hook" and is prepended in front of the
      dead-air-trimmed rest of the video (previously turning jump_cut on
      silently dropped hook_cut — now they compose).
    • normalize_audio        -> EBU R128 loudness normalization (-14 LUFS),
      on by default, matches the loudness social platforms expect.
    • caption keyword emphasis -> numbers, money, and punchy "power words"
      auto-highlight in captions even outside full word-by-word styles.
    • quality/caption_style inputs are now validated with a safe fallback
      instead of raising a raw KeyError if the frontend sends a typo'd key.
    • self-serves its own frontend (editor.html) at GET /api/editor/editor,
      the same "own file, same process/port" pattern as downloader.py.
 
DEPENDENCIES (install what you want to use — everything degrades gracefully
if a library is missing, see FEATURE FLAGS AT IMPORT TIME below):
    pip install opencv-python            # required for any face tracking
    pip install mediapipe                # optional, gives the "advanced" tracker
                                          # (falls back to Haar cascade otherwise)
    pip install faster-whisper           # installed-but-unused; USE_WHISPER=False by design,
                                          # see the "transcript source" section below
    pip install numpy

NO MORE WHISPER (by request): transcripts now come ONLY from the source
platform's own captions, pulled via yt-dlp (see get_transcript_words /
fetch_ytdlp_transcript). A video with no such captions (a plain PC upload,
or a platform video with captions disabled) simply has no transcript for
that run — subtitles + the AI viral-clip finder are skipped for it, while
face-tracking and the loudness/silence-based hook-cut & jump-cut (neither
needs a transcript) keep working exactly as before. faster-whisper's code
is left in the file (unused) rather than deleted, in case you want it back.

AI VIRAL-CLIP FINDER: when a transcript IS available, it is sent to the
first working provider in AI_PROVIDERS (Gemini -> Groq -> OpenRouter ->
Cerebras, in that order, all with free tiers) to rank every strong
short-form clip candidate. See AI_PROVIDERS near the top of the file to
paste in your own API keys (or set the matching environment variable).
 
DESIGN NOTES — how each part of the request maps to code:
  • "advance level face tracking"   -> _FaceTracker (mediapipe BlazeFace if present,
                                        else OpenCV Haar cascade), EMA-smoothed
                                        centroid so the crop doesn't jitter frame
                                        to frame, multi-face -> largest/most-central
                                        face wins.
  • "perfect advance subtitles,
     trending caption styles"       -> CAPTION_STYLES presets + build_ass() which
                                        groups words into short on-screen chunks and
                                        emits karaoke-style \\k tags for word-by-word
                                        pop/highlight animation, burned in with
                                        ffmpeg's `ass` filter (not a soft-sub track,
                                        so it always displays correctly everywhere).
  • "single high quality video+audio
     in one file"                   -> every render always outputs one .mp4 with
                                        both streams muxed, CRF-based high quality
                                        encode (see QUALITY_PRESETS).
  • "export in all ratio perfectly" -> RATIO_PRESETS, rendered in a loop, one file
                                        per ratio, each independently croppped using
                                        the SAME face track (recomputed crop window
                                        per target aspect).
  • "rearrange video manually /
     hook cut"                      -> job["segments"]: an ordered list of
                                        {start,end} clip ranges. Auto hook-cut picks
                                        the loudest window as segment 1 automatically;
                                        manual mode lets the caller add/remove/reorder
                                        segments directly (segment/add, segment/remove).
  • "add and remove option for all" -> every feature is a boolean flag in job config
                                        (features{}) that can be flipped and re-rendered,
                                        plus explicit add/remove endpoints for the two
                                        list-based tracks (crop keyframes, segments).
"""
 
import os
import re
import json
import uuid
import time
import struct
import threading
import subprocess
import urllib.request
import urllib.error
import concurrent.futures
from pathlib import Path
 
from flask import Blueprint, request, jsonify, send_file
 
editor_bp = Blueprint("editor_bp", __name__)
 
# ══════════════════════════════ embedded frontend ══════════════════════════════
# The whole editor UI lives right here as one string — no separate .html file
# to ship or lose track of. Served as-is by GET /api/editor/editor. Its fetch
# calls use relative paths (same-origin), so it works immediately wherever
# this blueprint is registered.
EDITOR_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reframe — AI video editor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  /* ── Theme — mirrors the main app's palette exactly (same variable
     VALUES, just different variable NAMES for this page's own CSS). Default
     below is "dark-violet", the main app's default. The <html data-theme="…">
     overrides further down are read straight off the SAME localStorage key
     the main app writes to (see the sync script at the bottom of <body>),
     so switching themes on Studio/Downloader also re-themes this page —
     no manual matching needed ever again. ── */
  :root{
    --bg:#0b0d12; --panel:#13151c; --panel-2:#191c25; --border:#262a36;
    --text:#eef0f6; --text-dim:#8a90a4; --text-faint:#8a90a4;
    --amber:#6e5bff; --teal:#22d3c4; --red:#ff5a6e;
    --amber-dim: color-mix(in srgb, var(--amber) 22%, #000);
    --teal-dim: color-mix(in srgb, var(--teal) 22%, #000);
    --radius:10px;
    font-size:15px;
  }
  html[data-theme="light"]{
    --bg:#f4f5f9; --panel:#ffffff; --panel-2:#eef0f6; --border:#dde1ea;
    --text:#171923; --text-dim:#5b6472; --text-faint:#5b6472;
    --amber:#5b46e0; --teal:#0891b2; --red:#e0324a;
  }
  html[data-theme="midnight-blue"]{
    --bg:#050a16; --panel:#0c1526; --panel-2:#12213a; --border:#1f3358;
    --text:#e8f1ff; --text-dim:#7f93b8; --text-faint:#7f93b8;
    --amber:#3b82f6; --teal:#22d3ee; --red:#ff5a6e;
  }
  html[data-theme="forest"]{
    --bg:#0a130d; --panel:#0f1e14; --panel-2:#16301f; --border:#234a30;
    --text:#eafbea; --text-dim:#86b399; --text-faint:#86b399;
    --amber:#22c55e; --teal:#a3e635; --red:#ff5a6e;
  }
  html[data-theme="sunset"]{
    --bg:#1a0e12; --panel:#241419; --panel-2:#331d27; --border:#4a2635;
    --text:#fdf1ee; --text-dim:#cb9aa3; --text-faint:#cb9aa3;
    --amber:#fb7185; --teal:#f59e0b; --red:#ff5a6e;
  }
  html[data-theme="ocean"]{
    --bg:#061319; --panel:#0b1f27; --panel-2:#123039; --border:#1e4a56;
    --text:#e6fbff; --text-dim:#7fb8c4; --text-faint:#7fb8c4;
    --amber:#06b6d4; --teal:#6366f1; --red:#ff5a6e;
  }
  html[data-theme="rose"]{
    --bg:#170f14; --panel:#221720; --panel-2:#2e1e2b; --border:#4a2f45;
    --text:#fdeef5; --text-dim:#c79ab0; --text-faint:#c79ab0;
    --amber:#ec4899; --teal:#f472b6; --red:#ff5a6e;
  }
  html[data-theme="amber"]{
    --bg:#160f05; --panel:#221708; --panel-2:#33230d; --border:#4d3416;
    --text:#fdf4e3; --text-dim:#c9a877; --text-faint:#c9a877;
    --amber:#f59e0b; --teal:#eab308; --red:#ff5a6e;
  }
  html[data-theme="slate"]{
    --bg:#101215; --panel:#181b1f; --panel-2:#20242a; --border:#333941;
    --text:#eceef1; --text-dim:#9aa2ad; --text-faint:#9aa2ad;
    --amber:#64748b; --teal:#94a3b8; --red:#ff5a6e;
  }
  html[data-theme="cyberpunk"]{
    --bg:#0a0014; --panel:#150124; --panel-2:#210235; --border:#4a0a6b;
    --text:#f5e6ff; --text-dim:#b083d6; --text-faint:#b083d6;
    --amber:#e91e9c; --teal:#00f0ff; --red:#ff5a6e;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; background:var(--bg); color:var(--text);
    font-family:'Inter',sans-serif; min-height:100vh;
  }
  h1,h2,h3,.display{font-family:'Space Grotesk',sans-serif;}
  .mono{font-family:'JetBrains Mono',monospace;}
  ::selection{background:var(--amber-dim); color:var(--amber);}
 
  /* ── top bar ── */
  .topbar{
    display:flex; align-items:center; gap:16px; padding:14px 22px;
    border-bottom:1px solid var(--border); background:var(--panel);
    position:sticky; top:0; z-index:20;
  }
  .brand{display:flex; align-items:center; gap:10px;}
  .brand .dot{width:10px; height:10px; border-radius:50%; background:var(--amber); box-shadow:0 0 10px var(--amber);}
  .brand h1{font-size:18px; font-weight:700; margin:0; letter-spacing:.2px;}
  .brand span{color:var(--text-dim); font-size:12px;}
  .topbar .spacer{flex:1;}
  .api-input{
    background:var(--panel-2); border:1px solid var(--border); color:var(--text-dim);
    border-radius:8px; padding:7px 10px; font-size:12px; font-family:'JetBrains Mono',monospace;
    width:220px;
  }
  .api-input:focus{outline:none; border-color:var(--amber); color:var(--text);}
 
  /* ── layout ── */
  .app{display:grid; grid-template-columns:1fr 380px; gap:1px; background:var(--border);}
  @media (max-width:980px){ .app{grid-template-columns:1fr;} }
  .col{background:var(--bg); padding:22px; min-width:0;}
 
  /* ── upload zone ── */
  .dropzone{
    border:1.5px dashed var(--border); border-radius:var(--radius);
    padding:40px 20px; text-align:center; cursor:pointer;
    transition:border-color .15s, background .15s;
  }
  .dropzone:hover, .dropzone.drag{border-color:var(--amber); background:rgba(255,176,32,.04);}
  .dropzone svg{opacity:.5; margin-bottom:10px;}
  .dropzone .hint{color:var(--text-dim); font-size:13px; margin-top:6px;}
 
  /* ── fetch-from-link (ytdlp-style resolver, built into the editor) ── */
  .fetch-section{border:1px solid var(--border); border-radius:var(--radius); padding:12px; background:var(--panel-2); margin-bottom:14px;}
  .fetch-row{display:flex; gap:8px; margin-top:8px;}
  .fetch-input{
    flex:1; background:var(--panel); border:1px solid var(--border); color:var(--text);
    border-radius:8px; padding:9px 10px; font-size:13px;
  }
  .fetch-input:focus{outline:none; border-color:var(--amber);}
  .btn-accent{background:var(--amber); color:#1a1200; border:none; font-weight:600;}
  .btn-accent:hover{filter:brightness(1.08);}
  .fetch-card{
    display:flex; gap:10px; align-items:center; margin-top:10px;
    border:1px solid var(--border); border-radius:8px; padding:10px; background:var(--panel);
  }
  .fetch-thumb{width:88px; height:56px; object-fit:cover; border-radius:6px; background:#000; flex-shrink:0;}
  .fetch-meta{flex:1; min-width:0;}
  .fetch-title{font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
  .fetch-sub{font-size:11.5px; color:var(--text-dim); margin-top:2px;}
  .fetch-status{font-size:12px; color:var(--text-dim); margin-top:8px; min-height:16px;}
 
  /* ── server library (downloader output + past uploads) ── */
  .or-divider{
    display:flex; align-items:center; gap:10px; margin:14px 0;
    color:var(--text-faint); font-size:12px; text-transform:uppercase; letter-spacing:.5px;
  }
  .or-divider::before, .or-divider::after{content:""; flex:1; height:1px; background:var(--border);}
  .lib-section{border:1px solid var(--border); border-radius:var(--radius); padding:12px; background:var(--panel-2);}
  .lib-header{display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; font-size:13px; color:var(--text-dim);}
  .lib-grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:8px; max-height:220px; overflow-y:auto;}
  .lib-empty{color:var(--text-faint); font-size:12.5px; grid-column:1/-1;}
  .lib-item{
    border:1px solid var(--border); border-radius:8px; padding:10px; cursor:pointer;
    background:var(--panel); transition:border-color .15s, transform .1s;
  }
  .lib-item:hover{border-color:var(--amber); transform:translateY(-1px);}
  .lib-item.active{border-color:var(--teal); background:var(--teal-dim);}
  .lib-item .lib-name{font-size:12.5px; font-weight:600; word-break:break-word; margin-bottom:4px;}
  .lib-item .lib-meta{font-size:11px; color:var(--text-dim); display:flex; justify-content:space-between;}
  .lib-item .lib-tag{
    display:inline-block; font-size:10px; padding:1px 6px; border-radius:20px;
    background:var(--amber-dim); color:var(--amber); margin-bottom:4px;
  }
 
  /* ── preview ── */
  .preview-wrap{position:relative; display:flex; justify-content:center; margin-bottom:16px;}
  .preview-frame{position:relative; background:#000; border-radius:8px; overflow:hidden; max-width:100%;}
  video{display:block; max-width:100%; max-height:60vh;}
  .crop-box{
    position:absolute; border:1.5px solid var(--amber); pointer-events:none;
    box-shadow:0 0 0 2000px rgba(0,0,0,.45);
  }
  .crop-box .corner{
    position:absolute; width:16px; height:16px; border-color:var(--amber); border-style:solid; border-width:0;
  }
  .crop-box .tl{top:-1.5px; left:-1.5px; border-top-width:3px; border-left-width:3px;}
  .crop-box .tr{top:-1.5px; right:-1.5px; border-top-width:3px; border-right-width:3px;}
  .crop-box .bl{bottom:-1.5px; left:-1.5px; border-bottom-width:3px; border-left-width:3px;}
  .crop-box .br{bottom:-1.5px; right:-1.5px; border-bottom-width:3px; border-right-width:3px;}
  .crop-box{cursor:grab;}
  .crop-box.dragging{cursor:grabbing;}
  .crop-readout{
    position:absolute; top:8px; left:8px; background:rgba(0,0,0,.6);
    color:var(--amber); font-size:11px; padding:3px 7px; border-radius:5px;
  }
 
  .preview-controls{display:flex; align-items:center; gap:10px; justify-content:center; margin-bottom:8px;}
  .btn-icon{
    background:var(--panel-2); border:1px solid var(--border); color:var(--text);
    width:34px; height:34px; border-radius:50%; cursor:pointer; display:flex;
    align-items:center; justify-content:center;
  }
  .btn-icon:hover{border-color:var(--amber);}
  .time{font-size:12px; color:var(--text-dim); min-width:96px; text-align:center;}
 
  /* ── timeline ── */
  .timeline-panel{background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:16px;}
  .timeline-panel h3{margin:0 0 4px; font-size:13px; letter-spacing:.4px; text-transform:uppercase; color:var(--text-dim);}
  .timeline{
    position:relative; height:56px; background:var(--panel-2); border-radius:6px;
    margin-top:10px; overflow:hidden; cursor:pointer;
  }
  .tl-face{position:absolute; top:0; height:18px; background:linear-gradient(90deg, transparent, var(--teal-dim)); opacity:.6;}
  .tl-seg{
    position:absolute; top:20px; height:18px; background:var(--teal); opacity:.85; border-radius:3px;
    display:flex; align-items:center; justify-content:center; font-size:10px; color:#04231D; font-weight:600;
    cursor:grab;
  }
  .tl-hook{position:absolute; top:38px; height:14px; background:var(--amber); opacity:.9; border-radius:3px;}
  .tl-playhead{position:absolute; top:0; bottom:0; width:2px; background:var(--red);}
  .tl-legend{display:flex; gap:16px; margin-top:8px; font-size:11px; color:var(--text-dim);}
  .tl-legend span{display:inline-flex; align-items:center; gap:5px;}
  .swatch{width:9px; height:9px; border-radius:2px; display:inline-block;}
 
  .seg-list{margin-top:10px; display:flex; flex-direction:column; gap:6px;}
  .seg-row{
    display:flex; align-items:center; gap:8px; background:var(--panel-2);
    border:1px solid var(--border); border-radius:7px; padding:6px 10px; font-size:12px;
  }
  .seg-row .mono{color:var(--teal);}
  .seg-row .spacer{flex:1;}
  .seg-row button{background:none; border:none; color:var(--text-faint); cursor:pointer; font-size:14px;}
  .seg-row button:hover{color:var(--red);}
  .btn-small{
    background:var(--panel-2); border:1px solid var(--border); color:var(--text-dim);
    border-radius:6px; padding:6px 10px; font-size:12px; cursor:pointer;
  }
  .btn-small:hover{border-color:var(--amber); color:var(--amber);}
 
  /* ── side panel ── */
  .section{margin-bottom:26px;}
  .section h3{
    font-size:12px; letter-spacing:.5px; text-transform:uppercase; color:var(--text-dim);
    margin:0 0 12px; display:flex; align-items:center; gap:8px;
  }
  .section h3 .num{
    width:18px; height:18px; border-radius:50%; background:var(--panel-2); border:1px solid var(--border);
    display:flex; align-items:center; justify-content:center; font-size:10px; color:var(--amber);
  }
 
  .toggle-row{
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 12px; background:var(--panel); border:1px solid var(--border);
    border-radius:8px; margin-bottom:8px;
  }
  .toggle-row .label{font-size:13px;}
  .toggle-row .desc{font-size:11px; color:var(--text-faint); margin-top:2px;}
  .toggle-row.disabled{opacity:.4;}
  .switch{position:relative; width:38px; height:21px; flex-shrink:0;}
  .switch input{opacity:0; width:0; height:0;}
  .slider{
    position:absolute; inset:0; background:var(--panel-2); border:1px solid var(--border);
    border-radius:20px; cursor:pointer; transition:.15s;
  }
  .slider::before{
    content:''; position:absolute; width:15px; height:15px; left:2px; top:2px;
    background:var(--text-dim); border-radius:50%; transition:.15s;
  }
  .switch input:checked + .slider{background:var(--amber-dim); border-color:var(--amber);}
  .switch input:checked + .slider::before{transform:translateX(17px); background:var(--amber);}
  .switch input:disabled + .slider{cursor:not-allowed;}
 
  .ratio-grid{display:grid; grid-template-columns:1fr 1fr; gap:8px;}
  .ratio-card{
    border:1px solid var(--border); background:var(--panel); border-radius:8px;
    padding:10px; cursor:pointer; text-align:center; position:relative;
  }
  .ratio-card.active{border-color:var(--amber); background:rgba(255,176,32,.06);}
  .ratio-card .shape{margin:0 auto 6px; background:var(--panel-2); border:1px solid var(--border); border-radius:3px;}
  .ratio-card .name{font-size:12px; font-weight:600;}
  .ratio-card .lbl{font-size:10px; color:var(--text-faint); margin-top:2px;}
 
  .style-list{display:flex; flex-direction:column; gap:8px;}
  .style-card{
    display:flex; align-items:center; gap:12px; border:1px solid var(--border);
    background:var(--panel); border-radius:8px; padding:10px 12px; cursor:pointer;
  }
  .style-card.active{border-color:var(--teal);}
  .style-swatch{
    flex-shrink:0; width:64px; height:36px; border-radius:6px; background:#000;
    display:flex; align-items:center; justify-content:center; font-size:9px; font-weight:800;
    letter-spacing:.5px;
  }
  .style-card .name{font-size:12.5px; font-weight:600;}
  .style-card .lbl{font-size:10.5px; color:var(--text-faint);}
 
  select, .fill-toggle{
    width:100%; background:var(--panel); border:1px solid var(--border); color:var(--text);
    padding:9px 10px; border-radius:8px; font-size:13px;
  }
  .fill-toggle{display:flex; gap:6px;}
  .fill-opt{flex:1; text-align:center; padding:8px; border-radius:6px; cursor:pointer; font-size:12px; border:1px solid var(--border);}
  .fill-opt.active{border-color:var(--amber); color:var(--amber); background:rgba(255,176,32,.06);}
 
  .render-btn{
    width:100%; background:var(--amber); color:#1A1204; border:none; border-radius:10px;
    padding:14px; font-size:14px; font-weight:700; cursor:pointer; font-family:'Space Grotesk',sans-serif;
    letter-spacing:.3px;
  }
  .render-btn:disabled{background:var(--panel-2); color:var(--text-faint); cursor:not-allowed;}
  .render-btn:not(:disabled):hover{filter:brightness(1.08);}
 
  .progress-wrap{margin-top:14px;}
  .progress-track{height:6px; background:var(--panel-2); border-radius:4px; overflow:hidden;}
  .progress-fill{height:100%; background:linear-gradient(90deg, var(--teal), var(--amber)); width:0%; transition:width .3s;}
  .progress-label{display:flex; justify-content:space-between; font-size:11px; color:var(--text-dim); margin-top:6px;}
  .log-console{margin-top:10px; max-height:150px; overflow-y:auto; background:var(--bg);
    border:1px solid var(--border); border-radius:8px; padding:8px 10px; font-family:'Consolas',monospace;
    font-size:11px; line-height:1.6; color:var(--text-dim);}
  .log-console .log-line{white-space:pre-wrap; word-break:break-word;}
  .log-console .log-line:last-child{color:var(--text);}
  .ai-clip-row{background:var(--panel-2); border:1px solid var(--border); border-radius:10px;
    padding:10px 12px; margin-bottom:8px;}
  .ai-clip-row .ai-clip-top{display:flex; justify-content:space-between; align-items:center; gap:8px;}
  .ai-clip-row .ai-clip-score{font-weight:800; color:var(--amber); font-size:13px;}
  .ai-clip-row .ai-clip-hook{font-weight:700; margin:4px 0;}
  .ai-clip-row .ai-clip-meta{font-size:11px; color:var(--text-dim);}
  .ai-clip-row .ai-clip-reason{font-size:12px; color:var(--text-dim); margin-top:4px;}
 
  .results{display:flex; flex-direction:column; gap:8px; margin-top:14px;}
  .result-row{
    display:flex; align-items:center; gap:10px; background:var(--panel); border:1px solid var(--border);
    border-radius:8px; padding:10px 12px;
  }
  .result-row .name{font-size:12.5px; font-weight:600; flex:1;}
  .result-row a{
    background:var(--panel-2); border:1px solid var(--border); color:var(--teal);
    text-decoration:none; font-size:11.5px; padding:6px 10px; border-radius:6px;
  }
  .result-row a:hover{border-color:var(--teal);}
 
  .status-line{font-size:11.5px; color:var(--text-faint); margin-top:10px; line-height:1.5;}
  .status-line b{color:var(--text-dim);}
  .feature-warn{
    font-size:11px; color:var(--amber); background:rgba(255,176,32,.08); border:1px solid var(--amber-dim);
    border-radius:6px; padding:8px 10px; margin-top:8px; display:none;
  }
</style>
</head>
<body>
<script>
// ── Theme sync — this page is served from the SAME Flask app/port as the
// main Studio (registered as a blueprint), and when opened as an <iframe> on
// that page it's also same-origin, so it shares localStorage automatically.
// Reading 'shortsStudioTheme' here — the exact key the main page's theme
// picker writes to — means this editor always matches whatever theme is
// active there, with zero manual setup. Runs immediately (before first
// paint) so there's no flash of the wrong palette, and again whenever the
// theme changes elsewhere (the 'storage' event fires in this frame whenever
// another same-origin window/frame updates localStorage).
(function(){
  function applyTheme(){
    const t = localStorage.getItem('shortsStudioTheme') || 'dark-violet';
    if(t === 'dark-violet') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', t);
  }
  applyTheme();
  window.addEventListener('storage', (e)=>{ if(e.key === 'shortsStudioTheme') applyTheme(); });
})();
</script>
 
<div class="topbar">
  <div class="brand"><span class="dot"></span><h1>Reframe</h1><span>face-tracked auto edit</span></div>
  <div class="spacer"></div>
  <span id="apiBaseHint" class="mono" style="font-size:11px; color:var(--text-faint); margin-right:6px;">same server</span>
  <button type="button" id="apiBaseToggle" class="btn-small" title="Only needed if this page is served from a different host than the API">⚙</button>
  <input id="apiBase" class="api-input mono" placeholder="leave blank — uses this same page's server" value="" style="display:none;">
</div>
 
<div class="app">
  <!-- LEFT: preview + timeline -->
  <div class="col">
    <div class="fetch-section">
      <div class="lib-header">
        <span>🔗 Fetch from a link (YouTube / Instagram / TikTok / X / Facebook…)</span>
      </div>
      <div class="fetch-row">
        <input type="text" id="fetchUrl" class="fetch-input mono" placeholder="Paste a video URL…">
        <button type="button" class="btn-small btn-accent" id="fetchInfoBtn">Fetch</button>
      </div>
      <div id="fetchCard" class="fetch-card" style="display:none;">
        <img id="fetchThumb" class="fetch-thumb" alt="">
        <div class="fetch-meta">
          <div id="fetchTitle" class="fetch-title"></div>
          <div id="fetchSub" class="fetch-sub"></div>
        </div>
        <button type="button" class="btn-small btn-accent" id="fetchUseBtn">⬇ Fetch high-quality &amp; use</button>
      </div>
      <div id="fetchStatus" class="fetch-status"></div>
    </div>
 
    <div class="or-divider">or drop / choose a video from this PC</div>
    <div id="dropzone" class="dropzone">
      <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8A93A3" stroke-width="1.5"><path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v3a2 2 0 002 2h12a2 2 0 002-2v-3"/></svg>
      <div>Drop a video, or click to choose one</div>
      <div class="hint">mp4 / mov · analyzed locally before render</div>
      <input type="file" id="fileInput" accept="video/*" style="display:none">
    </div>
 
    <div class="or-divider">or pick a video already fetched by the Downloader</div>
    <div class="lib-section">
      <div class="lib-header">
        <span>📥 From Downloader / server library</span>
        <button type="button" class="btn-small" id="refreshLibBtn">Refresh</button>
      </div>
      <div class="lib-grid" id="libGrid"><span class="lib-empty">Click Refresh to list videos already downloaded via the Downloader tab.</span></div>
    </div>
 
    <div id="previewSection" style="display:none;">
      <div class="preview-wrap">
        <div class="preview-frame" id="previewFrame">
          <video id="video"></video>
          <div class="crop-box" id="cropBox" style="display:none;">
            <div class="corner tl"></div><div class="corner tr"></div><div class="corner bl"></div><div class="corner br"></div>
            <div class="crop-readout" id="cropReadout">9:16</div>
          </div>
        </div>
      </div>
      <div class="preview-controls">
        <div class="btn-icon" id="playBtn">▶</div>
        <div class="btn-icon" id="muteBtn" title="Mute/unmute preview">🔊</div>
        <div class="time mono" id="timeLabel">0:00 / 0:00</div>
      </div>
 
      <div class="status-line" id="statusLine"></div>
      <div class="progress-wrap" id="analyzeProgressWrap" style="display:none;">
        <div class="progress-track"><div class="progress-fill" id="analyzeProgressFill"></div></div>
        <div class="progress-label"><span id="analyzeProgressStage">queued</span><span id="analyzeProgressPct">0%</span></div>
        <div class="log-console" id="analyzeLogConsole"></div>
      </div>
      <div id="sourceMetaPanel" class="lib-section" style="display:none; margin-top:10px;"></div>

      <div id="aiClipsPanel" class="lib-section" style="display:none; margin-top:10px;">
        <div class="lib-header"><span>✨ AI-suggested viral clips</span></div>
        <div class="hint" style="padding:8px 12px 0;">Ranked best → worst. Tap "Use this clip" to set it as your export segment — nothing else to configure.</div>
        <div id="aiClipsList" style="padding:10px 12px;"></div>
      </div>

      <div class="timeline-panel">
        <h3>Timeline</h3>
        <div class="hint" style="margin:-6px 0 10px;">
          This bar is the whole video. The buttons below it are all OPTIONAL manual overrides —
          if you don't touch anything here, Export still uses the whole video with auto face-tracking.
          Use these only if you want to hand-pick a moment:
        </div>
        <div class="timeline" id="timeline">
          <div class="tl-playhead" id="playhead" style="left:0%"></div>
        </div>
        <div class="tl-legend">
          <span><span class="swatch" style="background:var(--teal-dim)"></span>face track</span>
          <span><span class="swatch" style="background:var(--teal)"></span>segments</span>
          <span><span class="swatch" style="background:var(--amber)"></span>suggested hook</span>
        </div>
        <div class="seg-list" id="segList"></div>
        <div style="display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;">
          <button class="btn-small" id="addSegBtn" title="Click play, pause where you want a clip to start/end, then click this">+ add segment at playhead (±3s)</button>
          <button class="btn-small" id="useHookBtn" style="display:none;" title="Auto-picked loudest/most energetic moment as a cold open">use suggested hook</button>
          <button class="btn-small" id="useSilenceBtn" style="display:none;" title="Auto-removes dead air / silence between lines">use auto jump-cut segments</button>
        </div>
        <div style="display:flex; gap:8px; margin-top:8px; flex-wrap:wrap;">
          <button class="btn-small" id="addKfBtn" title="Locks the face-crop box to exactly where it is right now, at this timestamp">+ pin crop here (manual keyframe)</button>
          <button class="btn-small" id="clearKfBtn">clear manual crop</button>
        </div>
      </div>
    </div>
  </div>

 
  <!-- RIGHT: controls -->
  <div class="col">
    <div class="section">
      <h3><span class="num">1</span>Auto-reframe</h3>
      <div class="toggle-row" id="rowFace">
        <div><div class="label">Face tracking</div><div class="desc">Advanced (mediapipe) if installed, else standard cascade</div></div>
        <label class="switch"><input type="checkbox" id="chkFace" checked><span class="slider"></span></label>
      </div>
      <div class="fill-toggle" style="margin-top:8px;">
        <div class="fill-opt active" data-fill="crop">Crop to face</div>
        <div class="fill-opt" data-fill="blur_pad">Fit + blurred bars</div>
      </div>
    </div>
 
    <div class="section">
      <h3><span class="num">2</span>Captions</h3>
      <div class="toggle-row" id="rowSubs">
        <div><div class="label">Auto subtitles</div><div class="desc">Word-level, keyword emphasis auto-highlighted</div></div>
        <label class="switch"><input type="checkbox" id="chkSubs" checked><span class="slider"></span></label>
      </div>
      <div class="style-list" id="styleList"></div>
    </div>
 
    <div class="section">
      <h3><span class="num">3</span>Hook &amp; pacing</h3>
      <div class="toggle-row" id="rowHook">
        <div><div class="label">Hook-cut</div><div class="desc">Open on the loudest / most energetic moment</div></div>
        <label class="switch"><input type="checkbox" id="chkHook"><span class="slider"></span></label>
      </div>
      <div class="toggle-row" id="rowJump">
        <div><div class="label">Jump-cut silences</div><div class="desc">Auto-remove dead air between lines</div></div>
        <label class="switch"><input type="checkbox" id="chkJump"><span class="slider"></span></label>
      </div>
      <div class="toggle-row">
        <div><div class="label">Loudness normalize</div><div class="desc">-14 LUFS, matches platform playback volume</div></div>
        <label class="switch"><input type="checkbox" id="chkNorm" checked><span class="slider"></span></label>
      </div>
    </div>
 
    <div class="section">
      <h3><span class="num">4</span>Export ratios</h3>
      <div class="ratio-grid" id="ratioGrid"></div>
    </div>
 
    <div class="section">
      <h3><span class="num">5</span>Quality</h3>
      <select id="qualitySel">
        <option value="high">High (crf 18, slow) — best quality</option>
        <option value="medium">Medium (crf 21) — balanced</option>
        <option value="fast">Fast (crf 23) — quick preview</option>
      </select>
    </div>
 
    <button class="render-btn" id="renderBtn" disabled>Analyze a video first</button>
    <div class="feature-warn" id="featureWarn"></div>
    <div class="progress-wrap" id="progressWrap" style="display:none;">
      <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
      <div class="progress-label"><span id="progressStage">queued</span><span id="progressPct">0%</span></div>
      <div class="log-console" id="renderLogConsole"></div>
    </div>
    <div class="results" id="results"></div>
  </div>
</div>
 
<script>
const $ = (id) => document.getElementById(id);
const state = {
  sourcePath: null, sourceMeta: null, sourceOrigin: 'pc', duration: 0, srcW: 0, srcH: 0, srcFps: null,
  faceTrack: [], words: [], suggestedHook: null, suggestedSilences: null, aiClips: [],
  segments: [], manualKeyframes: [],
  // Multiple export ratios can be selected at once now (a Set of ratio
  // keys, e.g. {'9:16','1:1'}), not just one — "auto" is pre-selected by
  // default since it's the safest zero-quality-loss choice.
  selectedRatios: new Set(['auto']),
  previewRatio: '9:16', // drives the live crop box only, independent of export selection
  selectedStyle: 'bold_pop', fillMode: 'crop',
  jobId: null, dragging: false,
};
 
// Same-origin by default: this page is served BY the same Flask app/port
// that Reframe, the Downloader, and everything else run on (they're all
// registered on one Flask app as blueprints, per RenderDetect's main()),
// so relative paths already hit the right server. The "⚙" override is
// only for the rare case this HTML got copied to a different host — if
// left blank (the default), api() never leaves the current origin, which
// is what fixes the class of bug where a stray/incorrect API host caused
// every request (analyze, render, fetch…) to silently fail.
function apiBase(){ const v = $('apiBase').value.trim().replace(/\/$/, ''); return v; }
function api(path){ return apiBase() + path; }
$('apiBaseToggle').onclick = ()=>{
  const el = $('apiBase');
  const show = el.style.display === 'none';
  el.style.display = show ? 'inline-block' : 'none';
  $('apiBaseHint').style.display = show ? 'none' : 'inline';
};
 
const RATIO_META = {
  '9:16': {w:9,h:16}, '16:9': {w:16,h:9}, '1:1': {w:1,h:1}, '4:5': {w:4,h:5}, '4:3': {w:4,h:3},
};
const STYLE_META = {
  bold_pop:       {bg:'#111', color:'#fff', hi:'#FFD700', sample:'THIS IS'},
  karaoke_classic:{bg:'#111', color:'#fff', hi:'#FFA500', sample:'FILLS IN'},
  neon_glow:      {bg:'#111', color:'#fff', hi:'#00F0FF', sample:'GLOWS UP'},
  minimal_clean:  {bg:'#111', color:'#fff', hi:'#fff', sample:'stays quiet'},
  creator_bold:   {bg:'#111', color:'#FFFF00', hi:'#FFFF00', sample:'HUGE TEXT'},
};
 
// ── boot: load styles / ratios / features from backend ──
async function loadMeta(){
  try{
    const [styles, ratios, features] = await Promise.all([
      fetch(api('/api/editor/styles')).then(r=>r.json()),
      fetch(api('/api/editor/ratios')).then(r=>r.json()),
      fetch(api('/api/editor/features')).then(r=>r.json()),
    ]);
    renderStyles(styles);
    renderRatios(ratios);
    applyFeatureAvailability(features);
  }catch(e){
    $('featureWarn').style.display='block';
    $('featureWarn').textContent = 'Could not reach the backend at ' + api('') + ' — set the API base URL above and reload.';
  }
}
 
function renderStyles(styles){
  const list = $('styleList'); list.innerHTML='';
  Object.entries(styles).forEach(([key, meta])=>{
    const m = STYLE_META[key] || {bg:'#111', color:'#fff', hi:'#fff', sample:'Sample'};
    const card = document.createElement('div');
    card.className = 'style-card' + (key===state.selectedStyle ? ' active':'');
    card.innerHTML = `<div class="style-swatch" style="background:${m.bg}">
        <span style="color:${m.color}">${m.sample.split(' ')[0]}</span>&nbsp;<span style="color:${m.hi}">${m.sample.split(' ')[1]||''}</span>
      </div>
      <div><div class="name">${meta.label.split(' (')[0]}</div><div class="lbl">${(meta.label.match(/\((.*)\)/)||[,''])[1]}</div></div>`;
    card.onclick = ()=>{ state.selectedStyle = key; renderStyles(styles); };
    list.appendChild(card);
  });
}
 
// Export ratios are now MULTI-select — tap any number of cards and every
// one you pick gets rendered and offered as a download, all in the same
// render job. "Auto" (native size, no crop, zero quality loss) is its own
// card too. state.previewRatio (separate from the export selection) just
// drives which crop box is drawn live over the video for editing.
function renderRatios(ratios){
  const grid = $('ratioGrid'); grid.innerHTML='';
  Object.entries(ratios).forEach(([key, r])=>{
    const isAuto = key === 'auto';
    const srcAspect = (state.srcW && state.srcH) ? {w: state.srcW, h: state.srcH} : {w:9, h:16};
    const meta = isAuto ? srcAspect : (RATIO_META[key] || {w:1,h:1});
    const scale = 30 / Math.max(meta.w, meta.h);
    const selected = state.selectedRatios.has(key);
    const card = document.createElement('div');
    card.className = 'ratio-card' + (selected ? ' active':'');
    card.innerHTML = `<div class="shape" style="width:${meta.w*scale}px; height:${meta.h*scale}px;"></div>
      <div class="name">${selected ? '✓ ' : ''}${key === 'auto' ? 'Auto' : key}</div><div class="lbl">${r.label}</div>`;
    card.onclick = ()=>{
      if(state.selectedRatios.has(key)) state.selectedRatios.delete(key);
      else state.selectedRatios.add(key);
      if(!state.selectedRatios.size) state.selectedRatios.add(key); // keep at least one selected
      if(!isAuto){ state.previewRatio = key; drawCropForCurrentTime(); }
      renderRatios(ratios);
    };
    grid.appendChild(card);
  });
}
 
function applyFeatureAvailability(f){
  toggleRow('rowFace', 'chkFace', f.face_tracking, 'opencv-python not installed on the server');
  toggleRow('rowSubs', 'chkSubs', f.subtitles, 'faster-whisper not installed on the server');
  toggleRow('rowHook', 'chkHook', true, '');
  toggleRow('rowJump', 'chkJump', true, '');
  if (!f.face_tracking_advanced && f.face_tracking){
    $('rowFace').querySelector('.desc').textContent = 'Standard tracker active (install mediapipe for advanced mode)';
  }
}
function toggleRow(rowId, chkId, available, msg){
  if(!available){
    $(rowId).classList.add('disabled');
    $(chkId).checked = false;
    $(chkId).disabled = true;
    $(rowId).querySelector('.desc').textContent = msg;
  }
}
 
// ── fill mode ──
document.querySelectorAll('.fill-opt').forEach(el=>{
  el.onclick = ()=>{
    document.querySelectorAll('.fill-opt').forEach(o=>o.classList.remove('active'));
    el.classList.add('active');
    state.fillMode = el.dataset.fill;
    drawCropForCurrentTime();
  };
});
 
// ── fetch-from-link (built-in yt-dlp resolver, high quality single mp4) ──
// Mirrors the Downloader tab's speed trick (cached auth mode) but skips the
// full format-picker UI on purpose: this always grabs the single best
// available video+audio quality and merges it into one mp4, then feeds it
// straight into the exact same analyze/render pipeline as an upload.
let fetchState = { url: null };
 
$('fetchInfoBtn').onclick = async ()=>{
  const url = $('fetchUrl').value.trim();
  if(!url) return;
  $('fetchStatus').textContent = 'Resolving link…';
  $('fetchCard').style.display = 'none';
  try{
    const res = await fetch(api('/api/editor/fetch/info'), {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url})
    });
    const data = await res.json();
    if(data.error){ $('fetchStatus').textContent = data.error; return; }
    fetchState.url = url;
    fetchState.meta = data;   // keep the FULL metadata (views/likes/uploader/etc.) for later
    // Show it immediately — title/uploader/captions/chapters are all known
    // from this single resolve call, before any download has happened.
    state.sourceMeta = data;
    renderSourceMeta();
    $('fetchThumb').src = data.thumbnail || '';
    $('fetchTitle').textContent = data.title || 'Untitled video';
    $('fetchSub').textContent = [data.uploader, data.duration_str, data.resolution].filter(Boolean).join(' · ');
    $('fetchCard').style.display = 'flex';
    $('fetchStatus').textContent = '';
  }catch(e){
    $('fetchStatus').textContent = 'Could not reach the server to resolve that link. Make sure the app is running and reload this page.';
  }
};
 
$('fetchUseBtn').onclick = async ()=>{
  $('fetchUseBtn').disabled = true;
  $('fetchStatus').textContent = 'Fetching best quality (video + audio, single file)…';
  try{
    const res = await fetch(api('/api/editor/fetch/start'), {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url: fetchState.url})
    });
    const data = await res.json();
    if(data.error){ $('fetchStatus').textContent = data.error; $('fetchUseBtn').disabled = false; return; }
    pollFetchProgress(data.fetch_id);
  }catch(e){
    $('fetchStatus').textContent = 'Fetch failed to start — check the connection and try again.';
    $('fetchUseBtn').disabled = false;
  }
};
 
async function pollFetchProgress(fetchId){
  let st;
  try{
    const res = await fetch(api('/api/editor/fetch/progress/'+fetchId));
    st = await res.json();
  }catch(e){
    setTimeout(()=>pollFetchProgress(fetchId), 1200); // transient network hiccup, keep polling
    return;
  }
  if(st.status === 'error'){
    $('fetchStatus').textContent = 'Error: ' + st.error;
    $('fetchUseBtn').disabled = false;
    return;
  }
  if(st.status === 'done'){
    $('fetchStatus').textContent = 'Downloaded — analyzing…';
    $('fetchUseBtn').disabled = false;
    $('fetchCard').style.display = 'none';
    $('fetchUrl').value = '';
    loadSource(st.source_path, fetchState.meta, 'fetch');
    return;
  }
  const pct = st.percent != null ? st.percent + '%' : '';
  $('fetchStatus').textContent = `${st.stage || 'downloading'}… ${pct} ${st.speed_str || ''}`.trim();
  setTimeout(()=>pollFetchProgress(fetchId), 900);
}
 
// ── server library (downloader output + earlier uploads) ──
// Lets the user reuse a video already fetched via the Downloader tab (or an
// earlier upload) without picking it from disk again — same analyze/render
// pipeline as a fresh upload, just a different source_path origin.
$('refreshLibBtn').onclick = loadLibrary;
 
async function loadLibrary(){
  const grid = $('libGrid');
  grid.innerHTML = '<span class="lib-empty">Loading…</span>';
  try{
    const res = await fetch(api('/api/editor/sources'));
    const data = await res.json();
    const items = data.sources || [];
    if(!items.length){
      grid.innerHTML = '<span class="lib-empty">Nothing yet — fetch a video in the Downloader tab, or upload one above.</span>';
      return;
    }
    grid.innerHTML = '';
    items.forEach(it=>{
      const el = document.createElement('div');
      el.className = 'lib-item';
      el.innerHTML = `<span class="lib-tag">${it.origin === 'downloader' ? '📥 downloaded' : '📤 uploaded'}</span>
        <div class="lib-name">${it.name}</div>
        <div class="lib-meta"><span>${it.size_str||''}</span><span>${it.modified_str||''}</span></div>`;
      el.onclick = ()=>{
        document.querySelectorAll('.lib-item.active').forEach(n=>n.classList.remove('active'));
        el.classList.add('active');
        loadSource(it.path, null, it.origin === 'downloader' ? 'downloader' : 'pc');
      };
      grid.appendChild(el);
    });
  }catch(e){
    grid.innerHTML = '<span class="lib-empty">Could not reach the server — make sure the app is running, then hit Refresh again.</span>';
  }
}
 
loadLibrary();
 
// ── upload flow ──
$('dropzone').onclick = ()=> $('fileInput').click();
$('fileInput').onchange = (e)=> handleFile(e.target.files[0]);
['dragover','dragleave','drop'].forEach(evt=>{
  $('dropzone').addEventListener(evt, (e)=>{
    e.preventDefault();
    $('dropzone').classList.toggle('drag', evt==='dragover');
    if(evt==='drop' && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
});
 
async function handleFile(file){
  if(!file) return;
  $('video').src = URL.createObjectURL(file);
  $('previewSection').style.display = 'block';
  $('statusLine').textContent = 'Uploading…';
  try{
    const fd = new FormData(); fd.append('file', file);
    const res = await fetch(api('/api/editor/upload'), {method:'POST', body: fd});
    const data = await res.json();
    if(data.error){ showAnalysisError('Upload error: ' + data.error); return; }
    loadSource(data.source_path, null, 'pc');
  }catch(e){
    showAnalysisError('Could not reach the server to upload this file.');
  }
}
 
// ── unified source loader — every origin (fetched link, downloader
// library, or a file picked from this PC) funnels through here so
// analysis always runs the same way and the Export panel always ends up
// enabled the same way, no matter where the video came from. ──
async function loadSource(sourcePath, meta, origin){
  state.sourcePath = sourcePath;
  state.sourceMeta = meta || null;
  state.sourceOrigin = origin || 'pc';
  $('video').src = api('/api/editor/preview?path=' + encodeURIComponent(sourcePath));
  $('previewSection').style.display = 'block';
  renderSourceMeta();
  await runAnalysis();
}
 
// Shows the rich details panel. For links fetched from YouTube/Instagram/
// etc. this uses the metadata the SERVER already pulled during Fetch (no
// need to re-derive title/uploader/views by hand); for PC uploads or
// library items with no such metadata it just relies on the ffprobe-based
// technical details that come back from /api/editor/analyze.
function renderSourceMeta(){
  const el = $('sourceMetaPanel');
  if(!el) return;
  const m = state.sourceMeta || {};
  const captionLine = m.has_captions
    ? `Yes (${m.caption_kind || 'found'}${m.words_count ? `, ${m.words_count} words` : ''})`
    : (m.has_captions === false ? 'None found on the source platform' : null);
  const rows = [
    ['Title', m.title],
    ['Channel / uploader', m.uploader],
    ['Views', m.view_count_str],
    ['Likes', m.like_count_str],
    ['Uploaded', m.upload_date_str],
    // Known the INSTANT the link is resolved — before any video download —
    // which is also the direct answer to "does analyze need a download":
    // captions/chapters come from the platform's metadata, not the file.
    ['Captions on source', captionLine],
    ['Chapters on source', m.chapters_count ? `${m.chapters_count} chapter marker(s)` : null],
    // Technical rows come from ffmpeg probing (_probe on the server) and are
    // filled in for EVERY source — PC upload, Downloader-library pick, or a
    // freshly fetched link — not just ones that happen to have yt-dlp
    // metadata. This is what makes "same full details" true across origins.
    ['Resolution', state.srcW && state.srcH ? `${state.srcW}×${state.srcH}` : null],
    ['Duration', state.duration ? fmtTime(state.duration) : null],
    ['Frame rate', state.srcFps ? `${state.srcFps} fps` : null],
  ].filter(([,v]) => v);
  if(!rows.length){ el.style.display = 'none'; el.innerHTML = ''; return; }
  const heading = m.title ? 'ℹ️ Fetched video details' : 'ℹ️ Video details';
  el.style.display = 'block';
  el.innerHTML = `<div class="lib-header"><span>${heading}</span></div>` +
    rows.map(([k,v]) => `<div class="status-line"><b>${k}:</b> ${v}</div>`).join('');
}
 
function showAnalysisError(msg){
  $('statusLine').innerHTML = `<span style="color:var(--amber);">${msg}</span> ` +
    `<button type="button" class="btn-small" id="retryAnalysisBtn" style="margin-left:6px;">Retry analysis</button>`;
  const btn = $('retryAnalysisBtn');
  if(btn) btn.onclick = () => runAnalysis();
  $('renderBtn').disabled = true;
  $('renderBtn').textContent = 'Analyze a video first';
}
 
async function runAnalysis(){
  if(!state.sourcePath){ showAnalysisError('No video selected yet.'); return; }
  $('renderBtn').disabled = true;
  $('statusLine').textContent = 'Starting analysis…';
  $('analyzeProgressWrap').style.display = 'block';
  $('analyzeProgressFill').style.width = '0%';
  $('analyzeProgressPct').textContent = '0%';
  $('analyzeProgressStage').textContent = 'queued';
  $('analyzeLogConsole').innerHTML = '';
  $('aiClipsPanel').style.display = 'none';
  const body = {
    source_path: state.sourcePath,
    face_tracking: $('chkFace').checked && !$('chkFace').disabled,
    subtitles: $('chkSubs').checked && !$('chkSubs').disabled,
    hook_cut: true,
    jump_cut: true,
  };
  let jobId;
  try{
    const res = await fetch(api('/api/editor/analyze/start'), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    if(!res.ok){ showAnalysisError(`Analysis request failed (server said HTTP ${res.status}).`); return; }
    const started = await res.json();
    if(started.error){ showAnalysisError('Analysis error: ' + started.error); return; }
    jobId = started.job_id;
  }catch(e){
    showAnalysisError('Could not reach the server to start analysis — check it is still running, then retry.');
    return;
  }
  pollAnalyzeProgress(jobId);
}

function _renderLogLines(el, logs){
  if(!logs || !logs.length) return;
  el.innerHTML = logs.map(l => `<div class="log-line">${l.replace(/</g,'&lt;')}</div>`).join('');
  el.scrollTop = el.scrollHeight;
}

async function pollAnalyzeProgress(jobId){
  let job;
  try{
    const res = await fetch(api('/api/editor/analyze/progress/'+jobId));
    job = await res.json();
  }catch(e){
    setTimeout(()=>pollAnalyzeProgress(jobId), 1200); // transient hiccup, keep polling
    return;
  }
  $('analyzeProgressFill').style.width = (job.percent||0)+'%';
  $('analyzeProgressStage').textContent = job.stage || job.status;
  $('analyzeProgressPct').textContent = (job.percent||0)+'%';
  _renderLogLines($('analyzeLogConsole'), job.logs);

  if(job.status === 'error'){
    showAnalysisError('Analysis error: ' + job.error);
    return;
  }
  if(job.status !== 'done'){ setTimeout(()=>pollAnalyzeProgress(jobId), 900); return; }

  const data = job.result || {};
  state.duration = data.probe.duration || 0;
  state.srcW = (data.source_size && data.source_size.w) || data.probe.width;
  state.srcH = (data.source_size && data.source_size.h) || data.probe.height;
  state.srcFps = data.probe.fps ? Math.round(data.probe.fps * 100) / 100 : null;
  renderSourceMeta();
  state.faceTrack = data.face_track || [];
  state.faceTrackingMode = data.face_tracking_mode || null;
  state.words = data.words || [];
  state.suggestedHook = data.suggested_hook || null;
  state.suggestedSilences = data.suggested_segments || null;
  state.aiClips = data.ai_clips || [];
  $('useHookBtn').style.display = state.suggestedHook ? 'inline-block' : 'none';
  $('useSilenceBtn').style.display = state.suggestedSilences ? 'inline-block' : 'none';
  renderAiClips();
  drawTimeline();
  // Export becomes available the instant analysis finishes, identically
  // for an uploaded PC file, a Downloader-library pick, or a fresh link
  // fetch — there is no special-casing by source here. This is true even
  // if one sub-feature (e.g. subtitles) failed above — analyze always
  // finishes, so render is never blocked by one slow/failed piece.
  $('renderBtn').disabled = false;
  $('renderBtn').textContent = 'Render exports';
  const srcTag = state.sourceOrigin === 'fetch' ? 'fetched video' : (state.sourceOrigin === 'downloader' ? 'downloaded video' : 'uploaded video');
  const warnings = [data.face_tracking_error, data.hook_cut_error, data.jump_cut_error].filter(Boolean);
  let line = `Analyzed ${srcTag} — <b>${state.faceTrack.length}</b> face samples · <b>${state.words.length}</b> caption words · duration <b>${fmtTime(state.duration)}</b> · source <b>${state.srcW}×${state.srcH}</b>`;
  if(warnings.length) line += ` <span style="color:var(--amber);">(${warnings.join(' · ')})</span>`;
  $('statusLine').innerHTML = line;
}

// Shows the AI-ranked viral clip list (start/end/hook/score/reason) built
// from the source's real captions — nothing to configure, just tap a clip
// to make it the export segment.
function renderAiClips(){
  const panel = $('aiClipsPanel'), list = $('aiClipsList');
  if(!state.aiClips || !state.aiClips.length){ panel.style.display = 'none'; list.innerHTML=''; return; }
  panel.style.display = 'block';
  list.innerHTML = state.aiClips.map((c, i) => `
    <div class="ai-clip-row">
      <div class="ai-clip-top">
        <span class="ai-clip-score">Score ${c.score}</span>
        <span class="ai-clip-meta">${fmtTime(c.start_time)} → ${fmtTime(c.end_time)} · ${c.duration.toFixed(1)}s · ${c.category||''}</span>
      </div>
      <div class="ai-clip-hook">${(c.hook_text||'').replace(/</g,'&lt;')}</div>
      <div class="ai-clip-reason">${(c.reason||'').replace(/</g,'&lt;')} ${c.editing_suggestion ? '· ' + c.editing_suggestion.replace(/</g,'&lt;') : ''}</div>
      <button type="button" class="btn-small" style="margin-top:8px;" onclick="useAiClip(${i})">Use this clip</button>
    </div>
  `).join('');
}

function useAiClip(i){
  const c = state.aiClips[i];
  if(!c) return;
  state.segments = [{start: c.start_time, end: c.end_time}];
  drawTimeline();
  $('statusLine').innerHTML = `Using AI clip: <b>${fmtTime(c.start_time)}–${fmtTime(c.end_time)}</b> (score ${c.score}) as the export segment.`;
}
 
// ── video controls ──
const video = $('video');
$('playBtn').onclick = ()=>{ video.paused ? video.play() : video.pause(); };
video.onplay = ()=> $('playBtn').textContent = '❚❚';
video.onpause = ()=> $('playBtn').textContent = '▶';
$('muteBtn').onclick = ()=>{ video.muted = !video.muted; $('muteBtn').textContent = video.muted ? '🔇' : '🔊'; };
video.ontimeupdate = ()=>{
  $('timeLabel').textContent = `${fmtTime(video.currentTime)} / ${fmtTime(video.duration||0)}`;
  const pct = video.duration ? (video.currentTime/video.duration*100) : 0;
  $('playhead').style.left = pct + '%';
  drawCropForCurrentTime();
};
function fmtTime(t){ t=Math.max(0,t||0); const m=Math.floor(t/60), s=Math.floor(t%60); return `${m}:${String(s).padStart(2,'0')}`; }
 
// ── crop overlay: shows the current auto/manual crop box, draggable ──
function currentCrop(){
  const kfs = state.manualKeyframes.length ? state.manualKeyframes : state.faceTrack.map(p=>({t:p.t, x:p.cx, y:p.cy}));
  if(!kfs.length) return {cx:0.5, cy:0.5};
  const t = video.currentTime;
  let lo = kfs[0], hi = kfs[kfs.length-1];
  for(let i=0;i<kfs.length-1;i++){ if(kfs[i].t<=t && kfs[i+1].t>=t){ lo=kfs[i]; hi=kfs[i+1]; break; } }
  const span = (hi.t - lo.t) || 1;
  const f = Math.min(1, Math.max(0, (t - lo.t)/span));
  const cxKey = state.manualKeyframes.length ? 'x' : 'cx';
  const cyKey = state.manualKeyframes.length ? 'y' : 'cy';
  return { cx: lo[cxKey] + (hi[cxKey]-lo[cxKey])*f, cy: lo[cyKey] + (hi[cyKey]-lo[cyKey])*f };
}
 
function drawCropForCurrentTime(){
  const box = $('cropBox');
  if(state.fillMode === 'blur_pad' || !$('chkFace').checked){ box.style.display='none'; return; }
  if(!state.srcW || !video.videoWidth){ return; }
  const meta = RATIO_META[state.previewRatio || '9:16'];
  const targetAR = meta.w/meta.h;
  const srcAR = video.videoWidth/video.videoHeight;
  let cropWFrac, cropHFrac;
  if(srcAR > targetAR){ cropHFrac = 1; cropWFrac = targetAR/srcAR; }
  else { cropWFrac = 1; cropHFrac = srcAR/targetAR; }
  const {cx, cy} = currentCrop();
  let left = cx - cropWFrac/2, top = cy - cropHFrac/2;
  left = Math.min(1-cropWFrac, Math.max(0, left));
  top = Math.min(1-cropHFrac, Math.max(0, top));
 
  const frame = $('previewFrame').getBoundingClientRect();
  box.style.display = 'block';
  box.style.left = (left*100)+'%';
  box.style.top = (top*100)+'%';
  box.style.width = (cropWFrac*100)+'%';
  box.style.height = (cropHFrac*100)+'%';
  $('cropReadout').textContent = state.previewRatio || '9:16';
  box.dataset.cx = cx; box.dataset.cy = cy;
}
 
// dragging the crop box records a manual keyframe candidate (committed via "pin crop here")
let dragStart = null;
$('cropBox').addEventListener('mousedown', (e)=>{
  state.dragging = true; $('cropBox').classList.add('dragging');
  dragStart = {x:e.clientX, y:e.clientY, left:parseFloat($('cropBox').style.left), top:parseFloat($('cropBox').style.top)};
});
window.addEventListener('mousemove', (e)=>{
  if(!state.dragging) return;
  const frame = $('previewFrame').getBoundingClientRect();
  const dx = (e.clientX-dragStart.x)/frame.width*100;
  const dy = (e.clientY-dragStart.y)/frame.height*100;
  const box = $('cropBox');
  const newLeft = Math.min(100-parseFloat(box.style.width), Math.max(0, dragStart.left+dx));
  const newTop = Math.min(100-parseFloat(box.style.height), Math.max(0, dragStart.top+dy));
  box.style.left = newLeft+'%'; box.style.top = newTop+'%';
  box.dataset.cx = (newLeft + parseFloat(box.style.width)/2)/100;
  box.dataset.cy = (newTop + parseFloat(box.style.height)/2)/100;
});
window.addEventListener('mouseup', ()=>{ state.dragging=false; $('cropBox').classList.remove('dragging'); });
 
$('addKfBtn').onclick = ()=>{
  const box = $('cropBox');
  const meta = RATIO_META[state.previewRatio || '9:16'];
  const srcAR = video.videoWidth/video.videoHeight;
  const targetAR = meta.w/meta.h;
  let cropW, cropH;
  if(srcAR > targetAR){ cropH = state.srcH; cropW = Math.round(cropH*targetAR); }
  else { cropW = state.srcW; cropH = Math.round(cropW/targetAR); }
  const cx = parseFloat(box.dataset.cx||0.5), cy = parseFloat(box.dataset.cy||0.5);
  let x = Math.round(cx*state.srcW - cropW/2), y = Math.round(cy*state.srcH - cropH/2);
  x = Math.max(0, Math.min(x, state.srcW-cropW)); y = Math.max(0, Math.min(y, state.srcH-cropH));
  state.manualKeyframes = state.manualKeyframes.filter(k=>Math.abs(k.t-video.currentTime)>0.05);
  state.manualKeyframes.push({t: video.currentTime, x, y});
  state.manualKeyframes.sort((a,b)=>a.t-b.t);
  drawTimeline();
  $('statusLine').textContent = `Pinned manual crop at ${fmtTime(video.currentTime)} — ${state.manualKeyframes.length} manual keyframe(s) active (overrides auto tracking).`;
};
$('clearKfBtn').onclick = ()=>{ state.manualKeyframes=[]; drawTimeline(); drawCropForCurrentTime(); };
 
// ── timeline: face density, segments, hook suggestion ──
function drawTimeline(){
  const tl = $('timeline');
  [...tl.querySelectorAll('.tl-face,.tl-hook')].forEach(e=>e.remove());
  if(state.duration){
    state.faceTrack.forEach(p=>{
      const el = document.createElement('div');
      el.className='tl-face';
      el.style.left = (p.t/state.duration*100)+'%'; el.style.width='2px';
      tl.appendChild(el);
    });
    if(state.suggestedHook){
      const el = document.createElement('div'); el.className='tl-hook';
      el.style.left = (state.suggestedHook.start/state.duration*100)+'%';
      el.style.width = ((state.suggestedHook.end-state.suggestedHook.start)/state.duration*100)+'%';
      tl.appendChild(el);
    }
  }
  renderSegList();
}
function renderSegList(){
  const list = $('segList'); list.innerHTML='';
  state.segments.forEach((seg, i)=>{
    const row = document.createElement('div'); row.className='seg-row';
    row.innerHTML = `<span class="mono">#${i+1}</span><span class="mono">${fmtTime(seg.start)} → ${fmtTime(seg.end)}</span><span class="spacer"></span><button data-i="${i}">✕</button>`;
    row.querySelector('button').onclick = ()=>{ state.segments.splice(i,1); renderSegList(); };
    list.appendChild(row);
  });
}
$('timeline').addEventListener('click', (e)=>{
  const rect = $('timeline').getBoundingClientRect();
  const frac = (e.clientX-rect.left)/rect.width;
  video.currentTime = frac * (video.duration||0);
});
$('addSegBtn').onclick = ()=>{
  const t = video.currentTime;
  state.segments.push({start: Math.max(0, +(t-3).toFixed(2)), end: Math.min(state.duration, +(t+3).toFixed(2))});
  renderSegList();
};
$('useHookBtn').onclick = ()=>{
  if(!state.suggestedHook) return;
  state.segments = [state.suggestedHook, {start:0, end: state.duration}];
  renderSegList();
  $('chkHook').checked = true;
};
$('useSilenceBtn').onclick = ()=>{
  if(!state.suggestedSilences) return;
  state.segments = state.suggestedSilences;
  renderSegList();
  $('chkJump').checked = true;
};
 
// ── render ──
$('renderBtn').onclick = async ()=>{
  if(!state.selectedRatios.size){ $('statusLine').textContent = 'Pick at least one export ratio first.'; return; }
  $('renderBtn').disabled = true;
  $('progressWrap').style.display = 'block';
  $('renderLogConsole').innerHTML = '';
  $('results').innerHTML = '';
  const body = {
    source_path: state.sourcePath,
    export_ratios: Array.from(state.selectedRatios),
    face_tracking: $('chkFace').checked && !$('chkFace').disabled,
    subtitles: $('chkSubs').checked && !$('chkSubs').disabled,
    hook_cut: $('chkHook').checked,
    jump_cut: $('chkJump').checked,
    normalize_audio: $('chkNorm').checked,
    caption_style: state.selectedStyle,
    fill_mode: state.fillMode,
    quality: $('qualitySel').value,
    manual_crop_keyframes: state.manualKeyframes.length ? state.manualKeyframes : null,
    segments: state.segments.length ? state.segments : null,
    // Already computed during analyze — sending them lets render skip a
    // second full face-track scan and a second transcript fetch entirely
    // (server only falls back to recomputing if segments/hook/jump-cut
    // trimming changes the timeline, which invalidates these timestamps).
    face_track: state.faceTrack && state.faceTrack.length ? state.faceTrack : null,
    face_track_src_w: state.srcW || null,
    face_track_src_h: state.srcH || null,
    face_tracking_mode: state.faceTrackingMode || null,
    words: state.words && state.words.length ? state.words : null,
  };
  try{
    const res = await fetch(api('/api/editor/render'), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const data = await res.json();
    if(data.error){ $('statusLine').textContent = 'Render error: '+data.error; $('renderBtn').disabled=false; return; }
    state.jobId = data.job_id;
    pollProgress();
  }catch(e){
    $('statusLine').textContent = 'Could not reach the server to start the render — check it is still running and try again.';
    $('renderBtn').disabled = false;
  }
};
 
async function pollProgress(){
  let job;
  try{
    const res = await fetch(api('/api/editor/progress/'+state.jobId));
    job = await res.json();
  }catch(e){
    setTimeout(pollProgress, 1500); // transient hiccup, keep polling
    return;
  }
  $('progressFill').style.width = (job.percent||0)+'%';
  $('progressStage').textContent = job.stage || job.status;
  $('progressPct').textContent = (job.percent||0)+'%';
  _renderLogLines($('renderLogConsole'), job.logs);
  if(job.status === 'error'){
    $('statusLine').textContent = 'Render failed: '+job.error;
    $('renderBtn').disabled = false;
    return;
  }
  if(job.status !== 'done'){ setTimeout(pollProgress, 1200); return; }
  $('renderBtn').disabled = false;
  const results = $('results');
  Object.entries(job.ratios||{}).forEach(([ratio, fname])=>{
    const row = document.createElement('div'); row.className='result-row';
    const tag = ratio === 'auto' ? 'Auto (original size — lossless)' : `${ratio} export`;
    row.innerHTML = `<span class="name">${tag}</span><a href="${api('/api/editor/file/'+state.jobId+'/'+ratio)}" download>Download</a>`;
    results.appendChild(row);
  });
}
 
loadMeta();
</script>
</body>
</html>
"""
 
 
# Optional, best-effort link to downloader.py's in-memory job table so the
# editor can show nicer names for downloaded videos ("Video Title.mp4"
# instead of "<dl_id>_Video Title.mp4"). video_editor.py still works fine
# standalone if downloader.py isn't registered — this never raises.
def _downloader_jobs():
    try:
        import downloader
        return downloader.DL_JOBS
    except Exception:
        return {}
 
# ── configured once via init_editor() ──────────────────────────────
EDIT_DIR = None
SRC_DIR = None
FFMPEG_PATH = "ffmpeg"
FFPROBE_PATH = "ffprobe"
 
# job_id -> {status, stage, percent, error, ratios: {ratio: filename}, config}
EDIT_JOBS = {}

# job_id -> {status, stage, percent, error, logs: [...], result: {...}}
# Analyze runs as a background job too now (instead of blocking the HTTP
# request), specifically so the frontend can show a live progress bar +
# log console instead of a page that just "hangs" with no feedback while
# face-tracking / caption-fetch / AI clip-finding are running.
ANALYZE_JOBS = {}


def _job_log(job, msg, percent=None, stage=None):
    """Appends a timestamped line to a job's log console feed and, in the
    same call, optionally moves its progress bar — every stage of both
    analyze and render report through this one function."""
    job.setdefault("logs", []).append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    if percent is not None:
        job["percent"] = percent
    if stage is not None:
        job["stage"] = stage
 
 
def init_editor(base_dir, ffmpeg_path=None):
    """Call once at startup, e.g. init_editor(BASE, FFMPEG)."""
    global EDIT_DIR, SRC_DIR, FFMPEG_PATH, FFPROBE_PATH
    base_dir = Path(base_dir)
    EDIT_DIR = base_dir / "edited"
    EDIT_DIR.mkdir(exist_ok=True)
    SRC_DIR = base_dir / "downloads"   # reuse downloader's output as source pool
    (EDIT_DIR / "uploads").mkdir(exist_ok=True)
    if ffmpeg_path:
        FFMPEG_PATH = str(ffmpeg_path)
        # ffprobe normally sits next to ffmpeg
        cand = Path(ffmpeg_path).with_name(
            "ffprobe.exe" if str(ffmpeg_path).lower().endswith(".exe") else "ffprobe"
        )
        if cand.exists():
            FFPROBE_PATH = str(cand)

    # Whisper is intentionally NEVER auto-loaded anymore (see USE_WHISPER
    # below) — transcripts come only from the source platform's own
    # captions via yt-dlp, so there is nothing to pre-warm here.
 
 
def _no_console_kwargs():
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
 
 
def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, **_no_console_kwargs())
 
 
# ══════════════════════════════ feature availability ══════════════════════════════
 
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
 
try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
    # Importing successfully doesn't mean `mp.solutions.face_detection` still
    # exists — recent mediapipe releases dropped the legacy `solutions` API
    # in favor of the newer Tasks API. Probe it once here so feature_status()
    # (and every caller) reports the truth instead of assuming "advanced" is
    # available and crashing later inside _FaceTracker.
    try:
        mp.solutions.face_detection.FaceDetection
        _HAS_MEDIAPIPE_SOLUTIONS = True
    except AttributeError:
        _HAS_MEDIAPIPE_SOLUTIONS = False
except ImportError:
    _HAS_MEDIAPIPE = False
    _HAS_MEDIAPIPE_SOLUTIONS = False
 
try:
    from faster_whisper import WhisperModel
    _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False
 
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
 
try:
    import yt_dlp
    _HAS_YTDLP = True
except ImportError:
    _HAS_YTDLP = False
 
_WHISPER_MODEL = None  # lazy-loaded singleton
 
 
def feature_status():
    return {
        "face_tracking": _HAS_CV2,
        "face_tracking_advanced": _HAS_CV2 and _HAS_MEDIAPIPE_SOLUTIONS,
        # Subtitles now depend on a caption SOURCE (yt-dlp, for a fetched
        # video) being available, not on Whisper being installed — Whisper
        # is intentionally disabled everywhere (USE_WHISPER = False).
        "subtitles": _HAS_YTDLP,
        "hook_cut_auto": _HAS_NUMPY,
        "fetch_from_link": _HAS_YTDLP,
        "ai_clip_finder": any(_ai_key(p) for p in AI_PROVIDERS),
    }
 
 
# ══════════════════════════════ presets ══════════════════════════════
 
# width x height for each export ratio, chosen at "good enough to look sharp,
# small enough to encode fast" — bump these if you want 4K exports.
RATIO_PRESETS = {
    # "auto" is a sentinel: w/h are resolved at render time from the SOURCE
    # video's own dimensions (no target size baked in here). It means "keep
    # the video exactly as shot" — no crop, no letterbox, no rescale — and
    # whenever nothing else in the pipeline needs to touch the picture
    # (no captions/color/manual crop requested) it is exported via ffmpeg
    # stream-copy, i.e. the original video bytes are re-muxed, not
    # re-encoded, so there is truly zero generational quality loss.
    "auto":  {"w": None, "h": None, "label": "Auto — original size, no crop, zero quality loss"},
    "9:16":  {"w": 1080, "h": 1920, "label": "Reels / Shorts / TikTok"},
    "16:9":  {"w": 1920, "h": 1080, "label": "YouTube / Landscape"},
    "1:1":   {"w": 1080, "h": 1080, "label": "Square / Feed post"},
    "4:5":   {"w": 1080, "h": 1350, "label": "Instagram portrait"},
    "4:3":   {"w": 1440, "h": 1080, "label": "Classic / Facebook"},
}
 
QUALITY_PRESETS = {
    "high":   {"crf": 18, "preset": "slow"},
    "medium": {"crf": 21, "preset": "medium"},
    "fast":   {"crf": 23, "preset": "veryfast"},
}
 
# Words that get an automatic extra-emphasis color in captions even in
# styles that don't do full word-by-word highlighting — numbers, money and
# a short list of "power words" are what trending caption tools punch up.
_EMPHASIS_PATTERN = None  # compiled lazily, see _is_emphasis_word()
_POWER_WORDS = {
    "free", "now", "never", "always", "secret", "insane", "crazy", "huge",
    "warning", "stop", "wait", "new", "best", "worst", "you", "your",
}
 
 
def _is_emphasis_word(word):
    import re
    global _EMPHASIS_PATTERN
    if _EMPHASIS_PATTERN is None:
        _EMPHASIS_PATTERN = re.compile(r"[\d%$]|^\$")
    w = word.strip(".,!?").lower()
    return bool(_EMPHASIS_PATTERN.search(word)) or w in _POWER_WORDS
 
# Trending caption styles. Each is a full ASS "Style:" line plus a couple of
# behavior flags consumed by build_ass(). Colors are &HAABBGGRR (ASS order).
CAPTION_STYLES = {
    "bold_pop": {
        "label": "Bold Pop (white + yellow highlight)",
        "style": "Style: Default,Montserrat Black,84,&H00FFFFFF,&H0000D7FF,&H00101010,&H00000000,"
                  "-1,0,0,0,100,100,0,0,1,6,0,2,60,60,140,1",
        "highlight_color": "&H0000D7FF",   # active word turns yellow/gold
        "word_by_word": True,
        "pop_scale": True,
    },
    "karaoke_classic": {
        "label": "Karaoke Fill (progressive color wipe)",
        "style": "Style: Default,Montserrat SemiBold,72,&H00FFFFFF,&H0000A5FF,&H00202020,&H00000000,"
                  "-1,0,0,0,100,100,0,0,1,4,0,2,60,60,150,1",
        "highlight_color": "&H0000A5FF",
        "word_by_word": True,
        "karaoke_fill": True,
    },
    "neon_glow": {
        "label": "Neon Glow (cyan outline pop)",
        "style": "Style: Default,Montserrat Black,80,&H00FFFFFF,&H00FFF000,&H00902000,&H00000000,"
                  "-1,0,0,0,100,100,0,0,1,5,2,2,60,60,140,1",
        "highlight_color": "&H00FFF000",
        "word_by_word": True,
        "pop_scale": True,
    },
    "minimal_clean": {
        "label": "Minimal Clean (small centered white)",
        "style": "Style: Default,Inter Medium,54,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
                  "0,0,0,0,100,100,0,0,1,2,1,2,60,60,120,1",
        "highlight_color": "&H00FFFFFF",
        "word_by_word": False,
        "pop_scale": False,
    },
    "creator_bold": {
        "label": "Creator Bold (huge yellow, black outline)",
        "style": "Style: Default,Montserrat Black,92,&H0000FFFF,&H000000FF,&H00000000,&H00000000,"
                  "-1,0,0,0,100,100,0,0,1,8,0,2,50,50,160,1",
        "highlight_color": "&H000000FF",
        "word_by_word": True,
        "pop_scale": True,
    },
}
 
 
@editor_bp.route("/api/editor/styles")
def api_editor_styles():
    return jsonify({k: {"label": v["label"]} for k, v in CAPTION_STYLES.items()})
 
 
@editor_bp.route("/api/editor/ratios")
def api_editor_ratios():
    return jsonify(RATIO_PRESETS)
 
 
@editor_bp.route("/api/editor/editor")
def api_editor_page():
    """Serves the built-in editor UI — no separate .html file to manage,
    it's embedded in this module (EDITOR_HTML below) and rendered straight
    from memory. Open this URL in a tab; its fetch calls are same-origin."""
    from flask import Response
    return Response(EDITOR_HTML, mimetype="text/html")
 
 
def _fmt_size(n):
    if not n:
        return "0 B"
    n = float(n)
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
 
 
@editor_bp.route("/api/editor/sources")
def api_editor_sources():
    """Lists videos the editor can open WITHOUT a fresh upload: everything
    downloader.py has saved to DL_DIR/downloads (so a just-fetched video can
    be sent straight into the editor), plus anything uploaded here before.
    Both origins return the exact same shape and both feed the exact same
    source_path used by /analyze and /render — the editor doesn't care where
    a file came from."""
    import datetime
    jobs = _downloader_jobs()
    # dl_id -> nicer display title, when downloader.py is registered too
    title_by_stub = {}
    for job in jobs.values():
        fname = job.get("filename")
        if fname and "_" in fname:
            stub = fname.split("_", 1)[0]
            title_by_stub[stub] = fname.split("_", 1)[-1]
 
    sources = []
    for folder, origin in ((SRC_DIR, "downloader"), (EDIT_DIR / "uploads", "upload")):
        if not folder or not folder.exists():
            continue
        for p in sorted(folder.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_file() or p.suffix.lower() not in (".mp4", ".mov", ".mkv", ".webm", ".m4v"):
                continue
            stub = p.name.split("_", 1)[0]
            display = title_by_stub.get(stub) or (p.name.split("_", 1)[-1] if "_" in p.name else p.name)
            stat = p.stat()
            sources.append({
                "name": display,
                "path": str(p.resolve()),
                "origin": origin,
                "size": stat.st_size,
                "size_str": _fmt_size(stat.st_size),
                "modified": stat.st_mtime,
                "modified_str": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%d %b, %H:%M"),
            })
    sources.sort(key=lambda s: s["modified"], reverse=True)
    return jsonify({"sources": sources})
 
 
def _is_allowed_source(path):
    """Only ever stream files that live under the editor's own known roots
    (downloader's downloads folder, this module's uploads folder, or its
    own rendered-output folder) — never an arbitrary server path."""
    try:
        rp = Path(path).resolve()
    except Exception:
        return False
    for root in (SRC_DIR, EDIT_DIR / "uploads", EDIT_DIR):
        if root and root.exists():
            try:
                rp.relative_to(root.resolve())
                return True
            except ValueError:
                continue
    return False
 
 
@editor_bp.route("/api/editor/preview")
def api_editor_preview():
    """Streams a source (or rendered) video for <video> tag playback/scrub,
    so picking a file from the library previews instantly without a
    round-trip upload. Range requests are handled by Flask's send_file."""
    path = request.args.get("path", "")
    if not path or not _is_allowed_source(path) or not Path(path).exists():
        return "Not found or not allowed", 404
    return send_file(path, conditional=True)
 
 
@editor_bp.route("/api/editor/features")
def api_editor_features():
    return jsonify(feature_status())
 
 
# ══════════════════════════════ probing ══════════════════════════════
 
def _probe(path):
    """Returns {duration, width, height, fps} for any video — PC upload,
    Downloader-library pick, or freshly fetched link — using ONLY ffmpeg
    (FFMPEG_PATH), never ffprobe.

    Why: this whole app is wired up with `FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()`
    in the main file, which downloads/bundles an ffmpeg BINARY ONLY — it does
    not ship ffprobe next to it, and a bare "ffprobe" is not guaranteed to be
    on PATH on a fresh Windows machine (that mismatch is exactly what produced
    `FileNotFoundError: [WinError 2] The system cannot find the file
    specified` — the app was calling a program that doesn't exist on disk).
    `ffmpeg -i <file>` already prints duration/resolution/fps to stderr for
    every input, so we parse that instead — one binary, guaranteed present,
    identical accurate metadata for every source, no more ffprobe dependency.
    """
    cmd = [FFMPEG_PATH, "-hide_banner", "-i", str(path)]
    out = _run(cmd)
    text = (out.stderr or "") + (out.stdout or "")

    duration = 0.0
    dm = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if dm:
        hh, mm, ss = dm.groups()
        duration = int(hh) * 3600 + int(mm) * 60 + float(ss)

    width = height = None
    # e.g. "Stream #0:0: Video: h264 ..., yuv420p, 1080x1920 [SAR 1:1 DAR 9:16], 30 fps, 30 tbr, ..."
    vm = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", text)
    if vm:
        width, height = int(vm.group(1)), int(vm.group(2))

    fps = 25.0
    fm = re.search(r"([\d.]+)\s+fps", text)
    if fm:
        try:
            fps = float(fm.group(1))
        except ValueError:
            pass

    return {"duration": duration, "width": width, "height": height, "fps": fps}
 
 
# ══════════════════════════════ face tracking ══════════════════════════════
 
class _FaceTracker:
    """Samples frames, finds a face each time, and produces a smoothed
    (jitter-free) track of the subject's center point over time.
 
    Uses mediapipe's BlazeFace model when available ("advanced" mode — much
    more accurate on angled/small faces), otherwise falls back to OpenCV's
    bundled Haar cascade so face tracking still works with just opencv-python
    installed.
    """
 
    def __init__(self):
        # mediapipe importing successfully doesn't guarantee the legacy
        # `mp.solutions.face_detection` API is present — recent mediapipe
        # builds moved to a different Tasks API and dropped `.solutions`
        # entirely, which used to crash every analyze request with
        # `AttributeError: module 'mediapipe' has no attribute 'solutions'`.
        # _HAS_MEDIAPIPE_SOLUTIONS was probed once at startup, so this just
        # picks the mode that's actually usable instead of assuming.
        self.advanced = _HAS_MEDIAPIPE_SOLUTIONS
        if self.advanced:
            self._mp_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        elif _HAS_CV2:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        else:
            raise RuntimeError("opencv-python is required for face tracking")
 
    def _detect(self, frame_bgr):
        """Returns list of (cx, cy, w, h) in pixel coords, largest first."""
        h_img, w_img = frame_bgr.shape[:2]
        faces = []
        if self.advanced:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._mp_detector.process(rgb)
            if result.detections:
                for d in result.detections:
                    box = d.location_data.relative_bounding_box
                    fw, fh = box.width * w_img, box.height * h_img
                    fx = box.xmin * w_img + fw / 2
                    fy = box.ymin * h_img + fh / 2
                    faces.append((fx, fy, fw, fh))
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            dets = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50))
            for (x, y, w, h) in dets:
                faces.append((x + w / 2, y + h / 2, w, h))
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)  # largest face first
        return faces
 
    def track(self, video_path, sample_fps=2.0, progress_cb=None):
        """Returns a list of {t, cx, cy, w, h} in NORMALIZED (0-1) coords,
        exponentially smoothed so the crop pans instead of jumping."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video for face tracking")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, int(round(src_fps / sample_fps)))
 
        raw_points = []
        last_center = (0.5, 0.5)  # default: frame center when nobody's found yet
        frame_idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % step == 0:
                ok2, frame = cap.retrieve()
                if ok2:
                    faces = self._detect(frame)
                    if faces:
                        fx, fy, fw, fh = faces[0]
                        cx, cy = fx / w_img, fy / h_img
                        last_center = (cx, cy)
                    else:
                        cx, cy = last_center  # hold last known position
                    t = frame_idx / src_fps
                    raw_points.append({"t": t, "cx": cx, "cy": cy})
                if progress_cb and total_frames:
                    progress_cb(min(99, int(frame_idx * 100 / total_frames)))
            frame_idx += 1
        cap.release()
 
        if not raw_points:
            raw_points = [{"t": 0.0, "cx": 0.5, "cy": 0.5}]
 
        # EMA smoothing so the crop glides instead of snapping every sample
        alpha = 0.25
        smoothed = [raw_points[0]]
        for p in raw_points[1:]:
            prev = smoothed[-1]
            smoothed.append({
                "t": p["t"],
                "cx": prev["cx"] + alpha * (p["cx"] - prev["cx"]),
                "cy": prev["cy"] + alpha * (p["cy"] - prev["cy"]),
            })
        return smoothed, (w_img, h_img)
 
 
def analyze_face_track(video_path, sample_fps=2.0, progress_cb=None):
    if not _HAS_CV2:
        raise RuntimeError("opencv-python is not installed — face tracking unavailable")
    tracker = _FaceTracker()
    points, (w, h) = tracker.track(video_path, sample_fps=sample_fps, progress_cb=progress_cb)
    return points, w, h, tracker.advanced
 
 
# ══════════════════════════════ crop-window math ══════════════════════════════
 
def _crop_size_for_ratio(src_w, src_h, target_w, target_h):
    """Largest crop rectangle matching the target aspect that still fits
    inside the source frame (so we crop, never letterbox/pad)."""
    target_ar = target_w / target_h
    src_ar = src_w / src_h
    if src_ar > target_ar:
        # source is wider than target -> crop width, keep full height
        crop_h = src_h
        crop_w = int(round(crop_h * target_ar))
    else:
        # source is taller than target -> crop height, keep full width
        crop_w = src_w
        crop_h = int(round(crop_w / target_ar))
    return crop_w, crop_h
 
 
def _build_crop_keyframes(track_points, src_w, src_h, crop_w, crop_h):
    """Turns normalized face-center points into pixel-space crop x/y
    top-left keyframes, clamped so the crop never leaves the frame."""
    kfs = []
    for p in track_points:
        cx_px = p["cx"] * src_w
        cy_px = p["cy"] * src_h
        x = int(cx_px - crop_w / 2)
        y = int(cy_px - crop_h / 2)
        x = max(0, min(x, src_w - crop_w))
        y = max(0, min(y, src_h - crop_h))
        kfs.append({"t": p["t"], "x": x, "y": y})
    return _simplify_keyframes(kfs)


def _simplify_keyframes(kfs, max_keyframes=48, min_move_px=6):
    """Collapses near-static points and hard-caps the keyframe count.

    _expr_chain below turns every keyframe pair into one more nested
    if(lt(t,...), ramp, ...) level. Face-tracking samples at ~2 points/sec,
    so anything longer than ~30-40s of video produced 150-300+ keyframes —
    hundreds of nested if() calls — which ffmpeg's expression evaluator
    cannot parse reliably (it silently corrupts/truncates into exactly the
    runaway ')))))' string seen in "ffmpeg failed for 9:16"). Capping the
    keyframe count keeps every render — regardless of clip length — well
    inside what ffmpeg can actually parse, while (1) dropping points where
    the face barely moved and (2) evenly downsampling only if still over
    the cap keeps the crop motion visually smooth rather than jumpy.
    """
    if len(kfs) <= 2:
        return kfs
    simplified = [kfs[0]]
    for kf in kfs[1:-1]:
        last = simplified[-1]
        if abs(kf["x"] - last["x"]) >= min_move_px or abs(kf["y"] - last["y"]) >= min_move_px:
            simplified.append(kf)
    if simplified[-1] is not kfs[-1]:
        simplified.append(kfs[-1])
    if len(simplified) > max_keyframes:
        step = len(simplified) / max_keyframes
        picked = [simplified[int(i * step)] for i in range(max_keyframes)]
        if picked[-1] is not simplified[-1]:
            picked.append(simplified[-1])
        simplified = picked
    return simplified
 
 
def _expr_chain(keyframes, key):
    """Builds an ffmpeg time-expression string that linearly interpolates
    between keyframes for either 'x' or 'y' — this drives a moving crop
    window without needing any external filter graph patching."""
    if len(keyframes) == 1:
        return str(keyframes[0][key])
    expr = str(keyframes[-1][key])
    for i in range(len(keyframes) - 2, -1, -1):
        t0, v0 = keyframes[i]["t"], keyframes[i][key]
        t1, v1 = keyframes[i + 1]["t"], keyframes[i + 1][key]
        if t1 <= t0:
            continue
        # linear ramp between (t0,v0) and (t1,v1), holds v0 before t0
        ramp = f"({v0}+({v1}-{v0})*(t-{t0})/{(t1 - t0)})"
        expr = f"if(lt(t,{t1}),{ramp},{expr})"
    return expr
 
 
# ══════════════════════════════ subtitles ══════════════════════════════
 
def _load_whisper(model_size="small"):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = WhisperModel(model_size, compute_type="int8")
    return _WHISPER_MODEL
 
 
def transcribe_words(video_path, model_size="small", language=None):
    """Returns [{start, end, text}, ...] word-level timestamps."""
    if not _HAS_WHISPER:
        raise RuntimeError("faster-whisper is not installed — subtitles unavailable")
    model = _load_whisper(model_size)
    segments, _info = model.transcribe(str(video_path), word_timestamps=True, language=language)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            text = (w.word or "").strip()
            if text:
                words.append({"start": w.start, "end": w.end, "text": text})
    return words


# ══════════════ transcript source: platform captions via yt-dlp (NO WHISPER) ══════════════
# Whisper is intentionally never called automatically anymore. `transcribe_words`
# above is left in place (harmless, dead code) in case you ever want to wire it
# back in by hand — but every analyze/render call path now goes through
# get_transcript_words() below instead, which only ever pulls REAL captions the
# source platform already made (via yt-dlp), never runs local speech-to-text.
# A plain PC upload, or a fetched video whose platform has no captions on it,
# simply has no transcript: subtitles + the AI viral-clip finder are skipped
# for that one video, while face-tracking and the loudness/silence-based
# hook-cut & jump-cut (neither needs a transcript) keep working as before.
USE_WHISPER = False

SOURCE_URL_MAP = {}    # str(source_path) -> the URL it was fetched from (set in _run_fetch_job)
SOURCE_TITLE_MAP = {}  # str(source_path) -> the platform title (set in _run_fetch_job)
TRANSCRIPT_CACHE = {}  # str(source_path) -> {"words":[...], "chapters":[...], "source_kind":...} | None


def _vtt_to_words(vtt_text):
    """Turns a WebVTT captions file into an approximate word-level timeline
    (real platform captions are only cue-level — a few words per timestamp
    range — so words inside each cue are spread evenly across that cue's
    duration). Output uses the SAME {"start","end","text"} shape Whisper
    used to produce, so build_ass()/_chunk_words() need no changes."""
    cues = []
    blocks = re.split(r"\n\s*\n", vtt_text.strip())
    time_re = re.compile(r"(\d{2}:)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}:)?(\d{2}):(\d{2})[.,](\d{3})")

    def _t(h, m, s, ms):
        h = int(h.rstrip(":")) if h else 0
        return h * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    for block in blocks:
        m = time_re.search(block)
        if not m:
            continue
        start = _t(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _t(m.group(5), m.group(6), m.group(7), m.group(8))
        text_lines = block[m.end():].strip().splitlines()
        text = " ".join(l.strip() for l in text_lines if l.strip() and "-->" not in l)
        text = re.sub(r"<[^>]+>", "", text).strip()  # strip VTT tags (<c>, karaoke timing, etc.)
        if text and end > start:
            cues.append((start, end, text))

    words = []
    for start, end, text in cues:
        toks = text.split()
        if not toks:
            continue
        span = max(end - start, 0.2)
        step = span / len(toks)
        for i, tok in enumerate(toks):
            words.append({"start": round(start + i * step, 2), "end": round(start + (i + 1) * step, 2), "text": tok})
    return words


def _transcript_from_ytdlp_info(info, log_cb=None):
    """Pulls chapters + real captions (manual first, then auto-generated)
    out of a yt-dlp `info` dict that was ALREADY extracted — no network
    call of its own. yt-dlp always populates info['subtitles'] /
    info['automatic_captions'] / info['chapters'] during extraction
    regardless of the writesubtitles/writeautomaticsub options, so this can
    run against the very same info dict a download step already produced."""
    def _log(msg):
        if log_cb:
            log_cb(msg)
    chapters = info.get("chapters") or []
    subs = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    track, source_kind = None, None
    for lang_map, kind in ((subs, "manual"), (auto, "auto-generated")):
        for lang in ("en", "en-US", "en-GB", "en-orig"):
            if lang in lang_map:
                track, source_kind = lang_map[lang], kind
                break
        if track:
            break
    if not track:
        if chapters:
            _log(f"No captions, but found {len(chapters)} chapter marker(s) on the source platform.")
        else:
            _log("No English captions or chapters found on the source platform for this video — "
                 "subtitles and the AI clip-finder are skipped for it (Whisper stays off).")
        return {"words": [], "chapters": chapters, "source_kind": None}

    vtt_url = next((f["url"] for f in track if f.get("ext") == "vtt"), track[0].get("url"))
    with urllib.request.urlopen(vtt_url, timeout=15) as resp:
        vtt_text = resp.read().decode("utf-8", errors="ignore")
    words = _vtt_to_words(vtt_text)
    _log(f"Pulled {len(words)} caption word(s) [{source_kind}] + {len(chapters)} chapter(s) "
         f"straight from the source — no Whisper needed.")
    return {"words": words, "chapters": chapters, "source_kind": source_kind}


def fetch_ytdlp_transcript(url, log_cb=None):
    """Pulls real captions (manual first, then auto-generated) + chapters
    for a URL via yt-dlp WITHOUT downloading the video again — this is
    what replaces Whisper entirely for any video that came from a link.

    This does its OWN extract_info network round-trip, so it's only used
    as a fallback (e.g. analyzing a video whose transcript wasn't already
    captured at fetch time). Whenever possible — see _run_fetch_job — the
    info dict from the ORIGINAL download is reused instead via
    _transcript_from_ytdlp_info, to avoid a second request to the source
    platform in quick succession (back-to-back requests for the same video
    are exactly what triggers an HTTP 429 from platforms like YouTube)."""
    def _log(msg):
        if log_cb:
            log_cb(msg)
    if not _HAS_YTDLP:
        return None
    try:
        opts = {
            "quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True,
            "writesubtitles": True, "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB", "en-orig"], "subtitlesformat": "vtt",
            "socket_timeout": 15,
        }
        attempts, cookies_file = _fetch_auth_attempts()
        info = None
        for mode in attempts:
            try:
                o = _fetch_apply_auth(opts, mode, cookies_file)
                with yt_dlp.YoutubeDL(o) as ydl:
                    info = ydl.extract_info(url, download=False)
                break
            except Exception:
                continue
        if not info:
            _log("Could not resolve this link for captions.")
            return None
        return _transcript_from_ytdlp_info(info, log_cb=log_cb)
    except Exception as e:
        _log(f"Caption fetch failed ({e}) — continuing without a transcript for this video.")
        return None


def get_transcript_words(source_path, log_cb=None):
    """The ONE place analyze/render should call for a transcript. Looks up
    the URL this source came from (if it was fetched at all — see
    SOURCE_URL_MAP) and returns cached/fresh caption-derived words, or None
    if there's no caption source. Never touches Whisper."""
    key = str(source_path)
    if key in TRANSCRIPT_CACHE:
        return TRANSCRIPT_CACHE[key]
    url = SOURCE_URL_MAP.get(key)
    if not url:
        TRANSCRIPT_CACHE[key] = None
        return None
    data = fetch_ytdlp_transcript(url, log_cb=log_cb)
    TRANSCRIPT_CACHE[key] = data
    return data


# ══════════════════ AI providers — viral clip finder (transcript -> ranked clips) ══════════════════
# Paste your own FREE API keys below, OR set them as environment variables
# with the same name (env vars win if both are set). Tried top to bottom —
# the first provider with a key that actually succeeds is used; a missing
# key, a rate limit, a timeout, or a bad response all just fall through to
# the next one automatically, so one exhausted free quota doesn't stop the
# feature. All of these have a free tier as of writing:
#
#   Gemini      key_env=GEMINI_API_KEY      https://aistudio.google.com/apikey
#               free tier: ~15 req/min, ~1500 req/day, resets daily (midnight Pacific)
#   Groq        key_env=GROQ_API_KEY        https://console.groq.com/keys
#               free tier: very fast inference, generous daily token limit, resets daily
#   OpenRouter  key_env=OPENROUTER_API_KEY  https://openrouter.ai/keys
#               free tier: use any model ending in ":free", resets daily
#   Cerebras    key_env=CEREBRAS_API_KEY    https://cloud.cerebras.ai
#               free tier: fast Llama models, resets daily
#
# ── PASTE KEYS HERE (or leave "" and use the env vars instead) ──
AI_PROVIDERS = [
    {"name": "Gemini",     "key_env": "GEMINI_API_KEY_EDIT",     "key": "", "model": "gemini-2.0-flash"},
    {"name": "Groq",       "key_env": "GROQ_API_KEY",       "key": "", "model": "llama-3.3-70b-versatile"},
    {"name": "OpenRouter", "key_env": "OPENROUTER_API_KEY", "key": "", "model": "meta-llama/llama-3.3-70b-instruct:free"},
    {"name": "Cerebras",   "key_env": "CEREBRAS_API_KEY",   "key": "", "model": "llama-3.3-70b"},
]


def _ai_key(provider):
    return os.environ.get(provider["key_env"]) or provider.get("key") or ""


def _http_post_json(url, headers, body, timeout=90):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def _call_gemini(provider, prompt):
    key = _ai_key(provider)
    if not key:
        raise RuntimeError("no API key set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{provider['model']}:generateContent?key={key}"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}}
    data = _http_post_json(url, {}, body)
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_openai_compatible(provider, prompt, base_url):
    key = _ai_key(provider)
    if not key:
        raise RuntimeError("no API key set")
    body = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": "You reply with ONLY valid JSON. No markdown fences, no commentary."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }
    data = _http_post_json(f"{base_url}/chat/completions", {"Authorization": f"Bearer {key}"}, body)
    return data["choices"][0]["message"]["content"]


def _call_ai_provider(provider, prompt):
    name = provider["name"]
    if name == "Gemini":
        return _call_gemini(provider, prompt)
    if name == "Groq":
        return _call_openai_compatible(provider, prompt, "https://api.groq.com/openai/v1")
    if name == "OpenRouter":
        return _call_openai_compatible(provider, prompt, "https://openrouter.ai/api/v1")
    if name == "Cerebras":
        return _call_openai_compatible(provider, prompt, "https://api.cerebras.ai/v1")
    raise RuntimeError(f"Unknown provider {name}")


def call_ai_with_fallback(prompt, log_cb=None):
    def _log(msg):
        if log_cb:
            log_cb(msg)
    last_err = None
    for provider in AI_PROVIDERS:
        if not _ai_key(provider):
            _log(f"{provider['name']}: no API key set — skipping.")
            continue
        try:
            _log(f"{provider['name']}: sending transcript for viral-clip analysis…")
            text = _call_ai_provider(provider, prompt)
            _log(f"{provider['name']}: responded.")
            return text, provider["name"]
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")[:300]
            except Exception:
                pass
            last_err = f"{provider['name']} HTTP {e.code}: {body}"
            _log(f"{provider['name']} failed (HTTP {e.code}) — trying next provider…")
        except Exception as e:
            last_err = f"{provider['name']}: {e}"
            _log(f"{provider['name']} failed ({e}) — trying next provider…")
    raise RuntimeError(f"All AI providers failed or have no key set. Last error: {last_err}")


VIRAL_CLIP_PROMPT = """You are an expert short-form video editor who finds viral clips inside long-form video transcripts.

Video title: {title}

Transcript with timestamps (JSON array of objects with "start", "end", "text"):
{transcript_json}

Your task: Analyze the complete transcript with timestamps.
Find ALL potential short-form viral clips.
Do not limit to 20 clips. Return every strong candidate.
A good clip must have:
1. Strong opening hook
2. Curiosity gap
3. Emotional trigger
4. Useful information
5. Complete story within the clip
Avoid:
- introductions
- greetings
- boring explanations
- incomplete sentences

For every clip return an object with exactly these fields:
start_time, end_time, duration, hook_text, hook_type, score (0-100), reason, category, editing_suggestion

Rank all clips from best to worst (highest score first).

Reply with ONLY a JSON object of this exact shape: {{"clips": [ {{...}}, {{...}} ]}}
No markdown fences, no commentary — JSON only."""


def _words_to_caption_cues(words, max_gap=1.2, max_len=220):
    """Groups word-level timestamps back into readable lines (start/end/text)
    — what the AI prompt (and a human) actually wants to read, instead of
    one JSON object per single word."""
    cues, cur = [], None
    for w in words:
        if cur is None or w["start"] - cur["end"] > max_gap or len(cur["text"]) > max_len:
            if cur:
                cues.append(cur)
            cur = {"start": w["start"], "end": w["end"], "text": w["text"]}
        else:
            cur["end"] = w["end"]
            cur["text"] += " " + w["text"]
    if cur:
        cues.append(cur)
    return cues


def find_viral_clips(words, video_title=None, log_cb=None):
    """The AI hook/clip-finder. Feeds the (caption-derived) transcript to
    the first working AI_PROVIDERS entry and returns a ranked list of clip
    dicts. Returns [] (never raises) if there's no transcript or no AI
    provider is configured/working, so callers just fall back to the
    existing loudness/silence heuristics instead."""
    def _log(msg):
        if log_cb:
            log_cb(msg)
    if not words:
        return []
    cues = _words_to_caption_cues(words)
    prompt = VIRAL_CLIP_PROMPT.format(
        title=video_title or "Untitled",
        transcript_json=json.dumps(
            [{"start": round(c["start"], 2), "end": round(c["end"], 2), "text": c["text"]} for c in cues],
            ensure_ascii=False,
        ),
    )
    try:
        raw, provider_name = call_ai_with_fallback(prompt, log_cb=log_cb)
    except Exception as e:
        _log(f"AI clip-finder unavailable ({e}) — falling back to loudness/silence-based hook detection only.")
        return []

    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            _log("AI response wasn't valid JSON — skipping AI clips for this run.")
            return []
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            _log("AI response wasn't valid JSON even after cleanup — skipping AI clips for this run.")
            return []

    clips = parsed.get("clips") if isinstance(parsed, dict) else parsed
    if not isinstance(clips, list):
        return []
    cleaned = []
    for c in clips:
        try:
            start, end = float(c["start_time"]), float(c["end_time"])
            if end <= start:
                continue
            cleaned.append({
                "start_time": round(start, 2), "end_time": round(end, 2),
                "duration": round(end - start, 2),
                "hook_text": c.get("hook_text", ""), "hook_type": c.get("hook_type", ""),
                "score": max(0, min(100, int(c.get("score", 0)))),
                "reason": c.get("reason", ""), "category": c.get("category", ""),
                "editing_suggestion": c.get("editing_suggestion", ""),
            })
        except (KeyError, TypeError, ValueError):
            continue
    cleaned.sort(key=lambda c: c["score"], reverse=True)
    _log(f"AI ({provider_name}) found {len(cleaned)} candidate clip(s).")
    return cleaned
 
 
def _chunk_words(words, max_words=4, max_span=1.6):
    """Groups words into short on-screen caption chunks — the standard
    'trending shorts' style of 2-5 words on screen at a time, not full
    sentences."""
    chunks = []
    cur = []
    for w in words:
        if cur and (len(cur) >= max_words or (w["end"] - cur[0]["start"]) > max_span):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks
 
 
def _ass_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"
 
 
def build_ass(words, style_name, video_w, video_h):
    """Builds a full .ass subtitle document with word-by-word highlight
    animation for the chosen trending style, sized for the given output
    resolution so text scales correctly across aspect ratios."""
    style = CAPTION_STYLES.get(style_name, CAPTION_STYLES["bold_pop"])
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
ScaledBorderAndShadow: yes
 
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style["style"]}
 
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for chunk in _chunk_words(words):
        start, end = chunk[0]["start"], chunk[-1]["end"]
        if style.get("karaoke_fill"):
            # \k tags: whole line present, active word wipes to highlight color
            text = "".join(
                r"{\k%d}%s " % (max(1, int(round((w["end"] - w["start"]) * 100))), w["text"])
                for w in chunk
            )
            lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
        elif style.get("word_by_word"):
            # emit one Dialogue event per word so only the current word is
            # shown in the highlight color while the rest stay default —
            # gives the punchy "pop" caption look trending on shorts.
            for w in chunk:
                parts = []
                for w2 in chunk:
                    if w2 is w:
                        scale = r"\fscx115\fscy115" if style.get("pop_scale") else ""
                        parts.append(r"{\c%s%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], scale, w2["text"]))
                    elif _is_emphasis_word(w2["text"]):
                        # numbers / money / power-words get punched up even
                        # when they're not the "active" word of the moment
                        parts.append(r"{\c%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], w2["text"]))
                    else:
                        parts.append(w2["text"])
                text = " ".join(parts)
                lines.append(f"Dialogue: 0,{_ass_time(w['start'])},{_ass_time(w['end'])},Default,,0,0,0,,{text}")
        else:
            parts = []
            for w2 in chunk:
                if _is_emphasis_word(w2["text"]):
                    parts.append(r"{\c%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], w2["text"]))
                else:
                    parts.append(w2["text"])
            text = " ".join(parts)
            lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
    return header + "\n".join(lines) + "\n"
 
 
# ══════════════════════════════ hook-cut / segments ══════════════════════════════
 
def _read_pcm_mono(video_path, sr=8000):
    """Decodes a fast, low-res mono PCM stream via ffmpeg for lightweight
    loudness analysis (used only to pick the auto hook window)."""
    cmd = [FFMPEG_PATH, "-v", "error", "-i", str(video_path),
           "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True, **_no_console_kwargs())
    raw = proc.stdout
    count = len(raw) // 2
    samples = struct.unpack(f"<{count}h", raw[:count * 2])
    return np.array(samples, dtype=np.float32) / 32768.0, sr
 
 
def auto_hook_window(video_path, duration, hook_len=6.0):
    """Finds the loudest `hook_len`-second window in the clip — a cheap but
    effective proxy for 'most energetic / most likely to hook a viewer'
    moment, used to auto-suggest where the exported clip should start."""
    if not _HAS_NUMPY:
        raise RuntimeError("numpy is not installed — auto hook-cut unavailable")
    audio, sr = _read_pcm_mono(video_path)
    if len(audio) == 0:
        return {"start": 0.0, "end": min(hook_len, duration)}
    win = int(hook_len * sr)
    if win >= len(audio):
        return {"start": 0.0, "end": duration}
    energy = audio ** 2
    # cumulative sum for O(1) windowed average lookups
    cumsum = np.cumsum(np.insert(energy, 0, 0))
    window_energy = cumsum[win:] - cumsum[:-win]
    best_start_sample = int(np.argmax(window_energy))
    start = best_start_sample / sr
    return {"start": round(start, 2), "end": round(min(start + hook_len, duration), 2)}
 
 
def detect_silences(video_path, noise_db=-30, min_silence=0.5):
    """Runs ffmpeg's silencedetect and returns the list of NON-silent
    {start,end} ranges — i.e. the segments worth keeping. This powers
    automatic 'jump-cut' editing (removing dead air/pauses), the other
    trending short-form edit style besides the loudest-window hook-cut."""
    cmd = [FFMPEG_PATH, "-v", "info", "-i", str(video_path),
           "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
    result = _run(cmd)
    log = result.stderr or ""
    starts, ends = [], []
    for line in log.splitlines():
        if "silence_start" in line:
            starts.append(float(line.split("silence_start:")[1].strip().split(" ")[0]))
        elif "silence_end" in line:
            ends.append(float(line.split("silence_end:")[1].strip().split(" ")[0].split("|")[0]))
    info = _probe(video_path)
    duration = info["duration"]
    silences = list(zip(starts, ends[: len(starts)]))
    keep = []
    cursor = 0.0
    for s, e in silences:
        if s > cursor:
            keep.append({"start": round(cursor, 2), "end": round(s, 2)})
        cursor = max(cursor, e)
    if cursor < duration:
        keep.append({"start": round(cursor, 2), "end": round(duration, 2)})
    return [k for k in keep if k["end"] - k["start"] > 0.15]
 
 
def _build_concat_file(video_path, segments, tmp_dir):
    """Writes an ffmpeg concat-demuxer list after trimming each segment,
    so multiple {start,end} ranges can be stitched in a custom, manually
    chosen order (rearrange / hook-cut)."""
    parts = []
    for i, seg in enumerate(segments):
        out = tmp_dir / f"seg_{i}.mp4"
        cmd = [
            FFMPEG_PATH, "-y", "-v", "error",
            "-ss", str(seg["start"]), "-to", str(seg["end"]),
            "-i", str(video_path),
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            str(out),
        ]
        _run(cmd)
        parts.append(out)
    list_path = tmp_dir / "concat_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p.as_posix()}'\n")
    return list_path
 
 
def _measure_loudnorm(path):
    """First pass of two-pass EBU R128 loudness normalization: measures the
    input's actual loudness/true-peak/range so the second (real encode)
    pass can normalize with `linear=true` against real measured values
    instead of loudnorm's single-pass dynamic estimate — noticeably more
    accurate and avoids the pumping/gain-riding artifacts single-pass mode
    can introduce on speech-heavy short-form video."""
    cmd = [FFMPEG_PATH, "-v", "info", "-i", str(path),
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
           "-f", "null", "-"]
    result = _run(cmd)
    log = result.stderr or ""
    try:
        start = log.rindex("{")
        end = log.rindex("}") + 1
        return json.loads(log[start:end])
    except (ValueError, json.JSONDecodeError):
        return None  # falls back to single-pass below
 
 
# ══════════════════════════════ render pipeline ══════════════════════════════
 
def _render_one_ratio(job, source_path, ratio_key, track_points, src_w, src_h,
                       words, tmp_dir, loud_measured=None):
    cfg = job["config"]
    ratio = RATIO_PRESETS[ratio_key]
    is_auto = (ratio_key == "auto")
    # "auto" keeps the SOURCE's own dimensions — no target size is imposed.
    target_w, target_h = (src_w, src_h) if is_auto else (ratio["w"], ratio["h"])
    quality = QUALITY_PRESETS.get(cfg.get("quality", "high"), QUALITY_PRESETS["high"])
 
    fill_mode = cfg.get("fill_mode", "crop")  # "crop" (default) or "blur_pad"
    # "auto" never crops or letterboxes — it IS the full original frame —
    # so blur_pad/crop fill-mode choices simply don't apply to it.
    if is_auto:
        fill_mode = "none"
 
    # ── subtitles: build the .ass first so both fill-mode branches can use it ──
    ass_path = None
    if cfg["features"].get("subtitles") and words:
        ass_content = build_ass(words, cfg.get("caption_style", "bold_pop"), target_w, target_h)
        ass_path = tmp_dir / f"subs_{ratio_key.replace(':', 'x')}.ass"
        ass_path.write_text(ass_content, encoding="utf-8")
        ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
 
    if is_auto and not ass_path and not cfg.get("manual_crop_keyframes"):
        # Nothing needs to touch the picture at all: fastest AND highest
        # possible quality path is a pure stream-copy remux (no re-encode
        # -> byte-for-byte original video quality; only trimmed/normalized
        # audio may differ). This is what "zero quality loss" means here.
        out_name = f"{job['job_id']}_{ratio_key.replace(':', 'x')}.mp4"
        out_path = EDIT_DIR / out_name
        if cfg.get("normalize_audio", True):
            measured = loud_measured
            audio_filter = (
                "loudnorm=I=-14:TP=-1.5:LRA=11:"
                f"measured_I={measured.get('input_i', -14)}:"
                f"measured_TP={measured.get('input_tp', -1.5)}:"
                f"measured_LRA={measured.get('input_lra', 11)}:"
                f"measured_thresh={measured.get('input_thresh', -24)}:"
                "linear=true:print_format=summary"
            ) if measured else "loudnorm=I=-14:TP=-1.5:LRA=11"
            cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path),
                   "-c:v", "copy", "-af", audio_filter, "-c:a", "aac", "-b:a", "192k",
                   "-movflags", "+faststart", str(out_path)]
        else:
            cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path),
                   "-c", "copy", "-movflags", "+faststart", str(out_path)]
        result = _run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {ratio_key}: {result.stderr[-800:]}")
        return out_name
 
    if fill_mode == "blur_pad":
        # "Advanced" reframe mode: instead of cropping content away, fit the
        # WHOLE frame inside the target canvas and fill the leftover bars
        # with a blurred, zoomed copy of the same frame — the look used by
        # most professional auto-reframe tools when a hard crop would cut
        # off important content. No face tracking needed for this branch,
        # since nothing is being cropped out.
        fc = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h},gblur=sigma=25,eq=brightness=-0.05[bg2];"
            f"[fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg2];"
            f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2:format=auto[base]"
        )
        if ass_path:
            fc += f",[base]ass='{ass_escaped}'[vout]"
            map_v = "[vout]"
        else:
            fc += "[vout]"
            map_v = "[vout]"
        vf_args = ["-filter_complex", fc, "-map", map_v, "-map", "0:a?"]
    elif is_auto:
        # Captions and/or a manual crop keyframe are present, so a re-encode
        # can't be avoided — but the frame itself is never cropped or
        # rescaled; it stays at the source's own resolution. Quality is set
        # one notch above "high" (crf 16, effectively visually lossless)
        # specifically for this path so adding captions never visibly
        # degrades the original picture.
        filters = ["setsar=1"]
        if ass_path:
            filters.append(f"ass='{ass_escaped}'")
        vf_args = ["-vf", ",".join(filters)]
        quality = {"crf": 16, "preset": quality["preset"]}
    else:
        filters = []
        # ── crop (face-tracked or manual) ──
        if cfg["features"].get("face_tracking") or cfg.get("manual_crop_keyframes"):
            crop_w, crop_h = _crop_size_for_ratio(src_w, src_h, target_w, target_h)
            if cfg.get("manual_crop_keyframes"):
                kfs = _simplify_keyframes(sorted(cfg["manual_crop_keyframes"], key=lambda k: k["t"]))
            else:
                kfs = _build_crop_keyframes(track_points, src_w, src_h, crop_w, crop_h)
            x_expr = _expr_chain(kfs, "x")
            y_expr = _expr_chain(kfs, "y")
            filters.append(f"crop={crop_w}:{crop_h}:'{x_expr}':'{y_expr}'")
        else:
            # no face tracking requested -> simple centered crop to the target ratio
            crop_w, crop_h = _crop_size_for_ratio(src_w, src_h, target_w, target_h)
            filters.append(f"crop={crop_w}:{crop_h}:(in_w-out_w)/2:(in_h-out_h)/2")
 
        filters.append(f"scale={target_w}:{target_h}:flags=lanczos")
        filters.append("setsar=1")
        if ass_path:
            filters.append(f"ass='{ass_escaped}'")
        vf_args = ["-vf", ",".join(filters)]
 
    out_name = f"{job['job_id']}_{ratio_key.replace(':', 'x')}.mp4"
    out_path = EDIT_DIR / out_name
 
    # Loudness normalization (EBU R128, -14 LUFS is the standard target for
    # social platforms) so exported audio doesn't sound quiet/inconsistent
    # next to other content in-feed. On "high" quality we do a real
    # measure-then-normalize TWO-PASS pass for accuracy; cheaper qualities
    # use fast single-pass so exports stay quick.
    audio_filters = []
    if cfg.get("normalize_audio", True):
        measured = loud_measured
        if measured:
            audio_filters = [
                "loudnorm=I=-14:TP=-1.5:LRA=11:"
                f"measured_I={measured.get('input_i', -14)}:"
                f"measured_TP={measured.get('input_tp', -1.5)}:"
                f"measured_LRA={measured.get('input_lra', 11)}:"
                f"measured_thresh={measured.get('input_thresh', -24)}:"
                "linear=true:print_format=summary"
            ]
        else:
            audio_filters = ["loudnorm=I=-14:TP=-1.5:LRA=11"]
 
    cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path), *vf_args]
    if audio_filters:
        cmd += ["-af", ",".join(audio_filters)]
    cmd += [
        "-c:v", "libx264", "-crf", str(quality["crf"]), "-preset", quality["preset"],
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {ratio_key}: {result.stderr[-800:]}")
    return out_name
 
 
def _run_render_job(job_id):
    job = EDIT_JOBS[job_id]
    cfg = job["config"]
    job.setdefault("logs", [])
    log = lambda msg, **kw: _job_log(job, msg, **kw)
    tmp_dir = EDIT_DIR / f"tmp_{job_id}"
    tmp_dir.mkdir(exist_ok=True)
 
    try:
        original_source_path = Path(cfg["source_path"])
        source_path = original_source_path
        if not source_path.exists():
            raise RuntimeError("Source video not found")
        log(f"Starting render of {len(cfg.get('export_ratios') or ['9:16'])} ratio(s)...", stage="segments", percent=2)
 
        # ── 1. hook-cut / manual rearrange: trim+stitch segments first ──
        segments = cfg.get("segments")
        want_jump = cfg["features"].get("jump_cut")
        want_hook = cfg["features"].get("hook_cut")
        if not segments and (want_jump or want_hook):
            log("Auto-detecting hook/silence cuts...", percent=5)
            info = _probe(source_path)
            jump_segments = detect_silences(source_path) if want_jump else None
            if want_hook:
                hook = auto_hook_window(source_path, info["duration"], cfg.get("hook_len", 6.0))
                # Cold-open on the loudest window, then play the rest of the
                # clip — dead-air-trimmed if jump_cut is also on, otherwise
                # the full original timeline. Combining both no longer means
                # jump_cut silently wins.
                rest = jump_segments if jump_segments else [{"start": 0.0, "end": info["duration"]}]
                segments = [hook] + rest
            else:
                segments = jump_segments
        if segments:
            log(f"Trimming/stitching {len(segments)} segment(s)...", percent=8)
            list_path = _build_concat_file(source_path, segments, tmp_dir)
            stitched = tmp_dir / "stitched.mp4"
            _run([FFMPEG_PATH, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                  "-i", str(list_path), "-c", "copy", str(stitched)])
            source_path = stitched
        job["percent"] = 10
 
        info = _probe(source_path)
        src_w, src_h = info["width"], info["height"]
 
        # ── 2. face tracking (once, reused for every export ratio) ──
        job["stage"] = "face_tracking"
        track_points = []
        if cfg["features"].get("face_tracking"):
            reused = cfg.get("face_track")
            # Analyze already scanned the whole clip once for this exact
            # source. Reuse those points instead of scanning again — UNLESS
            # segments/hook/jump-cut trimming just rebuilt source_path into a
            # stitched file above, in which case the old timestamps no longer
            # line up with the new timeline and a fresh scan is required.
            if reused and source_path == original_source_path:
                track_points = reused
                src_w = cfg.get("face_track_src_w") or src_w
                src_h = cfg.get("face_track_src_h") or src_h
                job["face_tracking_mode"] = cfg.get("face_tracking_mode", "reused from analyze")
                log(f"Reusing {len(track_points)} face-track point(s) from analyze — no second full scan needed.", percent=40)
            else:
                log("Face-tracking the video...", percent=10)
                def cb(pct):
                    job["percent"] = 10 + int(pct * 0.3)
                track_points, src_w, src_h, advanced = analyze_face_track(
                    source_path, sample_fps=cfg.get("track_fps", 2.0), progress_cb=cb
                )
                job["face_tracking_mode"] = "advanced (mediapipe)" if advanced else "standard (haar cascade)"
                log(f"Face tracking done ({job['face_tracking_mode']}).", percent=40)
        job["percent"] = 40
 
        # ── 3. subtitles (once, reused for every export ratio) — from real
        # platform captions (yt-dlp), never Whisper. Captions are only used
        # when NO trim/stitch happened to source_path, since caption
        # timestamps are relative to the ORIGINAL full video and would be
        # misaligned against a stitched/trimmed timeline. ──
        job["stage"] = "subtitles"
        words = []
        if cfg["features"].get("subtitles"):
            if source_path != original_source_path:
                log("Skipping captions for this render: hook-cut/jump-cut trimmed the "
                    "timeline, and platform caption timestamps would no longer line up.", percent=55)
            elif cfg.get("words"):
                words = cfg["words"]
                log(f"Using {len(words)} caption word(s) from analyze — no second fetch needed.", percent=55)
            else:
                log("Fetching captions for subtitles...", percent=45)
                transcript = get_transcript_words(original_source_path, log_cb=log)
                words = (transcript or {}).get("words") or []
                if words:
                    log(f"Using {len(words)} caption word(s) for subtitles.", percent=55)
                else:
                    log("No captions available for this video — rendering without subtitles.", percent=55)
        job["percent"] = 60
 
        # ── 4. render each requested aspect ratio as one final file ──
        job["stage"] = "render"
        ratios = cfg.get("export_ratios") or ["9:16"]
        job["ratios"] = {}
        n = len(ratios)
 
        # measured once (not per-ratio, source audio is identical across
        # ratios) and only for "high" quality, where the extra ffmpeg pass
        # is worth the accuracy; cheaper qualities use fast single-pass.
        loud_measured = None
        if cfg.get("normalize_audio", True) and cfg.get("quality", "high") == "high":
            log("Measuring loudness for normalization...", percent=58)
            loud_measured = _measure_loudnorm(source_path)
 
        for i, ratio_key in enumerate(ratios):
            if ratio_key not in RATIO_PRESETS:
                continue
            log(f"Rendering {ratio_key} ({i+1}/{n})...", percent=60 + int(i * 40 / max(1, n)))
            out_name = _render_one_ratio(job, source_path, ratio_key, track_points,
                                          src_w, src_h, words, tmp_dir, loud_measured)
            job["ratios"][ratio_key] = out_name
            job["percent"] = 60 + int((i + 1) * 40 / max(1, n))
            log(f"{ratio_key} done.")
 
        job["status"] = "done"
        job["stage"] = "done"
        job["percent"] = 100
        log("Render complete.")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        log(f"Render failed: {e}", stage="error")

    finally:
        # scratch files (trimmed segments, stitched source, .ass files) —
        # keep only the final per-ratio outputs in EDIT_DIR
        try:
            for f in tmp_dir.glob("*"):
                f.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            pass
 
 
# ══════════════════════════════ routes ══════════════════════════════
 
def _run_analyze_job(job_id):
    job = ANALYZE_JOBS[job_id]
    cfg = job["config"]
    source_path = Path(cfg["source_path"])
    log = lambda msg, **kw: _job_log(job, msg, **kw)

    try:
        log("Reading video info (ffprobe)…", percent=2, stage="probe")
        info = _probe(source_path)
        result = {"probe": info, "features_available": feature_status()}
        log(f"Source is {info.get('width')}x{info.get('height')}, "
            f"{info.get('duration', 0):.1f}s.", percent=8)

        if cfg.get("face_tracking", True) and _HAS_CV2:
            log("Face-tracking the video (this scans the whole clip)…", stage="face_tracking", percent=12)
            try:
                points, w, h, advanced = analyze_face_track(source_path, sample_fps=cfg.get("track_fps", 2.0))
                result["face_track"] = points
                result["face_tracking_mode"] = "advanced" if advanced else "standard"
                result["source_size"] = {"w": w, "h": h}
                log(f"Face tracking done — {len(points)} sample point(s), "
                    f"{'advanced (mediapipe)' if advanced else 'standard (haar cascade)'} mode.", percent=40)
            except Exception as e:
                result["face_track"] = []
                result["face_tracking_error"] = str(e)
                log(f"Face tracking failed: {e} — continuing without it.", percent=40)
        else:
            result["face_track"] = []

        words, ai_clips = [], []
        if cfg.get("subtitles", True):
            log("Looking for real captions on the source platform (yt-dlp)…", stage="captions", percent=45)
            transcript = get_transcript_words(source_path, log_cb=log)
            if transcript and transcript.get("words"):
                words = transcript["words"]
                result["chapters"] = transcript.get("chapters") or []
                result["caption_source"] = transcript.get("source_kind")
                log(f"{len(words)} caption word(s) ready — Whisper was not used.", percent=55)
            elif transcript is not None:
                result["subtitle_note"] = "No English captions on the source for this video."
                log("No captions available for this video (Whisper is disabled by design) — "
                    "subtitles/AI clip-finder skipped, face-track + silence-based editing still work.", percent=55)
            else:
                result["subtitle_note"] = "This video has no linked URL (PC upload/library file) — no caption source."
                log("No source URL for this file, so no captions to pull (Whisper is disabled by design).", percent=55)
            result["words"] = words

            if words:
                log("Sending transcript to AI for viral-clip analysis…", stage="ai_clips", percent=60)
                title = SOURCE_TITLE_MAP.get(str(source_path))
                ai_clips = find_viral_clips(words, video_title=title, log_cb=log)
                result["ai_clips"] = ai_clips
                log(f"AI clip-finder returned {len(ai_clips)} ranked clip(s).", percent=78)
            else:
                result["ai_clips"] = []

        if cfg.get("hook_cut", False) and _HAS_NUMPY:
            log("Scanning loudness for a cold-open hook window…", stage="hook_cut", percent=85)
            try:
                result["suggested_hook"] = auto_hook_window(source_path, info["duration"], cfg.get("hook_len", 6.0))
            except Exception as e:
                result["hook_cut_error"] = str(e)
                log(f"Hook-cut detection failed: {e}")

        if cfg.get("jump_cut", False):
            log("Detecting dead-air / silence to auto-trim…", stage="jump_cut", percent=92)
            try:
                result["suggested_segments"] = detect_silences(source_path)
            except Exception as e:
                result["jump_cut_error"] = str(e)
                log(f"Silence detection failed: {e}")

        job["result"] = result
        job["status"] = "done"
        log("Analysis complete.", percent=100, stage="done")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        log(f"Analysis failed: {e}", stage="error")


@editor_bp.route("/api/editor/analyze/start", methods=["POST"])
def api_editor_analyze_start():
    """Kicks off analysis (face-track + real captions + AI viral-clip
    finder + hook/silence detection) as a background job instead of
    blocking the request, so the frontend can show a live progress bar and
    a scrolling log console instead of the page just sitting there with no
    feedback. Poll /api/editor/analyze/progress/<job_id> for updates."""
    data = request.json or {}
    source_path = Path(data.get("source_path", ""))
    if not source_path.exists():
        return jsonify({"error": "source_path not found"}), 400
    job_id = uuid.uuid4().hex[:10]
    ANALYZE_JOBS[job_id] = {
        "status": "running", "stage": "queued", "percent": 0, "error": None,
        "logs": [], "result": None,
        "config": {
            "source_path": str(source_path),
            "face_tracking": bool(data.get("face_tracking", True)),
            "subtitles": bool(data.get("subtitles", True)),
            "hook_cut": bool(data.get("hook_cut", False)),
            "jump_cut": bool(data.get("jump_cut", False)),
            "track_fps": data.get("track_fps", 2.0),
            "hook_len": data.get("hook_len", 6.0),
        },
    }
    threading.Thread(target=_run_analyze_job, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id})


@editor_bp.route("/api/editor/analyze/progress/<job_id>")
def api_editor_analyze_progress(job_id):
    job = ANALYZE_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown analyze job"}), 404
    return jsonify({k: v for k, v in job.items() if k != "config"})
 
 
@editor_bp.route("/api/editor/upload", methods=["POST"])
def api_editor_upload():
    """Lets the frontend upload a local file directly instead of only
    pointing at a path already on the server (e.g. a downloader.py output)."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file uploaded"}), 400
    safe_name = f"{uuid.uuid4().hex[:10]}_{Path(f.filename).name}"
    dest = EDIT_DIR / "uploads"
    dest.mkdir(exist_ok=True)
    path = dest / safe_name
    f.save(path)
    return jsonify({"source_path": str(path)})
 
 
# ══════════════════════════════ fetch-from-link (built-in yt-dlp) ══════════════════════════════
# A third source alongside "upload from PC" and "pick from downloader" —
# paste a link right here in the editor. Deliberately skips the full
# format-picker: it always resolves ONE single, best-available video+audio
# mp4 (never a video-only stream that would need a silent-audio placeholder,
# never separate files) and hands that straight to analyze/render.
#
# Speed trick mirrors downloader.py: remembers whichever auth mode (cookies
# file / browser cookies / plain / bypass) last worked and tries that first,
# so repeat fetches stay fast. If downloader.py is also registered in this
# process, its already-resolved mode is reused immediately instead of
# re-discovering it from scratch.
 
FETCH_JOBS = {}  # fetch_id -> {status, stage, percent, error, source_path, url}
_FETCH_RESOLVED_MODE = {"mode": None}
_FETCH_COOKIE_BROWSERS = ["chrome", "edge", "firefox", "brave"]
 
 
def _fetch_no_console_kwargs():
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
 
 
def _fetch_auth_attempts():
    attempts = []
    # reuse downloader.py's already-known-good mode first, if it's loaded
    try:
        import downloader
        if downloader._RESOLVED_MODE.get("mode"):
            attempts.append(downloader._RESOLVED_MODE["mode"])
    except Exception:
        pass
    if _FETCH_RESOLVED_MODE["mode"]:
        attempts.append(_FETCH_RESOLVED_MODE["mode"])
    cookies_file = SRC_DIR.parent / "cookies.txt" if SRC_DIR else None
    rest = []
    if cookies_file and cookies_file.exists():
        rest.append("cookies_file")
    rest.extend(_FETCH_COOKIE_BROWSERS)
    rest.append("default")
    rest.append("bypass")
    for m in rest:
        if m not in attempts:
            attempts.append(m)
    return attempts, cookies_file
 
 
def _fetch_apply_auth(opts, mode, cookies_file):
    opts = dict(opts)
    if mode == "cookies_file" and cookies_file:
        opts["cookiefile"] = str(cookies_file)
    elif mode in _FETCH_COOKIE_BROWSERS:
        opts["cookiesfrombrowser"] = (mode,)
    elif mode == "bypass":
        opts["extractor_args"] = {"youtube": {"player_client": ["ios", "android", "mweb"]}}
    return opts
 
 
def _fetch_extract(url, extra_opts=None, download=False):
    base_opts = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "socket_timeout": 15, "retries": 2, "extractor_retries": 1, "geo_bypass": True,
    }
    if FFMPEG_PATH:
        base_opts["ffmpeg_location"] = str(FFMPEG_PATH)
    if extra_opts:
        base_opts.update(extra_opts)
 
    attempts, cookies_file = _fetch_auth_attempts()
    last_err = None
    for mode in attempts:
        opts = _fetch_apply_auth(base_opts, mode, cookies_file)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=download)
            _FETCH_RESOLVED_MODE["mode"] = mode
            return info, ydl if download else None, None
        except Exception as e:
            last_err = e
            continue
    return None, None, last_err
 
 
def _fetch_fmt_duration(sec):
    if sec is None:
        return None
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
 
 
def _fetch_fmt_int(n):
    if n is None:
        return None
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)
 
 
def _fetch_fmt_upload_date(d):
    if not d:
        return None
    try:
        from datetime import datetime as _dt
        return _dt.strptime(str(d), "%Y%m%d").strftime("%d %b %Y")
    except ValueError:
        return str(d)
 
 
@editor_bp.route("/api/editor/fetch/info", methods=["POST"])
def api_fetch_info():
    """Resolves a link (YouTube/Instagram/TikTok/X/Facebook/etc.) and hands
    back the FULL set of metadata yt-dlp/the source platform already knows
    about it — title, uploader, view/like counts, upload date, best
    available resolution — so the UI never has to re-derive any of this by
    hand, and so a video that started life as a fetched link can show that
    rich context in the analysis panel too (a plain PC upload naturally
    won't have this — it only gets the ffprobe technical details)."""
    if not _HAS_YTDLP:
        return jsonify({"error": "yt-dlp is not installed on the server — pip install yt-dlp"}), 400
    data = request.json or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Paste a video URL first"}), 400
    info, _, err = _fetch_extract(url, extra_opts={"format": "bestvideo+bestaudio/best"})
    if info is None:
        return jsonify({"error": f"Could not resolve that link: {err}"}), 400
 
    best_h = best_w = None
    for f in info.get("formats", []) or []:
        if f.get("vcodec") not in (None, "none") and f.get("height"):
            if not best_h or f["height"] > best_h:
                best_h, best_w = f["height"], f.get("width")
 
    # Caption/chapter PRESENCE only (no vtt download here — keep this
    # endpoint fast) proves this doesn't need the video downloaded at all;
    # yt-dlp already knows this from the same resolve call above.
    chapters = info.get("chapters") or []
    subs, auto = info.get("subtitles") or {}, info.get("automatic_captions") or {}
    caption_langs = [l for l in ("en", "en-US", "en-GB", "en-orig") if l in subs or l in auto]
    caption_kind = "manual" if any(l in subs for l in caption_langs) else ("auto-generated" if caption_langs else None)

    return jsonify({
        "title": info.get("title"),
        "uploader": info.get("uploader") or info.get("channel"),
        "duration": info.get("duration"),
        "duration_str": _fetch_fmt_duration(info.get("duration")),
        "thumbnail": info.get("thumbnail"),
        "view_count": info.get("view_count"),
        "view_count_str": _fetch_fmt_int(info.get("view_count")),
        "like_count": info.get("like_count"),
        "like_count_str": _fetch_fmt_int(info.get("like_count")),
        "upload_date": info.get("upload_date"),
        "upload_date_str": _fetch_fmt_upload_date(info.get("upload_date")),
        "resolution": f"{best_w}x{best_h}" if best_w and best_h else None,
        "extractor": info.get("extractor_key"),
        "webpage_url": info.get("webpage_url"),
        "has_captions": bool(caption_langs),
        "caption_kind": caption_kind,
        "chapters_count": len(chapters),
    })
 
 
def _run_fetch_job(fetch_id, url):
    job = FETCH_JOBS[fetch_id]
    job["stage"] = "connect"
 
    def hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes") or 0
            speed = d.get("speed")
            job.update({
                "status": "downloading", "stage": "download",
                "percent": round(downloaded * 100 / total, 1) if total else None,
                "speed_str": (_fmt_size(speed) + "/s") if speed else None,
            })
        elif d.get("status") == "finished":
            job["status"] = "processing"
            job["stage"] = "merge"
 
    # Saved into SRC_DIR (the SAME "downloads" folder downloader.py uses) so
    # a link fetched here shows up in /api/editor/sources too, and nothing
    # about the analyze/render pipeline needs to know it came from here
    # instead of the Downloader tab or a PC upload.
    extra_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": str(SRC_DIR / f"editorfetch_{fetch_id}_%(title).60s.%(ext)s"),
        "progress_hooks": [hook],
        "merge_output_format": "mp4",   # always ONE muxed video+audio file
    }
 
    orig_popen = subprocess.Popen
    def quiet_popen(*args, **kwargs):
        kwargs.update(_fetch_no_console_kwargs())
        return orig_popen(*args, **kwargs)
    subprocess.Popen = quiet_popen
    try:
        info, ydl, err = _fetch_extract(url, extra_opts=extra_opts, download=True)
    finally:
        subprocess.Popen = orig_popen
 
    if info is None:
        job["status"] = "error"
        job["error"] = f"Fetch failed: {err}"
        return
    try:
        fname = ydl.prepare_filename(info)
        p = Path(fname)
        if not p.exists():
            p2 = p.with_suffix(".mp4")
            if p2.exists():
                p = p2
        job["source_path"] = str(p.resolve())
        SOURCE_URL_MAP[job["source_path"]] = url
        SOURCE_TITLE_MAP[job["source_path"]] = info.get("title")
        # Transcript metadata (captions/chapters) is already sitting inside
        # this SAME `info` dict yt-dlp just returned — no need to ever ask
        # the source platform for it again. Caching it here means analyze
        # later hits TRANSCRIPT_CACHE instantly instead of making a second,
        # back-to-back request that's exactly what triggers a 429.
        try:
            transcript = _transcript_from_ytdlp_info(info)
        except Exception:
            transcript = None
        TRANSCRIPT_CACHE[job["source_path"]] = transcript
        job["transcript_summary"] = {
            "has_captions": bool(transcript and transcript.get("words")),
            "caption_source": (transcript or {}).get("source_kind"),
            "words_count": len(transcript["words"]) if transcript else 0,
            "chapters_count": len((transcript or {}).get("chapters") or []),
        }
        job["status"] = "done"
        job["stage"] = "done"
        job["percent"] = 100
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"Could not finalize file: {e}"
 
 
@editor_bp.route("/api/editor/fetch/start", methods=["POST"])
def api_fetch_start():
    if not _HAS_YTDLP:
        return jsonify({"error": "yt-dlp is not installed on the server — pip install yt-dlp"}), 400
    data = request.json or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL given"}), 400
    fetch_id = uuid.uuid4().hex[:10]
    FETCH_JOBS[fetch_id] = {
        "status": "starting", "stage": "connect", "percent": 0,
        "speed_str": None, "error": None, "url": url, "source_path": None,
    }
    threading.Thread(target=_run_fetch_job, args=(fetch_id, url), daemon=True).start()
    return jsonify({"fetch_id": fetch_id})
 
 
@editor_bp.route("/api/editor/fetch/progress/<fetch_id>")
def api_fetch_progress(fetch_id):
    job = FETCH_JOBS.get(fetch_id)
    if not job:
        return jsonify({"error": "Unknown fetch job"}), 404
    return jsonify(job)
 
 
@editor_bp.route("/api/editor/render", methods=["POST"])
def api_editor_render():
    data = request.json or {}
    source_path = data.get("source_path", "")
    if not source_path or not Path(source_path).exists():
        return jsonify({"error": "source_path not found"}), 400
 
    export_ratios = data.get("export_ratios") or ["9:16"]
    bad = [r for r in export_ratios if r not in RATIO_PRESETS]
    if bad:
        return jsonify({"error": f"Unknown ratio(s): {bad}. Valid: {list(RATIO_PRESETS)}"}), 400
 
    features = {
        "face_tracking": bool(data.get("face_tracking", True)),
        "subtitles": bool(data.get("subtitles", True)),
        "hook_cut": bool(data.get("hook_cut", False)),
        "jump_cut": bool(data.get("jump_cut", False)),
    }
    if features["face_tracking"] and not _HAS_CV2:
        return jsonify({"error": "Face tracking requested but opencv-python is not installed"}), 400
    # Subtitles are no longer a hard requirement on anything being installed —
    # they simply depend on the source having real platform captions
    # available (see get_transcript_words); if not, rendering proceeds
    # without subtitles instead of failing the whole request.
    if features["hook_cut"] and not data.get("segments") and not _HAS_NUMPY:
        return jsonify({"error": "Auto hook-cut requested but numpy is not installed"}), 400
 
    job_id = uuid.uuid4().hex[:10]
    EDIT_JOBS[job_id] = {
        "job_id": job_id, "status": "starting", "stage": "queued", "percent": 0,
        "error": None, "ratios": {},
        "config": {
            "source_path": source_path,
            "export_ratios": export_ratios,
            "features": features,
            "caption_style": data.get("caption_style", "bold_pop"),
            "quality": data.get("quality", "high"),
            "track_fps": data.get("track_fps", 2.0),
            "face_track": data.get("face_track"),               # reuse analyze's scan if timeline is unchanged
            "face_track_src_w": data.get("face_track_src_w"),
            "face_track_src_h": data.get("face_track_src_h"),
            "face_tracking_mode": data.get("face_tracking_mode"),
            "words": data.get("words") or [],                    # transcript words gathered during analyze
            "language": data.get("language"),
            "hook_len": data.get("hook_len", 6.0),
            "manual_crop_keyframes": data.get("manual_crop_keyframes"),  # [{t,x,y}]
            "segments": data.get("segments"),  # [{start,end}] manual rearrange/trim order
            "fill_mode": data.get("fill_mode", "crop"),  # "crop" or "blur_pad"
            "normalize_audio": bool(data.get("normalize_audio", True)),
        },
    }
    threading.Thread(target=_run_render_job, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id})
 
 
@editor_bp.route("/api/editor/progress/<job_id>")
def api_editor_progress(job_id):
    job = EDIT_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown edit job"}), 404
    return jsonify({k: v for k, v in job.items() if k != "config"} | {
        "features": job["config"]["features"], "export_ratios": job["config"]["export_ratios"]
    })
 
 
@editor_bp.route("/api/editor/file/<job_id>/<ratio>")
def api_editor_file(job_id, ratio):
    job = EDIT_JOBS.get(job_id)
    if not job or job.get("status") != "done":
        return "Not ready", 404
    fname = job["ratios"].get(ratio)
    if not fname:
        return "Ratio not found for this job", 404
    p = EDIT_DIR / fname
    if not p.exists():
        return "Not found", 404
    return send_file(p, as_attachment=True, download_name=fname)
 
 
# ── manual crop keyframe add/remove (per not-yet-rendered job config) ──
 
@editor_bp.route("/api/editor/keyframe/add", methods=["POST"])
def api_keyframe_add():
    data = request.json or {}
    job = EDIT_JOBS.get(data.get("job_id"))
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    kf = {"t": float(data["t"]), "x": int(data["x"]), "y": int(data["y"])}
    job["config"].setdefault("manual_crop_keyframes", [])
    job["config"]["manual_crop_keyframes"] = job["config"]["manual_crop_keyframes"] or []
    job["config"]["manual_crop_keyframes"].append(kf)
    job["config"]["manual_crop_keyframes"].sort(key=lambda k: k["t"])
    return jsonify({"manual_crop_keyframes": job["config"]["manual_crop_keyframes"]})
 
 
@editor_bp.route("/api/editor/keyframe/remove", methods=["POST"])
def api_keyframe_remove():
    data = request.json or {}
    job = EDIT_JOBS.get(data.get("job_id"))
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    t = float(data["t"])
    kfs = job["config"].get("manual_crop_keyframes") or []
    job["config"]["manual_crop_keyframes"] = [k for k in kfs if abs(k["t"] - t) > 1e-6]
    return jsonify({"manual_crop_keyframes": job["config"]["manual_crop_keyframes"]})
 
 
# ── hook-cut / rearrange segment add/remove ──
 
@editor_bp.route("/api/editor/segment/add", methods=["POST"])
def api_segment_add():
    data = request.json or {}
    job = EDIT_JOBS.get(data.get("job_id"))
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    seg = {"start": float(data["start"]), "end": float(data["end"])}
    index = data.get("index")
    segs = job["config"].get("segments") or []
    if index is None or index >= len(segs):
        segs.append(seg)
    else:
        segs.insert(int(index), seg)
    job["config"]["segments"] = segs
    return jsonify({"segments": segs})
 
 
@editor_bp.route("/api/editor/segment/remove", methods=["POST"])
def api_segment_remove():
    data = request.json or {}
    job = EDIT_JOBS.get(data.get("job_id"))
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    index = int(data["index"])
    segs = job["config"].get("segments") or []
    if 0 <= index < len(segs):
        segs.pop(index)
    job["config"]["segments"] = segs
    return jsonify({"segments": segs})








# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# AutoShortAi — Video Editor module
# ──────────────────────────────────
# Adds AI auto-reframe (face tracking), trending animated captions,
# hook-cut / manual clip reordering, and one-click multi-aspect-ratio
# export — all producing a SINGLE muxed video+audio file per ratio.
 
# Wire it into the main app with:
 
#     from video_editor import editor_bp, init_editor
#     init_editor(BASE, FFMPEG)
#     app.register_blueprint(editor_bp)
 
# Routes exposed (all under /api/editor/...):
#     GET  /api/editor/editor             -> serves the standalone editor.html UI (open this in a tab)
#     POST /api/editor/upload             -> upload a local file, returns a source_path to use below
#     POST /api/editor/analyze            -> face-track + transcript + silence/hook preview (for the UI)
#     GET  /api/editor/styles             -> list of caption style presets
#     GET  /api/editor/ratios             -> list of export aspect-ratio presets
#     GET  /api/editor/features           -> which advanced features are available (installed libs)
#     POST /api/editor/render             -> kicks off a background render job
#     GET  /api/editor/progress/<job_id>  -> live progress
#     GET  /api/editor/file/<job_id>/<ratio> -> serves one finished ratio's file
#     POST /api/editor/keyframe/add       -> add a manual crop keyframe to a job's override track
#     POST /api/editor/keyframe/remove    -> remove a manual crop keyframe
#     POST /api/editor/segment/add        -> add a hook-cut / reorder segment
#     POST /api/editor/segment/remove     -> remove a hook-cut / reorder segment
 
# ADVANCED EXTRAS (on top of the original spec):
#     • fill_mode="blur_pad"   -> reframe by fitting the WHOLE frame + blurred
#       background bars instead of hard-cropping (no content ever cut off).
#     • jump_cut feature       -> auto-detects and removes silence/dead-air,
#       the other half of "trending" short-form editing besides hook-cut.
#     • hook_cut + jump_cut COMBINED -> if both are on, the loudest window is
#       used as the cold-open "hook" and is prepended in front of the
#       dead-air-trimmed rest of the video (previously turning jump_cut on
#       silently dropped hook_cut — now they compose).
#     • normalize_audio        -> EBU R128 loudness normalization (-14 LUFS),
#       on by default, matches the loudness social platforms expect.
#     • caption keyword emphasis -> numbers, money, and punchy "power words"
#       auto-highlight in captions even outside full word-by-word styles.
#     • quality/caption_style inputs are now validated with a safe fallback
#       instead of raising a raw KeyError if the frontend sends a typo'd key.
#     • self-serves its own frontend (editor.html) at GET /api/editor/editor,
#       the same "own file, same process/port" pattern as downloader.py.
 
# DEPENDENCIES (install what you want to use — everything degrades gracefully
# if a library is missing, see FEATURE FLAGS AT IMPORT TIME below):
#     pip install opencv-python            # required for any face tracking
#     pip install mediapipe                # optional, gives the "advanced" tracker
#                                           # (falls back to Haar cascade otherwise)
#     pip install faster-whisper           # required for subtitles (word timestamps)
#     pip install numpy
 
# DESIGN NOTES — how each part of the request maps to code:
#   • "advance level face tracking"   -> _FaceTracker (mediapipe BlazeFace if present,
#                                         else OpenCV Haar cascade), EMA-smoothed
#                                         centroid so the crop doesn't jitter frame
#                                         to frame, multi-face -> largest/most-central
#                                         face wins.
#   • "perfect advance subtitles,
#      trending caption styles"       -> CAPTION_STYLES presets + build_ass() which
#                                         groups words into short on-screen chunks and
#                                         emits karaoke-style \\k tags for word-by-word
#                                         pop/highlight animation, burned in with
#                                         ffmpeg's `ass` filter (not a soft-sub track,
#                                         so it always displays correctly everywhere).
#   • "single high quality video+audio
#      in one file"                   -> every render always outputs one .mp4 with
#                                         both streams muxed, CRF-based high quality
#                                         encode (see QUALITY_PRESETS).
#   • "export in all ratio perfectly" -> RATIO_PRESETS, rendered in a loop, one file
#                                         per ratio, each independently croppped using
#                                         the SAME face track (recomputed crop window
#                                         per target aspect).
#   • "rearrange video manually /
#      hook cut"                      -> job["segments"]: an ordered list of
#                                         {start,end} clip ranges. Auto hook-cut picks
#                                         the loudest window as segment 1 automatically;
#                                         manual mode lets the caller add/remove/reorder
#                                         segments directly (segment/add, segment/remove).
#   • "add and remove option for all" -> every feature is a boolean flag in job config
#                                         (features{}) that can be flipped and re-rendered,
#                                         plus explicit add/remove endpoints for the two
#                                         list-based tracks (crop keyframes, segments).
# """
 
# import os
# import re
# import json
# import uuid
# import struct
# import threading
# import subprocess
# import concurrent.futures
# from pathlib import Path
 
# from flask import Blueprint, request, jsonify, send_file
 
# editor_bp = Blueprint("editor_bp", __name__)
 
# # ══════════════════════════════ embedded frontend ══════════════════════════════
# # The whole editor UI lives right here as one string — no separate .html file
# # to ship or lose track of. Served as-is by GET /api/editor/editor. Its fetch
# # calls use relative paths (same-origin), so it works immediately wherever
# # this blueprint is registered.
# EDITOR_HTML = r"""<!doctype html>
# <html lang="en">
# <head>
# <meta charset="utf-8">
# <meta name="viewport" content="width=device-width, initial-scale=1">
# <title>Reframe — AI video editor</title>
# <link rel="preconnect" href="https://fonts.googleapis.com">
# <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
# <style>
#   /* ── Theme — mirrors the main app's palette exactly (same variable
#      VALUES, just different variable NAMES for this page's own CSS). Default
#      below is "dark-violet", the main app's default. The <html data-theme="…">
#      overrides further down are read straight off the SAME localStorage key
#      the main app writes to (see the sync script at the bottom of <body>),
#      so switching themes on Studio/Downloader also re-themes this page —
#      no manual matching needed ever again. ── */
#   :root{
#     --bg:#0b0d12; --panel:#13151c; --panel-2:#191c25; --border:#262a36;
#     --text:#eef0f6; --text-dim:#8a90a4; --text-faint:#8a90a4;
#     --amber:#6e5bff; --teal:#22d3c4; --red:#ff5a6e;
#     --amber-dim: color-mix(in srgb, var(--amber) 22%, #000);
#     --teal-dim: color-mix(in srgb, var(--teal) 22%, #000);
#     --radius:10px;
#     font-size:15px;
#   }
#   html[data-theme="light"]{
#     --bg:#f4f5f9; --panel:#ffffff; --panel-2:#eef0f6; --border:#dde1ea;
#     --text:#171923; --text-dim:#5b6472; --text-faint:#5b6472;
#     --amber:#5b46e0; --teal:#0891b2; --red:#e0324a;
#   }
#   html[data-theme="midnight-blue"]{
#     --bg:#050a16; --panel:#0c1526; --panel-2:#12213a; --border:#1f3358;
#     --text:#e8f1ff; --text-dim:#7f93b8; --text-faint:#7f93b8;
#     --amber:#3b82f6; --teal:#22d3ee; --red:#ff5a6e;
#   }
#   html[data-theme="forest"]{
#     --bg:#0a130d; --panel:#0f1e14; --panel-2:#16301f; --border:#234a30;
#     --text:#eafbea; --text-dim:#86b399; --text-faint:#86b399;
#     --amber:#22c55e; --teal:#a3e635; --red:#ff5a6e;
#   }
#   html[data-theme="sunset"]{
#     --bg:#1a0e12; --panel:#241419; --panel-2:#331d27; --border:#4a2635;
#     --text:#fdf1ee; --text-dim:#cb9aa3; --text-faint:#cb9aa3;
#     --amber:#fb7185; --teal:#f59e0b; --red:#ff5a6e;
#   }
#   html[data-theme="ocean"]{
#     --bg:#061319; --panel:#0b1f27; --panel-2:#123039; --border:#1e4a56;
#     --text:#e6fbff; --text-dim:#7fb8c4; --text-faint:#7fb8c4;
#     --amber:#06b6d4; --teal:#6366f1; --red:#ff5a6e;
#   }
#   html[data-theme="rose"]{
#     --bg:#170f14; --panel:#221720; --panel-2:#2e1e2b; --border:#4a2f45;
#     --text:#fdeef5; --text-dim:#c79ab0; --text-faint:#c79ab0;
#     --amber:#ec4899; --teal:#f472b6; --red:#ff5a6e;
#   }
#   html[data-theme="amber"]{
#     --bg:#160f05; --panel:#221708; --panel-2:#33230d; --border:#4d3416;
#     --text:#fdf4e3; --text-dim:#c9a877; --text-faint:#c9a877;
#     --amber:#f59e0b; --teal:#eab308; --red:#ff5a6e;
#   }
#   html[data-theme="slate"]{
#     --bg:#101215; --panel:#181b1f; --panel-2:#20242a; --border:#333941;
#     --text:#eceef1; --text-dim:#9aa2ad; --text-faint:#9aa2ad;
#     --amber:#64748b; --teal:#94a3b8; --red:#ff5a6e;
#   }
#   html[data-theme="cyberpunk"]{
#     --bg:#0a0014; --panel:#150124; --panel-2:#210235; --border:#4a0a6b;
#     --text:#f5e6ff; --text-dim:#b083d6; --text-faint:#b083d6;
#     --amber:#e91e9c; --teal:#00f0ff; --red:#ff5a6e;
#   }
#   *{box-sizing:border-box;}
#   body{
#     margin:0; background:var(--bg); color:var(--text);
#     font-family:'Inter',sans-serif; min-height:100vh;
#   }
#   h1,h2,h3,.display{font-family:'Space Grotesk',sans-serif;}
#   .mono{font-family:'JetBrains Mono',monospace;}
#   ::selection{background:var(--amber-dim); color:var(--amber);}
 
#   /* ── top bar ── */
#   .topbar{
#     display:flex; align-items:center; gap:16px; padding:14px 22px;
#     border-bottom:1px solid var(--border); background:var(--panel);
#     position:sticky; top:0; z-index:20;
#   }
#   .brand{display:flex; align-items:center; gap:10px;}
#   .brand .dot{width:10px; height:10px; border-radius:50%; background:var(--amber); box-shadow:0 0 10px var(--amber);}
#   .brand h1{font-size:18px; font-weight:700; margin:0; letter-spacing:.2px;}
#   .brand span{color:var(--text-dim); font-size:12px;}
#   .topbar .spacer{flex:1;}
#   .api-input{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text-dim);
#     border-radius:8px; padding:7px 10px; font-size:12px; font-family:'JetBrains Mono',monospace;
#     width:220px;
#   }
#   .api-input:focus{outline:none; border-color:var(--amber); color:var(--text);}
 
#   /* ── layout ── */
#   .app{display:grid; grid-template-columns:1fr 380px; gap:1px; background:var(--border);}
#   @media (max-width:980px){ .app{grid-template-columns:1fr;} }
#   .col{background:var(--bg); padding:22px; min-width:0;}
 
#   /* ── upload zone ── */
#   .dropzone{
#     border:1.5px dashed var(--border); border-radius:var(--radius);
#     padding:40px 20px; text-align:center; cursor:pointer;
#     transition:border-color .15s, background .15s;
#   }
#   .dropzone:hover, .dropzone.drag{border-color:var(--amber); background:rgba(255,176,32,.04);}
#   .dropzone svg{opacity:.5; margin-bottom:10px;}
#   .dropzone .hint{color:var(--text-dim); font-size:13px; margin-top:6px;}
 
#   /* ── fetch-from-link (ytdlp-style resolver, built into the editor) ── */
#   .fetch-section{border:1px solid var(--border); border-radius:var(--radius); padding:12px; background:var(--panel-2); margin-bottom:14px;}
#   .fetch-row{display:flex; gap:8px; margin-top:8px;}
#   .fetch-input{
#     flex:1; background:var(--panel); border:1px solid var(--border); color:var(--text);
#     border-radius:8px; padding:9px 10px; font-size:13px;
#   }
#   .fetch-input:focus{outline:none; border-color:var(--amber);}
#   .btn-accent{background:var(--amber); color:#1a1200; border:none; font-weight:600;}
#   .btn-accent:hover{filter:brightness(1.08);}
#   .fetch-card{
#     display:flex; gap:10px; align-items:center; margin-top:10px;
#     border:1px solid var(--border); border-radius:8px; padding:10px; background:var(--panel);
#   }
#   .fetch-thumb{width:88px; height:56px; object-fit:cover; border-radius:6px; background:#000; flex-shrink:0;}
#   .fetch-meta{flex:1; min-width:0;}
#   .fetch-title{font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
#   .fetch-sub{font-size:11.5px; color:var(--text-dim); margin-top:2px;}
#   .fetch-status{font-size:12px; color:var(--text-dim); margin-top:8px; min-height:16px;}
 
#   /* ── server library (downloader output + past uploads) ── */
#   .or-divider{
#     display:flex; align-items:center; gap:10px; margin:14px 0;
#     color:var(--text-faint); font-size:12px; text-transform:uppercase; letter-spacing:.5px;
#   }
#   .or-divider::before, .or-divider::after{content:""; flex:1; height:1px; background:var(--border);}
#   .lib-section{border:1px solid var(--border); border-radius:var(--radius); padding:12px; background:var(--panel-2);}
#   .lib-header{display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; font-size:13px; color:var(--text-dim);}
#   .lib-grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:8px; max-height:220px; overflow-y:auto;}
#   .lib-empty{color:var(--text-faint); font-size:12.5px; grid-column:1/-1;}
#   .lib-item{
#     border:1px solid var(--border); border-radius:8px; padding:10px; cursor:pointer;
#     background:var(--panel); transition:border-color .15s, transform .1s;
#   }
#   .lib-item:hover{border-color:var(--amber); transform:translateY(-1px);}
#   .lib-item.active{border-color:var(--teal); background:var(--teal-dim);}
#   .lib-item .lib-name{font-size:12.5px; font-weight:600; word-break:break-word; margin-bottom:4px;}
#   .lib-item .lib-meta{font-size:11px; color:var(--text-dim); display:flex; justify-content:space-between;}
#   .lib-item .lib-tag{
#     display:inline-block; font-size:10px; padding:1px 6px; border-radius:20px;
#     background:var(--amber-dim); color:var(--amber); margin-bottom:4px;
#   }
 
#   /* ── preview ── */
#   .preview-wrap{position:relative; display:flex; justify-content:center; margin-bottom:16px;}
#   .preview-frame{position:relative; background:#000; border-radius:8px; overflow:hidden; max-width:100%;}
#   video{display:block; max-width:100%; max-height:60vh;}
#   .crop-box{
#     position:absolute; border:1.5px solid var(--amber); pointer-events:none;
#     box-shadow:0 0 0 2000px rgba(0,0,0,.45);
#   }
#   .crop-box .corner{
#     position:absolute; width:16px; height:16px; border-color:var(--amber); border-style:solid; border-width:0;
#   }
#   .crop-box .tl{top:-1.5px; left:-1.5px; border-top-width:3px; border-left-width:3px;}
#   .crop-box .tr{top:-1.5px; right:-1.5px; border-top-width:3px; border-right-width:3px;}
#   .crop-box .bl{bottom:-1.5px; left:-1.5px; border-bottom-width:3px; border-left-width:3px;}
#   .crop-box .br{bottom:-1.5px; right:-1.5px; border-bottom-width:3px; border-right-width:3px;}
#   .crop-box{cursor:grab;}
#   .crop-box.dragging{cursor:grabbing;}
#   .crop-readout{
#     position:absolute; top:8px; left:8px; background:rgba(0,0,0,.6);
#     color:var(--amber); font-size:11px; padding:3px 7px; border-radius:5px;
#   }
 
#   .preview-controls{display:flex; align-items:center; gap:10px; justify-content:center; margin-bottom:8px;}
#   .btn-icon{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text);
#     width:34px; height:34px; border-radius:50%; cursor:pointer; display:flex;
#     align-items:center; justify-content:center;
#   }
#   .btn-icon:hover{border-color:var(--amber);}
#   .time{font-size:12px; color:var(--text-dim); min-width:96px; text-align:center;}
 
#   /* ── timeline ── */
#   .timeline-panel{background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:16px;}
#   .timeline-panel h3{margin:0 0 4px; font-size:13px; letter-spacing:.4px; text-transform:uppercase; color:var(--text-dim);}
#   .timeline{
#     position:relative; height:56px; background:var(--panel-2); border-radius:6px;
#     margin-top:10px; overflow:hidden; cursor:pointer;
#   }
#   .tl-face{position:absolute; top:0; height:18px; background:linear-gradient(90deg, transparent, var(--teal-dim)); opacity:.6;}
#   .tl-seg{
#     position:absolute; top:20px; height:18px; background:var(--teal); opacity:.85; border-radius:3px;
#     display:flex; align-items:center; justify-content:center; font-size:10px; color:#04231D; font-weight:600;
#     cursor:grab;
#   }
#   .tl-hook{position:absolute; top:38px; height:14px; background:var(--amber); opacity:.9; border-radius:3px;}
#   .tl-playhead{position:absolute; top:0; bottom:0; width:2px; background:var(--red);}
#   .tl-legend{display:flex; gap:16px; margin-top:8px; font-size:11px; color:var(--text-dim);}
#   .tl-legend span{display:inline-flex; align-items:center; gap:5px;}
#   .swatch{width:9px; height:9px; border-radius:2px; display:inline-block;}
 
#   .seg-list{margin-top:10px; display:flex; flex-direction:column; gap:6px;}
#   .seg-row{
#     display:flex; align-items:center; gap:8px; background:var(--panel-2);
#     border:1px solid var(--border); border-radius:7px; padding:6px 10px; font-size:12px;
#   }
#   .seg-row .mono{color:var(--teal);}
#   .seg-row .spacer{flex:1;}
#   .seg-row button{background:none; border:none; color:var(--text-faint); cursor:pointer; font-size:14px;}
#   .seg-row button:hover{color:var(--red);}
#   .btn-small{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text-dim);
#     border-radius:6px; padding:6px 10px; font-size:12px; cursor:pointer;
#   }
#   .btn-small:hover{border-color:var(--amber); color:var(--amber);}
 
#   /* ── side panel ── */
#   .section{margin-bottom:26px;}
#   .section h3{
#     font-size:12px; letter-spacing:.5px; text-transform:uppercase; color:var(--text-dim);
#     margin:0 0 12px; display:flex; align-items:center; gap:8px;
#   }
#   .section h3 .num{
#     width:18px; height:18px; border-radius:50%; background:var(--panel-2); border:1px solid var(--border);
#     display:flex; align-items:center; justify-content:center; font-size:10px; color:var(--amber);
#   }
 
#   .toggle-row{
#     display:flex; align-items:center; justify-content:space-between;
#     padding:10px 12px; background:var(--panel); border:1px solid var(--border);
#     border-radius:8px; margin-bottom:8px;
#   }
#   .toggle-row .label{font-size:13px;}
#   .toggle-row .desc{font-size:11px; color:var(--text-faint); margin-top:2px;}
#   .toggle-row.disabled{opacity:.4;}
#   .switch{position:relative; width:38px; height:21px; flex-shrink:0;}
#   .switch input{opacity:0; width:0; height:0;}
#   .slider{
#     position:absolute; inset:0; background:var(--panel-2); border:1px solid var(--border);
#     border-radius:20px; cursor:pointer; transition:.15s;
#   }
#   .slider::before{
#     content:''; position:absolute; width:15px; height:15px; left:2px; top:2px;
#     background:var(--text-dim); border-radius:50%; transition:.15s;
#   }
#   .switch input:checked + .slider{background:var(--amber-dim); border-color:var(--amber);}
#   .switch input:checked + .slider::before{transform:translateX(17px); background:var(--amber);}
#   .switch input:disabled + .slider{cursor:not-allowed;}
 
#   .ratio-grid{display:grid; grid-template-columns:1fr 1fr; gap:8px;}
#   .ratio-card{
#     border:1px solid var(--border); background:var(--panel); border-radius:8px;
#     padding:10px; cursor:pointer; text-align:center; position:relative;
#   }
#   .ratio-card.active{border-color:var(--amber); background:rgba(255,176,32,.06);}
#   .ratio-card .shape{margin:0 auto 6px; background:var(--panel-2); border:1px solid var(--border); border-radius:3px;}
#   .ratio-card .name{font-size:12px; font-weight:600;}
#   .ratio-card .lbl{font-size:10px; color:var(--text-faint); margin-top:2px;}
 
#   .style-list{display:flex; flex-direction:column; gap:8px;}
#   .style-card{
#     display:flex; align-items:center; gap:12px; border:1px solid var(--border);
#     background:var(--panel); border-radius:8px; padding:10px 12px; cursor:pointer;
#   }
#   .style-card.active{border-color:var(--teal);}
#   .style-swatch{
#     flex-shrink:0; width:64px; height:36px; border-radius:6px; background:#000;
#     display:flex; align-items:center; justify-content:center; font-size:9px; font-weight:800;
#     letter-spacing:.5px;
#   }
#   .style-card .name{font-size:12.5px; font-weight:600;}
#   .style-card .lbl{font-size:10.5px; color:var(--text-faint);}
 
#   select, .fill-toggle{
#     width:100%; background:var(--panel); border:1px solid var(--border); color:var(--text);
#     padding:9px 10px; border-radius:8px; font-size:13px;
#   }
#   .fill-toggle{display:flex; gap:6px;}
#   .fill-opt{flex:1; text-align:center; padding:8px; border-radius:6px; cursor:pointer; font-size:12px; border:1px solid var(--border);}
#   .fill-opt.active{border-color:var(--amber); color:var(--amber); background:rgba(255,176,32,.06);}
 
#   .render-btn{
#     width:100%; background:var(--amber); color:#1A1204; border:none; border-radius:10px;
#     padding:14px; font-size:14px; font-weight:700; cursor:pointer; font-family:'Space Grotesk',sans-serif;
#     letter-spacing:.3px;
#   }
#   .render-btn:disabled{background:var(--panel-2); color:var(--text-faint); cursor:not-allowed;}
#   .render-btn:not(:disabled):hover{filter:brightness(1.08);}
 
#   .progress-wrap{margin-top:14px;}
#   .progress-track{height:6px; background:var(--panel-2); border-radius:4px; overflow:hidden;}
#   .progress-fill{height:100%; background:linear-gradient(90deg, var(--teal), var(--amber)); width:0%; transition:width .3s;}
#   .progress-label{display:flex; justify-content:space-between; font-size:11px; color:var(--text-dim); margin-top:6px;}
 
#   .results{display:flex; flex-direction:column; gap:8px; margin-top:14px;}
#   .result-row{
#     display:flex; align-items:center; gap:10px; background:var(--panel); border:1px solid var(--border);
#     border-radius:8px; padding:10px 12px;
#   }
#   .result-row .name{font-size:12.5px; font-weight:600; flex:1;}
#   .result-row a{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--teal);
#     text-decoration:none; font-size:11.5px; padding:6px 10px; border-radius:6px;
#   }
#   .result-row a:hover{border-color:var(--teal);}
 
#   .status-line{font-size:11.5px; color:var(--text-faint); margin-top:10px; line-height:1.5;}
#   .status-line b{color:var(--text-dim);}
#   .feature-warn{
#     font-size:11px; color:var(--amber); background:rgba(255,176,32,.08); border:1px solid var(--amber-dim);
#     border-radius:6px; padding:8px 10px; margin-top:8px; display:none;
#   }
# </style>
# </head>
# <body>
# <script>
# // ── Theme sync — this page is served from the SAME Flask app/port as the
# // main Studio (registered as a blueprint), and when opened as an <iframe> on
# // that page it's also same-origin, so it shares localStorage automatically.
# // Reading 'shortsStudioTheme' here — the exact key the main page's theme
# // picker writes to — means this editor always matches whatever theme is
# // active there, with zero manual setup. Runs immediately (before first
# // paint) so there's no flash of the wrong palette, and again whenever the
# // theme changes elsewhere (the 'storage' event fires in this frame whenever
# // another same-origin window/frame updates localStorage).
# (function(){
#   function applyTheme(){
#     const t = localStorage.getItem('shortsStudioTheme') || 'dark-violet';
#     if(t === 'dark-violet') document.documentElement.removeAttribute('data-theme');
#     else document.documentElement.setAttribute('data-theme', t);
#   }
#   applyTheme();
#   window.addEventListener('storage', (e)=>{ if(e.key === 'shortsStudioTheme') applyTheme(); });
# })();
# </script>
 
# <div class="topbar">
#   <div class="brand"><span class="dot"></span><h1>Reframe</h1><span>face-tracked auto edit</span></div>
#   <div class="spacer"></div>
#   <span id="apiBaseHint" class="mono" style="font-size:11px; color:var(--text-faint); margin-right:6px;">same server</span>
#   <button type="button" id="apiBaseToggle" class="btn-small" title="Only needed if this page is served from a different host than the API">⚙</button>
#   <input id="apiBase" class="api-input mono" placeholder="leave blank — uses this same page's server" value="" style="display:none;">
# </div>
 
# <div class="app">
#   <!-- LEFT: preview + timeline -->
#   <div class="col">
#     <div class="fetch-section">
#       <div class="lib-header">
#         <span>🔗 Fetch from a link (YouTube / Instagram / TikTok / X / Facebook…)</span>
#       </div>
#       <div class="fetch-row">
#         <input type="text" id="fetchUrl" class="fetch-input mono" placeholder="Paste a video URL…">
#         <button type="button" class="btn-small btn-accent" id="fetchInfoBtn">Fetch</button>
#       </div>
#       <div id="fetchCard" class="fetch-card" style="display:none;">
#         <img id="fetchThumb" class="fetch-thumb" alt="">
#         <div class="fetch-meta">
#           <div id="fetchTitle" class="fetch-title"></div>
#           <div id="fetchSub" class="fetch-sub"></div>
#         </div>
#         <button type="button" class="btn-small btn-accent" id="fetchUseBtn">⬇ Fetch high-quality &amp; use</button>
#       </div>
#       <div id="fetchStatus" class="fetch-status"></div>
#     </div>
 
#     <div class="or-divider">or drop / choose a video from this PC</div>
#     <div id="dropzone" class="dropzone">
#       <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8A93A3" stroke-width="1.5"><path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v3a2 2 0 002 2h12a2 2 0 002-2v-3"/></svg>
#       <div>Drop a video, or click to choose one</div>
#       <div class="hint">mp4 / mov · analyzed locally before render</div>
#       <input type="file" id="fileInput" accept="video/*" style="display:none">
#     </div>
 
#     <div class="or-divider">or pick a video already fetched by the Downloader</div>
#     <div class="lib-section">
#       <div class="lib-header">
#         <span>📥 From Downloader / server library</span>
#         <button type="button" class="btn-small" id="refreshLibBtn">Refresh</button>
#       </div>
#       <div class="lib-grid" id="libGrid"><span class="lib-empty">Click Refresh to list videos already downloaded via the Downloader tab.</span></div>
#     </div>
 
#     <div id="previewSection" style="display:none;">
#       <div class="preview-wrap">
#         <div class="preview-frame" id="previewFrame">
#           <video id="video" muted></video>
#           <div class="crop-box" id="cropBox" style="display:none;">
#             <div class="corner tl"></div><div class="corner tr"></div><div class="corner bl"></div><div class="corner br"></div>
#             <div class="crop-readout" id="cropReadout">9:16</div>
#           </div>
#         </div>
#       </div>
#       <div class="preview-controls">
#         <div class="btn-icon" id="playBtn">▶</div>
#         <div class="time mono" id="timeLabel">0:00 / 0:00</div>
#       </div>
 
#       <div class="timeline-panel">
#         <h3>Timeline</h3>
#         <div class="timeline" id="timeline">
#           <div class="tl-playhead" id="playhead" style="left:0%"></div>
#         </div>
#         <div class="tl-legend">
#           <span><span class="swatch" style="background:var(--teal-dim)"></span>face track</span>
#           <span><span class="swatch" style="background:var(--teal)"></span>segments</span>
#           <span><span class="swatch" style="background:var(--amber)"></span>suggested hook</span>
#         </div>
#         <div class="seg-list" id="segList"></div>
#         <div style="display:flex; gap:8px; margin-top:10px;">
#           <button class="btn-small" id="addSegBtn">+ add segment at playhead (±3s)</button>
#           <button class="btn-small" id="useHookBtn" style="display:none;">use suggested hook</button>
#           <button class="btn-small" id="useSilenceBtn" style="display:none;">use auto jump-cut segments</button>
#         </div>
#         <div style="display:flex; gap:8px; margin-top:8px;">
#           <button class="btn-small" id="addKfBtn">+ pin crop here (manual keyframe)</button>
#           <button class="btn-small" id="clearKfBtn">clear manual crop</button>
#         </div>
#       </div>
#       <div class="status-line" id="statusLine"></div>
#       <div id="sourceMetaPanel" class="lib-section" style="display:none; margin-top:10px;"></div>
#     </div>
#   </div>
 
#   <!-- RIGHT: controls -->
#   <div class="col">
#     <div class="section">
#       <h3><span class="num">1</span>Auto-reframe</h3>
#       <div class="toggle-row" id="rowFace">
#         <div><div class="label">Face tracking</div><div class="desc">Advanced (mediapipe) if installed, else standard cascade</div></div>
#         <label class="switch"><input type="checkbox" id="chkFace" checked><span class="slider"></span></label>
#       </div>
#       <div class="fill-toggle" style="margin-top:8px;">
#         <div class="fill-opt active" data-fill="crop">Crop to face</div>
#         <div class="fill-opt" data-fill="blur_pad">Fit + blurred bars</div>
#       </div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">2</span>Captions</h3>
#       <div class="toggle-row" id="rowSubs">
#         <div><div class="label">Auto subtitles</div><div class="desc">Word-level, keyword emphasis auto-highlighted</div></div>
#         <label class="switch"><input type="checkbox" id="chkSubs" checked><span class="slider"></span></label>
#       </div>
#       <div class="style-list" id="styleList"></div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">3</span>Hook &amp; pacing</h3>
#       <div class="toggle-row" id="rowHook">
#         <div><div class="label">Hook-cut</div><div class="desc">Open on the loudest / most energetic moment</div></div>
#         <label class="switch"><input type="checkbox" id="chkHook"><span class="slider"></span></label>
#       </div>
#       <div class="toggle-row" id="rowJump">
#         <div><div class="label">Jump-cut silences</div><div class="desc">Auto-remove dead air between lines</div></div>
#         <label class="switch"><input type="checkbox" id="chkJump"><span class="slider"></span></label>
#       </div>
#       <div class="toggle-row">
#         <div><div class="label">Loudness normalize</div><div class="desc">-14 LUFS, matches platform playback volume</div></div>
#         <label class="switch"><input type="checkbox" id="chkNorm" checked><span class="slider"></span></label>
#       </div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">4</span>Export ratios</h3>
#       <div class="ratio-grid" id="ratioGrid"></div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">5</span>Quality</h3>
#       <select id="qualitySel">
#         <option value="high">High (crf 18, slow) — best quality</option>
#         <option value="medium">Medium (crf 21) — balanced</option>
#         <option value="fast">Fast (crf 23) — quick preview</option>
#       </select>
#     </div>
 
#     <button class="render-btn" id="renderBtn" disabled>Analyze a video first</button>
#     <div class="feature-warn" id="featureWarn"></div>
#     <div class="progress-wrap" id="progressWrap" style="display:none;">
#       <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
#       <div class="progress-label"><span id="progressStage">queued</span><span id="progressPct">0%</span></div>
#     </div>
#     <div class="results" id="results"></div>
#   </div>
# </div>
 
# <script>
# const $ = (id) => document.getElementById(id);
# const state = {
#   sourcePath: null, sourceMeta: null, sourceOrigin: 'pc', duration: 0, srcW: 0, srcH: 0, srcFps: null,
#   faceTrack: [], words: [], suggestedHook: null, suggestedSilences: null,
#   segments: [], manualKeyframes: [],
#   // Multiple export ratios can be selected at once now (a Set of ratio
#   // keys, e.g. {'9:16','1:1'}), not just one — "auto" is pre-selected by
#   // default since it's the safest zero-quality-loss choice.
#   selectedRatios: new Set(['auto']),
#   previewRatio: '9:16', // drives the live crop box only, independent of export selection
#   selectedStyle: 'bold_pop', fillMode: 'crop',
#   jobId: null, dragging: false,
# };
 
# // Same-origin by default: this page is served BY the same Flask app/port
# // that Reframe, the Downloader, and everything else run on (they're all
# // registered on one Flask app as blueprints, per RenderDetect's main()),
# // so relative paths already hit the right server. The "⚙" override is
# // only for the rare case this HTML got copied to a different host — if
# // left blank (the default), api() never leaves the current origin, which
# // is what fixes the class of bug where a stray/incorrect API host caused
# // every request (analyze, render, fetch…) to silently fail.
# function apiBase(){ const v = $('apiBase').value.trim().replace(/\/$/, ''); return v; }
# function api(path){ return apiBase() + path; }
# $('apiBaseToggle').onclick = ()=>{
#   const el = $('apiBase');
#   const show = el.style.display === 'none';
#   el.style.display = show ? 'inline-block' : 'none';
#   $('apiBaseHint').style.display = show ? 'none' : 'inline';
# };
 
# const RATIO_META = {
#   '9:16': {w:9,h:16}, '16:9': {w:16,h:9}, '1:1': {w:1,h:1}, '4:5': {w:4,h:5}, '4:3': {w:4,h:3},
# };
# const STYLE_META = {
#   bold_pop:       {bg:'#111', color:'#fff', hi:'#FFD700', sample:'THIS IS'},
#   karaoke_classic:{bg:'#111', color:'#fff', hi:'#FFA500', sample:'FILLS IN'},
#   neon_glow:      {bg:'#111', color:'#fff', hi:'#00F0FF', sample:'GLOWS UP'},
#   minimal_clean:  {bg:'#111', color:'#fff', hi:'#fff', sample:'stays quiet'},
#   creator_bold:   {bg:'#111', color:'#FFFF00', hi:'#FFFF00', sample:'HUGE TEXT'},
# };
 
# // ── boot: load styles / ratios / features from backend ──
# async function loadMeta(){
#   try{
#     const [styles, ratios, features] = await Promise.all([
#       fetch(api('/api/editor/styles')).then(r=>r.json()),
#       fetch(api('/api/editor/ratios')).then(r=>r.json()),
#       fetch(api('/api/editor/features')).then(r=>r.json()),
#     ]);
#     renderStyles(styles);
#     renderRatios(ratios);
#     applyFeatureAvailability(features);
#   }catch(e){
#     $('featureWarn').style.display='block';
#     $('featureWarn').textContent = 'Could not reach the backend at ' + api('') + ' — set the API base URL above and reload.';
#   }
# }
 
# function renderStyles(styles){
#   const list = $('styleList'); list.innerHTML='';
#   Object.entries(styles).forEach(([key, meta])=>{
#     const m = STYLE_META[key] || {bg:'#111', color:'#fff', hi:'#fff', sample:'Sample'};
#     const card = document.createElement('div');
#     card.className = 'style-card' + (key===state.selectedStyle ? ' active':'');
#     card.innerHTML = `<div class="style-swatch" style="background:${m.bg}">
#         <span style="color:${m.color}">${m.sample.split(' ')[0]}</span>&nbsp;<span style="color:${m.hi}">${m.sample.split(' ')[1]||''}</span>
#       </div>
#       <div><div class="name">${meta.label.split(' (')[0]}</div><div class="lbl">${(meta.label.match(/\((.*)\)/)||[,''])[1]}</div></div>`;
#     card.onclick = ()=>{ state.selectedStyle = key; renderStyles(styles); };
#     list.appendChild(card);
#   });
# }
 
# // Export ratios are now MULTI-select — tap any number of cards and every
# // one you pick gets rendered and offered as a download, all in the same
# // render job. "Auto" (native size, no crop, zero quality loss) is its own
# // card too. state.previewRatio (separate from the export selection) just
# // drives which crop box is drawn live over the video for editing.
# function renderRatios(ratios){
#   const grid = $('ratioGrid'); grid.innerHTML='';
#   Object.entries(ratios).forEach(([key, r])=>{
#     const isAuto = key === 'auto';
#     const srcAspect = (state.srcW && state.srcH) ? {w: state.srcW, h: state.srcH} : {w:9, h:16};
#     const meta = isAuto ? srcAspect : (RATIO_META[key] || {w:1,h:1});
#     const scale = 30 / Math.max(meta.w, meta.h);
#     const selected = state.selectedRatios.has(key);
#     const card = document.createElement('div');
#     card.className = 'ratio-card' + (selected ? ' active':'');
#     card.innerHTML = `<div class="shape" style="width:${meta.w*scale}px; height:${meta.h*scale}px;"></div>
#       <div class="name">${selected ? '✓ ' : ''}${key === 'auto' ? 'Auto' : key}</div><div class="lbl">${r.label}</div>`;
#     card.onclick = ()=>{
#       if(state.selectedRatios.has(key)) state.selectedRatios.delete(key);
#       else state.selectedRatios.add(key);
#       if(!state.selectedRatios.size) state.selectedRatios.add(key); // keep at least one selected
#       if(!isAuto){ state.previewRatio = key; drawCropForCurrentTime(); }
#       renderRatios(ratios);
#     };
#     grid.appendChild(card);
#   });
# }
 
# function applyFeatureAvailability(f){
#   toggleRow('rowFace', 'chkFace', f.face_tracking, 'opencv-python not installed on the server');
#   toggleRow('rowSubs', 'chkSubs', f.subtitles, 'faster-whisper not installed on the server');
#   toggleRow('rowHook', 'chkHook', true, '');
#   toggleRow('rowJump', 'chkJump', true, '');
#   if (!f.face_tracking_advanced && f.face_tracking){
#     $('rowFace').querySelector('.desc').textContent = 'Standard tracker active (install mediapipe for advanced mode)';
#   }
# }
# function toggleRow(rowId, chkId, available, msg){
#   if(!available){
#     $(rowId).classList.add('disabled');
#     $(chkId).checked = false;
#     $(chkId).disabled = true;
#     $(rowId).querySelector('.desc').textContent = msg;
#   }
# }
 
# // ── fill mode ──
# document.querySelectorAll('.fill-opt').forEach(el=>{
#   el.onclick = ()=>{
#     document.querySelectorAll('.fill-opt').forEach(o=>o.classList.remove('active'));
#     el.classList.add('active');
#     state.fillMode = el.dataset.fill;
#     drawCropForCurrentTime();
#   };
# });
 
# // ── fetch-from-link (built-in yt-dlp resolver, high quality single mp4) ──
# // Mirrors the Downloader tab's speed trick (cached auth mode) but skips the
# // full format-picker UI on purpose: this always grabs the single best
# // available video+audio quality and merges it into one mp4, then feeds it
# // straight into the exact same analyze/render pipeline as an upload.
# let fetchState = { url: null };
 
# $('fetchInfoBtn').onclick = async ()=>{
#   const url = $('fetchUrl').value.trim();
#   if(!url) return;
#   $('fetchStatus').textContent = 'Resolving link…';
#   $('fetchCard').style.display = 'none';
#   try{
#     const res = await fetch(api('/api/editor/fetch/info'), {
#       method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url})
#     });
#     const data = await res.json();
#     if(data.error){ $('fetchStatus').textContent = data.error; return; }
#     fetchState.url = url;
#     fetchState.meta = data;   // keep the FULL metadata (views/likes/uploader/etc.) for later
#     $('fetchThumb').src = data.thumbnail || '';
#     $('fetchTitle').textContent = data.title || 'Untitled video';
#     $('fetchSub').textContent = [data.uploader, data.duration_str, data.resolution].filter(Boolean).join(' · ');
#     $('fetchCard').style.display = 'flex';
#     $('fetchStatus').textContent = '';
#   }catch(e){
#     $('fetchStatus').textContent = 'Could not reach the server to resolve that link. Make sure the app is running and reload this page.';
#   }
# };
 
# $('fetchUseBtn').onclick = async ()=>{
#   $('fetchUseBtn').disabled = true;
#   $('fetchStatus').textContent = 'Fetching best quality (video + audio, single file)…';
#   try{
#     const res = await fetch(api('/api/editor/fetch/start'), {
#       method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url: fetchState.url})
#     });
#     const data = await res.json();
#     if(data.error){ $('fetchStatus').textContent = data.error; $('fetchUseBtn').disabled = false; return; }
#     pollFetchProgress(data.fetch_id);
#   }catch(e){
#     $('fetchStatus').textContent = 'Fetch failed to start — check the connection and try again.';
#     $('fetchUseBtn').disabled = false;
#   }
# };
 
# async function pollFetchProgress(fetchId){
#   let st;
#   try{
#     const res = await fetch(api('/api/editor/fetch/progress/'+fetchId));
#     st = await res.json();
#   }catch(e){
#     setTimeout(()=>pollFetchProgress(fetchId), 1200); // transient network hiccup, keep polling
#     return;
#   }
#   if(st.status === 'error'){
#     $('fetchStatus').textContent = 'Error: ' + st.error;
#     $('fetchUseBtn').disabled = false;
#     return;
#   }
#   if(st.status === 'done'){
#     $('fetchStatus').textContent = 'Downloaded — analyzing…';
#     $('fetchUseBtn').disabled = false;
#     $('fetchCard').style.display = 'none';
#     $('fetchUrl').value = '';
#     loadSource(st.source_path, fetchState.meta, 'fetch');
#     return;
#   }
#   const pct = st.percent != null ? st.percent + '%' : '';
#   $('fetchStatus').textContent = `${st.stage || 'downloading'}… ${pct} ${st.speed_str || ''}`.trim();
#   setTimeout(()=>pollFetchProgress(fetchId), 900);
# }
 
# // ── server library (downloader output + earlier uploads) ──
# // Lets the user reuse a video already fetched via the Downloader tab (or an
# // earlier upload) without picking it from disk again — same analyze/render
# // pipeline as a fresh upload, just a different source_path origin.
# $('refreshLibBtn').onclick = loadLibrary;
 
# async function loadLibrary(){
#   const grid = $('libGrid');
#   grid.innerHTML = '<span class="lib-empty">Loading…</span>';
#   try{
#     const res = await fetch(api('/api/editor/sources'));
#     const data = await res.json();
#     const items = data.sources || [];
#     if(!items.length){
#       grid.innerHTML = '<span class="lib-empty">Nothing yet — fetch a video in the Downloader tab, or upload one above.</span>';
#       return;
#     }
#     grid.innerHTML = '';
#     items.forEach(it=>{
#       const el = document.createElement('div');
#       el.className = 'lib-item';
#       el.innerHTML = `<span class="lib-tag">${it.origin === 'downloader' ? '📥 downloaded' : '📤 uploaded'}</span>
#         <div class="lib-name">${it.name}</div>
#         <div class="lib-meta"><span>${it.size_str||''}</span><span>${it.modified_str||''}</span></div>`;
#       el.onclick = ()=>{
#         document.querySelectorAll('.lib-item.active').forEach(n=>n.classList.remove('active'));
#         el.classList.add('active');
#         loadSource(it.path, null, it.origin === 'downloader' ? 'downloader' : 'pc');
#       };
#       grid.appendChild(el);
#     });
#   }catch(e){
#     grid.innerHTML = '<span class="lib-empty">Could not reach the server — make sure the app is running, then hit Refresh again.</span>';
#   }
# }
 
# loadLibrary();
 
# // ── upload flow ──
# $('dropzone').onclick = ()=> $('fileInput').click();
# $('fileInput').onchange = (e)=> handleFile(e.target.files[0]);
# ['dragover','dragleave','drop'].forEach(evt=>{
#   $('dropzone').addEventListener(evt, (e)=>{
#     e.preventDefault();
#     $('dropzone').classList.toggle('drag', evt==='dragover');
#     if(evt==='drop' && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
#   });
# });
 
# async function handleFile(file){
#   if(!file) return;
#   $('video').src = URL.createObjectURL(file);
#   $('previewSection').style.display = 'block';
#   $('statusLine').textContent = 'Uploading…';
#   try{
#     const fd = new FormData(); fd.append('file', file);
#     const res = await fetch(api('/api/editor/upload'), {method:'POST', body: fd});
#     const data = await res.json();
#     if(data.error){ showAnalysisError('Upload error: ' + data.error); return; }
#     loadSource(data.source_path, null, 'pc');
#   }catch(e){
#     showAnalysisError('Could not reach the server to upload this file.');
#   }
# }
 
# // ── unified source loader — every origin (fetched link, downloader
# // library, or a file picked from this PC) funnels through here so
# // analysis always runs the same way and the Export panel always ends up
# // enabled the same way, no matter where the video came from. ──
# async function loadSource(sourcePath, meta, origin){
#   state.sourcePath = sourcePath;
#   state.sourceMeta = meta || null;
#   state.sourceOrigin = origin || 'pc';
#   $('video').src = api('/api/editor/preview?path=' + encodeURIComponent(sourcePath));
#   $('previewSection').style.display = 'block';
#   renderSourceMeta();
#   await runAnalysis();
# }
 
# // Shows the rich details panel. For links fetched from YouTube/Instagram/
# // etc. this uses the metadata the SERVER already pulled during Fetch (no
# // need to re-derive title/uploader/views by hand); for PC uploads or
# // library items with no such metadata it just relies on the ffprobe-based
# // technical details that come back from /api/editor/analyze.
# function renderSourceMeta(){
#   const el = $('sourceMetaPanel');
#   if(!el) return;
#   const m = state.sourceMeta || {};
#   const rows = [
#     ['Title', m.title],
#     ['Channel / uploader', m.uploader],
#     ['Views', m.view_count_str],
#     ['Likes', m.like_count_str],
#     ['Uploaded', m.upload_date_str],
#     // Technical rows come from ffmpeg probing (_probe on the server) and are
#     // filled in for EVERY source — PC upload, Downloader-library pick, or a
#     // freshly fetched link — not just ones that happen to have yt-dlp
#     // metadata. This is what makes "same full details" true across origins.
#     ['Resolution', state.srcW && state.srcH ? `${state.srcW}×${state.srcH}` : null],
#     ['Duration', state.duration ? fmtTime(state.duration) : null],
#     ['Frame rate', state.srcFps ? `${state.srcFps} fps` : null],
#   ].filter(([,v]) => v);
#   if(!rows.length){ el.style.display = 'none'; el.innerHTML = ''; return; }
#   const heading = m.title ? 'ℹ️ Fetched video details' : 'ℹ️ Video details';
#   el.style.display = 'block';
#   el.innerHTML = `<div class="lib-header"><span>${heading}</span></div>` +
#     rows.map(([k,v]) => `<div class="status-line"><b>${k}:</b> ${v}</div>`).join('');
# }
 
# function showAnalysisError(msg){
#   $('statusLine').innerHTML = `<span style="color:var(--amber);">${msg}</span> ` +
#     `<button type="button" class="btn-small" id="retryAnalysisBtn" style="margin-left:6px;">Retry analysis</button>`;
#   const btn = $('retryAnalysisBtn');
#   if(btn) btn.onclick = () => runAnalysis();
#   $('renderBtn').disabled = true;
#   $('renderBtn').textContent = 'Analyze a video first';
# }
 
# async function runAnalysis(){
#   if(!state.sourcePath){ showAnalysisError('No video selected yet.'); return; }
#   $('renderBtn').disabled = true;
#   const startedAt = Date.now();
#   const tick = setInterval(()=>{
#     const secs = Math.round((Date.now()-startedAt)/1000);
#     $('statusLine').textContent = `Analyzing — face track + transcript + hook detection… (${secs}s`
#       + (secs > 20 ? ', first-time subtitle setup can take a couple minutes)' : ')');
#   }, 1000);
#   // Belt-and-braces client-side cap on top of the server's own 180s subtitle
#   // timeout: if the server itself never responds for any reason, this stops
#   // the button from staying disabled forever with no explanation.
#   const controller = new AbortController();
#   const abortTimer = setTimeout(()=>controller.abort(), 240000);
#   const body = {
#     source_path: state.sourcePath,
#     face_tracking: $('chkFace').checked && !$('chkFace').disabled,
#     subtitles: $('chkSubs').checked && !$('chkSubs').disabled,
#     hook_cut: true,
#     jump_cut: true,
#   };
#   let data;
#   try{
#     const res = await fetch(api('/api/editor/analyze'), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body), signal: controller.signal});
#     if(!res.ok){ showAnalysisError(`Analysis request failed (server said HTTP ${res.status}).`); return; }
#     data = await res.json();
#   }catch(e){
#     if(e.name === 'AbortError'){
#       showAnalysisError('Analysis took too long and was stopped — the server may still be finishing in the background. Wait a minute, then retry.');
#     } else {
#       showAnalysisError('Could not reach the server to analyze this video — check it is still running, then retry.');
#     }
#     return;
#   }finally{
#     clearInterval(tick);
#     clearTimeout(abortTimer);
#   }
#   if(data.error){ showAnalysisError('Analysis error: ' + data.error); return; }
#   state.duration = data.probe.duration || 0;
#   state.srcW = (data.source_size && data.source_size.w) || data.probe.width;
#   state.srcH = (data.source_size && data.source_size.h) || data.probe.height;
#   state.srcFps = data.probe.fps ? Math.round(data.probe.fps * 100) / 100 : null;
#   renderSourceMeta();
#   state.faceTrack = data.face_track || [];
#   state.words = data.words || [];
#   state.suggestedHook = data.suggested_hook || null;
#   state.suggestedSilences = data.suggested_segments || null;
#   $('useHookBtn').style.display = state.suggestedHook ? 'inline-block' : 'none';
#   $('useSilenceBtn').style.display = state.suggestedSilences ? 'inline-block' : 'none';
#   drawTimeline();
#   // Export becomes available the instant analysis finishes, identically
#   // for an uploaded PC file, a Downloader-library pick, or a fresh link
#   // fetch — there is no special-casing by source here. This is true even
#   // if one sub-feature (e.g. subtitles) timed out above — analyze always
#   // returns, so render is never blocked by one slow/failed piece.
#   $('renderBtn').disabled = false;
#   $('renderBtn').textContent = 'Render exports';
#   const srcTag = state.sourceOrigin === 'fetch' ? 'fetched video' : (state.sourceOrigin === 'downloader' ? 'downloaded video' : 'uploaded video');
#   const warnings = [data.subtitle_error, data.face_tracking_error, data.hook_cut_error, data.jump_cut_error].filter(Boolean);
#   let line = `Analyzed ${srcTag} — <b>${state.faceTrack.length}</b> face samples · <b>${state.words.length}</b> words transcribed · duration <b>${fmtTime(state.duration)}</b> · source <b>${state.srcW}×${state.srcH}</b>`;
#   if(warnings.length) line += ` <span style="color:var(--amber);">(${warnings.join(' · ')})</span>`;
#   $('statusLine').innerHTML = line;
# }
 
# // ── video controls ──
# const video = $('video');
# $('playBtn').onclick = ()=>{ video.paused ? video.play() : video.pause(); };
# video.onplay = ()=> $('playBtn').textContent = '❚❚';
# video.onpause = ()=> $('playBtn').textContent = '▶';
# video.ontimeupdate = ()=>{
#   $('timeLabel').textContent = `${fmtTime(video.currentTime)} / ${fmtTime(video.duration||0)}`;
#   const pct = video.duration ? (video.currentTime/video.duration*100) : 0;
#   $('playhead').style.left = pct + '%';
#   drawCropForCurrentTime();
# };
# function fmtTime(t){ t=Math.max(0,t||0); const m=Math.floor(t/60), s=Math.floor(t%60); return `${m}:${String(s).padStart(2,'0')}`; }
 
# // ── crop overlay: shows the current auto/manual crop box, draggable ──
# function currentCrop(){
#   const kfs = state.manualKeyframes.length ? state.manualKeyframes : state.faceTrack.map(p=>({t:p.t, x:p.cx, y:p.cy}));
#   if(!kfs.length) return {cx:0.5, cy:0.5};
#   const t = video.currentTime;
#   let lo = kfs[0], hi = kfs[kfs.length-1];
#   for(let i=0;i<kfs.length-1;i++){ if(kfs[i].t<=t && kfs[i+1].t>=t){ lo=kfs[i]; hi=kfs[i+1]; break; } }
#   const span = (hi.t - lo.t) || 1;
#   const f = Math.min(1, Math.max(0, (t - lo.t)/span));
#   const cxKey = state.manualKeyframes.length ? 'x' : 'cx';
#   const cyKey = state.manualKeyframes.length ? 'y' : 'cy';
#   return { cx: lo[cxKey] + (hi[cxKey]-lo[cxKey])*f, cy: lo[cyKey] + (hi[cyKey]-lo[cyKey])*f };
# }
 
# function drawCropForCurrentTime(){
#   const box = $('cropBox');
#   if(state.fillMode === 'blur_pad' || !$('chkFace').checked){ box.style.display='none'; return; }
#   if(!state.srcW || !video.videoWidth){ return; }
#   const meta = RATIO_META[state.previewRatio || '9:16'];
#   const targetAR = meta.w/meta.h;
#   const srcAR = video.videoWidth/video.videoHeight;
#   let cropWFrac, cropHFrac;
#   if(srcAR > targetAR){ cropHFrac = 1; cropWFrac = targetAR/srcAR; }
#   else { cropWFrac = 1; cropHFrac = srcAR/targetAR; }
#   const {cx, cy} = currentCrop();
#   let left = cx - cropWFrac/2, top = cy - cropHFrac/2;
#   left = Math.min(1-cropWFrac, Math.max(0, left));
#   top = Math.min(1-cropHFrac, Math.max(0, top));
 
#   const frame = $('previewFrame').getBoundingClientRect();
#   box.style.display = 'block';
#   box.style.left = (left*100)+'%';
#   box.style.top = (top*100)+'%';
#   box.style.width = (cropWFrac*100)+'%';
#   box.style.height = (cropHFrac*100)+'%';
#   $('cropReadout').textContent = state.previewRatio || '9:16';
#   box.dataset.cx = cx; box.dataset.cy = cy;
# }
 
# // dragging the crop box records a manual keyframe candidate (committed via "pin crop here")
# let dragStart = null;
# $('cropBox').addEventListener('mousedown', (e)=>{
#   state.dragging = true; $('cropBox').classList.add('dragging');
#   dragStart = {x:e.clientX, y:e.clientY, left:parseFloat($('cropBox').style.left), top:parseFloat($('cropBox').style.top)};
# });
# window.addEventListener('mousemove', (e)=>{
#   if(!state.dragging) return;
#   const frame = $('previewFrame').getBoundingClientRect();
#   const dx = (e.clientX-dragStart.x)/frame.width*100;
#   const dy = (e.clientY-dragStart.y)/frame.height*100;
#   const box = $('cropBox');
#   const newLeft = Math.min(100-parseFloat(box.style.width), Math.max(0, dragStart.left+dx));
#   const newTop = Math.min(100-parseFloat(box.style.height), Math.max(0, dragStart.top+dy));
#   box.style.left = newLeft+'%'; box.style.top = newTop+'%';
#   box.dataset.cx = (newLeft + parseFloat(box.style.width)/2)/100;
#   box.dataset.cy = (newTop + parseFloat(box.style.height)/2)/100;
# });
# window.addEventListener('mouseup', ()=>{ state.dragging=false; $('cropBox').classList.remove('dragging'); });
 
# $('addKfBtn').onclick = ()=>{
#   const box = $('cropBox');
#   const meta = RATIO_META[state.previewRatio || '9:16'];
#   const srcAR = video.videoWidth/video.videoHeight;
#   const targetAR = meta.w/meta.h;
#   let cropW, cropH;
#   if(srcAR > targetAR){ cropH = state.srcH; cropW = Math.round(cropH*targetAR); }
#   else { cropW = state.srcW; cropH = Math.round(cropW/targetAR); }
#   const cx = parseFloat(box.dataset.cx||0.5), cy = parseFloat(box.dataset.cy||0.5);
#   let x = Math.round(cx*state.srcW - cropW/2), y = Math.round(cy*state.srcH - cropH/2);
#   x = Math.max(0, Math.min(x, state.srcW-cropW)); y = Math.max(0, Math.min(y, state.srcH-cropH));
#   state.manualKeyframes = state.manualKeyframes.filter(k=>Math.abs(k.t-video.currentTime)>0.05);
#   state.manualKeyframes.push({t: video.currentTime, x, y});
#   state.manualKeyframes.sort((a,b)=>a.t-b.t);
#   drawTimeline();
#   $('statusLine').textContent = `Pinned manual crop at ${fmtTime(video.currentTime)} — ${state.manualKeyframes.length} manual keyframe(s) active (overrides auto tracking).`;
# };
# $('clearKfBtn').onclick = ()=>{ state.manualKeyframes=[]; drawTimeline(); drawCropForCurrentTime(); };
 
# // ── timeline: face density, segments, hook suggestion ──
# function drawTimeline(){
#   const tl = $('timeline');
#   [...tl.querySelectorAll('.tl-face,.tl-hook')].forEach(e=>e.remove());
#   if(state.duration){
#     state.faceTrack.forEach(p=>{
#       const el = document.createElement('div');
#       el.className='tl-face';
#       el.style.left = (p.t/state.duration*100)+'%'; el.style.width='2px';
#       tl.appendChild(el);
#     });
#     if(state.suggestedHook){
#       const el = document.createElement('div'); el.className='tl-hook';
#       el.style.left = (state.suggestedHook.start/state.duration*100)+'%';
#       el.style.width = ((state.suggestedHook.end-state.suggestedHook.start)/state.duration*100)+'%';
#       tl.appendChild(el);
#     }
#   }
#   renderSegList();
# }
# function renderSegList(){
#   const list = $('segList'); list.innerHTML='';
#   state.segments.forEach((seg, i)=>{
#     const row = document.createElement('div'); row.className='seg-row';
#     row.innerHTML = `<span class="mono">#${i+1}</span><span class="mono">${fmtTime(seg.start)} → ${fmtTime(seg.end)}</span><span class="spacer"></span><button data-i="${i}">✕</button>`;
#     row.querySelector('button').onclick = ()=>{ state.segments.splice(i,1); renderSegList(); };
#     list.appendChild(row);
#   });
# }
# $('timeline').addEventListener('click', (e)=>{
#   const rect = $('timeline').getBoundingClientRect();
#   const frac = (e.clientX-rect.left)/rect.width;
#   video.currentTime = frac * (video.duration||0);
# });
# $('addSegBtn').onclick = ()=>{
#   const t = video.currentTime;
#   state.segments.push({start: Math.max(0, +(t-3).toFixed(2)), end: Math.min(state.duration, +(t+3).toFixed(2))});
#   renderSegList();
# };
# $('useHookBtn').onclick = ()=>{
#   if(!state.suggestedHook) return;
#   state.segments = [state.suggestedHook, {start:0, end: state.duration}];
#   renderSegList();
#   $('chkHook').checked = true;
# };
# $('useSilenceBtn').onclick = ()=>{
#   if(!state.suggestedSilences) return;
#   state.segments = state.suggestedSilences;
#   renderSegList();
#   $('chkJump').checked = true;
# };
 
# // ── render ──
# $('renderBtn').onclick = async ()=>{
#   if(!state.selectedRatios.size){ $('statusLine').textContent = 'Pick at least one export ratio first.'; return; }
#   $('renderBtn').disabled = true;
#   $('progressWrap').style.display = 'block';
#   $('results').innerHTML = '';
#   const body = {
#     source_path: state.sourcePath,
#     export_ratios: Array.from(state.selectedRatios),
#     face_tracking: $('chkFace').checked && !$('chkFace').disabled,
#     subtitles: $('chkSubs').checked && !$('chkSubs').disabled,
#     hook_cut: $('chkHook').checked,
#     jump_cut: $('chkJump').checked,
#     normalize_audio: $('chkNorm').checked,
#     caption_style: state.selectedStyle,
#     fill_mode: state.fillMode,
#     quality: $('qualitySel').value,
#     manual_crop_keyframes: state.manualKeyframes.length ? state.manualKeyframes : null,
#     segments: state.segments.length ? state.segments : null,
#   };
#   try{
#     const res = await fetch(api('/api/editor/render'), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
#     const data = await res.json();
#     if(data.error){ $('statusLine').textContent = 'Render error: '+data.error; $('renderBtn').disabled=false; return; }
#     state.jobId = data.job_id;
#     pollProgress();
#   }catch(e){
#     $('statusLine').textContent = 'Could not reach the server to start the render — check it is still running and try again.';
#     $('renderBtn').disabled = false;
#   }
# };
 
# async function pollProgress(){
#   let job;
#   try{
#     const res = await fetch(api('/api/editor/progress/'+state.jobId));
#     job = await res.json();
#   }catch(e){
#     setTimeout(pollProgress, 1500); // transient hiccup, keep polling
#     return;
#   }
#   $('progressFill').style.width = (job.percent||0)+'%';
#   $('progressStage').textContent = job.stage || job.status;
#   $('progressPct').textContent = (job.percent||0)+'%';
#   if(job.status === 'error'){
#     $('statusLine').textContent = 'Render failed: '+job.error;
#     $('renderBtn').disabled = false;
#     return;
#   }
#   if(job.status !== 'done'){ setTimeout(pollProgress, 1200); return; }
#   $('renderBtn').disabled = false;
#   const results = $('results');
#   Object.entries(job.ratios||{}).forEach(([ratio, fname])=>{
#     const row = document.createElement('div'); row.className='result-row';
#     const tag = ratio === 'auto' ? 'Auto (original size — lossless)' : `${ratio} export`;
#     row.innerHTML = `<span class="name">${tag}</span><a href="${api('/api/editor/file/'+state.jobId+'/'+ratio)}" download>Download</a>`;
#     results.appendChild(row);
#   });
# }
 
# loadMeta();
# </script>
# </body>
# </html>
# """
 
 
# # Optional, best-effort link to downloader.py's in-memory job table so the
# # editor can show nicer names for downloaded videos ("Video Title.mp4"
# # instead of "<dl_id>_Video Title.mp4"). video_editor.py still works fine
# # standalone if downloader.py isn't registered — this never raises.
# def _downloader_jobs():
#     try:
#         import downloader
#         return downloader.DL_JOBS
#     except Exception:
#         return {}
 
# # ── configured once via init_editor() ──────────────────────────────
# EDIT_DIR = None
# SRC_DIR = None
# FFMPEG_PATH = "ffmpeg"
# FFPROBE_PATH = "ffprobe"
 
# # job_id -> {status, stage, percent, error, ratios: {ratio: filename}, config}
# EDIT_JOBS = {}
 
 
# def init_editor(base_dir, ffmpeg_path=None):
#     """Call once at startup, e.g. init_editor(BASE, FFMPEG)."""
#     global EDIT_DIR, SRC_DIR, FFMPEG_PATH, FFPROBE_PATH
#     base_dir = Path(base_dir)
#     EDIT_DIR = base_dir / "edited"
#     EDIT_DIR.mkdir(exist_ok=True)
#     SRC_DIR = base_dir / "downloads"   # reuse downloader's output as source pool
#     (EDIT_DIR / "uploads").mkdir(exist_ok=True)
#     if ffmpeg_path:
#         FFMPEG_PATH = str(ffmpeg_path)
#         # ffprobe normally sits next to ffmpeg
#         cand = Path(ffmpeg_path).with_name(
#             "ffprobe.exe" if str(ffmpeg_path).lower().endswith(".exe") else "ffprobe"
#         )
#         if cand.exists():
#             FFPROBE_PATH = str(cand)

#     # Pre-warm faster-whisper's model in the background at startup instead of
#     # on the user's first Analyze click — its first run downloads a
#     # multi-hundred-MB model from Hugging Face, which is exactly what made
#     # the very first analyze on a fresh install look "stuck forever". Doing
#     # it here means that download happens once, quietly, while the app is
#     # starting up, and every Analyze click afterward is fast.
#     if _HAS_WHISPER:
#         def _prewarm():
#             try:
#                 _load_whisper("small")
#             except Exception:
#                 pass  # subtitles will just be unavailable until this succeeds later
#         threading.Thread(target=_prewarm, daemon=True).start()
 
 
# def _no_console_kwargs():
#     if os.name == "nt":
#         return {"creationflags": subprocess.CREATE_NO_WINDOW}
#     return {}
 
 
# def _run(cmd):
#     return subprocess.run(cmd, capture_output=True, text=True, **_no_console_kwargs())
 
 
# # ══════════════════════════════ feature availability ══════════════════════════════
 
# try:
#     import cv2
#     _HAS_CV2 = True
# except ImportError:
#     _HAS_CV2 = False
 
# try:
#     import mediapipe as mp
#     _HAS_MEDIAPIPE = True
#     # Importing successfully doesn't mean `mp.solutions.face_detection` still
#     # exists — recent mediapipe releases dropped the legacy `solutions` API
#     # in favor of the newer Tasks API. Probe it once here so feature_status()
#     # (and every caller) reports the truth instead of assuming "advanced" is
#     # available and crashing later inside _FaceTracker.
#     try:
#         mp.solutions.face_detection.FaceDetection
#         _HAS_MEDIAPIPE_SOLUTIONS = True
#     except AttributeError:
#         _HAS_MEDIAPIPE_SOLUTIONS = False
# except ImportError:
#     _HAS_MEDIAPIPE = False
#     _HAS_MEDIAPIPE_SOLUTIONS = False
 
# try:
#     from faster_whisper import WhisperModel
#     _HAS_WHISPER = True
# except ImportError:
#     _HAS_WHISPER = False
 
# try:
#     import numpy as np
#     _HAS_NUMPY = True
# except ImportError:
#     _HAS_NUMPY = False
 
# try:
#     import yt_dlp
#     _HAS_YTDLP = True
# except ImportError:
#     _HAS_YTDLP = False
 
# _WHISPER_MODEL = None  # lazy-loaded singleton
 
 
# def feature_status():
#     return {
#         "face_tracking": _HAS_CV2,
#         "face_tracking_advanced": _HAS_CV2 and _HAS_MEDIAPIPE_SOLUTIONS,
#         "subtitles": _HAS_WHISPER,
#         "hook_cut_auto": _HAS_NUMPY,
#         "fetch_from_link": _HAS_YTDLP,
#     }
 
 
# # ══════════════════════════════ presets ══════════════════════════════
 
# # width x height for each export ratio, chosen at "good enough to look sharp,
# # small enough to encode fast" — bump these if you want 4K exports.
# RATIO_PRESETS = {
#     # "auto" is a sentinel: w/h are resolved at render time from the SOURCE
#     # video's own dimensions (no target size baked in here). It means "keep
#     # the video exactly as shot" — no crop, no letterbox, no rescale — and
#     # whenever nothing else in the pipeline needs to touch the picture
#     # (no captions/color/manual crop requested) it is exported via ffmpeg
#     # stream-copy, i.e. the original video bytes are re-muxed, not
#     # re-encoded, so there is truly zero generational quality loss.
#     "auto":  {"w": None, "h": None, "label": "Auto — original size, no crop, zero quality loss"},
#     "9:16":  {"w": 1080, "h": 1920, "label": "Reels / Shorts / TikTok"},
#     "16:9":  {"w": 1920, "h": 1080, "label": "YouTube / Landscape"},
#     "1:1":   {"w": 1080, "h": 1080, "label": "Square / Feed post"},
#     "4:5":   {"w": 1080, "h": 1350, "label": "Instagram portrait"},
#     "4:3":   {"w": 1440, "h": 1080, "label": "Classic / Facebook"},
# }
 
# QUALITY_PRESETS = {
#     "high":   {"crf": 18, "preset": "slow"},
#     "medium": {"crf": 21, "preset": "medium"},
#     "fast":   {"crf": 23, "preset": "veryfast"},
# }
 
# # Words that get an automatic extra-emphasis color in captions even in
# # styles that don't do full word-by-word highlighting — numbers, money and
# # a short list of "power words" are what trending caption tools punch up.
# _EMPHASIS_PATTERN = None  # compiled lazily, see _is_emphasis_word()
# _POWER_WORDS = {
#     "free", "now", "never", "always", "secret", "insane", "crazy", "huge",
#     "warning", "stop", "wait", "new", "best", "worst", "you", "your",
# }
 
 
# def _is_emphasis_word(word):
#     import re
#     global _EMPHASIS_PATTERN
#     if _EMPHASIS_PATTERN is None:
#         _EMPHASIS_PATTERN = re.compile(r"[\d%$]|^\$")
#     w = word.strip(".,!?").lower()
#     return bool(_EMPHASIS_PATTERN.search(word)) or w in _POWER_WORDS
 
# # Trending caption styles. Each is a full ASS "Style:" line plus a couple of
# # behavior flags consumed by build_ass(). Colors are &HAABBGGRR (ASS order).
# CAPTION_STYLES = {
#     "bold_pop": {
#         "label": "Bold Pop (white + yellow highlight)",
#         "style": "Style: Default,Montserrat Black,84,&H00FFFFFF,&H0000D7FF,&H00101010,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,6,0,2,60,60,140,1",
#         "highlight_color": "&H0000D7FF",   # active word turns yellow/gold
#         "word_by_word": True,
#         "pop_scale": True,
#     },
#     "karaoke_classic": {
#         "label": "Karaoke Fill (progressive color wipe)",
#         "style": "Style: Default,Montserrat SemiBold,72,&H00FFFFFF,&H0000A5FF,&H00202020,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,4,0,2,60,60,150,1",
#         "highlight_color": "&H0000A5FF",
#         "word_by_word": True,
#         "karaoke_fill": True,
#     },
#     "neon_glow": {
#         "label": "Neon Glow (cyan outline pop)",
#         "style": "Style: Default,Montserrat Black,80,&H00FFFFFF,&H00FFF000,&H00902000,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,5,2,2,60,60,140,1",
#         "highlight_color": "&H00FFF000",
#         "word_by_word": True,
#         "pop_scale": True,
#     },
#     "minimal_clean": {
#         "label": "Minimal Clean (small centered white)",
#         "style": "Style: Default,Inter Medium,54,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
#                   "0,0,0,0,100,100,0,0,1,2,1,2,60,60,120,1",
#         "highlight_color": "&H00FFFFFF",
#         "word_by_word": False,
#         "pop_scale": False,
#     },
#     "creator_bold": {
#         "label": "Creator Bold (huge yellow, black outline)",
#         "style": "Style: Default,Montserrat Black,92,&H0000FFFF,&H000000FF,&H00000000,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,8,0,2,50,50,160,1",
#         "highlight_color": "&H000000FF",
#         "word_by_word": True,
#         "pop_scale": True,
#     },
# }
 
 
# @editor_bp.route("/api/editor/styles")
# def api_editor_styles():
#     return jsonify({k: {"label": v["label"]} for k, v in CAPTION_STYLES.items()})
 
 
# @editor_bp.route("/api/editor/ratios")
# def api_editor_ratios():
#     return jsonify(RATIO_PRESETS)
 
 
# @editor_bp.route("/api/editor/editor")
# def api_editor_page():
#     """Serves the built-in editor UI — no separate .html file to manage,
#     it's embedded in this module (EDITOR_HTML below) and rendered straight
#     from memory. Open this URL in a tab; its fetch calls are same-origin."""
#     from flask import Response
#     return Response(EDITOR_HTML, mimetype="text/html")
 
 
# def _fmt_size(n):
#     if not n:
#         return "0 B"
#     n = float(n)
#     for unit in ["B", "KB", "MB", "GB"]:
#         if n < 1024:
#             return f"{n:.1f} {unit}"
#         n /= 1024
#     return f"{n:.1f} TB"
 
 
# @editor_bp.route("/api/editor/sources")
# def api_editor_sources():
#     """Lists videos the editor can open WITHOUT a fresh upload: everything
#     downloader.py has saved to DL_DIR/downloads (so a just-fetched video can
#     be sent straight into the editor), plus anything uploaded here before.
#     Both origins return the exact same shape and both feed the exact same
#     source_path used by /analyze and /render — the editor doesn't care where
#     a file came from."""
#     import datetime
#     jobs = _downloader_jobs()
#     # dl_id -> nicer display title, when downloader.py is registered too
#     title_by_stub = {}
#     for job in jobs.values():
#         fname = job.get("filename")
#         if fname and "_" in fname:
#             stub = fname.split("_", 1)[0]
#             title_by_stub[stub] = fname.split("_", 1)[-1]
 
#     sources = []
#     for folder, origin in ((SRC_DIR, "downloader"), (EDIT_DIR / "uploads", "upload")):
#         if not folder or not folder.exists():
#             continue
#         for p in sorted(folder.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
#             if not p.is_file() or p.suffix.lower() not in (".mp4", ".mov", ".mkv", ".webm", ".m4v"):
#                 continue
#             stub = p.name.split("_", 1)[0]
#             display = title_by_stub.get(stub) or (p.name.split("_", 1)[-1] if "_" in p.name else p.name)
#             stat = p.stat()
#             sources.append({
#                 "name": display,
#                 "path": str(p.resolve()),
#                 "origin": origin,
#                 "size": stat.st_size,
#                 "size_str": _fmt_size(stat.st_size),
#                 "modified": stat.st_mtime,
#                 "modified_str": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%d %b, %H:%M"),
#             })
#     sources.sort(key=lambda s: s["modified"], reverse=True)
#     return jsonify({"sources": sources})
 
 
# def _is_allowed_source(path):
#     """Only ever stream files that live under the editor's own known roots
#     (downloader's downloads folder, this module's uploads folder, or its
#     own rendered-output folder) — never an arbitrary server path."""
#     try:
#         rp = Path(path).resolve()
#     except Exception:
#         return False
#     for root in (SRC_DIR, EDIT_DIR / "uploads", EDIT_DIR):
#         if root and root.exists():
#             try:
#                 rp.relative_to(root.resolve())
#                 return True
#             except ValueError:
#                 continue
#     return False
 
 
# @editor_bp.route("/api/editor/preview")
# def api_editor_preview():
#     """Streams a source (or rendered) video for <video> tag playback/scrub,
#     so picking a file from the library previews instantly without a
#     round-trip upload. Range requests are handled by Flask's send_file."""
#     path = request.args.get("path", "")
#     if not path or not _is_allowed_source(path) or not Path(path).exists():
#         return "Not found or not allowed", 404
#     return send_file(path, conditional=True)
 
 
# @editor_bp.route("/api/editor/features")
# def api_editor_features():
#     return jsonify(feature_status())
 
 
# # ══════════════════════════════ probing ══════════════════════════════
 
# def _probe(path):
#     """Returns {duration, width, height, fps} for any video — PC upload,
#     Downloader-library pick, or freshly fetched link — using ONLY ffmpeg
#     (FFMPEG_PATH), never ffprobe.

#     Why: this whole app is wired up with `FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()`
#     in the main file, which downloads/bundles an ffmpeg BINARY ONLY — it does
#     not ship ffprobe next to it, and a bare "ffprobe" is not guaranteed to be
#     on PATH on a fresh Windows machine (that mismatch is exactly what produced
#     `FileNotFoundError: [WinError 2] The system cannot find the file
#     specified` — the app was calling a program that doesn't exist on disk).
#     `ffmpeg -i <file>` already prints duration/resolution/fps to stderr for
#     every input, so we parse that instead — one binary, guaranteed present,
#     identical accurate metadata for every source, no more ffprobe dependency.
#     """
#     cmd = [FFMPEG_PATH, "-hide_banner", "-i", str(path)]
#     out = _run(cmd)
#     text = (out.stderr or "") + (out.stdout or "")

#     duration = 0.0
#     dm = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
#     if dm:
#         hh, mm, ss = dm.groups()
#         duration = int(hh) * 3600 + int(mm) * 60 + float(ss)

#     width = height = None
#     # e.g. "Stream #0:0: Video: h264 ..., yuv420p, 1080x1920 [SAR 1:1 DAR 9:16], 30 fps, 30 tbr, ..."
#     vm = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", text)
#     if vm:
#         width, height = int(vm.group(1)), int(vm.group(2))

#     fps = 25.0
#     fm = re.search(r"([\d.]+)\s+fps", text)
#     if fm:
#         try:
#             fps = float(fm.group(1))
#         except ValueError:
#             pass

#     return {"duration": duration, "width": width, "height": height, "fps": fps}
 
 
# # ══════════════════════════════ face tracking ══════════════════════════════
 
# class _FaceTracker:
#     """Samples frames, finds a face each time, and produces a smoothed
#     (jitter-free) track of the subject's center point over time.
 
#     Uses mediapipe's BlazeFace model when available ("advanced" mode — much
#     more accurate on angled/small faces), otherwise falls back to OpenCV's
#     bundled Haar cascade so face tracking still works with just opencv-python
#     installed.
#     """
 
#     def __init__(self):
#         # mediapipe importing successfully doesn't guarantee the legacy
#         # `mp.solutions.face_detection` API is present — recent mediapipe
#         # builds moved to a different Tasks API and dropped `.solutions`
#         # entirely, which used to crash every analyze request with
#         # `AttributeError: module 'mediapipe' has no attribute 'solutions'`.
#         # _HAS_MEDIAPIPE_SOLUTIONS was probed once at startup, so this just
#         # picks the mode that's actually usable instead of assuming.
#         self.advanced = _HAS_MEDIAPIPE_SOLUTIONS
#         if self.advanced:
#             self._mp_detector = mp.solutions.face_detection.FaceDetection(
#                 model_selection=1, min_detection_confidence=0.5
#             )
#         elif _HAS_CV2:
#             self._cascade = cv2.CascadeClassifier(
#                 cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#             )
#         else:
#             raise RuntimeError("opencv-python is required for face tracking")
 
#     def _detect(self, frame_bgr):
#         """Returns list of (cx, cy, w, h) in pixel coords, largest first."""
#         h_img, w_img = frame_bgr.shape[:2]
#         faces = []
#         if self.advanced:
#             rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#             result = self._mp_detector.process(rgb)
#             if result.detections:
#                 for d in result.detections:
#                     box = d.location_data.relative_bounding_box
#                     fw, fh = box.width * w_img, box.height * h_img
#                     fx = box.xmin * w_img + fw / 2
#                     fy = box.ymin * h_img + fh / 2
#                     faces.append((fx, fy, fw, fh))
#         else:
#             gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#             dets = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50))
#             for (x, y, w, h) in dets:
#                 faces.append((x + w / 2, y + h / 2, w, h))
#         faces.sort(key=lambda f: f[2] * f[3], reverse=True)  # largest face first
#         return faces
 
#     def track(self, video_path, sample_fps=2.0, progress_cb=None):
#         """Returns a list of {t, cx, cy, w, h} in NORMALIZED (0-1) coords,
#         exponentially smoothed so the crop pans instead of jumping."""
#         cap = cv2.VideoCapture(str(video_path))
#         if not cap.isOpened():
#             raise RuntimeError("Could not open video for face tracking")
#         src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#         w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         step = max(1, int(round(src_fps / sample_fps)))
 
#         raw_points = []
#         last_center = (0.5, 0.5)  # default: frame center when nobody's found yet
#         frame_idx = 0
#         while True:
#             ok = cap.grab()
#             if not ok:
#                 break
#             if frame_idx % step == 0:
#                 ok2, frame = cap.retrieve()
#                 if ok2:
#                     faces = self._detect(frame)
#                     if faces:
#                         fx, fy, fw, fh = faces[0]
#                         cx, cy = fx / w_img, fy / h_img
#                         last_center = (cx, cy)
#                     else:
#                         cx, cy = last_center  # hold last known position
#                     t = frame_idx / src_fps
#                     raw_points.append({"t": t, "cx": cx, "cy": cy})
#                 if progress_cb and total_frames:
#                     progress_cb(min(99, int(frame_idx * 100 / total_frames)))
#             frame_idx += 1
#         cap.release()
 
#         if not raw_points:
#             raw_points = [{"t": 0.0, "cx": 0.5, "cy": 0.5}]
 
#         # EMA smoothing so the crop glides instead of snapping every sample
#         alpha = 0.25
#         smoothed = [raw_points[0]]
#         for p in raw_points[1:]:
#             prev = smoothed[-1]
#             smoothed.append({
#                 "t": p["t"],
#                 "cx": prev["cx"] + alpha * (p["cx"] - prev["cx"]),
#                 "cy": prev["cy"] + alpha * (p["cy"] - prev["cy"]),
#             })
#         return smoothed, (w_img, h_img)
 
 
# def analyze_face_track(video_path, sample_fps=2.0, progress_cb=None):
#     if not _HAS_CV2:
#         raise RuntimeError("opencv-python is not installed — face tracking unavailable")
#     tracker = _FaceTracker()
#     points, (w, h) = tracker.track(video_path, sample_fps=sample_fps, progress_cb=progress_cb)
#     return points, w, h, tracker.advanced
 
 
# # ══════════════════════════════ crop-window math ══════════════════════════════
 
# def _crop_size_for_ratio(src_w, src_h, target_w, target_h):
#     """Largest crop rectangle matching the target aspect that still fits
#     inside the source frame (so we crop, never letterbox/pad)."""
#     target_ar = target_w / target_h
#     src_ar = src_w / src_h
#     if src_ar > target_ar:
#         # source is wider than target -> crop width, keep full height
#         crop_h = src_h
#         crop_w = int(round(crop_h * target_ar))
#     else:
#         # source is taller than target -> crop height, keep full width
#         crop_w = src_w
#         crop_h = int(round(crop_w / target_ar))
#     return crop_w, crop_h
 
 
# def _build_crop_keyframes(track_points, src_w, src_h, crop_w, crop_h):
#     """Turns normalized face-center points into pixel-space crop x/y
#     top-left keyframes, clamped so the crop never leaves the frame."""
#     kfs = []
#     for p in track_points:
#         cx_px = p["cx"] * src_w
#         cy_px = p["cy"] * src_h
#         x = int(cx_px - crop_w / 2)
#         y = int(cy_px - crop_h / 2)
#         x = max(0, min(x, src_w - crop_w))
#         y = max(0, min(y, src_h - crop_h))
#         kfs.append({"t": p["t"], "x": x, "y": y})
#     return kfs
 
 
# def _expr_chain(keyframes, key):
#     """Builds an ffmpeg time-expression string that linearly interpolates
#     between keyframes for either 'x' or 'y' — this drives a moving crop
#     window without needing any external filter graph patching."""
#     if len(keyframes) == 1:
#         return str(keyframes[0][key])
#     expr = str(keyframes[-1][key])
#     for i in range(len(keyframes) - 2, -1, -1):
#         t0, v0 = keyframes[i]["t"], keyframes[i][key]
#         t1, v1 = keyframes[i + 1]["t"], keyframes[i + 1][key]
#         if t1 <= t0:
#             continue
#         # linear ramp between (t0,v0) and (t1,v1), holds v0 before t0
#         ramp = f"({v0}+({v1}-{v0})*(t-{t0})/{(t1 - t0)})"
#         expr = f"if(lt(t,{t1}),{ramp},{expr})"
#     return expr
 
 
# # ══════════════════════════════ subtitles ══════════════════════════════
 
# def _load_whisper(model_size="small"):
#     global _WHISPER_MODEL
#     if _WHISPER_MODEL is None:
#         _WHISPER_MODEL = WhisperModel(model_size, compute_type="int8")
#     return _WHISPER_MODEL
 
 
# def transcribe_words(video_path, model_size="small", language=None):
#     """Returns [{start, end, text}, ...] word-level timestamps."""
#     if not _HAS_WHISPER:
#         raise RuntimeError("faster-whisper is not installed — subtitles unavailable")
#     model = _load_whisper(model_size)
#     segments, _info = model.transcribe(str(video_path), word_timestamps=True, language=language)
#     words = []
#     for seg in segments:
#         for w in (seg.words or []):
#             text = (w.word or "").strip()
#             if text:
#                 words.append({"start": w.start, "end": w.end, "text": text})
#     return words
 
 
# def _chunk_words(words, max_words=4, max_span=1.6):
#     """Groups words into short on-screen caption chunks — the standard
#     'trending shorts' style of 2-5 words on screen at a time, not full
#     sentences."""
#     chunks = []
#     cur = []
#     for w in words:
#         if cur and (len(cur) >= max_words or (w["end"] - cur[0]["start"]) > max_span):
#             chunks.append(cur)
#             cur = []
#         cur.append(w)
#     if cur:
#         chunks.append(cur)
#     return chunks
 
 
# def _ass_time(t):
#     h = int(t // 3600)
#     m = int((t % 3600) // 60)
#     s = t % 60
#     return f"{h}:{m:02d}:{s:05.2f}"
 
 
# def build_ass(words, style_name, video_w, video_h):
#     """Builds a full .ass subtitle document with word-by-word highlight
#     animation for the chosen trending style, sized for the given output
#     resolution so text scales correctly across aspect ratios."""
#     style = CAPTION_STYLES.get(style_name, CAPTION_STYLES["bold_pop"])
#     header = f"""[Script Info]
# ScriptType: v4.00+
# PlayResX: {video_w}
# PlayResY: {video_h}
# ScaledBorderAndShadow: yes
 
# [V4+ Styles]
# Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
# {style["style"]}
 
# [Events]
# Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
# """
#     lines = []
#     for chunk in _chunk_words(words):
#         start, end = chunk[0]["start"], chunk[-1]["end"]
#         if style.get("karaoke_fill"):
#             # \k tags: whole line present, active word wipes to highlight color
#             text = "".join(
#                 r"{\k%d}%s " % (max(1, int(round((w["end"] - w["start"]) * 100))), w["text"])
#                 for w in chunk
#             )
#             lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
#         elif style.get("word_by_word"):
#             # emit one Dialogue event per word so only the current word is
#             # shown in the highlight color while the rest stay default —
#             # gives the punchy "pop" caption look trending on shorts.
#             for w in chunk:
#                 parts = []
#                 for w2 in chunk:
#                     if w2 is w:
#                         scale = r"\fscx115\fscy115" if style.get("pop_scale") else ""
#                         parts.append(r"{\c%s%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], scale, w2["text"]))
#                     elif _is_emphasis_word(w2["text"]):
#                         # numbers / money / power-words get punched up even
#                         # when they're not the "active" word of the moment
#                         parts.append(r"{\c%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], w2["text"]))
#                     else:
#                         parts.append(w2["text"])
#                 text = " ".join(parts)
#                 lines.append(f"Dialogue: 0,{_ass_time(w['start'])},{_ass_time(w['end'])},Default,,0,0,0,,{text}")
#         else:
#             parts = []
#             for w2 in chunk:
#                 if _is_emphasis_word(w2["text"]):
#                     parts.append(r"{\c%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], w2["text"]))
#                 else:
#                     parts.append(w2["text"])
#             text = " ".join(parts)
#             lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
#     return header + "\n".join(lines) + "\n"
 
 
# # ══════════════════════════════ hook-cut / segments ══════════════════════════════
 
# def _read_pcm_mono(video_path, sr=8000):
#     """Decodes a fast, low-res mono PCM stream via ffmpeg for lightweight
#     loudness analysis (used only to pick the auto hook window)."""
#     cmd = [FFMPEG_PATH, "-v", "error", "-i", str(video_path),
#            "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
#     proc = subprocess.run(cmd, capture_output=True, **_no_console_kwargs())
#     raw = proc.stdout
#     count = len(raw) // 2
#     samples = struct.unpack(f"<{count}h", raw[:count * 2])
#     return np.array(samples, dtype=np.float32) / 32768.0, sr
 
 
# def auto_hook_window(video_path, duration, hook_len=6.0):
#     """Finds the loudest `hook_len`-second window in the clip — a cheap but
#     effective proxy for 'most energetic / most likely to hook a viewer'
#     moment, used to auto-suggest where the exported clip should start."""
#     if not _HAS_NUMPY:
#         raise RuntimeError("numpy is not installed — auto hook-cut unavailable")
#     audio, sr = _read_pcm_mono(video_path)
#     if len(audio) == 0:
#         return {"start": 0.0, "end": min(hook_len, duration)}
#     win = int(hook_len * sr)
#     if win >= len(audio):
#         return {"start": 0.0, "end": duration}
#     energy = audio ** 2
#     # cumulative sum for O(1) windowed average lookups
#     cumsum = np.cumsum(np.insert(energy, 0, 0))
#     window_energy = cumsum[win:] - cumsum[:-win]
#     best_start_sample = int(np.argmax(window_energy))
#     start = best_start_sample / sr
#     return {"start": round(start, 2), "end": round(min(start + hook_len, duration), 2)}
 
 
# def detect_silences(video_path, noise_db=-30, min_silence=0.5):
#     """Runs ffmpeg's silencedetect and returns the list of NON-silent
#     {start,end} ranges — i.e. the segments worth keeping. This powers
#     automatic 'jump-cut' editing (removing dead air/pauses), the other
#     trending short-form edit style besides the loudest-window hook-cut."""
#     cmd = [FFMPEG_PATH, "-v", "info", "-i", str(video_path),
#            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
#     result = _run(cmd)
#     log = result.stderr or ""
#     starts, ends = [], []
#     for line in log.splitlines():
#         if "silence_start" in line:
#             starts.append(float(line.split("silence_start:")[1].strip().split(" ")[0]))
#         elif "silence_end" in line:
#             ends.append(float(line.split("silence_end:")[1].strip().split(" ")[0].split("|")[0]))
#     info = _probe(video_path)
#     duration = info["duration"]
#     silences = list(zip(starts, ends[: len(starts)]))
#     keep = []
#     cursor = 0.0
#     for s, e in silences:
#         if s > cursor:
#             keep.append({"start": round(cursor, 2), "end": round(s, 2)})
#         cursor = max(cursor, e)
#     if cursor < duration:
#         keep.append({"start": round(cursor, 2), "end": round(duration, 2)})
#     return [k for k in keep if k["end"] - k["start"] > 0.15]
 
 
# def _build_concat_file(video_path, segments, tmp_dir):
#     """Writes an ffmpeg concat-demuxer list after trimming each segment,
#     so multiple {start,end} ranges can be stitched in a custom, manually
#     chosen order (rearrange / hook-cut)."""
#     parts = []
#     for i, seg in enumerate(segments):
#         out = tmp_dir / f"seg_{i}.mp4"
#         cmd = [
#             FFMPEG_PATH, "-y", "-v", "error",
#             "-ss", str(seg["start"]), "-to", str(seg["end"]),
#             "-i", str(video_path),
#             "-c", "copy", "-avoid_negative_ts", "make_zero",
#             str(out),
#         ]
#         _run(cmd)
#         parts.append(out)
#     list_path = tmp_dir / "concat_list.txt"
#     with open(list_path, "w", encoding="utf-8") as f:
#         for p in parts:
#             f.write(f"file '{p.as_posix()}'\n")
#     return list_path
 
 
# def _measure_loudnorm(path):
#     """First pass of two-pass EBU R128 loudness normalization: measures the
#     input's actual loudness/true-peak/range so the second (real encode)
#     pass can normalize with `linear=true` against real measured values
#     instead of loudnorm's single-pass dynamic estimate — noticeably more
#     accurate and avoids the pumping/gain-riding artifacts single-pass mode
#     can introduce on speech-heavy short-form video."""
#     cmd = [FFMPEG_PATH, "-v", "info", "-i", str(path),
#            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
#            "-f", "null", "-"]
#     result = _run(cmd)
#     log = result.stderr or ""
#     try:
#         start = log.rindex("{")
#         end = log.rindex("}") + 1
#         return json.loads(log[start:end])
#     except (ValueError, json.JSONDecodeError):
#         return None  # falls back to single-pass below
 
 
# # ══════════════════════════════ render pipeline ══════════════════════════════
 
# def _render_one_ratio(job, source_path, ratio_key, track_points, src_w, src_h,
#                        words, tmp_dir, loud_measured=None):
#     cfg = job["config"]
#     ratio = RATIO_PRESETS[ratio_key]
#     is_auto = (ratio_key == "auto")
#     # "auto" keeps the SOURCE's own dimensions — no target size is imposed.
#     target_w, target_h = (src_w, src_h) if is_auto else (ratio["w"], ratio["h"])
#     quality = QUALITY_PRESETS.get(cfg.get("quality", "high"), QUALITY_PRESETS["high"])
 
#     fill_mode = cfg.get("fill_mode", "crop")  # "crop" (default) or "blur_pad"
#     # "auto" never crops or letterboxes — it IS the full original frame —
#     # so blur_pad/crop fill-mode choices simply don't apply to it.
#     if is_auto:
#         fill_mode = "none"
 
#     # ── subtitles: build the .ass first so both fill-mode branches can use it ──
#     ass_path = None
#     if cfg["features"].get("subtitles") and words:
#         ass_content = build_ass(words, cfg.get("caption_style", "bold_pop"), target_w, target_h)
#         ass_path = tmp_dir / f"subs_{ratio_key.replace(':', 'x')}.ass"
#         ass_path.write_text(ass_content, encoding="utf-8")
#         ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
 
#     if is_auto and not ass_path and not cfg.get("manual_crop_keyframes"):
#         # Nothing needs to touch the picture at all: fastest AND highest
#         # possible quality path is a pure stream-copy remux (no re-encode
#         # -> byte-for-byte original video quality; only trimmed/normalized
#         # audio may differ). This is what "zero quality loss" means here.
#         out_name = f"{job['job_id']}_{ratio_key.replace(':', 'x')}.mp4"
#         out_path = EDIT_DIR / out_name
#         if cfg.get("normalize_audio", True):
#             measured = loud_measured
#             audio_filter = (
#                 "loudnorm=I=-14:TP=-1.5:LRA=11:"
#                 f"measured_I={measured.get('input_i', -14)}:"
#                 f"measured_TP={measured.get('input_tp', -1.5)}:"
#                 f"measured_LRA={measured.get('input_lra', 11)}:"
#                 f"measured_thresh={measured.get('input_thresh', -24)}:"
#                 "linear=true:print_format=summary"
#             ) if measured else "loudnorm=I=-14:TP=-1.5:LRA=11"
#             cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path),
#                    "-c:v", "copy", "-af", audio_filter, "-c:a", "aac", "-b:a", "192k",
#                    "-movflags", "+faststart", str(out_path)]
#         else:
#             cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path),
#                    "-c", "copy", "-movflags", "+faststart", str(out_path)]
#         result = _run(cmd)
#         if result.returncode != 0:
#             raise RuntimeError(f"ffmpeg failed for {ratio_key}: {result.stderr[-800:]}")
#         return out_name
 
#     if fill_mode == "blur_pad":
#         # "Advanced" reframe mode: instead of cropping content away, fit the
#         # WHOLE frame inside the target canvas and fill the leftover bars
#         # with a blurred, zoomed copy of the same frame — the look used by
#         # most professional auto-reframe tools when a hard crop would cut
#         # off important content. No face tracking needed for this branch,
#         # since nothing is being cropped out.
#         fc = (
#             f"[0:v]split=2[bg][fg];"
#             f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
#             f"crop={target_w}:{target_h},gblur=sigma=25,eq=brightness=-0.05[bg2];"
#             f"[fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg2];"
#             f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2:format=auto[base]"
#         )
#         if ass_path:
#             fc += f",[base]ass='{ass_escaped}'[vout]"
#             map_v = "[vout]"
#         else:
#             fc += "[vout]"
#             map_v = "[vout]"
#         vf_args = ["-filter_complex", fc, "-map", map_v, "-map", "0:a?"]
#     elif is_auto:
#         # Captions and/or a manual crop keyframe are present, so a re-encode
#         # can't be avoided — but the frame itself is never cropped or
#         # rescaled; it stays at the source's own resolution. Quality is set
#         # one notch above "high" (crf 16, effectively visually lossless)
#         # specifically for this path so adding captions never visibly
#         # degrades the original picture.
#         filters = ["setsar=1"]
#         if ass_path:
#             filters.append(f"ass='{ass_escaped}'")
#         vf_args = ["-vf", ",".join(filters)]
#         quality = {"crf": 16, "preset": quality["preset"]}
#     else:
#         filters = []
#         # ── crop (face-tracked or manual) ──
#         if cfg["features"].get("face_tracking") or cfg.get("manual_crop_keyframes"):
#             crop_w, crop_h = _crop_size_for_ratio(src_w, src_h, target_w, target_h)
#             if cfg.get("manual_crop_keyframes"):
#                 kfs = sorted(cfg["manual_crop_keyframes"], key=lambda k: k["t"])
#             else:
#                 kfs = _build_crop_keyframes(track_points, src_w, src_h, crop_w, crop_h)
#             x_expr = _expr_chain(kfs, "x")
#             y_expr = _expr_chain(kfs, "y")
#             filters.append(f"crop={crop_w}:{crop_h}:'{x_expr}':'{y_expr}'")
#         else:
#             # no face tracking requested -> simple centered crop to the target ratio
#             crop_w, crop_h = _crop_size_for_ratio(src_w, src_h, target_w, target_h)
#             filters.append(f"crop={crop_w}:{crop_h}:(in_w-out_w)/2:(in_h-out_h)/2")
 
#         filters.append(f"scale={target_w}:{target_h}:flags=lanczos")
#         filters.append("setsar=1")
#         if ass_path:
#             filters.append(f"ass='{ass_escaped}'")
#         vf_args = ["-vf", ",".join(filters)]
 
#     out_name = f"{job['job_id']}_{ratio_key.replace(':', 'x')}.mp4"
#     out_path = EDIT_DIR / out_name
 
#     # Loudness normalization (EBU R128, -14 LUFS is the standard target for
#     # social platforms) so exported audio doesn't sound quiet/inconsistent
#     # next to other content in-feed. On "high" quality we do a real
#     # measure-then-normalize TWO-PASS pass for accuracy; cheaper qualities
#     # use fast single-pass so exports stay quick.
#     audio_filters = []
#     if cfg.get("normalize_audio", True):
#         measured = loud_measured
#         if measured:
#             audio_filters = [
#                 "loudnorm=I=-14:TP=-1.5:LRA=11:"
#                 f"measured_I={measured.get('input_i', -14)}:"
#                 f"measured_TP={measured.get('input_tp', -1.5)}:"
#                 f"measured_LRA={measured.get('input_lra', 11)}:"
#                 f"measured_thresh={measured.get('input_thresh', -24)}:"
#                 "linear=true:print_format=summary"
#             ]
#         else:
#             audio_filters = ["loudnorm=I=-14:TP=-1.5:LRA=11"]
 
#     cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path), *vf_args]
#     if audio_filters:
#         cmd += ["-af", ",".join(audio_filters)]
#     cmd += [
#         "-c:v", "libx264", "-crf", str(quality["crf"]), "-preset", quality["preset"],
#         "-c:a", "aac", "-b:a", "192k",
#         "-movflags", "+faststart",
#         str(out_path),
#     ]
#     result = _run(cmd)
#     if result.returncode != 0:
#         raise RuntimeError(f"ffmpeg failed for {ratio_key}: {result.stderr[-800:]}")
#     return out_name
 
 
# def _run_render_job(job_id):
#     job = EDIT_JOBS[job_id]
#     cfg = job["config"]
#     tmp_dir = EDIT_DIR / f"tmp_{job_id}"
#     tmp_dir.mkdir(exist_ok=True)
 
#     try:
#         source_path = Path(cfg["source_path"])
#         if not source_path.exists():
#             raise RuntimeError("Source video not found")
 
#         # ── 1. hook-cut / manual rearrange: trim+stitch segments first ──
#         job["stage"] = "segments"
#         segments = cfg.get("segments")
#         want_jump = cfg["features"].get("jump_cut")
#         want_hook = cfg["features"].get("hook_cut")
#         if not segments and (want_jump or want_hook):
#             info = _probe(source_path)
#             jump_segments = detect_silences(source_path) if want_jump else None
#             if want_hook:
#                 hook = auto_hook_window(source_path, info["duration"], cfg.get("hook_len", 6.0))
#                 # Cold-open on the loudest window, then play the rest of the
#                 # clip — dead-air-trimmed if jump_cut is also on, otherwise
#                 # the full original timeline. Combining both no longer means
#                 # jump_cut silently wins.
#                 rest = jump_segments if jump_segments else [{"start": 0.0, "end": info["duration"]}]
#                 segments = [hook] + rest
#             else:
#                 segments = jump_segments
#         if segments:
#             list_path = _build_concat_file(source_path, segments, tmp_dir)
#             stitched = tmp_dir / "stitched.mp4"
#             _run([FFMPEG_PATH, "-y", "-v", "error", "-f", "concat", "-safe", "0",
#                   "-i", str(list_path), "-c", "copy", str(stitched)])
#             source_path = stitched
#         job["percent"] = 10
 
#         info = _probe(source_path)
#         src_w, src_h = info["width"], info["height"]
 
#         # ── 2. face tracking (once, reused for every export ratio) ──
#         job["stage"] = "face_tracking"
#         track_points = []
#         if cfg["features"].get("face_tracking"):
#             def cb(pct):
#                 job["percent"] = 10 + int(pct * 0.3)
#             track_points, src_w, src_h, advanced = analyze_face_track(
#                 source_path, sample_fps=cfg.get("track_fps", 2.0), progress_cb=cb
#             )
#             job["face_tracking_mode"] = "advanced (mediapipe)" if advanced else "standard (haar cascade)"
#         job["percent"] = 40
 
#         # ── 3. subtitles (once, reused for every export ratio) ──
#         job["stage"] = "subtitles"
#         words = []
#         if cfg["features"].get("subtitles"):
#             words = transcribe_words(source_path, model_size=cfg.get("whisper_model", "small"),
#                                       language=cfg.get("language"))
#         job["percent"] = 60
 
#         # ── 4. render each requested aspect ratio as one final file ──
#         job["stage"] = "render"
#         ratios = cfg.get("export_ratios") or ["9:16"]
#         job["ratios"] = {}
#         n = len(ratios)
 
#         # measured once (not per-ratio, source audio is identical across
#         # ratios) and only for "high" quality, where the extra ffmpeg pass
#         # is worth the accuracy; cheaper qualities use fast single-pass.
#         loud_measured = None
#         if cfg.get("normalize_audio", True) and cfg.get("quality", "high") == "high":
#             loud_measured = _measure_loudnorm(source_path)
 
#         for i, ratio_key in enumerate(ratios):
#             if ratio_key not in RATIO_PRESETS:
#                 continue
#             out_name = _render_one_ratio(job, source_path, ratio_key, track_points,
#                                           src_w, src_h, words, tmp_dir, loud_measured)
#             job["ratios"][ratio_key] = out_name
#             job["percent"] = 60 + int((i + 1) * 40 / max(1, n))
 
#         job["status"] = "done"
#         job["stage"] = "done"
#         job["percent"] = 100
#     except Exception as e:
#         job["status"] = "error"
#         job["error"] = str(e)
#     finally:
#         # scratch files (trimmed segments, stitched source, .ass files) —
#         # keep only the final per-ratio outputs in EDIT_DIR
#         try:
#             for f in tmp_dir.glob("*"):
#                 f.unlink(missing_ok=True)
#             tmp_dir.rmdir()
#         except OSError:
#             pass
 
 
# # ══════════════════════════════ routes ══════════════════════════════
 
# @editor_bp.route("/api/editor/analyze", methods=["POST"])
# def api_editor_analyze():
#     """Runs face tracking + transcription WITHOUT rendering, so a frontend
#     can show a preview / let the user manually adjust crop keyframes or
#     hook segments before committing to a full render.

#     Every sub-feature below is individually wrapped in try/except (and, for
#     subtitles specifically, a hard time limit). Before this, a single slow
#     or failing piece — most commonly faster-whisper's first-run model
#     download hanging on a slow/no-token connection — silently hung the
#     ENTIRE request forever: no response, no error, nothing in the server
#     log, and the "Analyze a video first" button on the frontend stayed
#     disabled with no way to know why. Now the endpoint always returns
#     within a bounded time with whatever it managed to compute, plus an
#     "*_error" field per feature that failed, so the button reliably
#     re-enables and any failure is visible instead of an infinite hang.
#     """
#     data = request.json or {}
#     source_path = Path(data.get("source_path", ""))
#     if not source_path.exists():
#         return jsonify({"error": "source_path not found"}), 400

#     try:
#         info = _probe(source_path)
#     except Exception as e:
#         return jsonify({"error": f"Could not read this video (probe failed): {e}"}), 500

#     result = {"probe": info, "features_available": feature_status()}

#     if data.get("face_tracking", True) and _HAS_CV2:
#         try:
#             points, w, h, advanced = analyze_face_track(source_path, sample_fps=data.get("track_fps", 2.0))
#             result["face_track"] = points
#             result["face_tracking_mode"] = "advanced" if advanced else "standard"
#             result["source_size"] = {"w": w, "h": h}
#         except Exception as e:
#             result["face_track"] = []
#             result["face_tracking_error"] = str(e)

#     if data.get("subtitles", True) and _HAS_WHISPER:
#         # faster-whisper downloads its model from Hugging Face on first use,
#         # which can take minutes (or hang) on a slow/rate-limited/offline
#         # connection. Bound it in a worker thread so a stuck download can
#         # never hang analyze forever — if it doesn't finish in time we skip
#         # captions for THIS request and move on; the model keeps downloading
#         # in the background, so the NEXT analyze on any video will be fast
#         # and will pick subtitles back up automatically.
#         try:
#             with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
#                 future = ex.submit(transcribe_words, source_path, data.get("whisper_model", "small"))
#                 words = future.result(timeout=180)
#             result["words"] = words
#         except concurrent.futures.TimeoutError:
#             result["words"] = []
#             result["subtitle_error"] = (
#                 "Subtitles timed out (likely still downloading the whisper model in the "
#                 "background on first run) — continuing without captions. Try Analyze again "
#                 "in a minute or two."
#             )
#         except Exception as e:
#             result["words"] = []
#             result["subtitle_error"] = str(e)

#     if data.get("hook_cut", False) and _HAS_NUMPY:
#         try:
#             result["suggested_hook"] = auto_hook_window(source_path, info["duration"], data.get("hook_len", 6.0))
#         except Exception as e:
#             result["hook_cut_error"] = str(e)

#     if data.get("jump_cut", False):
#         try:
#             result["suggested_segments"] = detect_silences(source_path)
#         except Exception as e:
#             result["jump_cut_error"] = str(e)

#     return jsonify(result)
 
 
# @editor_bp.route("/api/editor/upload", methods=["POST"])
# def api_editor_upload():
#     """Lets the frontend upload a local file directly instead of only
#     pointing at a path already on the server (e.g. a downloader.py output)."""
#     f = request.files.get("file")
#     if not f or not f.filename:
#         return jsonify({"error": "No file uploaded"}), 400
#     safe_name = f"{uuid.uuid4().hex[:10]}_{Path(f.filename).name}"
#     dest = EDIT_DIR / "uploads"
#     dest.mkdir(exist_ok=True)
#     path = dest / safe_name
#     f.save(path)
#     return jsonify({"source_path": str(path)})
 
 
# # ══════════════════════════════ fetch-from-link (built-in yt-dlp) ══════════════════════════════
# # A third source alongside "upload from PC" and "pick from downloader" —
# # paste a link right here in the editor. Deliberately skips the full
# # format-picker: it always resolves ONE single, best-available video+audio
# # mp4 (never a video-only stream that would need a silent-audio placeholder,
# # never separate files) and hands that straight to analyze/render.
# #
# # Speed trick mirrors downloader.py: remembers whichever auth mode (cookies
# # file / browser cookies / plain / bypass) last worked and tries that first,
# # so repeat fetches stay fast. If downloader.py is also registered in this
# # process, its already-resolved mode is reused immediately instead of
# # re-discovering it from scratch.
 
# FETCH_JOBS = {}  # fetch_id -> {status, stage, percent, error, source_path, url}
# _FETCH_RESOLVED_MODE = {"mode": None}
# _FETCH_COOKIE_BROWSERS = ["chrome", "edge", "firefox", "brave"]
 
 
# def _fetch_no_console_kwargs():
#     if os.name == "nt":
#         return {"creationflags": subprocess.CREATE_NO_WINDOW}
#     return {}
 
 
# def _fetch_auth_attempts():
#     attempts = []
#     # reuse downloader.py's already-known-good mode first, if it's loaded
#     try:
#         import downloader
#         if downloader._RESOLVED_MODE.get("mode"):
#             attempts.append(downloader._RESOLVED_MODE["mode"])
#     except Exception:
#         pass
#     if _FETCH_RESOLVED_MODE["mode"]:
#         attempts.append(_FETCH_RESOLVED_MODE["mode"])
#     cookies_file = SRC_DIR.parent / "cookies.txt" if SRC_DIR else None
#     rest = []
#     if cookies_file and cookies_file.exists():
#         rest.append("cookies_file")
#     rest.extend(_FETCH_COOKIE_BROWSERS)
#     rest.append("default")
#     rest.append("bypass")
#     for m in rest:
#         if m not in attempts:
#             attempts.append(m)
#     return attempts, cookies_file
 
 
# def _fetch_apply_auth(opts, mode, cookies_file):
#     opts = dict(opts)
#     if mode == "cookies_file" and cookies_file:
#         opts["cookiefile"] = str(cookies_file)
#     elif mode in _FETCH_COOKIE_BROWSERS:
#         opts["cookiesfrombrowser"] = (mode,)
#     elif mode == "bypass":
#         opts["extractor_args"] = {"youtube": {"player_client": ["ios", "android", "mweb"]}}
#     return opts
 
 
# def _fetch_extract(url, extra_opts=None, download=False):
#     base_opts = {
#         "quiet": True, "no_warnings": True, "noplaylist": True,
#         "socket_timeout": 15, "retries": 2, "extractor_retries": 1, "geo_bypass": True,
#     }
#     if FFMPEG_PATH:
#         base_opts["ffmpeg_location"] = str(FFMPEG_PATH)
#     if extra_opts:
#         base_opts.update(extra_opts)
 
#     attempts, cookies_file = _fetch_auth_attempts()
#     last_err = None
#     for mode in attempts:
#         opts = _fetch_apply_auth(base_opts, mode, cookies_file)
#         try:
#             with yt_dlp.YoutubeDL(opts) as ydl:
#                 info = ydl.extract_info(url, download=download)
#             _FETCH_RESOLVED_MODE["mode"] = mode
#             return info, ydl if download else None, None
#         except Exception as e:
#             last_err = e
#             continue
#     return None, None, last_err
 
 
# def _fetch_fmt_duration(sec):
#     if sec is None:
#         return None
#     sec = int(sec)
#     h, rem = divmod(sec, 3600)
#     m, s = divmod(rem, 60)
#     return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
 
 
# def _fetch_fmt_int(n):
#     if n is None:
#         return None
#     try:
#         return f"{int(n):,}"
#     except (TypeError, ValueError):
#         return str(n)
 
 
# def _fetch_fmt_upload_date(d):
#     if not d:
#         return None
#     try:
#         from datetime import datetime as _dt
#         return _dt.strptime(str(d), "%Y%m%d").strftime("%d %b %Y")
#     except ValueError:
#         return str(d)
 
 
# @editor_bp.route("/api/editor/fetch/info", methods=["POST"])
# def api_fetch_info():
#     """Resolves a link (YouTube/Instagram/TikTok/X/Facebook/etc.) and hands
#     back the FULL set of metadata yt-dlp/the source platform already knows
#     about it — title, uploader, view/like counts, upload date, best
#     available resolution — so the UI never has to re-derive any of this by
#     hand, and so a video that started life as a fetched link can show that
#     rich context in the analysis panel too (a plain PC upload naturally
#     won't have this — it only gets the ffprobe technical details)."""
#     if not _HAS_YTDLP:
#         return jsonify({"error": "yt-dlp is not installed on the server — pip install yt-dlp"}), 400
#     data = request.json or {}
#     url = (data.get("url") or "").strip()
#     if not url:
#         return jsonify({"error": "Paste a video URL first"}), 400
#     info, _, err = _fetch_extract(url, extra_opts={"format": "bestvideo+bestaudio/best"})
#     if info is None:
#         return jsonify({"error": f"Could not resolve that link: {err}"}), 400
 
#     best_h = best_w = None
#     for f in info.get("formats", []) or []:
#         if f.get("vcodec") not in (None, "none") and f.get("height"):
#             if not best_h or f["height"] > best_h:
#                 best_h, best_w = f["height"], f.get("width")
 
#     return jsonify({
#         "title": info.get("title"),
#         "uploader": info.get("uploader") or info.get("channel"),
#         "duration": info.get("duration"),
#         "duration_str": _fetch_fmt_duration(info.get("duration")),
#         "thumbnail": info.get("thumbnail"),
#         "view_count": info.get("view_count"),
#         "view_count_str": _fetch_fmt_int(info.get("view_count")),
#         "like_count": info.get("like_count"),
#         "like_count_str": _fetch_fmt_int(info.get("like_count")),
#         "upload_date": info.get("upload_date"),
#         "upload_date_str": _fetch_fmt_upload_date(info.get("upload_date")),
#         "resolution": f"{best_w}x{best_h}" if best_w and best_h else None,
#         "extractor": info.get("extractor_key"),
#         "webpage_url": info.get("webpage_url"),
#     })
 
 
# def _run_fetch_job(fetch_id, url):
#     job = FETCH_JOBS[fetch_id]
#     job["stage"] = "connect"
 
#     def hook(d):
#         if d.get("status") == "downloading":
#             total = d.get("total_bytes") or d.get("total_bytes_estimate")
#             downloaded = d.get("downloaded_bytes") or 0
#             speed = d.get("speed")
#             job.update({
#                 "status": "downloading", "stage": "download",
#                 "percent": round(downloaded * 100 / total, 1) if total else None,
#                 "speed_str": (_fmt_size(speed) + "/s") if speed else None,
#             })
#         elif d.get("status") == "finished":
#             job["status"] = "processing"
#             job["stage"] = "merge"
 
#     # Saved into SRC_DIR (the SAME "downloads" folder downloader.py uses) so
#     # a link fetched here shows up in /api/editor/sources too, and nothing
#     # about the analyze/render pipeline needs to know it came from here
#     # instead of the Downloader tab or a PC upload.
#     extra_opts = {
#         "format": "bestvideo+bestaudio/best",
#         "outtmpl": str(SRC_DIR / f"editorfetch_{fetch_id}_%(title).60s.%(ext)s"),
#         "progress_hooks": [hook],
#         "merge_output_format": "mp4",   # always ONE muxed video+audio file
#     }
 
#     orig_popen = subprocess.Popen
#     def quiet_popen(*args, **kwargs):
#         kwargs.update(_fetch_no_console_kwargs())
#         return orig_popen(*args, **kwargs)
#     subprocess.Popen = quiet_popen
#     try:
#         info, ydl, err = _fetch_extract(url, extra_opts=extra_opts, download=True)
#     finally:
#         subprocess.Popen = orig_popen
 
#     if info is None:
#         job["status"] = "error"
#         job["error"] = f"Fetch failed: {err}"
#         return
#     try:
#         fname = ydl.prepare_filename(info)
#         p = Path(fname)
#         if not p.exists():
#             p2 = p.with_suffix(".mp4")
#             if p2.exists():
#                 p = p2
#         job["source_path"] = str(p.resolve())
#         job["status"] = "done"
#         job["stage"] = "done"
#         job["percent"] = 100
#     except Exception as e:
#         job["status"] = "error"
#         job["error"] = f"Could not finalize file: {e}"
 
 
# @editor_bp.route("/api/editor/fetch/start", methods=["POST"])
# def api_fetch_start():
#     if not _HAS_YTDLP:
#         return jsonify({"error": "yt-dlp is not installed on the server — pip install yt-dlp"}), 400
#     data = request.json or {}
#     url = (data.get("url") or "").strip()
#     if not url:
#         return jsonify({"error": "No URL given"}), 400
#     fetch_id = uuid.uuid4().hex[:10]
#     FETCH_JOBS[fetch_id] = {
#         "status": "starting", "stage": "connect", "percent": 0,
#         "speed_str": None, "error": None, "url": url, "source_path": None,
#     }
#     threading.Thread(target=_run_fetch_job, args=(fetch_id, url), daemon=True).start()
#     return jsonify({"fetch_id": fetch_id})
 
 
# @editor_bp.route("/api/editor/fetch/progress/<fetch_id>")
# def api_fetch_progress(fetch_id):
#     job = FETCH_JOBS.get(fetch_id)
#     if not job:
#         return jsonify({"error": "Unknown fetch job"}), 404
#     return jsonify(job)
 
 
# @editor_bp.route("/api/editor/render", methods=["POST"])
# def api_editor_render():
#     data = request.json or {}
#     source_path = data.get("source_path", "")
#     if not source_path or not Path(source_path).exists():
#         return jsonify({"error": "source_path not found"}), 400
 
#     export_ratios = data.get("export_ratios") or ["9:16"]
#     bad = [r for r in export_ratios if r not in RATIO_PRESETS]
#     if bad:
#         return jsonify({"error": f"Unknown ratio(s): {bad}. Valid: {list(RATIO_PRESETS)}"}), 400
 
#     features = {
#         "face_tracking": bool(data.get("face_tracking", True)),
#         "subtitles": bool(data.get("subtitles", True)),
#         "hook_cut": bool(data.get("hook_cut", False)),
#         "jump_cut": bool(data.get("jump_cut", False)),
#     }
#     if features["face_tracking"] and not _HAS_CV2:
#         return jsonify({"error": "Face tracking requested but opencv-python is not installed"}), 400
#     if features["subtitles"] and not _HAS_WHISPER:
#         return jsonify({"error": "Subtitles requested but faster-whisper is not installed"}), 400
#     if features["hook_cut"] and not data.get("segments") and not _HAS_NUMPY:
#         return jsonify({"error": "Auto hook-cut requested but numpy is not installed"}), 400
 
#     job_id = uuid.uuid4().hex[:10]
#     EDIT_JOBS[job_id] = {
#         "job_id": job_id, "status": "starting", "stage": "queued", "percent": 0,
#         "error": None, "ratios": {},
#         "config": {
#             "source_path": source_path,
#             "export_ratios": export_ratios,
#             "features": features,
#             "caption_style": data.get("caption_style", "bold_pop"),
#             "quality": data.get("quality", "high"),
#             "track_fps": data.get("track_fps", 2.0),
#             "whisper_model": data.get("whisper_model", "small"),
#             "language": data.get("language"),
#             "hook_len": data.get("hook_len", 6.0),
#             "manual_crop_keyframes": data.get("manual_crop_keyframes"),  # [{t,x,y}]
#             "segments": data.get("segments"),  # [{start,end}] manual rearrange/trim order
#             "fill_mode": data.get("fill_mode", "crop"),  # "crop" or "blur_pad"
#             "normalize_audio": bool(data.get("normalize_audio", True)),
#         },
#     }
#     threading.Thread(target=_run_render_job, args=(job_id,), daemon=True).start()
#     return jsonify({"job_id": job_id})
 
 
# @editor_bp.route("/api/editor/progress/<job_id>")
# def api_editor_progress(job_id):
#     job = EDIT_JOBS.get(job_id)
#     if not job:
#         return jsonify({"error": "Unknown edit job"}), 404
#     return jsonify({k: v for k, v in job.items() if k != "config"} | {
#         "features": job["config"]["features"], "export_ratios": job["config"]["export_ratios"]
#     })
 
 
# @editor_bp.route("/api/editor/file/<job_id>/<ratio>")
# def api_editor_file(job_id, ratio):
#     job = EDIT_JOBS.get(job_id)
#     if not job or job.get("status") != "done":
#         return "Not ready", 404
#     fname = job["ratios"].get(ratio)
#     if not fname:
#         return "Ratio not found for this job", 404
#     p = EDIT_DIR / fname
#     if not p.exists():
#         return "Not found", 404
#     return send_file(p, as_attachment=True, download_name=fname)
 
 
# # ── manual crop keyframe add/remove (per not-yet-rendered job config) ──
 
# @editor_bp.route("/api/editor/keyframe/add", methods=["POST"])
# def api_keyframe_add():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     kf = {"t": float(data["t"]), "x": int(data["x"]), "y": int(data["y"])}
#     job["config"].setdefault("manual_crop_keyframes", [])
#     job["config"]["manual_crop_keyframes"] = job["config"]["manual_crop_keyframes"] or []
#     job["config"]["manual_crop_keyframes"].append(kf)
#     job["config"]["manual_crop_keyframes"].sort(key=lambda k: k["t"])
#     return jsonify({"manual_crop_keyframes": job["config"]["manual_crop_keyframes"]})
 
 
# @editor_bp.route("/api/editor/keyframe/remove", methods=["POST"])
# def api_keyframe_remove():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     t = float(data["t"])
#     kfs = job["config"].get("manual_crop_keyframes") or []
#     job["config"]["manual_crop_keyframes"] = [k for k in kfs if abs(k["t"] - t) > 1e-6]
#     return jsonify({"manual_crop_keyframes": job["config"]["manual_crop_keyframes"]})
 
 
# # ── hook-cut / rearrange segment add/remove ──
 
# @editor_bp.route("/api/editor/segment/add", methods=["POST"])
# def api_segment_add():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     seg = {"start": float(data["start"]), "end": float(data["end"])}
#     index = data.get("index")
#     segs = job["config"].get("segments") or []
#     if index is None or index >= len(segs):
#         segs.append(seg)
#     else:
#         segs.insert(int(index), seg)
#     job["config"]["segments"] = segs
#     return jsonify({"segments": segs})
 
 
# @editor_bp.route("/api/editor/segment/remove", methods=["POST"])
# def api_segment_remove():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     index = int(data["index"])
#     segs = job["config"].get("segments") or []
#     if 0 <= index < len(segs):
#         segs.pop(index)
#     job["config"]["segments"] = segs
#     return jsonify({"segments": segs})










# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# AutoShortAi — Video Editor module
# ──────────────────────────────────
# Adds AI auto-reframe (face tracking), trending animated captions,
# hook-cut / manual clip reordering, and one-click multi-aspect-ratio
# export — all producing a SINGLE muxed video+audio file per ratio.
 
# Wire it into the main app with:
 
#     from video_editor import editor_bp, init_editor
#     init_editor(BASE, FFMPEG)
#     app.register_blueprint(editor_bp)
 
# Routes exposed (all under /api/editor/...):
#     GET  /api/editor/editor             -> serves the standalone editor.html UI (open this in a tab)
#     POST /api/editor/upload             -> upload a local file, returns a source_path to use below
#     POST /api/editor/analyze            -> face-track + transcript + silence/hook preview (for the UI)
#     GET  /api/editor/styles             -> list of caption style presets
#     GET  /api/editor/ratios             -> list of export aspect-ratio presets
#     GET  /api/editor/features           -> which advanced features are available (installed libs)
#     POST /api/editor/render             -> kicks off a background render job
#     GET  /api/editor/progress/<job_id>  -> live progress
#     GET  /api/editor/file/<job_id>/<ratio> -> serves one finished ratio's file
#     POST /api/editor/keyframe/add       -> add a manual crop keyframe to a job's override track
#     POST /api/editor/keyframe/remove    -> remove a manual crop keyframe
#     POST /api/editor/segment/add        -> add a hook-cut / reorder segment
#     POST /api/editor/segment/remove     -> remove a hook-cut / reorder segment
 
# ADVANCED EXTRAS (on top of the original spec):
#     • fill_mode="blur_pad"   -> reframe by fitting the WHOLE frame + blurred
#       background bars instead of hard-cropping (no content ever cut off).
#     • jump_cut feature       -> auto-detects and removes silence/dead-air,
#       the other half of "trending" short-form editing besides hook-cut.
#     • hook_cut + jump_cut COMBINED -> if both are on, the loudest window is
#       used as the cold-open "hook" and is prepended in front of the
#       dead-air-trimmed rest of the video (previously turning jump_cut on
#       silently dropped hook_cut — now they compose).
#     • normalize_audio        -> EBU R128 loudness normalization (-14 LUFS),
#       on by default, matches the loudness social platforms expect.
#     • caption keyword emphasis -> numbers, money, and punchy "power words"
#       auto-highlight in captions even outside full word-by-word styles.
#     • quality/caption_style inputs are now validated with a safe fallback
#       instead of raising a raw KeyError if the frontend sends a typo'd key.
#     • self-serves its own frontend (editor.html) at GET /api/editor/editor,
#       the same "own file, same process/port" pattern as downloader.py.
 
# DEPENDENCIES (install what you want to use — everything degrades gracefully
# if a library is missing, see FEATURE FLAGS AT IMPORT TIME below):
#     pip install opencv-python            # required for any face tracking
#     pip install mediapipe                # optional, gives the "advanced" tracker
#                                           # (falls back to Haar cascade otherwise)
#     pip install faster-whisper           # required for subtitles (word timestamps)
#     pip install numpy
 
# DESIGN NOTES — how each part of the request maps to code:
#   • "advance level face tracking"   -> _FaceTracker (mediapipe BlazeFace if present,
#                                         else OpenCV Haar cascade), EMA-smoothed
#                                         centroid so the crop doesn't jitter frame
#                                         to frame, multi-face -> largest/most-central
#                                         face wins.
#   • "perfect advance subtitles,
#      trending caption styles"       -> CAPTION_STYLES presets + build_ass() which
#                                         groups words into short on-screen chunks and
#                                         emits karaoke-style \\k tags for word-by-word
#                                         pop/highlight animation, burned in with
#                                         ffmpeg's `ass` filter (not a soft-sub track,
#                                         so it always displays correctly everywhere).
#   • "single high quality video+audio
#      in one file"                   -> every render always outputs one .mp4 with
#                                         both streams muxed, CRF-based high quality
#                                         encode (see QUALITY_PRESETS).
#   • "export in all ratio perfectly" -> RATIO_PRESETS, rendered in a loop, one file
#                                         per ratio, each independently croppped using
#                                         the SAME face track (recomputed crop window
#                                         per target aspect).
#   • "rearrange video manually /
#      hook cut"                      -> job["segments"]: an ordered list of
#                                         {start,end} clip ranges. Auto hook-cut picks
#                                         the loudest window as segment 1 automatically;
#                                         manual mode lets the caller add/remove/reorder
#                                         segments directly (segment/add, segment/remove).
#   • "add and remove option for all" -> every feature is a boolean flag in job config
#                                         (features{}) that can be flipped and re-rendered,
#                                         plus explicit add/remove endpoints for the two
#                                         list-based tracks (crop keyframes, segments).
# """
 
# import os
# import json
# import uuid
# import struct
# import threading
# import subprocess
# from pathlib import Path
 
# from flask import Blueprint, request, jsonify, send_file
 
# editor_bp = Blueprint("editor_bp", __name__)
 
# # ══════════════════════════════ embedded frontend ══════════════════════════════
# # The whole editor UI lives right here as one string — no separate .html file
# # to ship or lose track of. Served as-is by GET /api/editor/editor. Its fetch
# # calls use relative paths (same-origin), so it works immediately wherever
# # this blueprint is registered.
# EDITOR_HTML = r"""<!doctype html>
# <html lang="en">
# <head>
# <meta charset="utf-8">
# <meta name="viewport" content="width=device-width, initial-scale=1">
# <title>Reframe — AI video editor</title>
# <link rel="preconnect" href="https://fonts.googleapis.com">
# <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
# <style>
#   /* ── Theme — mirrors the main app's palette exactly (same variable
#      VALUES, just different variable NAMES for this page's own CSS). Default
#      below is "dark-violet", the main app's default. The <html data-theme="…">
#      overrides further down are read straight off the SAME localStorage key
#      the main app writes to (see the sync script at the bottom of <body>),
#      so switching themes on Studio/Downloader also re-themes this page —
#      no manual matching needed ever again. ── */
#   :root{
#     --bg:#0b0d12; --panel:#13151c; --panel-2:#191c25; --border:#262a36;
#     --text:#eef0f6; --text-dim:#8a90a4; --text-faint:#8a90a4;
#     --amber:#6e5bff; --teal:#22d3c4; --red:#ff5a6e;
#     --amber-dim: color-mix(in srgb, var(--amber) 22%, #000);
#     --teal-dim: color-mix(in srgb, var(--teal) 22%, #000);
#     --radius:10px;
#     font-size:15px;
#   }
#   html[data-theme="light"]{
#     --bg:#f4f5f9; --panel:#ffffff; --panel-2:#eef0f6; --border:#dde1ea;
#     --text:#171923; --text-dim:#5b6472; --text-faint:#5b6472;
#     --amber:#5b46e0; --teal:#0891b2; --red:#e0324a;
#   }
#   html[data-theme="midnight-blue"]{
#     --bg:#050a16; --panel:#0c1526; --panel-2:#12213a; --border:#1f3358;
#     --text:#e8f1ff; --text-dim:#7f93b8; --text-faint:#7f93b8;
#     --amber:#3b82f6; --teal:#22d3ee; --red:#ff5a6e;
#   }
#   html[data-theme="forest"]{
#     --bg:#0a130d; --panel:#0f1e14; --panel-2:#16301f; --border:#234a30;
#     --text:#eafbea; --text-dim:#86b399; --text-faint:#86b399;
#     --amber:#22c55e; --teal:#a3e635; --red:#ff5a6e;
#   }
#   html[data-theme="sunset"]{
#     --bg:#1a0e12; --panel:#241419; --panel-2:#331d27; --border:#4a2635;
#     --text:#fdf1ee; --text-dim:#cb9aa3; --text-faint:#cb9aa3;
#     --amber:#fb7185; --teal:#f59e0b; --red:#ff5a6e;
#   }
#   html[data-theme="ocean"]{
#     --bg:#061319; --panel:#0b1f27; --panel-2:#123039; --border:#1e4a56;
#     --text:#e6fbff; --text-dim:#7fb8c4; --text-faint:#7fb8c4;
#     --amber:#06b6d4; --teal:#6366f1; --red:#ff5a6e;
#   }
#   html[data-theme="rose"]{
#     --bg:#170f14; --panel:#221720; --panel-2:#2e1e2b; --border:#4a2f45;
#     --text:#fdeef5; --text-dim:#c79ab0; --text-faint:#c79ab0;
#     --amber:#ec4899; --teal:#f472b6; --red:#ff5a6e;
#   }
#   html[data-theme="amber"]{
#     --bg:#160f05; --panel:#221708; --panel-2:#33230d; --border:#4d3416;
#     --text:#fdf4e3; --text-dim:#c9a877; --text-faint:#c9a877;
#     --amber:#f59e0b; --teal:#eab308; --red:#ff5a6e;
#   }
#   html[data-theme="slate"]{
#     --bg:#101215; --panel:#181b1f; --panel-2:#20242a; --border:#333941;
#     --text:#eceef1; --text-dim:#9aa2ad; --text-faint:#9aa2ad;
#     --amber:#64748b; --teal:#94a3b8; --red:#ff5a6e;
#   }
#   html[data-theme="cyberpunk"]{
#     --bg:#0a0014; --panel:#150124; --panel-2:#210235; --border:#4a0a6b;
#     --text:#f5e6ff; --text-dim:#b083d6; --text-faint:#b083d6;
#     --amber:#e91e9c; --teal:#00f0ff; --red:#ff5a6e;
#   }
#   *{box-sizing:border-box;}
#   body{
#     margin:0; background:var(--bg); color:var(--text);
#     font-family:'Inter',sans-serif; min-height:100vh;
#   }
#   h1,h2,h3,.display{font-family:'Space Grotesk',sans-serif;}
#   .mono{font-family:'JetBrains Mono',monospace;}
#   ::selection{background:var(--amber-dim); color:var(--amber);}
 
#   /* ── top bar ── */
#   .topbar{
#     display:flex; align-items:center; gap:16px; padding:14px 22px;
#     border-bottom:1px solid var(--border); background:var(--panel);
#     position:sticky; top:0; z-index:20;
#   }
#   .brand{display:flex; align-items:center; gap:10px;}
#   .brand .dot{width:10px; height:10px; border-radius:50%; background:var(--amber); box-shadow:0 0 10px var(--amber);}
#   .brand h1{font-size:18px; font-weight:700; margin:0; letter-spacing:.2px;}
#   .brand span{color:var(--text-dim); font-size:12px;}
#   .topbar .spacer{flex:1;}
#   .api-input{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text-dim);
#     border-radius:8px; padding:7px 10px; font-size:12px; font-family:'JetBrains Mono',monospace;
#     width:220px;
#   }
#   .api-input:focus{outline:none; border-color:var(--amber); color:var(--text);}
 
#   /* ── layout ── */
#   .app{display:grid; grid-template-columns:1fr 380px; gap:1px; background:var(--border);}
#   @media (max-width:980px){ .app{grid-template-columns:1fr;} }
#   .col{background:var(--bg); padding:22px; min-width:0;}
 
#   /* ── upload zone ── */
#   .dropzone{
#     border:1.5px dashed var(--border); border-radius:var(--radius);
#     padding:40px 20px; text-align:center; cursor:pointer;
#     transition:border-color .15s, background .15s;
#   }
#   .dropzone:hover, .dropzone.drag{border-color:var(--amber); background:rgba(255,176,32,.04);}
#   .dropzone svg{opacity:.5; margin-bottom:10px;}
#   .dropzone .hint{color:var(--text-dim); font-size:13px; margin-top:6px;}
 
#   /* ── fetch-from-link (ytdlp-style resolver, built into the editor) ── */
#   .fetch-section{border:1px solid var(--border); border-radius:var(--radius); padding:12px; background:var(--panel-2); margin-bottom:14px;}
#   .fetch-row{display:flex; gap:8px; margin-top:8px;}
#   .fetch-input{
#     flex:1; background:var(--panel); border:1px solid var(--border); color:var(--text);
#     border-radius:8px; padding:9px 10px; font-size:13px;
#   }
#   .fetch-input:focus{outline:none; border-color:var(--amber);}
#   .btn-accent{background:var(--amber); color:#1a1200; border:none; font-weight:600;}
#   .btn-accent:hover{filter:brightness(1.08);}
#   .fetch-card{
#     display:flex; gap:10px; align-items:center; margin-top:10px;
#     border:1px solid var(--border); border-radius:8px; padding:10px; background:var(--panel);
#   }
#   .fetch-thumb{width:88px; height:56px; object-fit:cover; border-radius:6px; background:#000; flex-shrink:0;}
#   .fetch-meta{flex:1; min-width:0;}
#   .fetch-title{font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
#   .fetch-sub{font-size:11.5px; color:var(--text-dim); margin-top:2px;}
#   .fetch-status{font-size:12px; color:var(--text-dim); margin-top:8px; min-height:16px;}
 
#   /* ── server library (downloader output + past uploads) ── */
#   .or-divider{
#     display:flex; align-items:center; gap:10px; margin:14px 0;
#     color:var(--text-faint); font-size:12px; text-transform:uppercase; letter-spacing:.5px;
#   }
#   .or-divider::before, .or-divider::after{content:""; flex:1; height:1px; background:var(--border);}
#   .lib-section{border:1px solid var(--border); border-radius:var(--radius); padding:12px; background:var(--panel-2);}
#   .lib-header{display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; font-size:13px; color:var(--text-dim);}
#   .lib-grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:8px; max-height:220px; overflow-y:auto;}
#   .lib-empty{color:var(--text-faint); font-size:12.5px; grid-column:1/-1;}
#   .lib-item{
#     border:1px solid var(--border); border-radius:8px; padding:10px; cursor:pointer;
#     background:var(--panel); transition:border-color .15s, transform .1s;
#   }
#   .lib-item:hover{border-color:var(--amber); transform:translateY(-1px);}
#   .lib-item.active{border-color:var(--teal); background:var(--teal-dim);}
#   .lib-item .lib-name{font-size:12.5px; font-weight:600; word-break:break-word; margin-bottom:4px;}
#   .lib-item .lib-meta{font-size:11px; color:var(--text-dim); display:flex; justify-content:space-between;}
#   .lib-item .lib-tag{
#     display:inline-block; font-size:10px; padding:1px 6px; border-radius:20px;
#     background:var(--amber-dim); color:var(--amber); margin-bottom:4px;
#   }
 
#   /* ── preview ── */
#   .preview-wrap{position:relative; display:flex; justify-content:center; margin-bottom:16px;}
#   .preview-frame{position:relative; background:#000; border-radius:8px; overflow:hidden; max-width:100%;}
#   video{display:block; max-width:100%; max-height:60vh;}
#   .crop-box{
#     position:absolute; border:1.5px solid var(--amber); pointer-events:none;
#     box-shadow:0 0 0 2000px rgba(0,0,0,.45);
#   }
#   .crop-box .corner{
#     position:absolute; width:16px; height:16px; border-color:var(--amber); border-style:solid; border-width:0;
#   }
#   .crop-box .tl{top:-1.5px; left:-1.5px; border-top-width:3px; border-left-width:3px;}
#   .crop-box .tr{top:-1.5px; right:-1.5px; border-top-width:3px; border-right-width:3px;}
#   .crop-box .bl{bottom:-1.5px; left:-1.5px; border-bottom-width:3px; border-left-width:3px;}
#   .crop-box .br{bottom:-1.5px; right:-1.5px; border-bottom-width:3px; border-right-width:3px;}
#   .crop-box{cursor:grab;}
#   .crop-box.dragging{cursor:grabbing;}
#   .crop-readout{
#     position:absolute; top:8px; left:8px; background:rgba(0,0,0,.6);
#     color:var(--amber); font-size:11px; padding:3px 7px; border-radius:5px;
#   }
 
#   .preview-controls{display:flex; align-items:center; gap:10px; justify-content:center; margin-bottom:8px;}
#   .btn-icon{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text);
#     width:34px; height:34px; border-radius:50%; cursor:pointer; display:flex;
#     align-items:center; justify-content:center;
#   }
#   .btn-icon:hover{border-color:var(--amber);}
#   .time{font-size:12px; color:var(--text-dim); min-width:96px; text-align:center;}
 
#   /* ── timeline ── */
#   .timeline-panel{background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:16px;}
#   .timeline-panel h3{margin:0 0 4px; font-size:13px; letter-spacing:.4px; text-transform:uppercase; color:var(--text-dim);}
#   .timeline{
#     position:relative; height:56px; background:var(--panel-2); border-radius:6px;
#     margin-top:10px; overflow:hidden; cursor:pointer;
#   }
#   .tl-face{position:absolute; top:0; height:18px; background:linear-gradient(90deg, transparent, var(--teal-dim)); opacity:.6;}
#   .tl-seg{
#     position:absolute; top:20px; height:18px; background:var(--teal); opacity:.85; border-radius:3px;
#     display:flex; align-items:center; justify-content:center; font-size:10px; color:#04231D; font-weight:600;
#     cursor:grab;
#   }
#   .tl-hook{position:absolute; top:38px; height:14px; background:var(--amber); opacity:.9; border-radius:3px;}
#   .tl-playhead{position:absolute; top:0; bottom:0; width:2px; background:var(--red);}
#   .tl-legend{display:flex; gap:16px; margin-top:8px; font-size:11px; color:var(--text-dim);}
#   .tl-legend span{display:inline-flex; align-items:center; gap:5px;}
#   .swatch{width:9px; height:9px; border-radius:2px; display:inline-block;}
 
#   .seg-list{margin-top:10px; display:flex; flex-direction:column; gap:6px;}
#   .seg-row{
#     display:flex; align-items:center; gap:8px; background:var(--panel-2);
#     border:1px solid var(--border); border-radius:7px; padding:6px 10px; font-size:12px;
#   }
#   .seg-row .mono{color:var(--teal);}
#   .seg-row .spacer{flex:1;}
#   .seg-row button{background:none; border:none; color:var(--text-faint); cursor:pointer; font-size:14px;}
#   .seg-row button:hover{color:var(--red);}
#   .btn-small{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text-dim);
#     border-radius:6px; padding:6px 10px; font-size:12px; cursor:pointer;
#   }
#   .btn-small:hover{border-color:var(--amber); color:var(--amber);}
 
#   /* ── side panel ── */
#   .section{margin-bottom:26px;}
#   .section h3{
#     font-size:12px; letter-spacing:.5px; text-transform:uppercase; color:var(--text-dim);
#     margin:0 0 12px; display:flex; align-items:center; gap:8px;
#   }
#   .section h3 .num{
#     width:18px; height:18px; border-radius:50%; background:var(--panel-2); border:1px solid var(--border);
#     display:flex; align-items:center; justify-content:center; font-size:10px; color:var(--amber);
#   }
 
#   .toggle-row{
#     display:flex; align-items:center; justify-content:space-between;
#     padding:10px 12px; background:var(--panel); border:1px solid var(--border);
#     border-radius:8px; margin-bottom:8px;
#   }
#   .toggle-row .label{font-size:13px;}
#   .toggle-row .desc{font-size:11px; color:var(--text-faint); margin-top:2px;}
#   .toggle-row.disabled{opacity:.4;}
#   .switch{position:relative; width:38px; height:21px; flex-shrink:0;}
#   .switch input{opacity:0; width:0; height:0;}
#   .slider{
#     position:absolute; inset:0; background:var(--panel-2); border:1px solid var(--border);
#     border-radius:20px; cursor:pointer; transition:.15s;
#   }
#   .slider::before{
#     content:''; position:absolute; width:15px; height:15px; left:2px; top:2px;
#     background:var(--text-dim); border-radius:50%; transition:.15s;
#   }
#   .switch input:checked + .slider{background:var(--amber-dim); border-color:var(--amber);}
#   .switch input:checked + .slider::before{transform:translateX(17px); background:var(--amber);}
#   .switch input:disabled + .slider{cursor:not-allowed;}
 
#   .ratio-grid{display:grid; grid-template-columns:1fr 1fr; gap:8px;}
#   .ratio-card{
#     border:1px solid var(--border); background:var(--panel); border-radius:8px;
#     padding:10px; cursor:pointer; text-align:center; position:relative;
#   }
#   .ratio-card.active{border-color:var(--amber); background:rgba(255,176,32,.06);}
#   .ratio-card .shape{margin:0 auto 6px; background:var(--panel-2); border:1px solid var(--border); border-radius:3px;}
#   .ratio-card .name{font-size:12px; font-weight:600;}
#   .ratio-card .lbl{font-size:10px; color:var(--text-faint); margin-top:2px;}
 
#   .style-list{display:flex; flex-direction:column; gap:8px;}
#   .style-card{
#     display:flex; align-items:center; gap:12px; border:1px solid var(--border);
#     background:var(--panel); border-radius:8px; padding:10px 12px; cursor:pointer;
#   }
#   .style-card.active{border-color:var(--teal);}
#   .style-swatch{
#     flex-shrink:0; width:64px; height:36px; border-radius:6px; background:#000;
#     display:flex; align-items:center; justify-content:center; font-size:9px; font-weight:800;
#     letter-spacing:.5px;
#   }
#   .style-card .name{font-size:12.5px; font-weight:600;}
#   .style-card .lbl{font-size:10.5px; color:var(--text-faint);}
 
#   select, .fill-toggle{
#     width:100%; background:var(--panel); border:1px solid var(--border); color:var(--text);
#     padding:9px 10px; border-radius:8px; font-size:13px;
#   }
#   .fill-toggle{display:flex; gap:6px;}
#   .fill-opt{flex:1; text-align:center; padding:8px; border-radius:6px; cursor:pointer; font-size:12px; border:1px solid var(--border);}
#   .fill-opt.active{border-color:var(--amber); color:var(--amber); background:rgba(255,176,32,.06);}
 
#   .render-btn{
#     width:100%; background:var(--amber); color:#1A1204; border:none; border-radius:10px;
#     padding:14px; font-size:14px; font-weight:700; cursor:pointer; font-family:'Space Grotesk',sans-serif;
#     letter-spacing:.3px;
#   }
#   .render-btn:disabled{background:var(--panel-2); color:var(--text-faint); cursor:not-allowed;}
#   .render-btn:not(:disabled):hover{filter:brightness(1.08);}
 
#   .progress-wrap{margin-top:14px;}
#   .progress-track{height:6px; background:var(--panel-2); border-radius:4px; overflow:hidden;}
#   .progress-fill{height:100%; background:linear-gradient(90deg, var(--teal), var(--amber)); width:0%; transition:width .3s;}
#   .progress-label{display:flex; justify-content:space-between; font-size:11px; color:var(--text-dim); margin-top:6px;}
 
#   .results{display:flex; flex-direction:column; gap:8px; margin-top:14px;}
#   .result-row{
#     display:flex; align-items:center; gap:10px; background:var(--panel); border:1px solid var(--border);
#     border-radius:8px; padding:10px 12px;
#   }
#   .result-row .name{font-size:12.5px; font-weight:600; flex:1;}
#   .result-row a{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--teal);
#     text-decoration:none; font-size:11.5px; padding:6px 10px; border-radius:6px;
#   }
#   .result-row a:hover{border-color:var(--teal);}
 
#   .status-line{font-size:11.5px; color:var(--text-faint); margin-top:10px; line-height:1.5;}
#   .status-line b{color:var(--text-dim);}
#   .feature-warn{
#     font-size:11px; color:var(--amber); background:rgba(255,176,32,.08); border:1px solid var(--amber-dim);
#     border-radius:6px; padding:8px 10px; margin-top:8px; display:none;
#   }
# </style>
# </head>
# <body>
# <script>
# // ── Theme sync — this page is served from the SAME Flask app/port as the
# // main Studio (registered as a blueprint), and when opened as an <iframe> on
# // that page it's also same-origin, so it shares localStorage automatically.
# // Reading 'shortsStudioTheme' here — the exact key the main page's theme
# // picker writes to — means this editor always matches whatever theme is
# // active there, with zero manual setup. Runs immediately (before first
# // paint) so there's no flash of the wrong palette, and again whenever the
# // theme changes elsewhere (the 'storage' event fires in this frame whenever
# // another same-origin window/frame updates localStorage).
# (function(){
#   function applyTheme(){
#     const t = localStorage.getItem('shortsStudioTheme') || 'dark-violet';
#     if(t === 'dark-violet') document.documentElement.removeAttribute('data-theme');
#     else document.documentElement.setAttribute('data-theme', t);
#   }
#   applyTheme();
#   window.addEventListener('storage', (e)=>{ if(e.key === 'shortsStudioTheme') applyTheme(); });
# })();
# </script>
 
# <div class="topbar">
#   <div class="brand"><span class="dot"></span><h1>Reframe</h1><span>face-tracked auto edit</span></div>
#   <div class="spacer"></div>
#   <span id="apiBaseHint" class="mono" style="font-size:11px; color:var(--text-faint); margin-right:6px;">same server</span>
#   <button type="button" id="apiBaseToggle" class="btn-small" title="Only needed if this page is served from a different host than the API">⚙</button>
#   <input id="apiBase" class="api-input mono" placeholder="leave blank — uses this same page's server" value="" style="display:none;">
# </div>
 
# <div class="app">
#   <!-- LEFT: preview + timeline -->
#   <div class="col">
#     <div class="fetch-section">
#       <div class="lib-header">
#         <span>🔗 Fetch from a link (YouTube / Instagram / TikTok / X / Facebook…)</span>
#       </div>
#       <div class="fetch-row">
#         <input type="text" id="fetchUrl" class="fetch-input mono" placeholder="Paste a video URL…">
#         <button type="button" class="btn-small btn-accent" id="fetchInfoBtn">Fetch</button>
#       </div>
#       <div id="fetchCard" class="fetch-card" style="display:none;">
#         <img id="fetchThumb" class="fetch-thumb" alt="">
#         <div class="fetch-meta">
#           <div id="fetchTitle" class="fetch-title"></div>
#           <div id="fetchSub" class="fetch-sub"></div>
#         </div>
#         <button type="button" class="btn-small btn-accent" id="fetchUseBtn">⬇ Fetch high-quality &amp; use</button>
#       </div>
#       <div id="fetchStatus" class="fetch-status"></div>
#     </div>
 
#     <div class="or-divider">or drop / choose a video from this PC</div>
#     <div id="dropzone" class="dropzone">
#       <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8A93A3" stroke-width="1.5"><path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v3a2 2 0 002 2h12a2 2 0 002-2v-3"/></svg>
#       <div>Drop a video, or click to choose one</div>
#       <div class="hint">mp4 / mov · analyzed locally before render</div>
#       <input type="file" id="fileInput" accept="video/*" style="display:none">
#     </div>
 
#     <div class="or-divider">or pick a video already fetched by the Downloader</div>
#     <div class="lib-section">
#       <div class="lib-header">
#         <span>📥 From Downloader / server library</span>
#         <button type="button" class="btn-small" id="refreshLibBtn">Refresh</button>
#       </div>
#       <div class="lib-grid" id="libGrid"><span class="lib-empty">Click Refresh to list videos already downloaded via the Downloader tab.</span></div>
#     </div>
 
#     <div id="previewSection" style="display:none;">
#       <div class="preview-wrap">
#         <div class="preview-frame" id="previewFrame">
#           <video id="video" muted></video>
#           <div class="crop-box" id="cropBox" style="display:none;">
#             <div class="corner tl"></div><div class="corner tr"></div><div class="corner bl"></div><div class="corner br"></div>
#             <div class="crop-readout" id="cropReadout">9:16</div>
#           </div>
#         </div>
#       </div>
#       <div class="preview-controls">
#         <div class="btn-icon" id="playBtn">▶</div>
#         <div class="time mono" id="timeLabel">0:00 / 0:00</div>
#       </div>
 
#       <div class="timeline-panel">
#         <h3>Timeline</h3>
#         <div class="timeline" id="timeline">
#           <div class="tl-playhead" id="playhead" style="left:0%"></div>
#         </div>
#         <div class="tl-legend">
#           <span><span class="swatch" style="background:var(--teal-dim)"></span>face track</span>
#           <span><span class="swatch" style="background:var(--teal)"></span>segments</span>
#           <span><span class="swatch" style="background:var(--amber)"></span>suggested hook</span>
#         </div>
#         <div class="seg-list" id="segList"></div>
#         <div style="display:flex; gap:8px; margin-top:10px;">
#           <button class="btn-small" id="addSegBtn">+ add segment at playhead (±3s)</button>
#           <button class="btn-small" id="useHookBtn" style="display:none;">use suggested hook</button>
#           <button class="btn-small" id="useSilenceBtn" style="display:none;">use auto jump-cut segments</button>
#         </div>
#         <div style="display:flex; gap:8px; margin-top:8px;">
#           <button class="btn-small" id="addKfBtn">+ pin crop here (manual keyframe)</button>
#           <button class="btn-small" id="clearKfBtn">clear manual crop</button>
#         </div>
#       </div>
#       <div class="status-line" id="statusLine"></div>
#       <div id="sourceMetaPanel" class="lib-section" style="display:none; margin-top:10px;"></div>
#     </div>
#   </div>
 
#   <!-- RIGHT: controls -->
#   <div class="col">
#     <div class="section">
#       <h3><span class="num">1</span>Auto-reframe</h3>
#       <div class="toggle-row" id="rowFace">
#         <div><div class="label">Face tracking</div><div class="desc">Advanced (mediapipe) if installed, else standard cascade</div></div>
#         <label class="switch"><input type="checkbox" id="chkFace" checked><span class="slider"></span></label>
#       </div>
#       <div class="fill-toggle" style="margin-top:8px;">
#         <div class="fill-opt active" data-fill="crop">Crop to face</div>
#         <div class="fill-opt" data-fill="blur_pad">Fit + blurred bars</div>
#       </div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">2</span>Captions</h3>
#       <div class="toggle-row" id="rowSubs">
#         <div><div class="label">Auto subtitles</div><div class="desc">Word-level, keyword emphasis auto-highlighted</div></div>
#         <label class="switch"><input type="checkbox" id="chkSubs" checked><span class="slider"></span></label>
#       </div>
#       <div class="style-list" id="styleList"></div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">3</span>Hook &amp; pacing</h3>
#       <div class="toggle-row" id="rowHook">
#         <div><div class="label">Hook-cut</div><div class="desc">Open on the loudest / most energetic moment</div></div>
#         <label class="switch"><input type="checkbox" id="chkHook"><span class="slider"></span></label>
#       </div>
#       <div class="toggle-row" id="rowJump">
#         <div><div class="label">Jump-cut silences</div><div class="desc">Auto-remove dead air between lines</div></div>
#         <label class="switch"><input type="checkbox" id="chkJump"><span class="slider"></span></label>
#       </div>
#       <div class="toggle-row">
#         <div><div class="label">Loudness normalize</div><div class="desc">-14 LUFS, matches platform playback volume</div></div>
#         <label class="switch"><input type="checkbox" id="chkNorm" checked><span class="slider"></span></label>
#       </div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">4</span>Export ratios</h3>
#       <div class="ratio-grid" id="ratioGrid"></div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">5</span>Quality</h3>
#       <select id="qualitySel">
#         <option value="high">High (crf 18, slow) — best quality</option>
#         <option value="medium">Medium (crf 21) — balanced</option>
#         <option value="fast">Fast (crf 23) — quick preview</option>
#       </select>
#     </div>
 
#     <button class="render-btn" id="renderBtn" disabled>Analyze a video first</button>
#     <div class="feature-warn" id="featureWarn"></div>
#     <div class="progress-wrap" id="progressWrap" style="display:none;">
#       <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
#       <div class="progress-label"><span id="progressStage">queued</span><span id="progressPct">0%</span></div>
#     </div>
#     <div class="results" id="results"></div>
#   </div>
# </div>
 
# <script>
# const $ = (id) => document.getElementById(id);
# const state = {
#   sourcePath: null, sourceMeta: null, sourceOrigin: 'pc', duration: 0, srcW: 0, srcH: 0,
#   faceTrack: [], words: [], suggestedHook: null, suggestedSilences: null,
#   segments: [], manualKeyframes: [],
#   // Multiple export ratios can be selected at once now (a Set of ratio
#   // keys, e.g. {'9:16','1:1'}), not just one — "auto" is pre-selected by
#   // default since it's the safest zero-quality-loss choice.
#   selectedRatios: new Set(['auto']),
#   previewRatio: '9:16', // drives the live crop box only, independent of export selection
#   selectedStyle: 'bold_pop', fillMode: 'crop',
#   jobId: null, dragging: false,
# };
 
# // Same-origin by default: this page is served BY the same Flask app/port
# // that Reframe, the Downloader, and everything else run on (they're all
# // registered on one Flask app as blueprints, per RenderDetect's main()),
# // so relative paths already hit the right server. The "⚙" override is
# // only for the rare case this HTML got copied to a different host — if
# // left blank (the default), api() never leaves the current origin, which
# // is what fixes the class of bug where a stray/incorrect API host caused
# // every request (analyze, render, fetch…) to silently fail.
# function apiBase(){ const v = $('apiBase').value.trim().replace(/\/$/, ''); return v; }
# function api(path){ return apiBase() + path; }
# $('apiBaseToggle').onclick = ()=>{
#   const el = $('apiBase');
#   const show = el.style.display === 'none';
#   el.style.display = show ? 'inline-block' : 'none';
#   $('apiBaseHint').style.display = show ? 'none' : 'inline';
# };
 
# const RATIO_META = {
#   '9:16': {w:9,h:16}, '16:9': {w:16,h:9}, '1:1': {w:1,h:1}, '4:5': {w:4,h:5}, '4:3': {w:4,h:3},
# };
# const STYLE_META = {
#   bold_pop:       {bg:'#111', color:'#fff', hi:'#FFD700', sample:'THIS IS'},
#   karaoke_classic:{bg:'#111', color:'#fff', hi:'#FFA500', sample:'FILLS IN'},
#   neon_glow:      {bg:'#111', color:'#fff', hi:'#00F0FF', sample:'GLOWS UP'},
#   minimal_clean:  {bg:'#111', color:'#fff', hi:'#fff', sample:'stays quiet'},
#   creator_bold:   {bg:'#111', color:'#FFFF00', hi:'#FFFF00', sample:'HUGE TEXT'},
# };
 
# // ── boot: load styles / ratios / features from backend ──
# async function loadMeta(){
#   try{
#     const [styles, ratios, features] = await Promise.all([
#       fetch(api('/api/editor/styles')).then(r=>r.json()),
#       fetch(api('/api/editor/ratios')).then(r=>r.json()),
#       fetch(api('/api/editor/features')).then(r=>r.json()),
#     ]);
#     renderStyles(styles);
#     renderRatios(ratios);
#     applyFeatureAvailability(features);
#   }catch(e){
#     $('featureWarn').style.display='block';
#     $('featureWarn').textContent = 'Could not reach the backend at ' + api('') + ' — set the API base URL above and reload.';
#   }
# }
 
# function renderStyles(styles){
#   const list = $('styleList'); list.innerHTML='';
#   Object.entries(styles).forEach(([key, meta])=>{
#     const m = STYLE_META[key] || {bg:'#111', color:'#fff', hi:'#fff', sample:'Sample'};
#     const card = document.createElement('div');
#     card.className = 'style-card' + (key===state.selectedStyle ? ' active':'');
#     card.innerHTML = `<div class="style-swatch" style="background:${m.bg}">
#         <span style="color:${m.color}">${m.sample.split(' ')[0]}</span>&nbsp;<span style="color:${m.hi}">${m.sample.split(' ')[1]||''}</span>
#       </div>
#       <div><div class="name">${meta.label.split(' (')[0]}</div><div class="lbl">${(meta.label.match(/\((.*)\)/)||[,''])[1]}</div></div>`;
#     card.onclick = ()=>{ state.selectedStyle = key; renderStyles(styles); };
#     list.appendChild(card);
#   });
# }
 
# // Export ratios are now MULTI-select — tap any number of cards and every
# // one you pick gets rendered and offered as a download, all in the same
# // render job. "Auto" (native size, no crop, zero quality loss) is its own
# // card too. state.previewRatio (separate from the export selection) just
# // drives which crop box is drawn live over the video for editing.
# function renderRatios(ratios){
#   const grid = $('ratioGrid'); grid.innerHTML='';
#   Object.entries(ratios).forEach(([key, r])=>{
#     const isAuto = key === 'auto';
#     const srcAspect = (state.srcW && state.srcH) ? {w: state.srcW, h: state.srcH} : {w:9, h:16};
#     const meta = isAuto ? srcAspect : (RATIO_META[key] || {w:1,h:1});
#     const scale = 30 / Math.max(meta.w, meta.h);
#     const selected = state.selectedRatios.has(key);
#     const card = document.createElement('div');
#     card.className = 'ratio-card' + (selected ? ' active':'');
#     card.innerHTML = `<div class="shape" style="width:${meta.w*scale}px; height:${meta.h*scale}px;"></div>
#       <div class="name">${selected ? '✓ ' : ''}${key === 'auto' ? 'Auto' : key}</div><div class="lbl">${r.label}</div>`;
#     card.onclick = ()=>{
#       if(state.selectedRatios.has(key)) state.selectedRatios.delete(key);
#       else state.selectedRatios.add(key);
#       if(!state.selectedRatios.size) state.selectedRatios.add(key); // keep at least one selected
#       if(!isAuto){ state.previewRatio = key; drawCropForCurrentTime(); }
#       renderRatios(ratios);
#     };
#     grid.appendChild(card);
#   });
# }
 
# function applyFeatureAvailability(f){
#   toggleRow('rowFace', 'chkFace', f.face_tracking, 'opencv-python not installed on the server');
#   toggleRow('rowSubs', 'chkSubs', f.subtitles, 'faster-whisper not installed on the server');
#   toggleRow('rowHook', 'chkHook', true, '');
#   toggleRow('rowJump', 'chkJump', true, '');
#   if (!f.face_tracking_advanced && f.face_tracking){
#     $('rowFace').querySelector('.desc').textContent = 'Standard tracker active (install mediapipe for advanced mode)';
#   }
# }
# function toggleRow(rowId, chkId, available, msg){
#   if(!available){
#     $(rowId).classList.add('disabled');
#     $(chkId).checked = false;
#     $(chkId).disabled = true;
#     $(rowId).querySelector('.desc').textContent = msg;
#   }
# }
 
# // ── fill mode ──
# document.querySelectorAll('.fill-opt').forEach(el=>{
#   el.onclick = ()=>{
#     document.querySelectorAll('.fill-opt').forEach(o=>o.classList.remove('active'));
#     el.classList.add('active');
#     state.fillMode = el.dataset.fill;
#     drawCropForCurrentTime();
#   };
# });
 
# // ── fetch-from-link (built-in yt-dlp resolver, high quality single mp4) ──
# // Mirrors the Downloader tab's speed trick (cached auth mode) but skips the
# // full format-picker UI on purpose: this always grabs the single best
# // available video+audio quality and merges it into one mp4, then feeds it
# // straight into the exact same analyze/render pipeline as an upload.
# let fetchState = { url: null };
 
# $('fetchInfoBtn').onclick = async ()=>{
#   const url = $('fetchUrl').value.trim();
#   if(!url) return;
#   $('fetchStatus').textContent = 'Resolving link…';
#   $('fetchCard').style.display = 'none';
#   try{
#     const res = await fetch(api('/api/editor/fetch/info'), {
#       method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url})
#     });
#     const data = await res.json();
#     if(data.error){ $('fetchStatus').textContent = data.error; return; }
#     fetchState.url = url;
#     fetchState.meta = data;   // keep the FULL metadata (views/likes/uploader/etc.) for later
#     $('fetchThumb').src = data.thumbnail || '';
#     $('fetchTitle').textContent = data.title || 'Untitled video';
#     $('fetchSub').textContent = [data.uploader, data.duration_str, data.resolution].filter(Boolean).join(' · ');
#     $('fetchCard').style.display = 'flex';
#     $('fetchStatus').textContent = '';
#   }catch(e){
#     $('fetchStatus').textContent = 'Could not reach the server to resolve that link. Make sure the app is running and reload this page.';
#   }
# };
 
# $('fetchUseBtn').onclick = async ()=>{
#   $('fetchUseBtn').disabled = true;
#   $('fetchStatus').textContent = 'Fetching best quality (video + audio, single file)…';
#   try{
#     const res = await fetch(api('/api/editor/fetch/start'), {
#       method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url: fetchState.url})
#     });
#     const data = await res.json();
#     if(data.error){ $('fetchStatus').textContent = data.error; $('fetchUseBtn').disabled = false; return; }
#     pollFetchProgress(data.fetch_id);
#   }catch(e){
#     $('fetchStatus').textContent = 'Fetch failed to start — check the connection and try again.';
#     $('fetchUseBtn').disabled = false;
#   }
# };
 
# async function pollFetchProgress(fetchId){
#   let st;
#   try{
#     const res = await fetch(api('/api/editor/fetch/progress/'+fetchId));
#     st = await res.json();
#   }catch(e){
#     setTimeout(()=>pollFetchProgress(fetchId), 1200); // transient network hiccup, keep polling
#     return;
#   }
#   if(st.status === 'error'){
#     $('fetchStatus').textContent = 'Error: ' + st.error;
#     $('fetchUseBtn').disabled = false;
#     return;
#   }
#   if(st.status === 'done'){
#     $('fetchStatus').textContent = 'Downloaded — analyzing…';
#     $('fetchUseBtn').disabled = false;
#     $('fetchCard').style.display = 'none';
#     $('fetchUrl').value = '';
#     loadSource(st.source_path, fetchState.meta, 'fetch');
#     return;
#   }
#   const pct = st.percent != null ? st.percent + '%' : '';
#   $('fetchStatus').textContent = `${st.stage || 'downloading'}… ${pct} ${st.speed_str || ''}`.trim();
#   setTimeout(()=>pollFetchProgress(fetchId), 900);
# }
 
# // ── server library (downloader output + earlier uploads) ──
# // Lets the user reuse a video already fetched via the Downloader tab (or an
# // earlier upload) without picking it from disk again — same analyze/render
# // pipeline as a fresh upload, just a different source_path origin.
# $('refreshLibBtn').onclick = loadLibrary;
 
# async function loadLibrary(){
#   const grid = $('libGrid');
#   grid.innerHTML = '<span class="lib-empty">Loading…</span>';
#   try{
#     const res = await fetch(api('/api/editor/sources'));
#     const data = await res.json();
#     const items = data.sources || [];
#     if(!items.length){
#       grid.innerHTML = '<span class="lib-empty">Nothing yet — fetch a video in the Downloader tab, or upload one above.</span>';
#       return;
#     }
#     grid.innerHTML = '';
#     items.forEach(it=>{
#       const el = document.createElement('div');
#       el.className = 'lib-item';
#       el.innerHTML = `<span class="lib-tag">${it.origin === 'downloader' ? '📥 downloaded' : '📤 uploaded'}</span>
#         <div class="lib-name">${it.name}</div>
#         <div class="lib-meta"><span>${it.size_str||''}</span><span>${it.modified_str||''}</span></div>`;
#       el.onclick = ()=>{
#         document.querySelectorAll('.lib-item.active').forEach(n=>n.classList.remove('active'));
#         el.classList.add('active');
#         loadSource(it.path, null, it.origin === 'downloader' ? 'downloader' : 'pc');
#       };
#       grid.appendChild(el);
#     });
#   }catch(e){
#     grid.innerHTML = '<span class="lib-empty">Could not reach the server — make sure the app is running, then hit Refresh again.</span>';
#   }
# }
 
# loadLibrary();
 
# // ── upload flow ──
# $('dropzone').onclick = ()=> $('fileInput').click();
# $('fileInput').onchange = (e)=> handleFile(e.target.files[0]);
# ['dragover','dragleave','drop'].forEach(evt=>{
#   $('dropzone').addEventListener(evt, (e)=>{
#     e.preventDefault();
#     $('dropzone').classList.toggle('drag', evt==='dragover');
#     if(evt==='drop' && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
#   });
# });
 
# async function handleFile(file){
#   if(!file) return;
#   $('video').src = URL.createObjectURL(file);
#   $('previewSection').style.display = 'block';
#   $('statusLine').textContent = 'Uploading…';
#   try{
#     const fd = new FormData(); fd.append('file', file);
#     const res = await fetch(api('/api/editor/upload'), {method:'POST', body: fd});
#     const data = await res.json();
#     if(data.error){ showAnalysisError('Upload error: ' + data.error); return; }
#     loadSource(data.source_path, null, 'pc');
#   }catch(e){
#     showAnalysisError('Could not reach the server to upload this file.');
#   }
# }
 
# // ── unified source loader — every origin (fetched link, downloader
# // library, or a file picked from this PC) funnels through here so
# // analysis always runs the same way and the Export panel always ends up
# // enabled the same way, no matter where the video came from. ──
# async function loadSource(sourcePath, meta, origin){
#   state.sourcePath = sourcePath;
#   state.sourceMeta = meta || null;
#   state.sourceOrigin = origin || 'pc';
#   $('video').src = api('/api/editor/preview?path=' + encodeURIComponent(sourcePath));
#   $('previewSection').style.display = 'block';
#   renderSourceMeta();
#   await runAnalysis();
# }
 
# // Shows the rich details panel. For links fetched from YouTube/Instagram/
# // etc. this uses the metadata the SERVER already pulled during Fetch (no
# // need to re-derive title/uploader/views by hand); for PC uploads or
# // library items with no such metadata it just relies on the ffprobe-based
# // technical details that come back from /api/editor/analyze.
# function renderSourceMeta(){
#   const el = $('sourceMetaPanel');
#   if(!el) return;
#   const m = state.sourceMeta;
#   if(!m){ el.style.display = 'none'; el.innerHTML=''; return; }
#   const rows = [
#     ['Title', m.title],
#     ['Channel / uploader', m.uploader],
#     ['Views', m.view_count_str],
#     ['Likes', m.like_count_str],
#     ['Uploaded', m.upload_date_str],
#     ['Source resolution', m.resolution],
#   ].filter(([,v]) => v);
#   if(!rows.length){ el.style.display='none'; return; }
#   el.style.display = 'block';
#   el.innerHTML = '<div class="lib-header"><span>ℹ️ Fetched video details</span></div>' +
#     rows.map(([k,v]) => `<div class="status-line"><b>${k}:</b> ${v}</div>`).join('');
# }
 
# function showAnalysisError(msg){
#   $('statusLine').innerHTML = `<span style="color:var(--amber);">${msg}</span> ` +
#     `<button type="button" class="btn-small" id="retryAnalysisBtn" style="margin-left:6px;">Retry analysis</button>`;
#   const btn = $('retryAnalysisBtn');
#   if(btn) btn.onclick = () => runAnalysis();
#   $('renderBtn').disabled = true;
#   $('renderBtn').textContent = 'Analyze a video first';
# }
 
# async function runAnalysis(){
#   if(!state.sourcePath){ showAnalysisError('No video selected yet.'); return; }
#   $('statusLine').textContent = 'Analyzing — face track + transcript + hook detection…';
#   $('renderBtn').disabled = true;
#   const body = {
#     source_path: state.sourcePath,
#     face_tracking: $('chkFace').checked && !$('chkFace').disabled,
#     subtitles: $('chkSubs').checked && !$('chkSubs').disabled,
#     hook_cut: true,
#     jump_cut: true,
#   };
#   let data;
#   try{
#     const res = await fetch(api('/api/editor/analyze'), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
#     if(!res.ok){ showAnalysisError(`Analysis request failed (server said HTTP ${res.status}).`); return; }
#     data = await res.json();
#   }catch(e){
#     showAnalysisError('Could not reach the server to analyze this video — check it is still running, then retry.');
#     return;
#   }
#   if(data.error){ showAnalysisError('Analysis error: ' + data.error); return; }
#   state.duration = data.probe.duration || 0;
#   state.srcW = (data.source_size && data.source_size.w) || data.probe.width;
#   state.srcH = (data.source_size && data.source_size.h) || data.probe.height;
#   state.faceTrack = data.face_track || [];
#   state.words = data.words || [];
#   state.suggestedHook = data.suggested_hook || null;
#   state.suggestedSilences = data.suggested_segments || null;
#   $('useHookBtn').style.display = state.suggestedHook ? 'inline-block' : 'none';
#   $('useSilenceBtn').style.display = state.suggestedSilences ? 'inline-block' : 'none';
#   drawTimeline();
#   // Export becomes available the instant analysis finishes, identically
#   // for an uploaded PC file, a Downloader-library pick, or a fresh link
#   // fetch — there is no special-casing by source here.
#   $('renderBtn').disabled = false;
#   $('renderBtn').textContent = 'Render exports';
#   const srcTag = state.sourceOrigin === 'fetch' ? 'fetched video' : (state.sourceOrigin === 'downloader' ? 'downloaded video' : 'uploaded video');
#   $('statusLine').innerHTML = `Analyzed ${srcTag} — <b>${state.faceTrack.length}</b> face samples · <b>${state.words.length}</b> words transcribed · duration <b>${fmtTime(state.duration)}</b> · source <b>${state.srcW}×${state.srcH}</b>`;
# }
 
# // ── video controls ──
# const video = $('video');
# $('playBtn').onclick = ()=>{ video.paused ? video.play() : video.pause(); };
# video.onplay = ()=> $('playBtn').textContent = '❚❚';
# video.onpause = ()=> $('playBtn').textContent = '▶';
# video.ontimeupdate = ()=>{
#   $('timeLabel').textContent = `${fmtTime(video.currentTime)} / ${fmtTime(video.duration||0)}`;
#   const pct = video.duration ? (video.currentTime/video.duration*100) : 0;
#   $('playhead').style.left = pct + '%';
#   drawCropForCurrentTime();
# };
# function fmtTime(t){ t=Math.max(0,t||0); const m=Math.floor(t/60), s=Math.floor(t%60); return `${m}:${String(s).padStart(2,'0')}`; }
 
# // ── crop overlay: shows the current auto/manual crop box, draggable ──
# function currentCrop(){
#   const kfs = state.manualKeyframes.length ? state.manualKeyframes : state.faceTrack.map(p=>({t:p.t, x:p.cx, y:p.cy}));
#   if(!kfs.length) return {cx:0.5, cy:0.5};
#   const t = video.currentTime;
#   let lo = kfs[0], hi = kfs[kfs.length-1];
#   for(let i=0;i<kfs.length-1;i++){ if(kfs[i].t<=t && kfs[i+1].t>=t){ lo=kfs[i]; hi=kfs[i+1]; break; } }
#   const span = (hi.t - lo.t) || 1;
#   const f = Math.min(1, Math.max(0, (t - lo.t)/span));
#   const cxKey = state.manualKeyframes.length ? 'x' : 'cx';
#   const cyKey = state.manualKeyframes.length ? 'y' : 'cy';
#   return { cx: lo[cxKey] + (hi[cxKey]-lo[cxKey])*f, cy: lo[cyKey] + (hi[cyKey]-lo[cyKey])*f };
# }
 
# function drawCropForCurrentTime(){
#   const box = $('cropBox');
#   if(state.fillMode === 'blur_pad' || !$('chkFace').checked){ box.style.display='none'; return; }
#   if(!state.srcW || !video.videoWidth){ return; }
#   const meta = RATIO_META[state.previewRatio || '9:16'];
#   const targetAR = meta.w/meta.h;
#   const srcAR = video.videoWidth/video.videoHeight;
#   let cropWFrac, cropHFrac;
#   if(srcAR > targetAR){ cropHFrac = 1; cropWFrac = targetAR/srcAR; }
#   else { cropWFrac = 1; cropHFrac = srcAR/targetAR; }
#   const {cx, cy} = currentCrop();
#   let left = cx - cropWFrac/2, top = cy - cropHFrac/2;
#   left = Math.min(1-cropWFrac, Math.max(0, left));
#   top = Math.min(1-cropHFrac, Math.max(0, top));
 
#   const frame = $('previewFrame').getBoundingClientRect();
#   box.style.display = 'block';
#   box.style.left = (left*100)+'%';
#   box.style.top = (top*100)+'%';
#   box.style.width = (cropWFrac*100)+'%';
#   box.style.height = (cropHFrac*100)+'%';
#   $('cropReadout').textContent = state.previewRatio || '9:16';
#   box.dataset.cx = cx; box.dataset.cy = cy;
# }
 
# // dragging the crop box records a manual keyframe candidate (committed via "pin crop here")
# let dragStart = null;
# $('cropBox').addEventListener('mousedown', (e)=>{
#   state.dragging = true; $('cropBox').classList.add('dragging');
#   dragStart = {x:e.clientX, y:e.clientY, left:parseFloat($('cropBox').style.left), top:parseFloat($('cropBox').style.top)};
# });
# window.addEventListener('mousemove', (e)=>{
#   if(!state.dragging) return;
#   const frame = $('previewFrame').getBoundingClientRect();
#   const dx = (e.clientX-dragStart.x)/frame.width*100;
#   const dy = (e.clientY-dragStart.y)/frame.height*100;
#   const box = $('cropBox');
#   const newLeft = Math.min(100-parseFloat(box.style.width), Math.max(0, dragStart.left+dx));
#   const newTop = Math.min(100-parseFloat(box.style.height), Math.max(0, dragStart.top+dy));
#   box.style.left = newLeft+'%'; box.style.top = newTop+'%';
#   box.dataset.cx = (newLeft + parseFloat(box.style.width)/2)/100;
#   box.dataset.cy = (newTop + parseFloat(box.style.height)/2)/100;
# });
# window.addEventListener('mouseup', ()=>{ state.dragging=false; $('cropBox').classList.remove('dragging'); });
 
# $('addKfBtn').onclick = ()=>{
#   const box = $('cropBox');
#   const meta = RATIO_META[state.previewRatio || '9:16'];
#   const srcAR = video.videoWidth/video.videoHeight;
#   const targetAR = meta.w/meta.h;
#   let cropW, cropH;
#   if(srcAR > targetAR){ cropH = state.srcH; cropW = Math.round(cropH*targetAR); }
#   else { cropW = state.srcW; cropH = Math.round(cropW/targetAR); }
#   const cx = parseFloat(box.dataset.cx||0.5), cy = parseFloat(box.dataset.cy||0.5);
#   let x = Math.round(cx*state.srcW - cropW/2), y = Math.round(cy*state.srcH - cropH/2);
#   x = Math.max(0, Math.min(x, state.srcW-cropW)); y = Math.max(0, Math.min(y, state.srcH-cropH));
#   state.manualKeyframes = state.manualKeyframes.filter(k=>Math.abs(k.t-video.currentTime)>0.05);
#   state.manualKeyframes.push({t: video.currentTime, x, y});
#   state.manualKeyframes.sort((a,b)=>a.t-b.t);
#   drawTimeline();
#   $('statusLine').textContent = `Pinned manual crop at ${fmtTime(video.currentTime)} — ${state.manualKeyframes.length} manual keyframe(s) active (overrides auto tracking).`;
# };
# $('clearKfBtn').onclick = ()=>{ state.manualKeyframes=[]; drawTimeline(); drawCropForCurrentTime(); };
 
# // ── timeline: face density, segments, hook suggestion ──
# function drawTimeline(){
#   const tl = $('timeline');
#   [...tl.querySelectorAll('.tl-face,.tl-hook')].forEach(e=>e.remove());
#   if(state.duration){
#     state.faceTrack.forEach(p=>{
#       const el = document.createElement('div');
#       el.className='tl-face';
#       el.style.left = (p.t/state.duration*100)+'%'; el.style.width='2px';
#       tl.appendChild(el);
#     });
#     if(state.suggestedHook){
#       const el = document.createElement('div'); el.className='tl-hook';
#       el.style.left = (state.suggestedHook.start/state.duration*100)+'%';
#       el.style.width = ((state.suggestedHook.end-state.suggestedHook.start)/state.duration*100)+'%';
#       tl.appendChild(el);
#     }
#   }
#   renderSegList();
# }
# function renderSegList(){
#   const list = $('segList'); list.innerHTML='';
#   state.segments.forEach((seg, i)=>{
#     const row = document.createElement('div'); row.className='seg-row';
#     row.innerHTML = `<span class="mono">#${i+1}</span><span class="mono">${fmtTime(seg.start)} → ${fmtTime(seg.end)}</span><span class="spacer"></span><button data-i="${i}">✕</button>`;
#     row.querySelector('button').onclick = ()=>{ state.segments.splice(i,1); renderSegList(); };
#     list.appendChild(row);
#   });
# }
# $('timeline').addEventListener('click', (e)=>{
#   const rect = $('timeline').getBoundingClientRect();
#   const frac = (e.clientX-rect.left)/rect.width;
#   video.currentTime = frac * (video.duration||0);
# });
# $('addSegBtn').onclick = ()=>{
#   const t = video.currentTime;
#   state.segments.push({start: Math.max(0, +(t-3).toFixed(2)), end: Math.min(state.duration, +(t+3).toFixed(2))});
#   renderSegList();
# };
# $('useHookBtn').onclick = ()=>{
#   if(!state.suggestedHook) return;
#   state.segments = [state.suggestedHook, {start:0, end: state.duration}];
#   renderSegList();
#   $('chkHook').checked = true;
# };
# $('useSilenceBtn').onclick = ()=>{
#   if(!state.suggestedSilences) return;
#   state.segments = state.suggestedSilences;
#   renderSegList();
#   $('chkJump').checked = true;
# };
 
# // ── render ──
# $('renderBtn').onclick = async ()=>{
#   if(!state.selectedRatios.size){ $('statusLine').textContent = 'Pick at least one export ratio first.'; return; }
#   $('renderBtn').disabled = true;
#   $('progressWrap').style.display = 'block';
#   $('results').innerHTML = '';
#   const body = {
#     source_path: state.sourcePath,
#     export_ratios: Array.from(state.selectedRatios),
#     face_tracking: $('chkFace').checked && !$('chkFace').disabled,
#     subtitles: $('chkSubs').checked && !$('chkSubs').disabled,
#     hook_cut: $('chkHook').checked,
#     jump_cut: $('chkJump').checked,
#     normalize_audio: $('chkNorm').checked,
#     caption_style: state.selectedStyle,
#     fill_mode: state.fillMode,
#     quality: $('qualitySel').value,
#     manual_crop_keyframes: state.manualKeyframes.length ? state.manualKeyframes : null,
#     segments: state.segments.length ? state.segments : null,
#   };
#   try{
#     const res = await fetch(api('/api/editor/render'), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
#     const data = await res.json();
#     if(data.error){ $('statusLine').textContent = 'Render error: '+data.error; $('renderBtn').disabled=false; return; }
#     state.jobId = data.job_id;
#     pollProgress();
#   }catch(e){
#     $('statusLine').textContent = 'Could not reach the server to start the render — check it is still running and try again.';
#     $('renderBtn').disabled = false;
#   }
# };
 
# async function pollProgress(){
#   let job;
#   try{
#     const res = await fetch(api('/api/editor/progress/'+state.jobId));
#     job = await res.json();
#   }catch(e){
#     setTimeout(pollProgress, 1500); // transient hiccup, keep polling
#     return;
#   }
#   $('progressFill').style.width = (job.percent||0)+'%';
#   $('progressStage').textContent = job.stage || job.status;
#   $('progressPct').textContent = (job.percent||0)+'%';
#   if(job.status === 'error'){
#     $('statusLine').textContent = 'Render failed: '+job.error;
#     $('renderBtn').disabled = false;
#     return;
#   }
#   if(job.status !== 'done'){ setTimeout(pollProgress, 1200); return; }
#   $('renderBtn').disabled = false;
#   const results = $('results');
#   Object.entries(job.ratios||{}).forEach(([ratio, fname])=>{
#     const row = document.createElement('div'); row.className='result-row';
#     const tag = ratio === 'auto' ? 'Auto (original size — lossless)' : `${ratio} export`;
#     row.innerHTML = `<span class="name">${tag}</span><a href="${api('/api/editor/file/'+state.jobId+'/'+ratio)}" download>Download</a>`;
#     results.appendChild(row);
#   });
# }
 
# loadMeta();
# </script>
# </body>
# </html>
# """
 
 
# # Optional, best-effort link to downloader.py's in-memory job table so the
# # editor can show nicer names for downloaded videos ("Video Title.mp4"
# # instead of "<dl_id>_Video Title.mp4"). video_editor.py still works fine
# # standalone if downloader.py isn't registered — this never raises.
# def _downloader_jobs():
#     try:
#         import downloader
#         return downloader.DL_JOBS
#     except Exception:
#         return {}
 
# # ── configured once via init_editor() ──────────────────────────────
# EDIT_DIR = None
# SRC_DIR = None
# FFMPEG_PATH = "ffmpeg"
# FFPROBE_PATH = "ffprobe"
 
# # job_id -> {status, stage, percent, error, ratios: {ratio: filename}, config}
# EDIT_JOBS = {}
 
 
# def init_editor(base_dir, ffmpeg_path=None):
#     """Call once at startup, e.g. init_editor(BASE, FFMPEG)."""
#     global EDIT_DIR, SRC_DIR, FFMPEG_PATH, FFPROBE_PATH
#     base_dir = Path(base_dir)
#     EDIT_DIR = base_dir / "edited"
#     EDIT_DIR.mkdir(exist_ok=True)
#     SRC_DIR = base_dir / "downloads"   # reuse downloader's output as source pool
#     (EDIT_DIR / "uploads").mkdir(exist_ok=True)
#     if ffmpeg_path:
#         FFMPEG_PATH = str(ffmpeg_path)
#         # ffprobe normally sits next to ffmpeg
#         cand = Path(ffmpeg_path).with_name(
#             "ffprobe.exe" if str(ffmpeg_path).lower().endswith(".exe") else "ffprobe"
#         )
#         if cand.exists():
#             FFPROBE_PATH = str(cand)
 
 
# def _no_console_kwargs():
#     if os.name == "nt":
#         return {"creationflags": subprocess.CREATE_NO_WINDOW}
#     return {}
 
 
# def _run(cmd):
#     return subprocess.run(cmd, capture_output=True, text=True, **_no_console_kwargs())
 
 
# # ══════════════════════════════ feature availability ══════════════════════════════
 
# try:
#     import cv2
#     _HAS_CV2 = True
# except ImportError:
#     _HAS_CV2 = False
 
# try:
#     import mediapipe as mp
#     _HAS_MEDIAPIPE = True
# except ImportError:
#     _HAS_MEDIAPIPE = False
 
# try:
#     from faster_whisper import WhisperModel
#     _HAS_WHISPER = True
# except ImportError:
#     _HAS_WHISPER = False
 
# try:
#     import numpy as np
#     _HAS_NUMPY = True
# except ImportError:
#     _HAS_NUMPY = False
 
# try:
#     import yt_dlp
#     _HAS_YTDLP = True
# except ImportError:
#     _HAS_YTDLP = False
 
# _WHISPER_MODEL = None  # lazy-loaded singleton
 
 
# def feature_status():
#     return {
#         "face_tracking": _HAS_CV2,
#         "face_tracking_advanced": _HAS_CV2 and _HAS_MEDIAPIPE,
#         "subtitles": _HAS_WHISPER,
#         "hook_cut_auto": _HAS_NUMPY,
#         "fetch_from_link": _HAS_YTDLP,
#     }
 
 
# # ══════════════════════════════ presets ══════════════════════════════
 
# # width x height for each export ratio, chosen at "good enough to look sharp,
# # small enough to encode fast" — bump these if you want 4K exports.
# RATIO_PRESETS = {
#     # "auto" is a sentinel: w/h are resolved at render time from the SOURCE
#     # video's own dimensions (no target size baked in here). It means "keep
#     # the video exactly as shot" — no crop, no letterbox, no rescale — and
#     # whenever nothing else in the pipeline needs to touch the picture
#     # (no captions/color/manual crop requested) it is exported via ffmpeg
#     # stream-copy, i.e. the original video bytes are re-muxed, not
#     # re-encoded, so there is truly zero generational quality loss.
#     "auto":  {"w": None, "h": None, "label": "Auto — original size, no crop, zero quality loss"},
#     "9:16":  {"w": 1080, "h": 1920, "label": "Reels / Shorts / TikTok"},
#     "16:9":  {"w": 1920, "h": 1080, "label": "YouTube / Landscape"},
#     "1:1":   {"w": 1080, "h": 1080, "label": "Square / Feed post"},
#     "4:5":   {"w": 1080, "h": 1350, "label": "Instagram portrait"},
#     "4:3":   {"w": 1440, "h": 1080, "label": "Classic / Facebook"},
# }
 
# QUALITY_PRESETS = {
#     "high":   {"crf": 18, "preset": "slow"},
#     "medium": {"crf": 21, "preset": "medium"},
#     "fast":   {"crf": 23, "preset": "veryfast"},
# }
 
# # Words that get an automatic extra-emphasis color in captions even in
# # styles that don't do full word-by-word highlighting — numbers, money and
# # a short list of "power words" are what trending caption tools punch up.
# _EMPHASIS_PATTERN = None  # compiled lazily, see _is_emphasis_word()
# _POWER_WORDS = {
#     "free", "now", "never", "always", "secret", "insane", "crazy", "huge",
#     "warning", "stop", "wait", "new", "best", "worst", "you", "your",
# }
 
 
# def _is_emphasis_word(word):
#     import re
#     global _EMPHASIS_PATTERN
#     if _EMPHASIS_PATTERN is None:
#         _EMPHASIS_PATTERN = re.compile(r"[\d%$]|^\$")
#     w = word.strip(".,!?").lower()
#     return bool(_EMPHASIS_PATTERN.search(word)) or w in _POWER_WORDS
 
# # Trending caption styles. Each is a full ASS "Style:" line plus a couple of
# # behavior flags consumed by build_ass(). Colors are &HAABBGGRR (ASS order).
# CAPTION_STYLES = {
#     "bold_pop": {
#         "label": "Bold Pop (white + yellow highlight)",
#         "style": "Style: Default,Montserrat Black,84,&H00FFFFFF,&H0000D7FF,&H00101010,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,6,0,2,60,60,140,1",
#         "highlight_color": "&H0000D7FF",   # active word turns yellow/gold
#         "word_by_word": True,
#         "pop_scale": True,
#     },
#     "karaoke_classic": {
#         "label": "Karaoke Fill (progressive color wipe)",
#         "style": "Style: Default,Montserrat SemiBold,72,&H00FFFFFF,&H0000A5FF,&H00202020,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,4,0,2,60,60,150,1",
#         "highlight_color": "&H0000A5FF",
#         "word_by_word": True,
#         "karaoke_fill": True,
#     },
#     "neon_glow": {
#         "label": "Neon Glow (cyan outline pop)",
#         "style": "Style: Default,Montserrat Black,80,&H00FFFFFF,&H00FFF000,&H00902000,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,5,2,2,60,60,140,1",
#         "highlight_color": "&H00FFF000",
#         "word_by_word": True,
#         "pop_scale": True,
#     },
#     "minimal_clean": {
#         "label": "Minimal Clean (small centered white)",
#         "style": "Style: Default,Inter Medium,54,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
#                   "0,0,0,0,100,100,0,0,1,2,1,2,60,60,120,1",
#         "highlight_color": "&H00FFFFFF",
#         "word_by_word": False,
#         "pop_scale": False,
#     },
#     "creator_bold": {
#         "label": "Creator Bold (huge yellow, black outline)",
#         "style": "Style: Default,Montserrat Black,92,&H0000FFFF,&H000000FF,&H00000000,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,8,0,2,50,50,160,1",
#         "highlight_color": "&H000000FF",
#         "word_by_word": True,
#         "pop_scale": True,
#     },
# }
 
 
# @editor_bp.route("/api/editor/styles")
# def api_editor_styles():
#     return jsonify({k: {"label": v["label"]} for k, v in CAPTION_STYLES.items()})
 
 
# @editor_bp.route("/api/editor/ratios")
# def api_editor_ratios():
#     return jsonify(RATIO_PRESETS)
 
 
# @editor_bp.route("/api/editor/editor")
# def api_editor_page():
#     """Serves the built-in editor UI — no separate .html file to manage,
#     it's embedded in this module (EDITOR_HTML below) and rendered straight
#     from memory. Open this URL in a tab; its fetch calls are same-origin."""
#     from flask import Response
#     return Response(EDITOR_HTML, mimetype="text/html")
 
 
# def _fmt_size(n):
#     if not n:
#         return "0 B"
#     n = float(n)
#     for unit in ["B", "KB", "MB", "GB"]:
#         if n < 1024:
#             return f"{n:.1f} {unit}"
#         n /= 1024
#     return f"{n:.1f} TB"
 
 
# @editor_bp.route("/api/editor/sources")
# def api_editor_sources():
#     """Lists videos the editor can open WITHOUT a fresh upload: everything
#     downloader.py has saved to DL_DIR/downloads (so a just-fetched video can
#     be sent straight into the editor), plus anything uploaded here before.
#     Both origins return the exact same shape and both feed the exact same
#     source_path used by /analyze and /render — the editor doesn't care where
#     a file came from."""
#     import datetime
#     jobs = _downloader_jobs()
#     # dl_id -> nicer display title, when downloader.py is registered too
#     title_by_stub = {}
#     for job in jobs.values():
#         fname = job.get("filename")
#         if fname and "_" in fname:
#             stub = fname.split("_", 1)[0]
#             title_by_stub[stub] = fname.split("_", 1)[-1]
 
#     sources = []
#     for folder, origin in ((SRC_DIR, "downloader"), (EDIT_DIR / "uploads", "upload")):
#         if not folder or not folder.exists():
#             continue
#         for p in sorted(folder.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
#             if not p.is_file() or p.suffix.lower() not in (".mp4", ".mov", ".mkv", ".webm", ".m4v"):
#                 continue
#             stub = p.name.split("_", 1)[0]
#             display = title_by_stub.get(stub) or (p.name.split("_", 1)[-1] if "_" in p.name else p.name)
#             stat = p.stat()
#             sources.append({
#                 "name": display,
#                 "path": str(p.resolve()),
#                 "origin": origin,
#                 "size": stat.st_size,
#                 "size_str": _fmt_size(stat.st_size),
#                 "modified": stat.st_mtime,
#                 "modified_str": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%d %b, %H:%M"),
#             })
#     sources.sort(key=lambda s: s["modified"], reverse=True)
#     return jsonify({"sources": sources})
 
 
# def _is_allowed_source(path):
#     """Only ever stream files that live under the editor's own known roots
#     (downloader's downloads folder, this module's uploads folder, or its
#     own rendered-output folder) — never an arbitrary server path."""
#     try:
#         rp = Path(path).resolve()
#     except Exception:
#         return False
#     for root in (SRC_DIR, EDIT_DIR / "uploads", EDIT_DIR):
#         if root and root.exists():
#             try:
#                 rp.relative_to(root.resolve())
#                 return True
#             except ValueError:
#                 continue
#     return False
 
 
# @editor_bp.route("/api/editor/preview")
# def api_editor_preview():
#     """Streams a source (or rendered) video for <video> tag playback/scrub,
#     so picking a file from the library previews instantly without a
#     round-trip upload. Range requests are handled by Flask's send_file."""
#     path = request.args.get("path", "")
#     if not path or not _is_allowed_source(path) or not Path(path).exists():
#         return "Not found or not allowed", 404
#     return send_file(path, conditional=True)
 
 
# @editor_bp.route("/api/editor/features")
# def api_editor_features():
#     return jsonify(feature_status())
 
 
# # ══════════════════════════════ probing ══════════════════════════════
 
# def _probe(path):
#     """Returns {duration, width, height, fps} via ffprobe."""
#     cmd = [
#         FFPROBE_PATH, "-v", "error", "-select_streams", "v:0",
#         "-show_entries", "stream=width,height,avg_frame_rate,duration",
#         "-show_entries", "format=duration",
#         "-of", "json", str(path),
#     ]
#     out = _run(cmd)
#     data = json.loads(out.stdout or "{}")
#     stream = (data.get("streams") or [{}])[0]
#     w = stream.get("width")
#     h = stream.get("height")
#     fps_raw = stream.get("avg_frame_rate") or "25/1"
#     try:
#         num, den = fps_raw.split("/")
#         fps = float(num) / float(den) if float(den) else 25.0
#     except Exception:
#         fps = 25.0
#     duration = stream.get("duration") or (data.get("format") or {}).get("duration")
#     try:
#         duration = float(duration)
#     except (TypeError, ValueError):
#         duration = 0.0
#     return {"duration": duration, "width": w, "height": h, "fps": fps}
 
 
# # ══════════════════════════════ face tracking ══════════════════════════════
 
# class _FaceTracker:
#     """Samples frames, finds a face each time, and produces a smoothed
#     (jitter-free) track of the subject's center point over time.
 
#     Uses mediapipe's BlazeFace model when available ("advanced" mode — much
#     more accurate on angled/small faces), otherwise falls back to OpenCV's
#     bundled Haar cascade so face tracking still works with just opencv-python
#     installed.
#     """
 
#     def __init__(self):
#         self.advanced = _HAS_MEDIAPIPE
#         if self.advanced:
#             self._mp_detector = mp.solutions.face_detection.FaceDetection(
#                 model_selection=1, min_detection_confidence=0.5
#             )
#         elif _HAS_CV2:
#             self._cascade = cv2.CascadeClassifier(
#                 cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#             )
#         else:
#             raise RuntimeError("opencv-python is required for face tracking")
 
#     def _detect(self, frame_bgr):
#         """Returns list of (cx, cy, w, h) in pixel coords, largest first."""
#         h_img, w_img = frame_bgr.shape[:2]
#         faces = []
#         if self.advanced:
#             rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#             result = self._mp_detector.process(rgb)
#             if result.detections:
#                 for d in result.detections:
#                     box = d.location_data.relative_bounding_box
#                     fw, fh = box.width * w_img, box.height * h_img
#                     fx = box.xmin * w_img + fw / 2
#                     fy = box.ymin * h_img + fh / 2
#                     faces.append((fx, fy, fw, fh))
#         else:
#             gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#             dets = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50))
#             for (x, y, w, h) in dets:
#                 faces.append((x + w / 2, y + h / 2, w, h))
#         faces.sort(key=lambda f: f[2] * f[3], reverse=True)  # largest face first
#         return faces
 
#     def track(self, video_path, sample_fps=2.0, progress_cb=None):
#         """Returns a list of {t, cx, cy, w, h} in NORMALIZED (0-1) coords,
#         exponentially smoothed so the crop pans instead of jumping."""
#         cap = cv2.VideoCapture(str(video_path))
#         if not cap.isOpened():
#             raise RuntimeError("Could not open video for face tracking")
#         src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#         w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         step = max(1, int(round(src_fps / sample_fps)))
 
#         raw_points = []
#         last_center = (0.5, 0.5)  # default: frame center when nobody's found yet
#         frame_idx = 0
#         while True:
#             ok = cap.grab()
#             if not ok:
#                 break
#             if frame_idx % step == 0:
#                 ok2, frame = cap.retrieve()
#                 if ok2:
#                     faces = self._detect(frame)
#                     if faces:
#                         fx, fy, fw, fh = faces[0]
#                         cx, cy = fx / w_img, fy / h_img
#                         last_center = (cx, cy)
#                     else:
#                         cx, cy = last_center  # hold last known position
#                     t = frame_idx / src_fps
#                     raw_points.append({"t": t, "cx": cx, "cy": cy})
#                 if progress_cb and total_frames:
#                     progress_cb(min(99, int(frame_idx * 100 / total_frames)))
#             frame_idx += 1
#         cap.release()
 
#         if not raw_points:
#             raw_points = [{"t": 0.0, "cx": 0.5, "cy": 0.5}]
 
#         # EMA smoothing so the crop glides instead of snapping every sample
#         alpha = 0.25
#         smoothed = [raw_points[0]]
#         for p in raw_points[1:]:
#             prev = smoothed[-1]
#             smoothed.append({
#                 "t": p["t"],
#                 "cx": prev["cx"] + alpha * (p["cx"] - prev["cx"]),
#                 "cy": prev["cy"] + alpha * (p["cy"] - prev["cy"]),
#             })
#         return smoothed, (w_img, h_img)
 
 
# def analyze_face_track(video_path, sample_fps=2.0, progress_cb=None):
#     if not _HAS_CV2:
#         raise RuntimeError("opencv-python is not installed — face tracking unavailable")
#     tracker = _FaceTracker()
#     points, (w, h) = tracker.track(video_path, sample_fps=sample_fps, progress_cb=progress_cb)
#     return points, w, h, tracker.advanced
 
 
# # ══════════════════════════════ crop-window math ══════════════════════════════
 
# def _crop_size_for_ratio(src_w, src_h, target_w, target_h):
#     """Largest crop rectangle matching the target aspect that still fits
#     inside the source frame (so we crop, never letterbox/pad)."""
#     target_ar = target_w / target_h
#     src_ar = src_w / src_h
#     if src_ar > target_ar:
#         # source is wider than target -> crop width, keep full height
#         crop_h = src_h
#         crop_w = int(round(crop_h * target_ar))
#     else:
#         # source is taller than target -> crop height, keep full width
#         crop_w = src_w
#         crop_h = int(round(crop_w / target_ar))
#     return crop_w, crop_h
 
 
# def _build_crop_keyframes(track_points, src_w, src_h, crop_w, crop_h):
#     """Turns normalized face-center points into pixel-space crop x/y
#     top-left keyframes, clamped so the crop never leaves the frame."""
#     kfs = []
#     for p in track_points:
#         cx_px = p["cx"] * src_w
#         cy_px = p["cy"] * src_h
#         x = int(cx_px - crop_w / 2)
#         y = int(cy_px - crop_h / 2)
#         x = max(0, min(x, src_w - crop_w))
#         y = max(0, min(y, src_h - crop_h))
#         kfs.append({"t": p["t"], "x": x, "y": y})
#     return kfs
 
 
# def _expr_chain(keyframes, key):
#     """Builds an ffmpeg time-expression string that linearly interpolates
#     between keyframes for either 'x' or 'y' — this drives a moving crop
#     window without needing any external filter graph patching."""
#     if len(keyframes) == 1:
#         return str(keyframes[0][key])
#     expr = str(keyframes[-1][key])
#     for i in range(len(keyframes) - 2, -1, -1):
#         t0, v0 = keyframes[i]["t"], keyframes[i][key]
#         t1, v1 = keyframes[i + 1]["t"], keyframes[i + 1][key]
#         if t1 <= t0:
#             continue
#         # linear ramp between (t0,v0) and (t1,v1), holds v0 before t0
#         ramp = f"({v0}+({v1}-{v0})*(t-{t0})/{(t1 - t0)})"
#         expr = f"if(lt(t,{t1}),{ramp},{expr})"
#     return expr
 
 
# # ══════════════════════════════ subtitles ══════════════════════════════
 
# def _load_whisper(model_size="small"):
#     global _WHISPER_MODEL
#     if _WHISPER_MODEL is None:
#         _WHISPER_MODEL = WhisperModel(model_size, compute_type="int8")
#     return _WHISPER_MODEL
 
 
# def transcribe_words(video_path, model_size="small", language=None):
#     """Returns [{start, end, text}, ...] word-level timestamps."""
#     if not _HAS_WHISPER:
#         raise RuntimeError("faster-whisper is not installed — subtitles unavailable")
#     model = _load_whisper(model_size)
#     segments, _info = model.transcribe(str(video_path), word_timestamps=True, language=language)
#     words = []
#     for seg in segments:
#         for w in (seg.words or []):
#             text = (w.word or "").strip()
#             if text:
#                 words.append({"start": w.start, "end": w.end, "text": text})
#     return words
 
 
# def _chunk_words(words, max_words=4, max_span=1.6):
#     """Groups words into short on-screen caption chunks — the standard
#     'trending shorts' style of 2-5 words on screen at a time, not full
#     sentences."""
#     chunks = []
#     cur = []
#     for w in words:
#         if cur and (len(cur) >= max_words or (w["end"] - cur[0]["start"]) > max_span):
#             chunks.append(cur)
#             cur = []
#         cur.append(w)
#     if cur:
#         chunks.append(cur)
#     return chunks
 
 
# def _ass_time(t):
#     h = int(t // 3600)
#     m = int((t % 3600) // 60)
#     s = t % 60
#     return f"{h}:{m:02d}:{s:05.2f}"
 
 
# def build_ass(words, style_name, video_w, video_h):
#     """Builds a full .ass subtitle document with word-by-word highlight
#     animation for the chosen trending style, sized for the given output
#     resolution so text scales correctly across aspect ratios."""
#     style = CAPTION_STYLES.get(style_name, CAPTION_STYLES["bold_pop"])
#     header = f"""[Script Info]
# ScriptType: v4.00+
# PlayResX: {video_w}
# PlayResY: {video_h}
# ScaledBorderAndShadow: yes
 
# [V4+ Styles]
# Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
# {style["style"]}
 
# [Events]
# Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
# """
#     lines = []
#     for chunk in _chunk_words(words):
#         start, end = chunk[0]["start"], chunk[-1]["end"]
#         if style.get("karaoke_fill"):
#             # \k tags: whole line present, active word wipes to highlight color
#             text = "".join(
#                 r"{\k%d}%s " % (max(1, int(round((w["end"] - w["start"]) * 100))), w["text"])
#                 for w in chunk
#             )
#             lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
#         elif style.get("word_by_word"):
#             # emit one Dialogue event per word so only the current word is
#             # shown in the highlight color while the rest stay default —
#             # gives the punchy "pop" caption look trending on shorts.
#             for w in chunk:
#                 parts = []
#                 for w2 in chunk:
#                     if w2 is w:
#                         scale = r"\fscx115\fscy115" if style.get("pop_scale") else ""
#                         parts.append(r"{\c%s%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], scale, w2["text"]))
#                     elif _is_emphasis_word(w2["text"]):
#                         # numbers / money / power-words get punched up even
#                         # when they're not the "active" word of the moment
#                         parts.append(r"{\c%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], w2["text"]))
#                     else:
#                         parts.append(w2["text"])
#                 text = " ".join(parts)
#                 lines.append(f"Dialogue: 0,{_ass_time(w['start'])},{_ass_time(w['end'])},Default,,0,0,0,,{text}")
#         else:
#             parts = []
#             for w2 in chunk:
#                 if _is_emphasis_word(w2["text"]):
#                     parts.append(r"{\c%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], w2["text"]))
#                 else:
#                     parts.append(w2["text"])
#             text = " ".join(parts)
#             lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
#     return header + "\n".join(lines) + "\n"
 
 
# # ══════════════════════════════ hook-cut / segments ══════════════════════════════
 
# def _read_pcm_mono(video_path, sr=8000):
#     """Decodes a fast, low-res mono PCM stream via ffmpeg for lightweight
#     loudness analysis (used only to pick the auto hook window)."""
#     cmd = [FFMPEG_PATH, "-v", "error", "-i", str(video_path),
#            "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
#     proc = subprocess.run(cmd, capture_output=True, **_no_console_kwargs())
#     raw = proc.stdout
#     count = len(raw) // 2
#     samples = struct.unpack(f"<{count}h", raw[:count * 2])
#     return np.array(samples, dtype=np.float32) / 32768.0, sr
 
 
# def auto_hook_window(video_path, duration, hook_len=6.0):
#     """Finds the loudest `hook_len`-second window in the clip — a cheap but
#     effective proxy for 'most energetic / most likely to hook a viewer'
#     moment, used to auto-suggest where the exported clip should start."""
#     if not _HAS_NUMPY:
#         raise RuntimeError("numpy is not installed — auto hook-cut unavailable")
#     audio, sr = _read_pcm_mono(video_path)
#     if len(audio) == 0:
#         return {"start": 0.0, "end": min(hook_len, duration)}
#     win = int(hook_len * sr)
#     if win >= len(audio):
#         return {"start": 0.0, "end": duration}
#     energy = audio ** 2
#     # cumulative sum for O(1) windowed average lookups
#     cumsum = np.cumsum(np.insert(energy, 0, 0))
#     window_energy = cumsum[win:] - cumsum[:-win]
#     best_start_sample = int(np.argmax(window_energy))
#     start = best_start_sample / sr
#     return {"start": round(start, 2), "end": round(min(start + hook_len, duration), 2)}
 
 
# def detect_silences(video_path, noise_db=-30, min_silence=0.5):
#     """Runs ffmpeg's silencedetect and returns the list of NON-silent
#     {start,end} ranges — i.e. the segments worth keeping. This powers
#     automatic 'jump-cut' editing (removing dead air/pauses), the other
#     trending short-form edit style besides the loudest-window hook-cut."""
#     cmd = [FFMPEG_PATH, "-v", "info", "-i", str(video_path),
#            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
#     result = _run(cmd)
#     log = result.stderr or ""
#     starts, ends = [], []
#     for line in log.splitlines():
#         if "silence_start" in line:
#             starts.append(float(line.split("silence_start:")[1].strip().split(" ")[0]))
#         elif "silence_end" in line:
#             ends.append(float(line.split("silence_end:")[1].strip().split(" ")[0].split("|")[0]))
#     info = _probe(video_path)
#     duration = info["duration"]
#     silences = list(zip(starts, ends[: len(starts)]))
#     keep = []
#     cursor = 0.0
#     for s, e in silences:
#         if s > cursor:
#             keep.append({"start": round(cursor, 2), "end": round(s, 2)})
#         cursor = max(cursor, e)
#     if cursor < duration:
#         keep.append({"start": round(cursor, 2), "end": round(duration, 2)})
#     return [k for k in keep if k["end"] - k["start"] > 0.15]
 
 
# def _build_concat_file(video_path, segments, tmp_dir):
#     """Writes an ffmpeg concat-demuxer list after trimming each segment,
#     so multiple {start,end} ranges can be stitched in a custom, manually
#     chosen order (rearrange / hook-cut)."""
#     parts = []
#     for i, seg in enumerate(segments):
#         out = tmp_dir / f"seg_{i}.mp4"
#         cmd = [
#             FFMPEG_PATH, "-y", "-v", "error",
#             "-ss", str(seg["start"]), "-to", str(seg["end"]),
#             "-i", str(video_path),
#             "-c", "copy", "-avoid_negative_ts", "make_zero",
#             str(out),
#         ]
#         _run(cmd)
#         parts.append(out)
#     list_path = tmp_dir / "concat_list.txt"
#     with open(list_path, "w", encoding="utf-8") as f:
#         for p in parts:
#             f.write(f"file '{p.as_posix()}'\n")
#     return list_path
 
 
# def _measure_loudnorm(path):
#     """First pass of two-pass EBU R128 loudness normalization: measures the
#     input's actual loudness/true-peak/range so the second (real encode)
#     pass can normalize with `linear=true` against real measured values
#     instead of loudnorm's single-pass dynamic estimate — noticeably more
#     accurate and avoids the pumping/gain-riding artifacts single-pass mode
#     can introduce on speech-heavy short-form video."""
#     cmd = [FFMPEG_PATH, "-v", "info", "-i", str(path),
#            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
#            "-f", "null", "-"]
#     result = _run(cmd)
#     log = result.stderr or ""
#     try:
#         start = log.rindex("{")
#         end = log.rindex("}") + 1
#         return json.loads(log[start:end])
#     except (ValueError, json.JSONDecodeError):
#         return None  # falls back to single-pass below
 
 
# # ══════════════════════════════ render pipeline ══════════════════════════════
 
# def _render_one_ratio(job, source_path, ratio_key, track_points, src_w, src_h,
#                        words, tmp_dir, loud_measured=None):
#     cfg = job["config"]
#     ratio = RATIO_PRESETS[ratio_key]
#     is_auto = (ratio_key == "auto")
#     # "auto" keeps the SOURCE's own dimensions — no target size is imposed.
#     target_w, target_h = (src_w, src_h) if is_auto else (ratio["w"], ratio["h"])
#     quality = QUALITY_PRESETS.get(cfg.get("quality", "high"), QUALITY_PRESETS["high"])
 
#     fill_mode = cfg.get("fill_mode", "crop")  # "crop" (default) or "blur_pad"
#     # "auto" never crops or letterboxes — it IS the full original frame —
#     # so blur_pad/crop fill-mode choices simply don't apply to it.
#     if is_auto:
#         fill_mode = "none"
 
#     # ── subtitles: build the .ass first so both fill-mode branches can use it ──
#     ass_path = None
#     if cfg["features"].get("subtitles") and words:
#         ass_content = build_ass(words, cfg.get("caption_style", "bold_pop"), target_w, target_h)
#         ass_path = tmp_dir / f"subs_{ratio_key.replace(':', 'x')}.ass"
#         ass_path.write_text(ass_content, encoding="utf-8")
#         ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
 
#     if is_auto and not ass_path and not cfg.get("manual_crop_keyframes"):
#         # Nothing needs to touch the picture at all: fastest AND highest
#         # possible quality path is a pure stream-copy remux (no re-encode
#         # -> byte-for-byte original video quality; only trimmed/normalized
#         # audio may differ). This is what "zero quality loss" means here.
#         out_name = f"{job['job_id']}_{ratio_key.replace(':', 'x')}.mp4"
#         out_path = EDIT_DIR / out_name
#         if cfg.get("normalize_audio", True):
#             measured = loud_measured
#             audio_filter = (
#                 "loudnorm=I=-14:TP=-1.5:LRA=11:"
#                 f"measured_I={measured.get('input_i', -14)}:"
#                 f"measured_TP={measured.get('input_tp', -1.5)}:"
#                 f"measured_LRA={measured.get('input_lra', 11)}:"
#                 f"measured_thresh={measured.get('input_thresh', -24)}:"
#                 "linear=true:print_format=summary"
#             ) if measured else "loudnorm=I=-14:TP=-1.5:LRA=11"
#             cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path),
#                    "-c:v", "copy", "-af", audio_filter, "-c:a", "aac", "-b:a", "192k",
#                    "-movflags", "+faststart", str(out_path)]
#         else:
#             cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path),
#                    "-c", "copy", "-movflags", "+faststart", str(out_path)]
#         result = _run(cmd)
#         if result.returncode != 0:
#             raise RuntimeError(f"ffmpeg failed for {ratio_key}: {result.stderr[-800:]}")
#         return out_name
 
#     if fill_mode == "blur_pad":
#         # "Advanced" reframe mode: instead of cropping content away, fit the
#         # WHOLE frame inside the target canvas and fill the leftover bars
#         # with a blurred, zoomed copy of the same frame — the look used by
#         # most professional auto-reframe tools when a hard crop would cut
#         # off important content. No face tracking needed for this branch,
#         # since nothing is being cropped out.
#         fc = (
#             f"[0:v]split=2[bg][fg];"
#             f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
#             f"crop={target_w}:{target_h},gblur=sigma=25,eq=brightness=-0.05[bg2];"
#             f"[fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg2];"
#             f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2:format=auto[base]"
#         )
#         if ass_path:
#             fc += f",[base]ass='{ass_escaped}'[vout]"
#             map_v = "[vout]"
#         else:
#             fc += "[vout]"
#             map_v = "[vout]"
#         vf_args = ["-filter_complex", fc, "-map", map_v, "-map", "0:a?"]
#     elif is_auto:
#         # Captions and/or a manual crop keyframe are present, so a re-encode
#         # can't be avoided — but the frame itself is never cropped or
#         # rescaled; it stays at the source's own resolution. Quality is set
#         # one notch above "high" (crf 16, effectively visually lossless)
#         # specifically for this path so adding captions never visibly
#         # degrades the original picture.
#         filters = ["setsar=1"]
#         if ass_path:
#             filters.append(f"ass='{ass_escaped}'")
#         vf_args = ["-vf", ",".join(filters)]
#         quality = {"crf": 16, "preset": quality["preset"]}
#     else:
#         filters = []
#         # ── crop (face-tracked or manual) ──
#         if cfg["features"].get("face_tracking") or cfg.get("manual_crop_keyframes"):
#             crop_w, crop_h = _crop_size_for_ratio(src_w, src_h, target_w, target_h)
#             if cfg.get("manual_crop_keyframes"):
#                 kfs = sorted(cfg["manual_crop_keyframes"], key=lambda k: k["t"])
#             else:
#                 kfs = _build_crop_keyframes(track_points, src_w, src_h, crop_w, crop_h)
#             x_expr = _expr_chain(kfs, "x")
#             y_expr = _expr_chain(kfs, "y")
#             filters.append(f"crop={crop_w}:{crop_h}:'{x_expr}':'{y_expr}'")
#         else:
#             # no face tracking requested -> simple centered crop to the target ratio
#             crop_w, crop_h = _crop_size_for_ratio(src_w, src_h, target_w, target_h)
#             filters.append(f"crop={crop_w}:{crop_h}:(in_w-out_w)/2:(in_h-out_h)/2")
 
#         filters.append(f"scale={target_w}:{target_h}:flags=lanczos")
#         filters.append("setsar=1")
#         if ass_path:
#             filters.append(f"ass='{ass_escaped}'")
#         vf_args = ["-vf", ",".join(filters)]
 
#     out_name = f"{job['job_id']}_{ratio_key.replace(':', 'x')}.mp4"
#     out_path = EDIT_DIR / out_name
 
#     # Loudness normalization (EBU R128, -14 LUFS is the standard target for
#     # social platforms) so exported audio doesn't sound quiet/inconsistent
#     # next to other content in-feed. On "high" quality we do a real
#     # measure-then-normalize TWO-PASS pass for accuracy; cheaper qualities
#     # use fast single-pass so exports stay quick.
#     audio_filters = []
#     if cfg.get("normalize_audio", True):
#         measured = loud_measured
#         if measured:
#             audio_filters = [
#                 "loudnorm=I=-14:TP=-1.5:LRA=11:"
#                 f"measured_I={measured.get('input_i', -14)}:"
#                 f"measured_TP={measured.get('input_tp', -1.5)}:"
#                 f"measured_LRA={measured.get('input_lra', 11)}:"
#                 f"measured_thresh={measured.get('input_thresh', -24)}:"
#                 "linear=true:print_format=summary"
#             ]
#         else:
#             audio_filters = ["loudnorm=I=-14:TP=-1.5:LRA=11"]
 
#     cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path), *vf_args]
#     if audio_filters:
#         cmd += ["-af", ",".join(audio_filters)]
#     cmd += [
#         "-c:v", "libx264", "-crf", str(quality["crf"]), "-preset", quality["preset"],
#         "-c:a", "aac", "-b:a", "192k",
#         "-movflags", "+faststart",
#         str(out_path),
#     ]
#     result = _run(cmd)
#     if result.returncode != 0:
#         raise RuntimeError(f"ffmpeg failed for {ratio_key}: {result.stderr[-800:]}")
#     return out_name
 
 
# def _run_render_job(job_id):
#     job = EDIT_JOBS[job_id]
#     cfg = job["config"]
#     tmp_dir = EDIT_DIR / f"tmp_{job_id}"
#     tmp_dir.mkdir(exist_ok=True)
 
#     try:
#         source_path = Path(cfg["source_path"])
#         if not source_path.exists():
#             raise RuntimeError("Source video not found")
 
#         # ── 1. hook-cut / manual rearrange: trim+stitch segments first ──
#         job["stage"] = "segments"
#         segments = cfg.get("segments")
#         want_jump = cfg["features"].get("jump_cut")
#         want_hook = cfg["features"].get("hook_cut")
#         if not segments and (want_jump or want_hook):
#             info = _probe(source_path)
#             jump_segments = detect_silences(source_path) if want_jump else None
#             if want_hook:
#                 hook = auto_hook_window(source_path, info["duration"], cfg.get("hook_len", 6.0))
#                 # Cold-open on the loudest window, then play the rest of the
#                 # clip — dead-air-trimmed if jump_cut is also on, otherwise
#                 # the full original timeline. Combining both no longer means
#                 # jump_cut silently wins.
#                 rest = jump_segments if jump_segments else [{"start": 0.0, "end": info["duration"]}]
#                 segments = [hook] + rest
#             else:
#                 segments = jump_segments
#         if segments:
#             list_path = _build_concat_file(source_path, segments, tmp_dir)
#             stitched = tmp_dir / "stitched.mp4"
#             _run([FFMPEG_PATH, "-y", "-v", "error", "-f", "concat", "-safe", "0",
#                   "-i", str(list_path), "-c", "copy", str(stitched)])
#             source_path = stitched
#         job["percent"] = 10
 
#         info = _probe(source_path)
#         src_w, src_h = info["width"], info["height"]
 
#         # ── 2. face tracking (once, reused for every export ratio) ──
#         job["stage"] = "face_tracking"
#         track_points = []
#         if cfg["features"].get("face_tracking"):
#             def cb(pct):
#                 job["percent"] = 10 + int(pct * 0.3)
#             track_points, src_w, src_h, advanced = analyze_face_track(
#                 source_path, sample_fps=cfg.get("track_fps", 2.0), progress_cb=cb
#             )
#             job["face_tracking_mode"] = "advanced (mediapipe)" if advanced else "standard (haar cascade)"
#         job["percent"] = 40
 
#         # ── 3. subtitles (once, reused for every export ratio) ──
#         job["stage"] = "subtitles"
#         words = []
#         if cfg["features"].get("subtitles"):
#             words = transcribe_words(source_path, model_size=cfg.get("whisper_model", "small"),
#                                       language=cfg.get("language"))
#         job["percent"] = 60
 
#         # ── 4. render each requested aspect ratio as one final file ──
#         job["stage"] = "render"
#         ratios = cfg.get("export_ratios") or ["9:16"]
#         job["ratios"] = {}
#         n = len(ratios)
 
#         # measured once (not per-ratio, source audio is identical across
#         # ratios) and only for "high" quality, where the extra ffmpeg pass
#         # is worth the accuracy; cheaper qualities use fast single-pass.
#         loud_measured = None
#         if cfg.get("normalize_audio", True) and cfg.get("quality", "high") == "high":
#             loud_measured = _measure_loudnorm(source_path)
 
#         for i, ratio_key in enumerate(ratios):
#             if ratio_key not in RATIO_PRESETS:
#                 continue
#             out_name = _render_one_ratio(job, source_path, ratio_key, track_points,
#                                           src_w, src_h, words, tmp_dir, loud_measured)
#             job["ratios"][ratio_key] = out_name
#             job["percent"] = 60 + int((i + 1) * 40 / max(1, n))
 
#         job["status"] = "done"
#         job["stage"] = "done"
#         job["percent"] = 100
#     except Exception as e:
#         job["status"] = "error"
#         job["error"] = str(e)
#     finally:
#         # scratch files (trimmed segments, stitched source, .ass files) —
#         # keep only the final per-ratio outputs in EDIT_DIR
#         try:
#             for f in tmp_dir.glob("*"):
#                 f.unlink(missing_ok=True)
#             tmp_dir.rmdir()
#         except OSError:
#             pass
 
 
# # ══════════════════════════════ routes ══════════════════════════════
 
# @editor_bp.route("/api/editor/analyze", methods=["POST"])
# def api_editor_analyze():
#     """Runs face tracking + transcription WITHOUT rendering, so a frontend
#     can show a preview / let the user manually adjust crop keyframes or
#     hook segments before committing to a full render."""
#     data = request.json or {}
#     source_path = Path(data.get("source_path", ""))
#     if not source_path.exists():
#         return jsonify({"error": "source_path not found"}), 400
 
#     info = _probe(source_path)
#     result = {"probe": info, "features_available": feature_status()}
 
#     if data.get("face_tracking", True) and _HAS_CV2:
#         points, w, h, advanced = analyze_face_track(source_path, sample_fps=data.get("track_fps", 2.0))
#         result["face_track"] = points
#         result["face_tracking_mode"] = "advanced" if advanced else "standard"
#         result["source_size"] = {"w": w, "h": h}
 
#     if data.get("subtitles", True) and _HAS_WHISPER:
#         words = transcribe_words(source_path, model_size=data.get("whisper_model", "small"))
#         result["words"] = words
 
#     if data.get("hook_cut", False) and _HAS_NUMPY:
#         result["suggested_hook"] = auto_hook_window(source_path, info["duration"], data.get("hook_len", 6.0))
 
#     if data.get("jump_cut", False):
#         result["suggested_segments"] = detect_silences(source_path)
 
#     return jsonify(result)
 
 
# @editor_bp.route("/api/editor/upload", methods=["POST"])
# def api_editor_upload():
#     """Lets the frontend upload a local file directly instead of only
#     pointing at a path already on the server (e.g. a downloader.py output)."""
#     f = request.files.get("file")
#     if not f or not f.filename:
#         return jsonify({"error": "No file uploaded"}), 400
#     safe_name = f"{uuid.uuid4().hex[:10]}_{Path(f.filename).name}"
#     dest = EDIT_DIR / "uploads"
#     dest.mkdir(exist_ok=True)
#     path = dest / safe_name
#     f.save(path)
#     return jsonify({"source_path": str(path)})
 
 
# # ══════════════════════════════ fetch-from-link (built-in yt-dlp) ══════════════════════════════
# # A third source alongside "upload from PC" and "pick from downloader" —
# # paste a link right here in the editor. Deliberately skips the full
# # format-picker: it always resolves ONE single, best-available video+audio
# # mp4 (never a video-only stream that would need a silent-audio placeholder,
# # never separate files) and hands that straight to analyze/render.
# #
# # Speed trick mirrors downloader.py: remembers whichever auth mode (cookies
# # file / browser cookies / plain / bypass) last worked and tries that first,
# # so repeat fetches stay fast. If downloader.py is also registered in this
# # process, its already-resolved mode is reused immediately instead of
# # re-discovering it from scratch.
 
# FETCH_JOBS = {}  # fetch_id -> {status, stage, percent, error, source_path, url}
# _FETCH_RESOLVED_MODE = {"mode": None}
# _FETCH_COOKIE_BROWSERS = ["chrome", "edge", "firefox", "brave"]
 
 
# def _fetch_no_console_kwargs():
#     if os.name == "nt":
#         return {"creationflags": subprocess.CREATE_NO_WINDOW}
#     return {}
 
 
# def _fetch_auth_attempts():
#     attempts = []
#     # reuse downloader.py's already-known-good mode first, if it's loaded
#     try:
#         import downloader
#         if downloader._RESOLVED_MODE.get("mode"):
#             attempts.append(downloader._RESOLVED_MODE["mode"])
#     except Exception:
#         pass
#     if _FETCH_RESOLVED_MODE["mode"]:
#         attempts.append(_FETCH_RESOLVED_MODE["mode"])
#     cookies_file = SRC_DIR.parent / "cookies.txt" if SRC_DIR else None
#     rest = []
#     if cookies_file and cookies_file.exists():
#         rest.append("cookies_file")
#     rest.extend(_FETCH_COOKIE_BROWSERS)
#     rest.append("default")
#     rest.append("bypass")
#     for m in rest:
#         if m not in attempts:
#             attempts.append(m)
#     return attempts, cookies_file
 
 
# def _fetch_apply_auth(opts, mode, cookies_file):
#     opts = dict(opts)
#     if mode == "cookies_file" and cookies_file:
#         opts["cookiefile"] = str(cookies_file)
#     elif mode in _FETCH_COOKIE_BROWSERS:
#         opts["cookiesfrombrowser"] = (mode,)
#     elif mode == "bypass":
#         opts["extractor_args"] = {"youtube": {"player_client": ["ios", "android", "mweb"]}}
#     return opts
 
 
# def _fetch_extract(url, extra_opts=None, download=False):
#     base_opts = {
#         "quiet": True, "no_warnings": True, "noplaylist": True,
#         "socket_timeout": 15, "retries": 2, "extractor_retries": 1, "geo_bypass": True,
#     }
#     if FFMPEG_PATH:
#         base_opts["ffmpeg_location"] = str(FFMPEG_PATH)
#     if extra_opts:
#         base_opts.update(extra_opts)
 
#     attempts, cookies_file = _fetch_auth_attempts()
#     last_err = None
#     for mode in attempts:
#         opts = _fetch_apply_auth(base_opts, mode, cookies_file)
#         try:
#             with yt_dlp.YoutubeDL(opts) as ydl:
#                 info = ydl.extract_info(url, download=download)
#             _FETCH_RESOLVED_MODE["mode"] = mode
#             return info, ydl if download else None, None
#         except Exception as e:
#             last_err = e
#             continue
#     return None, None, last_err
 
 
# def _fetch_fmt_duration(sec):
#     if sec is None:
#         return None
#     sec = int(sec)
#     h, rem = divmod(sec, 3600)
#     m, s = divmod(rem, 60)
#     return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
 
 
# def _fetch_fmt_int(n):
#     if n is None:
#         return None
#     try:
#         return f"{int(n):,}"
#     except (TypeError, ValueError):
#         return str(n)
 
 
# def _fetch_fmt_upload_date(d):
#     if not d:
#         return None
#     try:
#         from datetime import datetime as _dt
#         return _dt.strptime(str(d), "%Y%m%d").strftime("%d %b %Y")
#     except ValueError:
#         return str(d)
 
 
# @editor_bp.route("/api/editor/fetch/info", methods=["POST"])
# def api_fetch_info():
#     """Resolves a link (YouTube/Instagram/TikTok/X/Facebook/etc.) and hands
#     back the FULL set of metadata yt-dlp/the source platform already knows
#     about it — title, uploader, view/like counts, upload date, best
#     available resolution — so the UI never has to re-derive any of this by
#     hand, and so a video that started life as a fetched link can show that
#     rich context in the analysis panel too (a plain PC upload naturally
#     won't have this — it only gets the ffprobe technical details)."""
#     if not _HAS_YTDLP:
#         return jsonify({"error": "yt-dlp is not installed on the server — pip install yt-dlp"}), 400
#     data = request.json or {}
#     url = (data.get("url") or "").strip()
#     if not url:
#         return jsonify({"error": "Paste a video URL first"}), 400
#     info, _, err = _fetch_extract(url, extra_opts={"format": "bestvideo+bestaudio/best"})
#     if info is None:
#         return jsonify({"error": f"Could not resolve that link: {err}"}), 400
 
#     best_h = best_w = None
#     for f in info.get("formats", []) or []:
#         if f.get("vcodec") not in (None, "none") and f.get("height"):
#             if not best_h or f["height"] > best_h:
#                 best_h, best_w = f["height"], f.get("width")
 
#     return jsonify({
#         "title": info.get("title"),
#         "uploader": info.get("uploader") or info.get("channel"),
#         "duration": info.get("duration"),
#         "duration_str": _fetch_fmt_duration(info.get("duration")),
#         "thumbnail": info.get("thumbnail"),
#         "view_count": info.get("view_count"),
#         "view_count_str": _fetch_fmt_int(info.get("view_count")),
#         "like_count": info.get("like_count"),
#         "like_count_str": _fetch_fmt_int(info.get("like_count")),
#         "upload_date": info.get("upload_date"),
#         "upload_date_str": _fetch_fmt_upload_date(info.get("upload_date")),
#         "resolution": f"{best_w}x{best_h}" if best_w and best_h else None,
#         "extractor": info.get("extractor_key"),
#         "webpage_url": info.get("webpage_url"),
#     })
 
 
# def _run_fetch_job(fetch_id, url):
#     job = FETCH_JOBS[fetch_id]
#     job["stage"] = "connect"
 
#     def hook(d):
#         if d.get("status") == "downloading":
#             total = d.get("total_bytes") or d.get("total_bytes_estimate")
#             downloaded = d.get("downloaded_bytes") or 0
#             speed = d.get("speed")
#             job.update({
#                 "status": "downloading", "stage": "download",
#                 "percent": round(downloaded * 100 / total, 1) if total else None,
#                 "speed_str": (_fmt_size(speed) + "/s") if speed else None,
#             })
#         elif d.get("status") == "finished":
#             job["status"] = "processing"
#             job["stage"] = "merge"
 
#     # Saved into SRC_DIR (the SAME "downloads" folder downloader.py uses) so
#     # a link fetched here shows up in /api/editor/sources too, and nothing
#     # about the analyze/render pipeline needs to know it came from here
#     # instead of the Downloader tab or a PC upload.
#     extra_opts = {
#         "format": "bestvideo+bestaudio/best",
#         "outtmpl": str(SRC_DIR / f"editorfetch_{fetch_id}_%(title).60s.%(ext)s"),
#         "progress_hooks": [hook],
#         "merge_output_format": "mp4",   # always ONE muxed video+audio file
#     }
 
#     orig_popen = subprocess.Popen
#     def quiet_popen(*args, **kwargs):
#         kwargs.update(_fetch_no_console_kwargs())
#         return orig_popen(*args, **kwargs)
#     subprocess.Popen = quiet_popen
#     try:
#         info, ydl, err = _fetch_extract(url, extra_opts=extra_opts, download=True)
#     finally:
#         subprocess.Popen = orig_popen
 
#     if info is None:
#         job["status"] = "error"
#         job["error"] = f"Fetch failed: {err}"
#         return
#     try:
#         fname = ydl.prepare_filename(info)
#         p = Path(fname)
#         if not p.exists():
#             p2 = p.with_suffix(".mp4")
#             if p2.exists():
#                 p = p2
#         job["source_path"] = str(p.resolve())
#         job["status"] = "done"
#         job["stage"] = "done"
#         job["percent"] = 100
#     except Exception as e:
#         job["status"] = "error"
#         job["error"] = f"Could not finalize file: {e}"
 
 
# @editor_bp.route("/api/editor/fetch/start", methods=["POST"])
# def api_fetch_start():
#     if not _HAS_YTDLP:
#         return jsonify({"error": "yt-dlp is not installed on the server — pip install yt-dlp"}), 400
#     data = request.json or {}
#     url = (data.get("url") or "").strip()
#     if not url:
#         return jsonify({"error": "No URL given"}), 400
#     fetch_id = uuid.uuid4().hex[:10]
#     FETCH_JOBS[fetch_id] = {
#         "status": "starting", "stage": "connect", "percent": 0,
#         "speed_str": None, "error": None, "url": url, "source_path": None,
#     }
#     threading.Thread(target=_run_fetch_job, args=(fetch_id, url), daemon=True).start()
#     return jsonify({"fetch_id": fetch_id})
 
 
# @editor_bp.route("/api/editor/fetch/progress/<fetch_id>")
# def api_fetch_progress(fetch_id):
#     job = FETCH_JOBS.get(fetch_id)
#     if not job:
#         return jsonify({"error": "Unknown fetch job"}), 404
#     return jsonify(job)
 
 
# @editor_bp.route("/api/editor/render", methods=["POST"])
# def api_editor_render():
#     data = request.json or {}
#     source_path = data.get("source_path", "")
#     if not source_path or not Path(source_path).exists():
#         return jsonify({"error": "source_path not found"}), 400
 
#     export_ratios = data.get("export_ratios") or ["9:16"]
#     bad = [r for r in export_ratios if r not in RATIO_PRESETS]
#     if bad:
#         return jsonify({"error": f"Unknown ratio(s): {bad}. Valid: {list(RATIO_PRESETS)}"}), 400
 
#     features = {
#         "face_tracking": bool(data.get("face_tracking", True)),
#         "subtitles": bool(data.get("subtitles", True)),
#         "hook_cut": bool(data.get("hook_cut", False)),
#         "jump_cut": bool(data.get("jump_cut", False)),
#     }
#     if features["face_tracking"] and not _HAS_CV2:
#         return jsonify({"error": "Face tracking requested but opencv-python is not installed"}), 400
#     if features["subtitles"] and not _HAS_WHISPER:
#         return jsonify({"error": "Subtitles requested but faster-whisper is not installed"}), 400
#     if features["hook_cut"] and not data.get("segments") and not _HAS_NUMPY:
#         return jsonify({"error": "Auto hook-cut requested but numpy is not installed"}), 400
 
#     job_id = uuid.uuid4().hex[:10]
#     EDIT_JOBS[job_id] = {
#         "job_id": job_id, "status": "starting", "stage": "queued", "percent": 0,
#         "error": None, "ratios": {},
#         "config": {
#             "source_path": source_path,
#             "export_ratios": export_ratios,
#             "features": features,
#             "caption_style": data.get("caption_style", "bold_pop"),
#             "quality": data.get("quality", "high"),
#             "track_fps": data.get("track_fps", 2.0),
#             "whisper_model": data.get("whisper_model", "small"),
#             "language": data.get("language"),
#             "hook_len": data.get("hook_len", 6.0),
#             "manual_crop_keyframes": data.get("manual_crop_keyframes"),  # [{t,x,y}]
#             "segments": data.get("segments"),  # [{start,end}] manual rearrange/trim order
#             "fill_mode": data.get("fill_mode", "crop"),  # "crop" or "blur_pad"
#             "normalize_audio": bool(data.get("normalize_audio", True)),
#         },
#     }
#     threading.Thread(target=_run_render_job, args=(job_id,), daemon=True).start()
#     return jsonify({"job_id": job_id})
 
 
# @editor_bp.route("/api/editor/progress/<job_id>")
# def api_editor_progress(job_id):
#     job = EDIT_JOBS.get(job_id)
#     if not job:
#         return jsonify({"error": "Unknown edit job"}), 404
#     return jsonify({k: v for k, v in job.items() if k != "config"} | {
#         "features": job["config"]["features"], "export_ratios": job["config"]["export_ratios"]
#     })
 
 
# @editor_bp.route("/api/editor/file/<job_id>/<ratio>")
# def api_editor_file(job_id, ratio):
#     job = EDIT_JOBS.get(job_id)
#     if not job or job.get("status") != "done":
#         return "Not ready", 404
#     fname = job["ratios"].get(ratio)
#     if not fname:
#         return "Ratio not found for this job", 404
#     p = EDIT_DIR / fname
#     if not p.exists():
#         return "Not found", 404
#     return send_file(p, as_attachment=True, download_name=fname)
 
 
# # ── manual crop keyframe add/remove (per not-yet-rendered job config) ──
 
# @editor_bp.route("/api/editor/keyframe/add", methods=["POST"])
# def api_keyframe_add():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     kf = {"t": float(data["t"]), "x": int(data["x"]), "y": int(data["y"])}
#     job["config"].setdefault("manual_crop_keyframes", [])
#     job["config"]["manual_crop_keyframes"] = job["config"]["manual_crop_keyframes"] or []
#     job["config"]["manual_crop_keyframes"].append(kf)
#     job["config"]["manual_crop_keyframes"].sort(key=lambda k: k["t"])
#     return jsonify({"manual_crop_keyframes": job["config"]["manual_crop_keyframes"]})
 
 
# @editor_bp.route("/api/editor/keyframe/remove", methods=["POST"])
# def api_keyframe_remove():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     t = float(data["t"])
#     kfs = job["config"].get("manual_crop_keyframes") or []
#     job["config"]["manual_crop_keyframes"] = [k for k in kfs if abs(k["t"] - t) > 1e-6]
#     return jsonify({"manual_crop_keyframes": job["config"]["manual_crop_keyframes"]})
 
 
# # ── hook-cut / rearrange segment add/remove ──
 
# @editor_bp.route("/api/editor/segment/add", methods=["POST"])
# def api_segment_add():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     seg = {"start": float(data["start"]), "end": float(data["end"])}
#     index = data.get("index")
#     segs = job["config"].get("segments") or []
#     if index is None or index >= len(segs):
#         segs.append(seg)
#     else:
#         segs.insert(int(index), seg)
#     job["config"]["segments"] = segs
#     return jsonify({"segments": segs})
 
 
# @editor_bp.route("/api/editor/segment/remove", methods=["POST"])
# def api_segment_remove():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     index = int(data["index"])
#     segs = job["config"].get("segments") or []
#     if 0 <= index < len(segs):
#         segs.pop(index)
#     job["config"]["segments"] = segs
#     return jsonify({"segments": segs})








# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# AutoShortAi — Video Editor module
# ──────────────────────────────────
# Adds AI auto-reframe (face tracking), trending animated captions,
# hook-cut / manual clip reordering, and one-click multi-aspect-ratio
# export — all producing a SINGLE muxed video+audio file per ratio.
 
# Wire it into the main app with:
 
#     from video_editor import editor_bp, init_editor
#     init_editor(BASE, FFMPEG)
#     app.register_blueprint(editor_bp)
 
# Routes exposed (all under /api/editor/...):
#     GET  /api/editor/editor             -> serves the standalone editor.html UI (open this in a tab)
#     POST /api/editor/upload             -> upload a local file, returns a source_path to use below
#     POST /api/editor/analyze/start       -> kicks off a background analyze job (face-track + real captions + AI viral-clip finder + hook/silence detection)
#     GET  /api/editor/analyze/progress/<job_id> -> live analyze progress + log console + final result
#     GET  /api/editor/styles             -> list of caption style presets
#     GET  /api/editor/ratios             -> list of export aspect-ratio presets
#     GET  /api/editor/features           -> which advanced features are available (installed libs)
#     POST /api/editor/render             -> kicks off a background render job
#     GET  /api/editor/progress/<job_id>  -> live progress
#     GET  /api/editor/file/<job_id>/<ratio> -> serves one finished ratio's file
#     POST /api/editor/keyframe/add       -> add a manual crop keyframe to a job's override track
#     POST /api/editor/keyframe/remove    -> remove a manual crop keyframe
#     POST /api/editor/segment/add        -> add a hook-cut / reorder segment
#     POST /api/editor/segment/remove     -> remove a hook-cut / reorder segment
 
# ADVANCED EXTRAS (on top of the original spec):
#     • fill_mode="blur_pad"   -> reframe by fitting the WHOLE frame + blurred
#       background bars instead of hard-cropping (no content ever cut off).
#     • jump_cut feature       -> auto-detects and removes silence/dead-air,
#       the other half of "trending" short-form editing besides hook-cut.
#     • hook_cut + jump_cut COMBINED -> if both are on, the loudest window is
#       used as the cold-open "hook" and is prepended in front of the
#       dead-air-trimmed rest of the video (previously turning jump_cut on
#       silently dropped hook_cut — now they compose).
#     • normalize_audio        -> EBU R128 loudness normalization (-14 LUFS),
#       on by default, matches the loudness social platforms expect.
#     • caption keyword emphasis -> numbers, money, and punchy "power words"
#       auto-highlight in captions even outside full word-by-word styles.
#     • quality/caption_style inputs are now validated with a safe fallback
#       instead of raising a raw KeyError if the frontend sends a typo'd key.
#     • self-serves its own frontend (editor.html) at GET /api/editor/editor,
#       the same "own file, same process/port" pattern as downloader.py.
 
# DEPENDENCIES (install what you want to use — everything degrades gracefully
# if a library is missing, see FEATURE FLAGS AT IMPORT TIME below):
#     pip install opencv-python            # required for any face tracking
#     pip install mediapipe                # optional, gives the "advanced" tracker
#                                           # (falls back to Haar cascade otherwise)
#     pip install faster-whisper           # installed-but-unused; USE_WHISPER=False by design,
#                                           # see the "transcript source" section below
#     pip install numpy

# NO MORE WHISPER (by request): transcripts now come ONLY from the source
# platform's own captions, pulled via yt-dlp (see get_transcript_words /
# fetch_ytdlp_transcript). A video with no such captions (a plain PC upload,
# or a platform video with captions disabled) simply has no transcript for
# that run — subtitles + the AI viral-clip finder are skipped for it, while
# face-tracking and the loudness/silence-based hook-cut & jump-cut (neither
# needs a transcript) keep working exactly as before. faster-whisper's code
# is left in the file (unused) rather than deleted, in case you want it back.

# AI VIRAL-CLIP FINDER: when a transcript IS available, it is sent to the
# first working provider in AI_PROVIDERS (Gemini -> Groq -> OpenRouter ->
# Cerebras, in that order, all with free tiers) to rank every strong
# short-form clip candidate. See AI_PROVIDERS near the top of the file to
# paste in your own API keys (or set the matching environment variable).
 
# DESIGN NOTES — how each part of the request maps to code:
#   • "advance level face tracking"   -> _FaceTracker (mediapipe BlazeFace if present,
#                                         else OpenCV Haar cascade), EMA-smoothed
#                                         centroid so the crop doesn't jitter frame
#                                         to frame, multi-face -> largest/most-central
#                                         face wins.
#   • "perfect advance subtitles,
#      trending caption styles"       -> CAPTION_STYLES presets + build_ass() which
#                                         groups words into short on-screen chunks and
#                                         emits karaoke-style \\k tags for word-by-word
#                                         pop/highlight animation, burned in with
#                                         ffmpeg's `ass` filter (not a soft-sub track,
#                                         so it always displays correctly everywhere).
#   • "single high quality video+audio
#      in one file"                   -> every render always outputs one .mp4 with
#                                         both streams muxed, CRF-based high quality
#                                         encode (see QUALITY_PRESETS).
#   • "export in all ratio perfectly" -> RATIO_PRESETS, rendered in a loop, one file
#                                         per ratio, each independently croppped using
#                                         the SAME face track (recomputed crop window
#                                         per target aspect).
#   • "rearrange video manually /
#      hook cut"                      -> job["segments"]: an ordered list of
#                                         {start,end} clip ranges. Auto hook-cut picks
#                                         the loudest window as segment 1 automatically;
#                                         manual mode lets the caller add/remove/reorder
#                                         segments directly (segment/add, segment/remove).
#   • "add and remove option for all" -> every feature is a boolean flag in job config
#                                         (features{}) that can be flipped and re-rendered,
#                                         plus explicit add/remove endpoints for the two
#                                         list-based tracks (crop keyframes, segments).
# """
 
# import os
# import re
# import json
# import uuid
# import time
# import struct
# import threading
# import subprocess
# import urllib.request
# import urllib.error
# import concurrent.futures
# from pathlib import Path
 
# from flask import Blueprint, request, jsonify, send_file
 
# editor_bp = Blueprint("editor_bp", __name__)
 
# # ══════════════════════════════ embedded frontend ══════════════════════════════
# # The whole editor UI lives right here as one string — no separate .html file
# # to ship or lose track of. Served as-is by GET /api/editor/editor. Its fetch
# # calls use relative paths (same-origin), so it works immediately wherever
# # this blueprint is registered.
# EDITOR_HTML = r"""<!doctype html>
# <html lang="en">
# <head>
# <meta charset="utf-8">
# <meta name="viewport" content="width=device-width, initial-scale=1">
# <title>Reframe — AI video editor</title>
# <link rel="preconnect" href="https://fonts.googleapis.com">
# <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
# <style>
#   /* ── Theme — mirrors the main app's palette exactly (same variable
#      VALUES, just different variable NAMES for this page's own CSS). Default
#      below is "dark-violet", the main app's default. The <html data-theme="…">
#      overrides further down are read straight off the SAME localStorage key
#      the main app writes to (see the sync script at the bottom of <body>),
#      so switching themes on Studio/Downloader also re-themes this page —
#      no manual matching needed ever again. ── */
#   :root{
#     --bg:#0b0d12; --panel:#13151c; --panel-2:#191c25; --border:#262a36;
#     --text:#eef0f6; --text-dim:#8a90a4; --text-faint:#8a90a4;
#     --amber:#6e5bff; --teal:#22d3c4; --red:#ff5a6e;
#     --amber-dim: color-mix(in srgb, var(--amber) 22%, #000);
#     --teal-dim: color-mix(in srgb, var(--teal) 22%, #000);
#     --radius:10px;
#     font-size:15px;
#   }
#   html[data-theme="light"]{
#     --bg:#f4f5f9; --panel:#ffffff; --panel-2:#eef0f6; --border:#dde1ea;
#     --text:#171923; --text-dim:#5b6472; --text-faint:#5b6472;
#     --amber:#5b46e0; --teal:#0891b2; --red:#e0324a;
#   }
#   html[data-theme="midnight-blue"]{
#     --bg:#050a16; --panel:#0c1526; --panel-2:#12213a; --border:#1f3358;
#     --text:#e8f1ff; --text-dim:#7f93b8; --text-faint:#7f93b8;
#     --amber:#3b82f6; --teal:#22d3ee; --red:#ff5a6e;
#   }
#   html[data-theme="forest"]{
#     --bg:#0a130d; --panel:#0f1e14; --panel-2:#16301f; --border:#234a30;
#     --text:#eafbea; --text-dim:#86b399; --text-faint:#86b399;
#     --amber:#22c55e; --teal:#a3e635; --red:#ff5a6e;
#   }
#   html[data-theme="sunset"]{
#     --bg:#1a0e12; --panel:#241419; --panel-2:#331d27; --border:#4a2635;
#     --text:#fdf1ee; --text-dim:#cb9aa3; --text-faint:#cb9aa3;
#     --amber:#fb7185; --teal:#f59e0b; --red:#ff5a6e;
#   }
#   html[data-theme="ocean"]{
#     --bg:#061319; --panel:#0b1f27; --panel-2:#123039; --border:#1e4a56;
#     --text:#e6fbff; --text-dim:#7fb8c4; --text-faint:#7fb8c4;
#     --amber:#06b6d4; --teal:#6366f1; --red:#ff5a6e;
#   }
#   html[data-theme="rose"]{
#     --bg:#170f14; --panel:#221720; --panel-2:#2e1e2b; --border:#4a2f45;
#     --text:#fdeef5; --text-dim:#c79ab0; --text-faint:#c79ab0;
#     --amber:#ec4899; --teal:#f472b6; --red:#ff5a6e;
#   }
#   html[data-theme="amber"]{
#     --bg:#160f05; --panel:#221708; --panel-2:#33230d; --border:#4d3416;
#     --text:#fdf4e3; --text-dim:#c9a877; --text-faint:#c9a877;
#     --amber:#f59e0b; --teal:#eab308; --red:#ff5a6e;
#   }
#   html[data-theme="slate"]{
#     --bg:#101215; --panel:#181b1f; --panel-2:#20242a; --border:#333941;
#     --text:#eceef1; --text-dim:#9aa2ad; --text-faint:#9aa2ad;
#     --amber:#64748b; --teal:#94a3b8; --red:#ff5a6e;
#   }
#   html[data-theme="cyberpunk"]{
#     --bg:#0a0014; --panel:#150124; --panel-2:#210235; --border:#4a0a6b;
#     --text:#f5e6ff; --text-dim:#b083d6; --text-faint:#b083d6;
#     --amber:#e91e9c; --teal:#00f0ff; --red:#ff5a6e;
#   }
#   *{box-sizing:border-box;}
#   body{
#     margin:0; background:var(--bg); color:var(--text);
#     font-family:'Inter',sans-serif; min-height:100vh;
#   }
#   h1,h2,h3,.display{font-family:'Space Grotesk',sans-serif;}
#   .mono{font-family:'JetBrains Mono',monospace;}
#   ::selection{background:var(--amber-dim); color:var(--amber);}
 
#   /* ── top bar ── */
#   .topbar{
#     display:flex; align-items:center; gap:16px; padding:14px 22px;
#     border-bottom:1px solid var(--border); background:var(--panel);
#     position:sticky; top:0; z-index:20;
#   }
#   .brand{display:flex; align-items:center; gap:10px;}
#   .brand .dot{width:10px; height:10px; border-radius:50%; background:var(--amber); box-shadow:0 0 10px var(--amber);}
#   .brand h1{font-size:18px; font-weight:700; margin:0; letter-spacing:.2px;}
#   .brand span{color:var(--text-dim); font-size:12px;}
#   .topbar .spacer{flex:1;}
#   .api-input{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text-dim);
#     border-radius:8px; padding:7px 10px; font-size:12px; font-family:'JetBrains Mono',monospace;
#     width:220px;
#   }
#   .api-input:focus{outline:none; border-color:var(--amber); color:var(--text);}
 
#   /* ── layout ── */
#   .app{display:grid; grid-template-columns:1fr 380px; gap:1px; background:var(--border);}
#   @media (max-width:980px){ .app{grid-template-columns:1fr;} }
#   .col{background:var(--bg); padding:22px; min-width:0;}
 
#   /* ── upload zone ── */
#   .dropzone{
#     border:1.5px dashed var(--border); border-radius:var(--radius);
#     padding:40px 20px; text-align:center; cursor:pointer;
#     transition:border-color .15s, background .15s;
#   }
#   .dropzone:hover, .dropzone.drag{border-color:var(--amber); background:rgba(255,176,32,.04);}
#   .dropzone svg{opacity:.5; margin-bottom:10px;}
#   .dropzone .hint{color:var(--text-dim); font-size:13px; margin-top:6px;}
 
#   /* ── fetch-from-link (ytdlp-style resolver, built into the editor) ── */
#   .fetch-section{border:1px solid var(--border); border-radius:var(--radius); padding:12px; background:var(--panel-2); margin-bottom:14px;}
#   .fetch-row{display:flex; gap:8px; margin-top:8px;}
#   .fetch-input{
#     flex:1; background:var(--panel); border:1px solid var(--border); color:var(--text);
#     border-radius:8px; padding:9px 10px; font-size:13px;
#   }
#   .fetch-input:focus{outline:none; border-color:var(--amber);}
#   .btn-accent{background:var(--amber); color:#1a1200; border:none; font-weight:600;}
#   .btn-accent:hover{filter:brightness(1.08);}
#   .fetch-card{
#     display:flex; gap:10px; align-items:center; margin-top:10px;
#     border:1px solid var(--border); border-radius:8px; padding:10px; background:var(--panel);
#   }
#   .fetch-thumb{width:88px; height:56px; object-fit:cover; border-radius:6px; background:#000; flex-shrink:0;}
#   .fetch-meta{flex:1; min-width:0;}
#   .fetch-title{font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
#   .fetch-sub{font-size:11.5px; color:var(--text-dim); margin-top:2px;}
#   .fetch-status{font-size:12px; color:var(--text-dim); margin-top:8px; min-height:16px;}
 
#   /* ── server library (downloader output + past uploads) ── */
#   .or-divider{
#     display:flex; align-items:center; gap:10px; margin:14px 0;
#     color:var(--text-faint); font-size:12px; text-transform:uppercase; letter-spacing:.5px;
#   }
#   .or-divider::before, .or-divider::after{content:""; flex:1; height:1px; background:var(--border);}
#   .lib-section{border:1px solid var(--border); border-radius:var(--radius); padding:12px; background:var(--panel-2);}
#   .lib-header{display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; font-size:13px; color:var(--text-dim);}
#   .lib-grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:8px; max-height:220px; overflow-y:auto;}
#   .lib-empty{color:var(--text-faint); font-size:12.5px; grid-column:1/-1;}
#   .lib-item{
#     border:1px solid var(--border); border-radius:8px; padding:10px; cursor:pointer;
#     background:var(--panel); transition:border-color .15s, transform .1s;
#   }
#   .lib-item:hover{border-color:var(--amber); transform:translateY(-1px);}
#   .lib-item.active{border-color:var(--teal); background:var(--teal-dim);}
#   .lib-item .lib-name{font-size:12.5px; font-weight:600; word-break:break-word; margin-bottom:4px;}
#   .lib-item .lib-meta{font-size:11px; color:var(--text-dim); display:flex; justify-content:space-between;}
#   .lib-item .lib-tag{
#     display:inline-block; font-size:10px; padding:1px 6px; border-radius:20px;
#     background:var(--amber-dim); color:var(--amber); margin-bottom:4px;
#   }
 
#   /* ── preview ── */
#   .preview-wrap{position:relative; display:flex; justify-content:center; margin-bottom:16px;}
#   .preview-frame{position:relative; background:#000; border-radius:8px; overflow:hidden; max-width:100%;}
#   video{display:block; max-width:100%; max-height:60vh;}
#   .crop-box{
#     position:absolute; border:1.5px solid var(--amber); pointer-events:none;
#     box-shadow:0 0 0 2000px rgba(0,0,0,.45);
#   }
#   .crop-box .corner{
#     position:absolute; width:16px; height:16px; border-color:var(--amber); border-style:solid; border-width:0;
#   }
#   .crop-box .tl{top:-1.5px; left:-1.5px; border-top-width:3px; border-left-width:3px;}
#   .crop-box .tr{top:-1.5px; right:-1.5px; border-top-width:3px; border-right-width:3px;}
#   .crop-box .bl{bottom:-1.5px; left:-1.5px; border-bottom-width:3px; border-left-width:3px;}
#   .crop-box .br{bottom:-1.5px; right:-1.5px; border-bottom-width:3px; border-right-width:3px;}
#   .crop-box{cursor:grab;}
#   .crop-box.dragging{cursor:grabbing;}
#   .crop-readout{
#     position:absolute; top:8px; left:8px; background:rgba(0,0,0,.6);
#     color:var(--amber); font-size:11px; padding:3px 7px; border-radius:5px;
#   }
 
#   .preview-controls{display:flex; align-items:center; gap:10px; justify-content:center; margin-bottom:8px;}
#   .btn-icon{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text);
#     width:34px; height:34px; border-radius:50%; cursor:pointer; display:flex;
#     align-items:center; justify-content:center;
#   }
#   .btn-icon:hover{border-color:var(--amber);}
#   .time{font-size:12px; color:var(--text-dim); min-width:96px; text-align:center;}
 
#   /* ── timeline ── */
#   .timeline-panel{background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:16px;}
#   .timeline-panel h3{margin:0 0 4px; font-size:13px; letter-spacing:.4px; text-transform:uppercase; color:var(--text-dim);}
#   .timeline{
#     position:relative; height:56px; background:var(--panel-2); border-radius:6px;
#     margin-top:10px; overflow:hidden; cursor:pointer;
#   }
#   .tl-face{position:absolute; top:0; height:18px; background:linear-gradient(90deg, transparent, var(--teal-dim)); opacity:.6;}
#   .tl-seg{
#     position:absolute; top:20px; height:18px; background:var(--teal); opacity:.85; border-radius:3px;
#     display:flex; align-items:center; justify-content:center; font-size:10px; color:#04231D; font-weight:600;
#     cursor:grab;
#   }
#   .tl-hook{position:absolute; top:38px; height:14px; background:var(--amber); opacity:.9; border-radius:3px;}
#   .tl-playhead{position:absolute; top:0; bottom:0; width:2px; background:var(--red);}
#   .tl-legend{display:flex; gap:16px; margin-top:8px; font-size:11px; color:var(--text-dim);}
#   .tl-legend span{display:inline-flex; align-items:center; gap:5px;}
#   .swatch{width:9px; height:9px; border-radius:2px; display:inline-block;}
 
#   .seg-list{margin-top:10px; display:flex; flex-direction:column; gap:6px;}
#   .seg-row{
#     display:flex; align-items:center; gap:8px; background:var(--panel-2);
#     border:1px solid var(--border); border-radius:7px; padding:6px 10px; font-size:12px;
#   }
#   .seg-row .mono{color:var(--teal);}
#   .seg-row .spacer{flex:1;}
#   .seg-row button{background:none; border:none; color:var(--text-faint); cursor:pointer; font-size:14px;}
#   .seg-row button:hover{color:var(--red);}
#   .btn-small{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text-dim);
#     border-radius:6px; padding:6px 10px; font-size:12px; cursor:pointer;
#   }
#   .btn-small:hover{border-color:var(--amber); color:var(--amber);}
 
#   /* ── side panel ── */
#   .section{margin-bottom:26px;}
#   .section h3{
#     font-size:12px; letter-spacing:.5px; text-transform:uppercase; color:var(--text-dim);
#     margin:0 0 12px; display:flex; align-items:center; gap:8px;
#   }
#   .section h3 .num{
#     width:18px; height:18px; border-radius:50%; background:var(--panel-2); border:1px solid var(--border);
#     display:flex; align-items:center; justify-content:center; font-size:10px; color:var(--amber);
#   }
 
#   .toggle-row{
#     display:flex; align-items:center; justify-content:space-between;
#     padding:10px 12px; background:var(--panel); border:1px solid var(--border);
#     border-radius:8px; margin-bottom:8px;
#   }
#   .toggle-row .label{font-size:13px;}
#   .toggle-row .desc{font-size:11px; color:var(--text-faint); margin-top:2px;}
#   .toggle-row.disabled{opacity:.4;}
#   .switch{position:relative; width:38px; height:21px; flex-shrink:0;}
#   .switch input{opacity:0; width:0; height:0;}
#   .slider{
#     position:absolute; inset:0; background:var(--panel-2); border:1px solid var(--border);
#     border-radius:20px; cursor:pointer; transition:.15s;
#   }
#   .slider::before{
#     content:''; position:absolute; width:15px; height:15px; left:2px; top:2px;
#     background:var(--text-dim); border-radius:50%; transition:.15s;
#   }
#   .switch input:checked + .slider{background:var(--amber-dim); border-color:var(--amber);}
#   .switch input:checked + .slider::before{transform:translateX(17px); background:var(--amber);}
#   .switch input:disabled + .slider{cursor:not-allowed;}
 
#   .ratio-grid{display:grid; grid-template-columns:1fr 1fr; gap:8px;}
#   .ratio-card{
#     border:1px solid var(--border); background:var(--panel); border-radius:8px;
#     padding:10px; cursor:pointer; text-align:center; position:relative;
#   }
#   .ratio-card.active{border-color:var(--amber); background:rgba(255,176,32,.06);}
#   .ratio-card .shape{margin:0 auto 6px; background:var(--panel-2); border:1px solid var(--border); border-radius:3px;}
#   .ratio-card .name{font-size:12px; font-weight:600;}
#   .ratio-card .lbl{font-size:10px; color:var(--text-faint); margin-top:2px;}
 
#   .style-list{display:flex; flex-direction:column; gap:8px;}
#   .style-card{
#     display:flex; align-items:center; gap:12px; border:1px solid var(--border);
#     background:var(--panel); border-radius:8px; padding:10px 12px; cursor:pointer;
#   }
#   .style-card.active{border-color:var(--teal);}
#   .style-swatch{
#     flex-shrink:0; width:64px; height:36px; border-radius:6px; background:#000;
#     display:flex; align-items:center; justify-content:center; font-size:9px; font-weight:800;
#     letter-spacing:.5px;
#   }
#   .style-card .name{font-size:12.5px; font-weight:600;}
#   .style-card .lbl{font-size:10.5px; color:var(--text-faint);}
 
#   select, .fill-toggle{
#     width:100%; background:var(--panel); border:1px solid var(--border); color:var(--text);
#     padding:9px 10px; border-radius:8px; font-size:13px;
#   }
#   .fill-toggle{display:flex; gap:6px;}
#   .fill-opt{flex:1; text-align:center; padding:8px; border-radius:6px; cursor:pointer; font-size:12px; border:1px solid var(--border);}
#   .fill-opt.active{border-color:var(--amber); color:var(--amber); background:rgba(255,176,32,.06);}
 
#   .render-btn{
#     width:100%; background:var(--amber); color:#1A1204; border:none; border-radius:10px;
#     padding:14px; font-size:14px; font-weight:700; cursor:pointer; font-family:'Space Grotesk',sans-serif;
#     letter-spacing:.3px;
#   }
#   .render-btn:disabled{background:var(--panel-2); color:var(--text-faint); cursor:not-allowed;}
#   .render-btn:not(:disabled):hover{filter:brightness(1.08);}
 
#   .progress-wrap{margin-top:14px;}
#   .progress-track{height:6px; background:var(--panel-2); border-radius:4px; overflow:hidden;}
#   .progress-fill{height:100%; background:linear-gradient(90deg, var(--teal), var(--amber)); width:0%; transition:width .3s;}
#   .progress-label{display:flex; justify-content:space-between; font-size:11px; color:var(--text-dim); margin-top:6px;}
#   .log-console{margin-top:10px; max-height:150px; overflow-y:auto; background:var(--bg);
#     border:1px solid var(--border); border-radius:8px; padding:8px 10px; font-family:'Consolas',monospace;
#     font-size:11px; line-height:1.6; color:var(--text-dim);}
#   .log-console .log-line{white-space:pre-wrap; word-break:break-word;}
#   .log-console .log-line:last-child{color:var(--text);}
#   .ai-clip-row{background:var(--panel-2); border:1px solid var(--border); border-radius:10px;
#     padding:10px 12px; margin-bottom:8px;}
#   .ai-clip-row .ai-clip-top{display:flex; justify-content:space-between; align-items:center; gap:8px;}
#   .ai-clip-row .ai-clip-score{font-weight:800; color:var(--amber); font-size:13px;}
#   .ai-clip-row .ai-clip-hook{font-weight:700; margin:4px 0;}
#   .ai-clip-row .ai-clip-meta{font-size:11px; color:var(--text-dim);}
#   .ai-clip-row .ai-clip-reason{font-size:12px; color:var(--text-dim); margin-top:4px;}
 
#   .results{display:flex; flex-direction:column; gap:8px; margin-top:14px;}
#   .result-row{
#     display:flex; align-items:center; gap:10px; background:var(--panel); border:1px solid var(--border);
#     border-radius:8px; padding:10px 12px;
#   }
#   .result-row .name{font-size:12.5px; font-weight:600; flex:1;}
#   .result-row a{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--teal);
#     text-decoration:none; font-size:11.5px; padding:6px 10px; border-radius:6px;
#   }
#   .result-row a:hover{border-color:var(--teal);}
 
#   .status-line{font-size:11.5px; color:var(--text-faint); margin-top:10px; line-height:1.5;}
#   .status-line b{color:var(--text-dim);}
#   .feature-warn{
#     font-size:11px; color:var(--amber); background:rgba(255,176,32,.08); border:1px solid var(--amber-dim);
#     border-radius:6px; padding:8px 10px; margin-top:8px; display:none;
#   }
# </style>
# </head>
# <body>
# <script>
# // ── Theme sync — this page is served from the SAME Flask app/port as the
# // main Studio (registered as a blueprint), and when opened as an <iframe> on
# // that page it's also same-origin, so it shares localStorage automatically.
# // Reading 'shortsStudioTheme' here — the exact key the main page's theme
# // picker writes to — means this editor always matches whatever theme is
# // active there, with zero manual setup. Runs immediately (before first
# // paint) so there's no flash of the wrong palette, and again whenever the
# // theme changes elsewhere (the 'storage' event fires in this frame whenever
# // another same-origin window/frame updates localStorage).
# (function(){
#   function applyTheme(){
#     const t = localStorage.getItem('shortsStudioTheme') || 'dark-violet';
#     if(t === 'dark-violet') document.documentElement.removeAttribute('data-theme');
#     else document.documentElement.setAttribute('data-theme', t);
#   }
#   applyTheme();
#   window.addEventListener('storage', (e)=>{ if(e.key === 'shortsStudioTheme') applyTheme(); });
# })();
# </script>
 
# <div class="topbar">
#   <div class="brand"><span class="dot"></span><h1>Reframe</h1><span>face-tracked auto edit</span></div>
#   <div class="spacer"></div>
#   <span id="apiBaseHint" class="mono" style="font-size:11px; color:var(--text-faint); margin-right:6px;">same server</span>
#   <button type="button" id="apiBaseToggle" class="btn-small" title="Only needed if this page is served from a different host than the API">⚙</button>
#   <input id="apiBase" class="api-input mono" placeholder="leave blank — uses this same page's server" value="" style="display:none;">
# </div>
 
# <div class="app">
#   <!-- LEFT: preview + timeline -->
#   <div class="col">
#     <div class="fetch-section">
#       <div class="lib-header">
#         <span>🔗 Fetch from a link (YouTube / Instagram / TikTok / X / Facebook…)</span>
#       </div>
#       <div class="fetch-row">
#         <input type="text" id="fetchUrl" class="fetch-input mono" placeholder="Paste a video URL…">
#         <button type="button" class="btn-small btn-accent" id="fetchInfoBtn">Fetch</button>
#       </div>
#       <div id="fetchCard" class="fetch-card" style="display:none;">
#         <img id="fetchThumb" class="fetch-thumb" alt="">
#         <div class="fetch-meta">
#           <div id="fetchTitle" class="fetch-title"></div>
#           <div id="fetchSub" class="fetch-sub"></div>
#         </div>
#         <button type="button" class="btn-small btn-accent" id="fetchUseBtn">⬇ Fetch high-quality &amp; use</button>
#       </div>
#       <div id="fetchStatus" class="fetch-status"></div>
#     </div>
 
#     <div class="or-divider">or drop / choose a video from this PC</div>
#     <div id="dropzone" class="dropzone">
#       <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8A93A3" stroke-width="1.5"><path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v3a2 2 0 002 2h12a2 2 0 002-2v-3"/></svg>
#       <div>Drop a video, or click to choose one</div>
#       <div class="hint">mp4 / mov · analyzed locally before render</div>
#       <input type="file" id="fileInput" accept="video/*" style="display:none">
#     </div>
 
#     <div class="or-divider">or pick a video already fetched by the Downloader</div>
#     <div class="lib-section">
#       <div class="lib-header">
#         <span>📥 From Downloader / server library</span>
#         <button type="button" class="btn-small" id="refreshLibBtn">Refresh</button>
#       </div>
#       <div class="lib-grid" id="libGrid"><span class="lib-empty">Click Refresh to list videos already downloaded via the Downloader tab.</span></div>
#     </div>
 
#     <div id="previewSection" style="display:none;">
#       <div class="preview-wrap">
#         <div class="preview-frame" id="previewFrame">
#           <video id="video" muted></video>
#           <div class="crop-box" id="cropBox" style="display:none;">
#             <div class="corner tl"></div><div class="corner tr"></div><div class="corner bl"></div><div class="corner br"></div>
#             <div class="crop-readout" id="cropReadout">9:16</div>
#           </div>
#         </div>
#       </div>
#       <div class="preview-controls">
#         <div class="btn-icon" id="playBtn">▶</div>
#         <div class="time mono" id="timeLabel">0:00 / 0:00</div>
#       </div>
 
#       <div class="status-line" id="statusLine"></div>
#       <div class="progress-wrap" id="analyzeProgressWrap" style="display:none;">
#         <div class="progress-track"><div class="progress-fill" id="analyzeProgressFill"></div></div>
#         <div class="progress-label"><span id="analyzeProgressStage">queued</span><span id="analyzeProgressPct">0%</span></div>
#         <div class="log-console" id="analyzeLogConsole"></div>
#       </div>
#       <div id="sourceMetaPanel" class="lib-section" style="display:none; margin-top:10px;"></div>

#       <div id="aiClipsPanel" class="lib-section" style="display:none; margin-top:10px;">
#         <div class="lib-header"><span>✨ AI-suggested viral clips</span></div>
#         <div class="hint" style="padding:8px 12px 0;">Ranked best → worst. Tap "Use this clip" to set it as your export segment — nothing else to configure.</div>
#         <div id="aiClipsList" style="padding:10px 12px;"></div>
#       </div>

#       <div class="timeline-panel">
#         <h3>Timeline</h3>
#         <div class="hint" style="margin:-6px 0 10px;">
#           This bar is the whole video. The buttons below it are all OPTIONAL manual overrides —
#           if you don't touch anything here, Export still uses the whole video with auto face-tracking.
#           Use these only if you want to hand-pick a moment:
#         </div>
#         <div class="timeline" id="timeline">
#           <div class="tl-playhead" id="playhead" style="left:0%"></div>
#         </div>
#         <div class="tl-legend">
#           <span><span class="swatch" style="background:var(--teal-dim)"></span>face track</span>
#           <span><span class="swatch" style="background:var(--teal)"></span>segments</span>
#           <span><span class="swatch" style="background:var(--amber)"></span>suggested hook</span>
#         </div>
#         <div class="seg-list" id="segList"></div>
#         <div style="display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;">
#           <button class="btn-small" id="addSegBtn" title="Click play, pause where you want a clip to start/end, then click this">+ add segment at playhead (±3s)</button>
#           <button class="btn-small" id="useHookBtn" style="display:none;" title="Auto-picked loudest/most energetic moment as a cold open">use suggested hook</button>
#           <button class="btn-small" id="useSilenceBtn" style="display:none;" title="Auto-removes dead air / silence between lines">use auto jump-cut segments</button>
#         </div>
#         <div style="display:flex; gap:8px; margin-top:8px; flex-wrap:wrap;">
#           <button class="btn-small" id="addKfBtn" title="Locks the face-crop box to exactly where it is right now, at this timestamp">+ pin crop here (manual keyframe)</button>
#           <button class="btn-small" id="clearKfBtn">clear manual crop</button>
#         </div>
#       </div>
#     </div>
#   </div>

 
#   <!-- RIGHT: controls -->
#   <div class="col">
#     <div class="section">
#       <h3><span class="num">1</span>Auto-reframe</h3>
#       <div class="toggle-row" id="rowFace">
#         <div><div class="label">Face tracking</div><div class="desc">Advanced (mediapipe) if installed, else standard cascade</div></div>
#         <label class="switch"><input type="checkbox" id="chkFace" checked><span class="slider"></span></label>
#       </div>
#       <div class="fill-toggle" style="margin-top:8px;">
#         <div class="fill-opt active" data-fill="crop">Crop to face</div>
#         <div class="fill-opt" data-fill="blur_pad">Fit + blurred bars</div>
#       </div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">2</span>Captions</h3>
#       <div class="toggle-row" id="rowSubs">
#         <div><div class="label">Auto subtitles</div><div class="desc">Word-level, keyword emphasis auto-highlighted</div></div>
#         <label class="switch"><input type="checkbox" id="chkSubs" checked><span class="slider"></span></label>
#       </div>
#       <div class="style-list" id="styleList"></div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">3</span>Hook &amp; pacing</h3>
#       <div class="toggle-row" id="rowHook">
#         <div><div class="label">Hook-cut</div><div class="desc">Open on the loudest / most energetic moment</div></div>
#         <label class="switch"><input type="checkbox" id="chkHook"><span class="slider"></span></label>
#       </div>
#       <div class="toggle-row" id="rowJump">
#         <div><div class="label">Jump-cut silences</div><div class="desc">Auto-remove dead air between lines</div></div>
#         <label class="switch"><input type="checkbox" id="chkJump"><span class="slider"></span></label>
#       </div>
#       <div class="toggle-row">
#         <div><div class="label">Loudness normalize</div><div class="desc">-14 LUFS, matches platform playback volume</div></div>
#         <label class="switch"><input type="checkbox" id="chkNorm" checked><span class="slider"></span></label>
#       </div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">4</span>Export ratios</h3>
#       <div class="ratio-grid" id="ratioGrid"></div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">5</span>Quality</h3>
#       <select id="qualitySel">
#         <option value="high">High (crf 18, slow) — best quality</option>
#         <option value="medium">Medium (crf 21) — balanced</option>
#         <option value="fast">Fast (crf 23) — quick preview</option>
#       </select>
#     </div>
 
#     <button class="render-btn" id="renderBtn" disabled>Analyze a video first</button>
#     <div class="feature-warn" id="featureWarn"></div>
#     <div class="progress-wrap" id="progressWrap" style="display:none;">
#       <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
#       <div class="progress-label"><span id="progressStage">queued</span><span id="progressPct">0%</span></div>
#       <div class="log-console" id="renderLogConsole"></div>
#     </div>
#     <div class="results" id="results"></div>
#   </div>
# </div>
 
# <script>
# const $ = (id) => document.getElementById(id);
# const state = {
#   sourcePath: null, sourceMeta: null, sourceOrigin: 'pc', duration: 0, srcW: 0, srcH: 0, srcFps: null,
#   faceTrack: [], words: [], suggestedHook: null, suggestedSilences: null, aiClips: [],
#   segments: [], manualKeyframes: [],
#   // Multiple export ratios can be selected at once now (a Set of ratio
#   // keys, e.g. {'9:16','1:1'}), not just one — "auto" is pre-selected by
#   // default since it's the safest zero-quality-loss choice.
#   selectedRatios: new Set(['auto']),
#   previewRatio: '9:16', // drives the live crop box only, independent of export selection
#   selectedStyle: 'bold_pop', fillMode: 'crop',
#   jobId: null, dragging: false,
# };
 
# // Same-origin by default: this page is served BY the same Flask app/port
# // that Reframe, the Downloader, and everything else run on (they're all
# // registered on one Flask app as blueprints, per RenderDetect's main()),
# // so relative paths already hit the right server. The "⚙" override is
# // only for the rare case this HTML got copied to a different host — if
# // left blank (the default), api() never leaves the current origin, which
# // is what fixes the class of bug where a stray/incorrect API host caused
# // every request (analyze, render, fetch…) to silently fail.
# function apiBase(){ const v = $('apiBase').value.trim().replace(/\/$/, ''); return v; }
# function api(path){ return apiBase() + path; }
# $('apiBaseToggle').onclick = ()=>{
#   const el = $('apiBase');
#   const show = el.style.display === 'none';
#   el.style.display = show ? 'inline-block' : 'none';
#   $('apiBaseHint').style.display = show ? 'none' : 'inline';
# };
 
# const RATIO_META = {
#   '9:16': {w:9,h:16}, '16:9': {w:16,h:9}, '1:1': {w:1,h:1}, '4:5': {w:4,h:5}, '4:3': {w:4,h:3},
# };
# const STYLE_META = {
#   bold_pop:       {bg:'#111', color:'#fff', hi:'#FFD700', sample:'THIS IS'},
#   karaoke_classic:{bg:'#111', color:'#fff', hi:'#FFA500', sample:'FILLS IN'},
#   neon_glow:      {bg:'#111', color:'#fff', hi:'#00F0FF', sample:'GLOWS UP'},
#   minimal_clean:  {bg:'#111', color:'#fff', hi:'#fff', sample:'stays quiet'},
#   creator_bold:   {bg:'#111', color:'#FFFF00', hi:'#FFFF00', sample:'HUGE TEXT'},
# };
 
# // ── boot: load styles / ratios / features from backend ──
# async function loadMeta(){
#   try{
#     const [styles, ratios, features] = await Promise.all([
#       fetch(api('/api/editor/styles')).then(r=>r.json()),
#       fetch(api('/api/editor/ratios')).then(r=>r.json()),
#       fetch(api('/api/editor/features')).then(r=>r.json()),
#     ]);
#     renderStyles(styles);
#     renderRatios(ratios);
#     applyFeatureAvailability(features);
#   }catch(e){
#     $('featureWarn').style.display='block';
#     $('featureWarn').textContent = 'Could not reach the backend at ' + api('') + ' — set the API base URL above and reload.';
#   }
# }
 
# function renderStyles(styles){
#   const list = $('styleList'); list.innerHTML='';
#   Object.entries(styles).forEach(([key, meta])=>{
#     const m = STYLE_META[key] || {bg:'#111', color:'#fff', hi:'#fff', sample:'Sample'};
#     const card = document.createElement('div');
#     card.className = 'style-card' + (key===state.selectedStyle ? ' active':'');
#     card.innerHTML = `<div class="style-swatch" style="background:${m.bg}">
#         <span style="color:${m.color}">${m.sample.split(' ')[0]}</span>&nbsp;<span style="color:${m.hi}">${m.sample.split(' ')[1]||''}</span>
#       </div>
#       <div><div class="name">${meta.label.split(' (')[0]}</div><div class="lbl">${(meta.label.match(/\((.*)\)/)||[,''])[1]}</div></div>`;
#     card.onclick = ()=>{ state.selectedStyle = key; renderStyles(styles); };
#     list.appendChild(card);
#   });
# }
 
# // Export ratios are now MULTI-select — tap any number of cards and every
# // one you pick gets rendered and offered as a download, all in the same
# // render job. "Auto" (native size, no crop, zero quality loss) is its own
# // card too. state.previewRatio (separate from the export selection) just
# // drives which crop box is drawn live over the video for editing.
# function renderRatios(ratios){
#   const grid = $('ratioGrid'); grid.innerHTML='';
#   Object.entries(ratios).forEach(([key, r])=>{
#     const isAuto = key === 'auto';
#     const srcAspect = (state.srcW && state.srcH) ? {w: state.srcW, h: state.srcH} : {w:9, h:16};
#     const meta = isAuto ? srcAspect : (RATIO_META[key] || {w:1,h:1});
#     const scale = 30 / Math.max(meta.w, meta.h);
#     const selected = state.selectedRatios.has(key);
#     const card = document.createElement('div');
#     card.className = 'ratio-card' + (selected ? ' active':'');
#     card.innerHTML = `<div class="shape" style="width:${meta.w*scale}px; height:${meta.h*scale}px;"></div>
#       <div class="name">${selected ? '✓ ' : ''}${key === 'auto' ? 'Auto' : key}</div><div class="lbl">${r.label}</div>`;
#     card.onclick = ()=>{
#       if(state.selectedRatios.has(key)) state.selectedRatios.delete(key);
#       else state.selectedRatios.add(key);
#       if(!state.selectedRatios.size) state.selectedRatios.add(key); // keep at least one selected
#       if(!isAuto){ state.previewRatio = key; drawCropForCurrentTime(); }
#       renderRatios(ratios);
#     };
#     grid.appendChild(card);
#   });
# }
 
# function applyFeatureAvailability(f){
#   toggleRow('rowFace', 'chkFace', f.face_tracking, 'opencv-python not installed on the server');
#   toggleRow('rowSubs', 'chkSubs', f.subtitles, 'faster-whisper not installed on the server');
#   toggleRow('rowHook', 'chkHook', true, '');
#   toggleRow('rowJump', 'chkJump', true, '');
#   if (!f.face_tracking_advanced && f.face_tracking){
#     $('rowFace').querySelector('.desc').textContent = 'Standard tracker active (install mediapipe for advanced mode)';
#   }
# }
# function toggleRow(rowId, chkId, available, msg){
#   if(!available){
#     $(rowId).classList.add('disabled');
#     $(chkId).checked = false;
#     $(chkId).disabled = true;
#     $(rowId).querySelector('.desc').textContent = msg;
#   }
# }
 
# // ── fill mode ──
# document.querySelectorAll('.fill-opt').forEach(el=>{
#   el.onclick = ()=>{
#     document.querySelectorAll('.fill-opt').forEach(o=>o.classList.remove('active'));
#     el.classList.add('active');
#     state.fillMode = el.dataset.fill;
#     drawCropForCurrentTime();
#   };
# });
 
# // ── fetch-from-link (built-in yt-dlp resolver, high quality single mp4) ──
# // Mirrors the Downloader tab's speed trick (cached auth mode) but skips the
# // full format-picker UI on purpose: this always grabs the single best
# // available video+audio quality and merges it into one mp4, then feeds it
# // straight into the exact same analyze/render pipeline as an upload.
# let fetchState = { url: null };
 
# $('fetchInfoBtn').onclick = async ()=>{
#   const url = $('fetchUrl').value.trim();
#   if(!url) return;
#   $('fetchStatus').textContent = 'Resolving link…';
#   $('fetchCard').style.display = 'none';
#   try{
#     const res = await fetch(api('/api/editor/fetch/info'), {
#       method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url})
#     });
#     const data = await res.json();
#     if(data.error){ $('fetchStatus').textContent = data.error; return; }
#     fetchState.url = url;
#     fetchState.meta = data;   // keep the FULL metadata (views/likes/uploader/etc.) for later
#     $('fetchThumb').src = data.thumbnail || '';
#     $('fetchTitle').textContent = data.title || 'Untitled video';
#     $('fetchSub').textContent = [data.uploader, data.duration_str, data.resolution].filter(Boolean).join(' · ');
#     $('fetchCard').style.display = 'flex';
#     $('fetchStatus').textContent = '';
#   }catch(e){
#     $('fetchStatus').textContent = 'Could not reach the server to resolve that link. Make sure the app is running and reload this page.';
#   }
# };
 
# $('fetchUseBtn').onclick = async ()=>{
#   $('fetchUseBtn').disabled = true;
#   $('fetchStatus').textContent = 'Fetching best quality (video + audio, single file)…';
#   try{
#     const res = await fetch(api('/api/editor/fetch/start'), {
#       method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url: fetchState.url})
#     });
#     const data = await res.json();
#     if(data.error){ $('fetchStatus').textContent = data.error; $('fetchUseBtn').disabled = false; return; }
#     pollFetchProgress(data.fetch_id);
#   }catch(e){
#     $('fetchStatus').textContent = 'Fetch failed to start — check the connection and try again.';
#     $('fetchUseBtn').disabled = false;
#   }
# };
 
# async function pollFetchProgress(fetchId){
#   let st;
#   try{
#     const res = await fetch(api('/api/editor/fetch/progress/'+fetchId));
#     st = await res.json();
#   }catch(e){
#     setTimeout(()=>pollFetchProgress(fetchId), 1200); // transient network hiccup, keep polling
#     return;
#   }
#   if(st.status === 'error'){
#     $('fetchStatus').textContent = 'Error: ' + st.error;
#     $('fetchUseBtn').disabled = false;
#     return;
#   }
#   if(st.status === 'done'){
#     $('fetchStatus').textContent = 'Downloaded — analyzing…';
#     $('fetchUseBtn').disabled = false;
#     $('fetchCard').style.display = 'none';
#     $('fetchUrl').value = '';
#     loadSource(st.source_path, fetchState.meta, 'fetch');
#     return;
#   }
#   const pct = st.percent != null ? st.percent + '%' : '';
#   $('fetchStatus').textContent = `${st.stage || 'downloading'}… ${pct} ${st.speed_str || ''}`.trim();
#   setTimeout(()=>pollFetchProgress(fetchId), 900);
# }
 
# // ── server library (downloader output + earlier uploads) ──
# // Lets the user reuse a video already fetched via the Downloader tab (or an
# // earlier upload) without picking it from disk again — same analyze/render
# // pipeline as a fresh upload, just a different source_path origin.
# $('refreshLibBtn').onclick = loadLibrary;
 
# async function loadLibrary(){
#   const grid = $('libGrid');
#   grid.innerHTML = '<span class="lib-empty">Loading…</span>';
#   try{
#     const res = await fetch(api('/api/editor/sources'));
#     const data = await res.json();
#     const items = data.sources || [];
#     if(!items.length){
#       grid.innerHTML = '<span class="lib-empty">Nothing yet — fetch a video in the Downloader tab, or upload one above.</span>';
#       return;
#     }
#     grid.innerHTML = '';
#     items.forEach(it=>{
#       const el = document.createElement('div');
#       el.className = 'lib-item';
#       el.innerHTML = `<span class="lib-tag">${it.origin === 'downloader' ? '📥 downloaded' : '📤 uploaded'}</span>
#         <div class="lib-name">${it.name}</div>
#         <div class="lib-meta"><span>${it.size_str||''}</span><span>${it.modified_str||''}</span></div>`;
#       el.onclick = ()=>{
#         document.querySelectorAll('.lib-item.active').forEach(n=>n.classList.remove('active'));
#         el.classList.add('active');
#         loadSource(it.path, null, it.origin === 'downloader' ? 'downloader' : 'pc');
#       };
#       grid.appendChild(el);
#     });
#   }catch(e){
#     grid.innerHTML = '<span class="lib-empty">Could not reach the server — make sure the app is running, then hit Refresh again.</span>';
#   }
# }
 
# loadLibrary();
 
# // ── upload flow ──
# $('dropzone').onclick = ()=> $('fileInput').click();
# $('fileInput').onchange = (e)=> handleFile(e.target.files[0]);
# ['dragover','dragleave','drop'].forEach(evt=>{
#   $('dropzone').addEventListener(evt, (e)=>{
#     e.preventDefault();
#     $('dropzone').classList.toggle('drag', evt==='dragover');
#     if(evt==='drop' && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
#   });
# });
 
# async function handleFile(file){
#   if(!file) return;
#   $('video').src = URL.createObjectURL(file);
#   $('previewSection').style.display = 'block';
#   $('statusLine').textContent = 'Uploading…';
#   try{
#     const fd = new FormData(); fd.append('file', file);
#     const res = await fetch(api('/api/editor/upload'), {method:'POST', body: fd});
#     const data = await res.json();
#     if(data.error){ showAnalysisError('Upload error: ' + data.error); return; }
#     loadSource(data.source_path, null, 'pc');
#   }catch(e){
#     showAnalysisError('Could not reach the server to upload this file.');
#   }
# }
 
# // ── unified source loader — every origin (fetched link, downloader
# // library, or a file picked from this PC) funnels through here so
# // analysis always runs the same way and the Export panel always ends up
# // enabled the same way, no matter where the video came from. ──
# async function loadSource(sourcePath, meta, origin){
#   state.sourcePath = sourcePath;
#   state.sourceMeta = meta || null;
#   state.sourceOrigin = origin || 'pc';
#   $('video').src = api('/api/editor/preview?path=' + encodeURIComponent(sourcePath));
#   $('previewSection').style.display = 'block';
#   renderSourceMeta();
#   await runAnalysis();
# }
 
# // Shows the rich details panel. For links fetched from YouTube/Instagram/
# // etc. this uses the metadata the SERVER already pulled during Fetch (no
# // need to re-derive title/uploader/views by hand); for PC uploads or
# // library items with no such metadata it just relies on the ffprobe-based
# // technical details that come back from /api/editor/analyze.
# function renderSourceMeta(){
#   const el = $('sourceMetaPanel');
#   if(!el) return;
#   const m = state.sourceMeta || {};
#   const rows = [
#     ['Title', m.title],
#     ['Channel / uploader', m.uploader],
#     ['Views', m.view_count_str],
#     ['Likes', m.like_count_str],
#     ['Uploaded', m.upload_date_str],
#     // Technical rows come from ffmpeg probing (_probe on the server) and are
#     // filled in for EVERY source — PC upload, Downloader-library pick, or a
#     // freshly fetched link — not just ones that happen to have yt-dlp
#     // metadata. This is what makes "same full details" true across origins.
#     ['Resolution', state.srcW && state.srcH ? `${state.srcW}×${state.srcH}` : null],
#     ['Duration', state.duration ? fmtTime(state.duration) : null],
#     ['Frame rate', state.srcFps ? `${state.srcFps} fps` : null],
#   ].filter(([,v]) => v);
#   if(!rows.length){ el.style.display = 'none'; el.innerHTML = ''; return; }
#   const heading = m.title ? 'ℹ️ Fetched video details' : 'ℹ️ Video details';
#   el.style.display = 'block';
#   el.innerHTML = `<div class="lib-header"><span>${heading}</span></div>` +
#     rows.map(([k,v]) => `<div class="status-line"><b>${k}:</b> ${v}</div>`).join('');
# }
 
# function showAnalysisError(msg){
#   $('statusLine').innerHTML = `<span style="color:var(--amber);">${msg}</span> ` +
#     `<button type="button" class="btn-small" id="retryAnalysisBtn" style="margin-left:6px;">Retry analysis</button>`;
#   const btn = $('retryAnalysisBtn');
#   if(btn) btn.onclick = () => runAnalysis();
#   $('renderBtn').disabled = true;
#   $('renderBtn').textContent = 'Analyze a video first';
# }
 
# async function runAnalysis(){
#   if(!state.sourcePath){ showAnalysisError('No video selected yet.'); return; }
#   $('renderBtn').disabled = true;
#   $('statusLine').textContent = 'Starting analysis…';
#   $('analyzeProgressWrap').style.display = 'block';
#   $('analyzeProgressFill').style.width = '0%';
#   $('analyzeProgressPct').textContent = '0%';
#   $('analyzeProgressStage').textContent = 'queued';
#   $('analyzeLogConsole').innerHTML = '';
#   $('aiClipsPanel').style.display = 'none';
#   const body = {
#     source_path: state.sourcePath,
#     face_tracking: $('chkFace').checked && !$('chkFace').disabled,
#     subtitles: $('chkSubs').checked && !$('chkSubs').disabled,
#     hook_cut: true,
#     jump_cut: true,
#   };
#   let jobId;
#   try{
#     const res = await fetch(api('/api/editor/analyze/start'), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
#     if(!res.ok){ showAnalysisError(`Analysis request failed (server said HTTP ${res.status}).`); return; }
#     const started = await res.json();
#     if(started.error){ showAnalysisError('Analysis error: ' + started.error); return; }
#     jobId = started.job_id;
#   }catch(e){
#     showAnalysisError('Could not reach the server to start analysis — check it is still running, then retry.');
#     return;
#   }
#   pollAnalyzeProgress(jobId);
# }

# function _renderLogLines(el, logs){
#   if(!logs || !logs.length) return;
#   el.innerHTML = logs.map(l => `<div class="log-line">${l.replace(/</g,'&lt;')}</div>`).join('');
#   el.scrollTop = el.scrollHeight;
# }

# async function pollAnalyzeProgress(jobId){
#   let job;
#   try{
#     const res = await fetch(api('/api/editor/analyze/progress/'+jobId));
#     job = await res.json();
#   }catch(e){
#     setTimeout(()=>pollAnalyzeProgress(jobId), 1200); // transient hiccup, keep polling
#     return;
#   }
#   $('analyzeProgressFill').style.width = (job.percent||0)+'%';
#   $('analyzeProgressStage').textContent = job.stage || job.status;
#   $('analyzeProgressPct').textContent = (job.percent||0)+'%';
#   _renderLogLines($('analyzeLogConsole'), job.logs);

#   if(job.status === 'error'){
#     showAnalysisError('Analysis error: ' + job.error);
#     return;
#   }
#   if(job.status !== 'done'){ setTimeout(()=>pollAnalyzeProgress(jobId), 900); return; }

#   const data = job.result || {};
#   state.duration = data.probe.duration || 0;
#   state.srcW = (data.source_size && data.source_size.w) || data.probe.width;
#   state.srcH = (data.source_size && data.source_size.h) || data.probe.height;
#   state.srcFps = data.probe.fps ? Math.round(data.probe.fps * 100) / 100 : null;
#   renderSourceMeta();
#   state.faceTrack = data.face_track || [];
#   state.words = data.words || [];
#   state.suggestedHook = data.suggested_hook || null;
#   state.suggestedSilences = data.suggested_segments || null;
#   state.aiClips = data.ai_clips || [];
#   $('useHookBtn').style.display = state.suggestedHook ? 'inline-block' : 'none';
#   $('useSilenceBtn').style.display = state.suggestedSilences ? 'inline-block' : 'none';
#   renderAiClips();
#   drawTimeline();
#   // Export becomes available the instant analysis finishes, identically
#   // for an uploaded PC file, a Downloader-library pick, or a fresh link
#   // fetch — there is no special-casing by source here. This is true even
#   // if one sub-feature (e.g. subtitles) failed above — analyze always
#   // finishes, so render is never blocked by one slow/failed piece.
#   $('renderBtn').disabled = false;
#   $('renderBtn').textContent = 'Render exports';
#   const srcTag = state.sourceOrigin === 'fetch' ? 'fetched video' : (state.sourceOrigin === 'downloader' ? 'downloaded video' : 'uploaded video');
#   const warnings = [data.face_tracking_error, data.hook_cut_error, data.jump_cut_error].filter(Boolean);
#   let line = `Analyzed ${srcTag} — <b>${state.faceTrack.length}</b> face samples · <b>${state.words.length}</b> caption words · duration <b>${fmtTime(state.duration)}</b> · source <b>${state.srcW}×${state.srcH}</b>`;
#   if(warnings.length) line += ` <span style="color:var(--amber);">(${warnings.join(' · ')})</span>`;
#   $('statusLine').innerHTML = line;
# }

# // Shows the AI-ranked viral clip list (start/end/hook/score/reason) built
# // from the source's real captions — nothing to configure, just tap a clip
# // to make it the export segment.
# function renderAiClips(){
#   const panel = $('aiClipsPanel'), list = $('aiClipsList');
#   if(!state.aiClips || !state.aiClips.length){ panel.style.display = 'none'; list.innerHTML=''; return; }
#   panel.style.display = 'block';
#   list.innerHTML = state.aiClips.map((c, i) => `
#     <div class="ai-clip-row">
#       <div class="ai-clip-top">
#         <span class="ai-clip-score">Score ${c.score}</span>
#         <span class="ai-clip-meta">${fmtTime(c.start_time)} → ${fmtTime(c.end_time)} · ${c.duration.toFixed(1)}s · ${c.category||''}</span>
#       </div>
#       <div class="ai-clip-hook">${(c.hook_text||'').replace(/</g,'&lt;')}</div>
#       <div class="ai-clip-reason">${(c.reason||'').replace(/</g,'&lt;')} ${c.editing_suggestion ? '· ' + c.editing_suggestion.replace(/</g,'&lt;') : ''}</div>
#       <button type="button" class="btn-small" style="margin-top:8px;" onclick="useAiClip(${i})">Use this clip</button>
#     </div>
#   `).join('');
# }

# function useAiClip(i){
#   const c = state.aiClips[i];
#   if(!c) return;
#   state.segments = [{start: c.start_time, end: c.end_time}];
#   drawTimeline();
#   $('statusLine').innerHTML = `Using AI clip: <b>${fmtTime(c.start_time)}–${fmtTime(c.end_time)}</b> (score ${c.score}) as the export segment.`;
# }
 
# // ── video controls ──
# const video = $('video');
# $('playBtn').onclick = ()=>{ video.paused ? video.play() : video.pause(); };
# video.onplay = ()=> $('playBtn').textContent = '❚❚';
# video.onpause = ()=> $('playBtn').textContent = '▶';
# video.ontimeupdate = ()=>{
#   $('timeLabel').textContent = `${fmtTime(video.currentTime)} / ${fmtTime(video.duration||0)}`;
#   const pct = video.duration ? (video.currentTime/video.duration*100) : 0;
#   $('playhead').style.left = pct + '%';
#   drawCropForCurrentTime();
# };
# function fmtTime(t){ t=Math.max(0,t||0); const m=Math.floor(t/60), s=Math.floor(t%60); return `${m}:${String(s).padStart(2,'0')}`; }
 
# // ── crop overlay: shows the current auto/manual crop box, draggable ──
# function currentCrop(){
#   const kfs = state.manualKeyframes.length ? state.manualKeyframes : state.faceTrack.map(p=>({t:p.t, x:p.cx, y:p.cy}));
#   if(!kfs.length) return {cx:0.5, cy:0.5};
#   const t = video.currentTime;
#   let lo = kfs[0], hi = kfs[kfs.length-1];
#   for(let i=0;i<kfs.length-1;i++){ if(kfs[i].t<=t && kfs[i+1].t>=t){ lo=kfs[i]; hi=kfs[i+1]; break; } }
#   const span = (hi.t - lo.t) || 1;
#   const f = Math.min(1, Math.max(0, (t - lo.t)/span));
#   const cxKey = state.manualKeyframes.length ? 'x' : 'cx';
#   const cyKey = state.manualKeyframes.length ? 'y' : 'cy';
#   return { cx: lo[cxKey] + (hi[cxKey]-lo[cxKey])*f, cy: lo[cyKey] + (hi[cyKey]-lo[cyKey])*f };
# }
 
# function drawCropForCurrentTime(){
#   const box = $('cropBox');
#   if(state.fillMode === 'blur_pad' || !$('chkFace').checked){ box.style.display='none'; return; }
#   if(!state.srcW || !video.videoWidth){ return; }
#   const meta = RATIO_META[state.previewRatio || '9:16'];
#   const targetAR = meta.w/meta.h;
#   const srcAR = video.videoWidth/video.videoHeight;
#   let cropWFrac, cropHFrac;
#   if(srcAR > targetAR){ cropHFrac = 1; cropWFrac = targetAR/srcAR; }
#   else { cropWFrac = 1; cropHFrac = srcAR/targetAR; }
#   const {cx, cy} = currentCrop();
#   let left = cx - cropWFrac/2, top = cy - cropHFrac/2;
#   left = Math.min(1-cropWFrac, Math.max(0, left));
#   top = Math.min(1-cropHFrac, Math.max(0, top));
 
#   const frame = $('previewFrame').getBoundingClientRect();
#   box.style.display = 'block';
#   box.style.left = (left*100)+'%';
#   box.style.top = (top*100)+'%';
#   box.style.width = (cropWFrac*100)+'%';
#   box.style.height = (cropHFrac*100)+'%';
#   $('cropReadout').textContent = state.previewRatio || '9:16';
#   box.dataset.cx = cx; box.dataset.cy = cy;
# }
 
# // dragging the crop box records a manual keyframe candidate (committed via "pin crop here")
# let dragStart = null;
# $('cropBox').addEventListener('mousedown', (e)=>{
#   state.dragging = true; $('cropBox').classList.add('dragging');
#   dragStart = {x:e.clientX, y:e.clientY, left:parseFloat($('cropBox').style.left), top:parseFloat($('cropBox').style.top)};
# });
# window.addEventListener('mousemove', (e)=>{
#   if(!state.dragging) return;
#   const frame = $('previewFrame').getBoundingClientRect();
#   const dx = (e.clientX-dragStart.x)/frame.width*100;
#   const dy = (e.clientY-dragStart.y)/frame.height*100;
#   const box = $('cropBox');
#   const newLeft = Math.min(100-parseFloat(box.style.width), Math.max(0, dragStart.left+dx));
#   const newTop = Math.min(100-parseFloat(box.style.height), Math.max(0, dragStart.top+dy));
#   box.style.left = newLeft+'%'; box.style.top = newTop+'%';
#   box.dataset.cx = (newLeft + parseFloat(box.style.width)/2)/100;
#   box.dataset.cy = (newTop + parseFloat(box.style.height)/2)/100;
# });
# window.addEventListener('mouseup', ()=>{ state.dragging=false; $('cropBox').classList.remove('dragging'); });
 
# $('addKfBtn').onclick = ()=>{
#   const box = $('cropBox');
#   const meta = RATIO_META[state.previewRatio || '9:16'];
#   const srcAR = video.videoWidth/video.videoHeight;
#   const targetAR = meta.w/meta.h;
#   let cropW, cropH;
#   if(srcAR > targetAR){ cropH = state.srcH; cropW = Math.round(cropH*targetAR); }
#   else { cropW = state.srcW; cropH = Math.round(cropW/targetAR); }
#   const cx = parseFloat(box.dataset.cx||0.5), cy = parseFloat(box.dataset.cy||0.5);
#   let x = Math.round(cx*state.srcW - cropW/2), y = Math.round(cy*state.srcH - cropH/2);
#   x = Math.max(0, Math.min(x, state.srcW-cropW)); y = Math.max(0, Math.min(y, state.srcH-cropH));
#   state.manualKeyframes = state.manualKeyframes.filter(k=>Math.abs(k.t-video.currentTime)>0.05);
#   state.manualKeyframes.push({t: video.currentTime, x, y});
#   state.manualKeyframes.sort((a,b)=>a.t-b.t);
#   drawTimeline();
#   $('statusLine').textContent = `Pinned manual crop at ${fmtTime(video.currentTime)} — ${state.manualKeyframes.length} manual keyframe(s) active (overrides auto tracking).`;
# };
# $('clearKfBtn').onclick = ()=>{ state.manualKeyframes=[]; drawTimeline(); drawCropForCurrentTime(); };
 
# // ── timeline: face density, segments, hook suggestion ──
# function drawTimeline(){
#   const tl = $('timeline');
#   [...tl.querySelectorAll('.tl-face,.tl-hook')].forEach(e=>e.remove());
#   if(state.duration){
#     state.faceTrack.forEach(p=>{
#       const el = document.createElement('div');
#       el.className='tl-face';
#       el.style.left = (p.t/state.duration*100)+'%'; el.style.width='2px';
#       tl.appendChild(el);
#     });
#     if(state.suggestedHook){
#       const el = document.createElement('div'); el.className='tl-hook';
#       el.style.left = (state.suggestedHook.start/state.duration*100)+'%';
#       el.style.width = ((state.suggestedHook.end-state.suggestedHook.start)/state.duration*100)+'%';
#       tl.appendChild(el);
#     }
#   }
#   renderSegList();
# }
# function renderSegList(){
#   const list = $('segList'); list.innerHTML='';
#   state.segments.forEach((seg, i)=>{
#     const row = document.createElement('div'); row.className='seg-row';
#     row.innerHTML = `<span class="mono">#${i+1}</span><span class="mono">${fmtTime(seg.start)} → ${fmtTime(seg.end)}</span><span class="spacer"></span><button data-i="${i}">✕</button>`;
#     row.querySelector('button').onclick = ()=>{ state.segments.splice(i,1); renderSegList(); };
#     list.appendChild(row);
#   });
# }
# $('timeline').addEventListener('click', (e)=>{
#   const rect = $('timeline').getBoundingClientRect();
#   const frac = (e.clientX-rect.left)/rect.width;
#   video.currentTime = frac * (video.duration||0);
# });
# $('addSegBtn').onclick = ()=>{
#   const t = video.currentTime;
#   state.segments.push({start: Math.max(0, +(t-3).toFixed(2)), end: Math.min(state.duration, +(t+3).toFixed(2))});
#   renderSegList();
# };
# $('useHookBtn').onclick = ()=>{
#   if(!state.suggestedHook) return;
#   state.segments = [state.suggestedHook, {start:0, end: state.duration}];
#   renderSegList();
#   $('chkHook').checked = true;
# };
# $('useSilenceBtn').onclick = ()=>{
#   if(!state.suggestedSilences) return;
#   state.segments = state.suggestedSilences;
#   renderSegList();
#   $('chkJump').checked = true;
# };
 
# // ── render ──
# $('renderBtn').onclick = async ()=>{
#   if(!state.selectedRatios.size){ $('statusLine').textContent = 'Pick at least one export ratio first.'; return; }
#   $('renderBtn').disabled = true;
#   $('progressWrap').style.display = 'block';
#   $('renderLogConsole').innerHTML = '';
#   $('results').innerHTML = '';
#   const body = {
#     source_path: state.sourcePath,
#     export_ratios: Array.from(state.selectedRatios),
#     face_tracking: $('chkFace').checked && !$('chkFace').disabled,
#     subtitles: $('chkSubs').checked && !$('chkSubs').disabled,
#     hook_cut: $('chkHook').checked,
#     jump_cut: $('chkJump').checked,
#     normalize_audio: $('chkNorm').checked,
#     caption_style: state.selectedStyle,
#     fill_mode: state.fillMode,
#     quality: $('qualitySel').value,
#     manual_crop_keyframes: state.manualKeyframes.length ? state.manualKeyframes : null,
#     segments: state.segments.length ? state.segments : null,
#   };
#   try{
#     const res = await fetch(api('/api/editor/render'), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
#     const data = await res.json();
#     if(data.error){ $('statusLine').textContent = 'Render error: '+data.error; $('renderBtn').disabled=false; return; }
#     state.jobId = data.job_id;
#     pollProgress();
#   }catch(e){
#     $('statusLine').textContent = 'Could not reach the server to start the render — check it is still running and try again.';
#     $('renderBtn').disabled = false;
#   }
# };
 
# async function pollProgress(){
#   let job;
#   try{
#     const res = await fetch(api('/api/editor/progress/'+state.jobId));
#     job = await res.json();
#   }catch(e){
#     setTimeout(pollProgress, 1500); // transient hiccup, keep polling
#     return;
#   }
#   $('progressFill').style.width = (job.percent||0)+'%';
#   $('progressStage').textContent = job.stage || job.status;
#   $('progressPct').textContent = (job.percent||0)+'%';
#   _renderLogLines($('renderLogConsole'), job.logs);
#   if(job.status === 'error'){
#     $('statusLine').textContent = 'Render failed: '+job.error;
#     $('renderBtn').disabled = false;
#     return;
#   }
#   if(job.status !== 'done'){ setTimeout(pollProgress, 1200); return; }
#   $('renderBtn').disabled = false;
#   const results = $('results');
#   Object.entries(job.ratios||{}).forEach(([ratio, fname])=>{
#     const row = document.createElement('div'); row.className='result-row';
#     const tag = ratio === 'auto' ? 'Auto (original size — lossless)' : `${ratio} export`;
#     row.innerHTML = `<span class="name">${tag}</span><a href="${api('/api/editor/file/'+state.jobId+'/'+ratio)}" download>Download</a>`;
#     results.appendChild(row);
#   });
# }
 
# loadMeta();
# </script>
# </body>
# </html>
# """
 
 
# # Optional, best-effort link to downloader.py's in-memory job table so the
# # editor can show nicer names for downloaded videos ("Video Title.mp4"
# # instead of "<dl_id>_Video Title.mp4"). video_editor.py still works fine
# # standalone if downloader.py isn't registered — this never raises.
# def _downloader_jobs():
#     try:
#         import downloader
#         return downloader.DL_JOBS
#     except Exception:
#         return {}
 
# # ── configured once via init_editor() ──────────────────────────────
# EDIT_DIR = None
# SRC_DIR = None
# FFMPEG_PATH = "ffmpeg"
# FFPROBE_PATH = "ffprobe"
 
# # job_id -> {status, stage, percent, error, ratios: {ratio: filename}, config}
# EDIT_JOBS = {}

# # job_id -> {status, stage, percent, error, logs: [...], result: {...}}
# # Analyze runs as a background job too now (instead of blocking the HTTP
# # request), specifically so the frontend can show a live progress bar +
# # log console instead of a page that just "hangs" with no feedback while
# # face-tracking / caption-fetch / AI clip-finding are running.
# ANALYZE_JOBS = {}


# def _job_log(job, msg, percent=None, stage=None):
#     """Appends a timestamped line to a job's log console feed and, in the
#     same call, optionally moves its progress bar — every stage of both
#     analyze and render report through this one function."""
#     job.setdefault("logs", []).append(f"[{time.strftime('%H:%M:%S')}] {msg}")
#     if percent is not None:
#         job["percent"] = percent
#     if stage is not None:
#         job["stage"] = stage
 
 
# def init_editor(base_dir, ffmpeg_path=None):
#     """Call once at startup, e.g. init_editor(BASE, FFMPEG)."""
#     global EDIT_DIR, SRC_DIR, FFMPEG_PATH, FFPROBE_PATH
#     base_dir = Path(base_dir)
#     EDIT_DIR = base_dir / "edited"
#     EDIT_DIR.mkdir(exist_ok=True)
#     SRC_DIR = base_dir / "downloads"   # reuse downloader's output as source pool
#     (EDIT_DIR / "uploads").mkdir(exist_ok=True)
#     if ffmpeg_path:
#         FFMPEG_PATH = str(ffmpeg_path)
#         # ffprobe normally sits next to ffmpeg
#         cand = Path(ffmpeg_path).with_name(
#             "ffprobe.exe" if str(ffmpeg_path).lower().endswith(".exe") else "ffprobe"
#         )
#         if cand.exists():
#             FFPROBE_PATH = str(cand)

#     # Whisper is intentionally NEVER auto-loaded anymore (see USE_WHISPER
#     # below) — transcripts come only from the source platform's own
#     # captions via yt-dlp, so there is nothing to pre-warm here.
 
 
# def _no_console_kwargs():
#     if os.name == "nt":
#         return {"creationflags": subprocess.CREATE_NO_WINDOW}
#     return {}
 
 
# def _run(cmd):
#     return subprocess.run(cmd, capture_output=True, text=True, **_no_console_kwargs())
 
 
# # ══════════════════════════════ feature availability ══════════════════════════════
 
# try:
#     import cv2
#     _HAS_CV2 = True
# except ImportError:
#     _HAS_CV2 = False
 
# try:
#     import mediapipe as mp
#     _HAS_MEDIAPIPE = True
#     # Importing successfully doesn't mean `mp.solutions.face_detection` still
#     # exists — recent mediapipe releases dropped the legacy `solutions` API
#     # in favor of the newer Tasks API. Probe it once here so feature_status()
#     # (and every caller) reports the truth instead of assuming "advanced" is
#     # available and crashing later inside _FaceTracker.
#     try:
#         mp.solutions.face_detection.FaceDetection
#         _HAS_MEDIAPIPE_SOLUTIONS = True
#     except AttributeError:
#         _HAS_MEDIAPIPE_SOLUTIONS = False
# except ImportError:
#     _HAS_MEDIAPIPE = False
#     _HAS_MEDIAPIPE_SOLUTIONS = False
 
# try:
#     from faster_whisper import WhisperModel
#     _HAS_WHISPER = True
# except ImportError:
#     _HAS_WHISPER = False
 
# try:
#     import numpy as np
#     _HAS_NUMPY = True
# except ImportError:
#     _HAS_NUMPY = False
 
# try:
#     import yt_dlp
#     _HAS_YTDLP = True
# except ImportError:
#     _HAS_YTDLP = False
 
# _WHISPER_MODEL = None  # lazy-loaded singleton
 
 
# def feature_status():
#     return {
#         "face_tracking": _HAS_CV2,
#         "face_tracking_advanced": _HAS_CV2 and _HAS_MEDIAPIPE_SOLUTIONS,
#         # Subtitles now depend on a caption SOURCE (yt-dlp, for a fetched
#         # video) being available, not on Whisper being installed — Whisper
#         # is intentionally disabled everywhere (USE_WHISPER = False).
#         "subtitles": _HAS_YTDLP,
#         "hook_cut_auto": _HAS_NUMPY,
#         "fetch_from_link": _HAS_YTDLP,
#         "ai_clip_finder": any(_ai_key(p) for p in AI_PROVIDERS),
#     }
 
 
# # ══════════════════════════════ presets ══════════════════════════════
 
# # width x height for each export ratio, chosen at "good enough to look sharp,
# # small enough to encode fast" — bump these if you want 4K exports.
# RATIO_PRESETS = {
#     # "auto" is a sentinel: w/h are resolved at render time from the SOURCE
#     # video's own dimensions (no target size baked in here). It means "keep
#     # the video exactly as shot" — no crop, no letterbox, no rescale — and
#     # whenever nothing else in the pipeline needs to touch the picture
#     # (no captions/color/manual crop requested) it is exported via ffmpeg
#     # stream-copy, i.e. the original video bytes are re-muxed, not
#     # re-encoded, so there is truly zero generational quality loss.
#     "auto":  {"w": None, "h": None, "label": "Auto — original size, no crop, zero quality loss"},
#     "9:16":  {"w": 1080, "h": 1920, "label": "Reels / Shorts / TikTok"},
#     "16:9":  {"w": 1920, "h": 1080, "label": "YouTube / Landscape"},
#     "1:1":   {"w": 1080, "h": 1080, "label": "Square / Feed post"},
#     "4:5":   {"w": 1080, "h": 1350, "label": "Instagram portrait"},
#     "4:3":   {"w": 1440, "h": 1080, "label": "Classic / Facebook"},
# }
 
# QUALITY_PRESETS = {
#     "high":   {"crf": 18, "preset": "slow"},
#     "medium": {"crf": 21, "preset": "medium"},
#     "fast":   {"crf": 23, "preset": "veryfast"},
# }
 
# # Words that get an automatic extra-emphasis color in captions even in
# # styles that don't do full word-by-word highlighting — numbers, money and
# # a short list of "power words" are what trending caption tools punch up.
# _EMPHASIS_PATTERN = None  # compiled lazily, see _is_emphasis_word()
# _POWER_WORDS = {
#     "free", "now", "never", "always", "secret", "insane", "crazy", "huge",
#     "warning", "stop", "wait", "new", "best", "worst", "you", "your",
# }
 
 
# def _is_emphasis_word(word):
#     import re
#     global _EMPHASIS_PATTERN
#     if _EMPHASIS_PATTERN is None:
#         _EMPHASIS_PATTERN = re.compile(r"[\d%$]|^\$")
#     w = word.strip(".,!?").lower()
#     return bool(_EMPHASIS_PATTERN.search(word)) or w in _POWER_WORDS
 
# # Trending caption styles. Each is a full ASS "Style:" line plus a couple of
# # behavior flags consumed by build_ass(). Colors are &HAABBGGRR (ASS order).
# CAPTION_STYLES = {
#     "bold_pop": {
#         "label": "Bold Pop (white + yellow highlight)",
#         "style": "Style: Default,Montserrat Black,84,&H00FFFFFF,&H0000D7FF,&H00101010,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,6,0,2,60,60,140,1",
#         "highlight_color": "&H0000D7FF",   # active word turns yellow/gold
#         "word_by_word": True,
#         "pop_scale": True,
#     },
#     "karaoke_classic": {
#         "label": "Karaoke Fill (progressive color wipe)",
#         "style": "Style: Default,Montserrat SemiBold,72,&H00FFFFFF,&H0000A5FF,&H00202020,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,4,0,2,60,60,150,1",
#         "highlight_color": "&H0000A5FF",
#         "word_by_word": True,
#         "karaoke_fill": True,
#     },
#     "neon_glow": {
#         "label": "Neon Glow (cyan outline pop)",
#         "style": "Style: Default,Montserrat Black,80,&H00FFFFFF,&H00FFF000,&H00902000,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,5,2,2,60,60,140,1",
#         "highlight_color": "&H00FFF000",
#         "word_by_word": True,
#         "pop_scale": True,
#     },
#     "minimal_clean": {
#         "label": "Minimal Clean (small centered white)",
#         "style": "Style: Default,Inter Medium,54,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
#                   "0,0,0,0,100,100,0,0,1,2,1,2,60,60,120,1",
#         "highlight_color": "&H00FFFFFF",
#         "word_by_word": False,
#         "pop_scale": False,
#     },
#     "creator_bold": {
#         "label": "Creator Bold (huge yellow, black outline)",
#         "style": "Style: Default,Montserrat Black,92,&H0000FFFF,&H000000FF,&H00000000,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,8,0,2,50,50,160,1",
#         "highlight_color": "&H000000FF",
#         "word_by_word": True,
#         "pop_scale": True,
#     },
# }
 
 
# @editor_bp.route("/api/editor/styles")
# def api_editor_styles():
#     return jsonify({k: {"label": v["label"]} for k, v in CAPTION_STYLES.items()})
 
 
# @editor_bp.route("/api/editor/ratios")
# def api_editor_ratios():
#     return jsonify(RATIO_PRESETS)
 
 
# @editor_bp.route("/api/editor/editor")
# def api_editor_page():
#     """Serves the built-in editor UI — no separate .html file to manage,
#     it's embedded in this module (EDITOR_HTML below) and rendered straight
#     from memory. Open this URL in a tab; its fetch calls are same-origin."""
#     from flask import Response
#     return Response(EDITOR_HTML, mimetype="text/html")
 
 
# def _fmt_size(n):
#     if not n:
#         return "0 B"
#     n = float(n)
#     for unit in ["B", "KB", "MB", "GB"]:
#         if n < 1024:
#             return f"{n:.1f} {unit}"
#         n /= 1024
#     return f"{n:.1f} TB"
 
 
# @editor_bp.route("/api/editor/sources")
# def api_editor_sources():
#     """Lists videos the editor can open WITHOUT a fresh upload: everything
#     downloader.py has saved to DL_DIR/downloads (so a just-fetched video can
#     be sent straight into the editor), plus anything uploaded here before.
#     Both origins return the exact same shape and both feed the exact same
#     source_path used by /analyze and /render — the editor doesn't care where
#     a file came from."""
#     import datetime
#     jobs = _downloader_jobs()
#     # dl_id -> nicer display title, when downloader.py is registered too
#     title_by_stub = {}
#     for job in jobs.values():
#         fname = job.get("filename")
#         if fname and "_" in fname:
#             stub = fname.split("_", 1)[0]
#             title_by_stub[stub] = fname.split("_", 1)[-1]
 
#     sources = []
#     for folder, origin in ((SRC_DIR, "downloader"), (EDIT_DIR / "uploads", "upload")):
#         if not folder or not folder.exists():
#             continue
#         for p in sorted(folder.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
#             if not p.is_file() or p.suffix.lower() not in (".mp4", ".mov", ".mkv", ".webm", ".m4v"):
#                 continue
#             stub = p.name.split("_", 1)[0]
#             display = title_by_stub.get(stub) or (p.name.split("_", 1)[-1] if "_" in p.name else p.name)
#             stat = p.stat()
#             sources.append({
#                 "name": display,
#                 "path": str(p.resolve()),
#                 "origin": origin,
#                 "size": stat.st_size,
#                 "size_str": _fmt_size(stat.st_size),
#                 "modified": stat.st_mtime,
#                 "modified_str": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%d %b, %H:%M"),
#             })
#     sources.sort(key=lambda s: s["modified"], reverse=True)
#     return jsonify({"sources": sources})
 
 
# def _is_allowed_source(path):
#     """Only ever stream files that live under the editor's own known roots
#     (downloader's downloads folder, this module's uploads folder, or its
#     own rendered-output folder) — never an arbitrary server path."""
#     try:
#         rp = Path(path).resolve()
#     except Exception:
#         return False
#     for root in (SRC_DIR, EDIT_DIR / "uploads", EDIT_DIR):
#         if root and root.exists():
#             try:
#                 rp.relative_to(root.resolve())
#                 return True
#             except ValueError:
#                 continue
#     return False
 
 
# @editor_bp.route("/api/editor/preview")
# def api_editor_preview():
#     """Streams a source (or rendered) video for <video> tag playback/scrub,
#     so picking a file from the library previews instantly without a
#     round-trip upload. Range requests are handled by Flask's send_file."""
#     path = request.args.get("path", "")
#     if not path or not _is_allowed_source(path) or not Path(path).exists():
#         return "Not found or not allowed", 404
#     return send_file(path, conditional=True)
 
 
# @editor_bp.route("/api/editor/features")
# def api_editor_features():
#     return jsonify(feature_status())
 
 
# # ══════════════════════════════ probing ══════════════════════════════
 
# def _probe(path):
#     """Returns {duration, width, height, fps} for any video — PC upload,
#     Downloader-library pick, or freshly fetched link — using ONLY ffmpeg
#     (FFMPEG_PATH), never ffprobe.

#     Why: this whole app is wired up with `FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()`
#     in the main file, which downloads/bundles an ffmpeg BINARY ONLY — it does
#     not ship ffprobe next to it, and a bare "ffprobe" is not guaranteed to be
#     on PATH on a fresh Windows machine (that mismatch is exactly what produced
#     `FileNotFoundError: [WinError 2] The system cannot find the file
#     specified` — the app was calling a program that doesn't exist on disk).
#     `ffmpeg -i <file>` already prints duration/resolution/fps to stderr for
#     every input, so we parse that instead — one binary, guaranteed present,
#     identical accurate metadata for every source, no more ffprobe dependency.
#     """
#     cmd = [FFMPEG_PATH, "-hide_banner", "-i", str(path)]
#     out = _run(cmd)
#     text = (out.stderr or "") + (out.stdout or "")

#     duration = 0.0
#     dm = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
#     if dm:
#         hh, mm, ss = dm.groups()
#         duration = int(hh) * 3600 + int(mm) * 60 + float(ss)

#     width = height = None
#     # e.g. "Stream #0:0: Video: h264 ..., yuv420p, 1080x1920 [SAR 1:1 DAR 9:16], 30 fps, 30 tbr, ..."
#     vm = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", text)
#     if vm:
#         width, height = int(vm.group(1)), int(vm.group(2))

#     fps = 25.0
#     fm = re.search(r"([\d.]+)\s+fps", text)
#     if fm:
#         try:
#             fps = float(fm.group(1))
#         except ValueError:
#             pass

#     return {"duration": duration, "width": width, "height": height, "fps": fps}
 
 
# # ══════════════════════════════ face tracking ══════════════════════════════
 
# class _FaceTracker:
#     """Samples frames, finds a face each time, and produces a smoothed
#     (jitter-free) track of the subject's center point over time.
 
#     Uses mediapipe's BlazeFace model when available ("advanced" mode — much
#     more accurate on angled/small faces), otherwise falls back to OpenCV's
#     bundled Haar cascade so face tracking still works with just opencv-python
#     installed.
#     """
 
#     def __init__(self):
#         # mediapipe importing successfully doesn't guarantee the legacy
#         # `mp.solutions.face_detection` API is present — recent mediapipe
#         # builds moved to a different Tasks API and dropped `.solutions`
#         # entirely, which used to crash every analyze request with
#         # `AttributeError: module 'mediapipe' has no attribute 'solutions'`.
#         # _HAS_MEDIAPIPE_SOLUTIONS was probed once at startup, so this just
#         # picks the mode that's actually usable instead of assuming.
#         self.advanced = _HAS_MEDIAPIPE_SOLUTIONS
#         if self.advanced:
#             self._mp_detector = mp.solutions.face_detection.FaceDetection(
#                 model_selection=1, min_detection_confidence=0.5
#             )
#         elif _HAS_CV2:
#             self._cascade = cv2.CascadeClassifier(
#                 cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#             )
#         else:
#             raise RuntimeError("opencv-python is required for face tracking")
 
#     def _detect(self, frame_bgr):
#         """Returns list of (cx, cy, w, h) in pixel coords, largest first."""
#         h_img, w_img = frame_bgr.shape[:2]
#         faces = []
#         if self.advanced:
#             rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#             result = self._mp_detector.process(rgb)
#             if result.detections:
#                 for d in result.detections:
#                     box = d.location_data.relative_bounding_box
#                     fw, fh = box.width * w_img, box.height * h_img
#                     fx = box.xmin * w_img + fw / 2
#                     fy = box.ymin * h_img + fh / 2
#                     faces.append((fx, fy, fw, fh))
#         else:
#             gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#             dets = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50))
#             for (x, y, w, h) in dets:
#                 faces.append((x + w / 2, y + h / 2, w, h))
#         faces.sort(key=lambda f: f[2] * f[3], reverse=True)  # largest face first
#         return faces
 
#     def track(self, video_path, sample_fps=2.0, progress_cb=None):
#         """Returns a list of {t, cx, cy, w, h} in NORMALIZED (0-1) coords,
#         exponentially smoothed so the crop pans instead of jumping."""
#         cap = cv2.VideoCapture(str(video_path))
#         if not cap.isOpened():
#             raise RuntimeError("Could not open video for face tracking")
#         src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#         w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         step = max(1, int(round(src_fps / sample_fps)))
 
#         raw_points = []
#         last_center = (0.5, 0.5)  # default: frame center when nobody's found yet
#         frame_idx = 0
#         while True:
#             ok = cap.grab()
#             if not ok:
#                 break
#             if frame_idx % step == 0:
#                 ok2, frame = cap.retrieve()
#                 if ok2:
#                     faces = self._detect(frame)
#                     if faces:
#                         fx, fy, fw, fh = faces[0]
#                         cx, cy = fx / w_img, fy / h_img
#                         last_center = (cx, cy)
#                     else:
#                         cx, cy = last_center  # hold last known position
#                     t = frame_idx / src_fps
#                     raw_points.append({"t": t, "cx": cx, "cy": cy})
#                 if progress_cb and total_frames:
#                     progress_cb(min(99, int(frame_idx * 100 / total_frames)))
#             frame_idx += 1
#         cap.release()
 
#         if not raw_points:
#             raw_points = [{"t": 0.0, "cx": 0.5, "cy": 0.5}]
 
#         # EMA smoothing so the crop glides instead of snapping every sample
#         alpha = 0.25
#         smoothed = [raw_points[0]]
#         for p in raw_points[1:]:
#             prev = smoothed[-1]
#             smoothed.append({
#                 "t": p["t"],
#                 "cx": prev["cx"] + alpha * (p["cx"] - prev["cx"]),
#                 "cy": prev["cy"] + alpha * (p["cy"] - prev["cy"]),
#             })
#         return smoothed, (w_img, h_img)
 
 
# def analyze_face_track(video_path, sample_fps=2.0, progress_cb=None):
#     if not _HAS_CV2:
#         raise RuntimeError("opencv-python is not installed — face tracking unavailable")
#     tracker = _FaceTracker()
#     points, (w, h) = tracker.track(video_path, sample_fps=sample_fps, progress_cb=progress_cb)
#     return points, w, h, tracker.advanced
 
 
# # ══════════════════════════════ crop-window math ══════════════════════════════
 
# def _crop_size_for_ratio(src_w, src_h, target_w, target_h):
#     """Largest crop rectangle matching the target aspect that still fits
#     inside the source frame (so we crop, never letterbox/pad)."""
#     target_ar = target_w / target_h
#     src_ar = src_w / src_h
#     if src_ar > target_ar:
#         # source is wider than target -> crop width, keep full height
#         crop_h = src_h
#         crop_w = int(round(crop_h * target_ar))
#     else:
#         # source is taller than target -> crop height, keep full width
#         crop_w = src_w
#         crop_h = int(round(crop_w / target_ar))
#     return crop_w, crop_h
 
 
# def _build_crop_keyframes(track_points, src_w, src_h, crop_w, crop_h):
#     """Turns normalized face-center points into pixel-space crop x/y
#     top-left keyframes, clamped so the crop never leaves the frame."""
#     kfs = []
#     for p in track_points:
#         cx_px = p["cx"] * src_w
#         cy_px = p["cy"] * src_h
#         x = int(cx_px - crop_w / 2)
#         y = int(cy_px - crop_h / 2)
#         x = max(0, min(x, src_w - crop_w))
#         y = max(0, min(y, src_h - crop_h))
#         kfs.append({"t": p["t"], "x": x, "y": y})
#     return kfs
 
 
# def _expr_chain(keyframes, key):
#     """Builds an ffmpeg time-expression string that linearly interpolates
#     between keyframes for either 'x' or 'y' — this drives a moving crop
#     window without needing any external filter graph patching."""
#     if len(keyframes) == 1:
#         return str(keyframes[0][key])
#     expr = str(keyframes[-1][key])
#     for i in range(len(keyframes) - 2, -1, -1):
#         t0, v0 = keyframes[i]["t"], keyframes[i][key]
#         t1, v1 = keyframes[i + 1]["t"], keyframes[i + 1][key]
#         if t1 <= t0:
#             continue
#         # linear ramp between (t0,v0) and (t1,v1), holds v0 before t0
#         ramp = f"({v0}+({v1}-{v0})*(t-{t0})/{(t1 - t0)})"
#         expr = f"if(lt(t,{t1}),{ramp},{expr})"
#     return expr
 
 
# # ══════════════════════════════ subtitles ══════════════════════════════
 
# def _load_whisper(model_size="small"):
#     global _WHISPER_MODEL
#     if _WHISPER_MODEL is None:
#         _WHISPER_MODEL = WhisperModel(model_size, compute_type="int8")
#     return _WHISPER_MODEL
 
 
# def transcribe_words(video_path, model_size="small", language=None):
#     """Returns [{start, end, text}, ...] word-level timestamps."""
#     if not _HAS_WHISPER:
#         raise RuntimeError("faster-whisper is not installed — subtitles unavailable")
#     model = _load_whisper(model_size)
#     segments, _info = model.transcribe(str(video_path), word_timestamps=True, language=language)
#     words = []
#     for seg in segments:
#         for w in (seg.words or []):
#             text = (w.word or "").strip()
#             if text:
#                 words.append({"start": w.start, "end": w.end, "text": text})
#     return words


# # ══════════════ transcript source: platform captions via yt-dlp (NO WHISPER) ══════════════
# # Whisper is intentionally never called automatically anymore. `transcribe_words`
# # above is left in place (harmless, dead code) in case you ever want to wire it
# # back in by hand — but every analyze/render call path now goes through
# # get_transcript_words() below instead, which only ever pulls REAL captions the
# # source platform already made (via yt-dlp), never runs local speech-to-text.
# # A plain PC upload, or a fetched video whose platform has no captions on it,
# # simply has no transcript: subtitles + the AI viral-clip finder are skipped
# # for that one video, while face-tracking and the loudness/silence-based
# # hook-cut & jump-cut (neither needs a transcript) keep working as before.
# USE_WHISPER = False

# SOURCE_URL_MAP = {}    # str(source_path) -> the URL it was fetched from (set in _run_fetch_job)
# SOURCE_TITLE_MAP = {}  # str(source_path) -> the platform title (set in _run_fetch_job)
# TRANSCRIPT_CACHE = {}  # str(source_path) -> {"words":[...], "chapters":[...], "source_kind":...} | None


# def _vtt_to_words(vtt_text):
#     """Turns a WebVTT captions file into an approximate word-level timeline
#     (real platform captions are only cue-level — a few words per timestamp
#     range — so words inside each cue are spread evenly across that cue's
#     duration). Output uses the SAME {"start","end","text"} shape Whisper
#     used to produce, so build_ass()/_chunk_words() need no changes."""
#     cues = []
#     blocks = re.split(r"\n\s*\n", vtt_text.strip())
#     time_re = re.compile(r"(\d{2}:)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}:)?(\d{2}):(\d{2})[.,](\d{3})")

#     def _t(h, m, s, ms):
#         h = int(h.rstrip(":")) if h else 0
#         return h * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

#     for block in blocks:
#         m = time_re.search(block)
#         if not m:
#             continue
#         start = _t(m.group(1), m.group(2), m.group(3), m.group(4))
#         end = _t(m.group(5), m.group(6), m.group(7), m.group(8))
#         text_lines = block[m.end():].strip().splitlines()
#         text = " ".join(l.strip() for l in text_lines if l.strip() and "-->" not in l)
#         text = re.sub(r"<[^>]+>", "", text).strip()  # strip VTT tags (<c>, karaoke timing, etc.)
#         if text and end > start:
#             cues.append((start, end, text))

#     words = []
#     for start, end, text in cues:
#         toks = text.split()
#         if not toks:
#             continue
#         span = max(end - start, 0.2)
#         step = span / len(toks)
#         for i, tok in enumerate(toks):
#             words.append({"start": round(start + i * step, 2), "end": round(start + (i + 1) * step, 2), "text": tok})
#     return words


# def fetch_ytdlp_transcript(url, log_cb=None):
#     """Pulls real captions (manual first, then auto-generated) + chapters
#     for a URL via yt-dlp WITHOUT downloading the video again — this is
#     what replaces Whisper entirely for any video that came from a link."""
#     def _log(msg):
#         if log_cb:
#             log_cb(msg)
#     if not _HAS_YTDLP:
#         return None
#     try:
#         opts = {
#             "quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True,
#             "writesubtitles": True, "writeautomaticsub": True,
#             "subtitleslangs": ["en", "en-US", "en-GB", "en-orig"], "subtitlesformat": "vtt",
#             "socket_timeout": 15,
#         }
#         attempts, cookies_file = _fetch_auth_attempts()
#         info = None
#         for mode in attempts:
#             try:
#                 o = _fetch_apply_auth(opts, mode, cookies_file)
#                 with yt_dlp.YoutubeDL(o) as ydl:
#                     info = ydl.extract_info(url, download=False)
#                 break
#             except Exception:
#                 continue
#         if not info:
#             _log("Could not resolve this link for captions.")
#             return None

#         chapters = info.get("chapters") or []
#         subs = info.get("subtitles") or {}
#         auto = info.get("automatic_captions") or {}
#         track, source_kind = None, None
#         for lang_map, kind in ((subs, "manual"), (auto, "auto-generated")):
#             for lang in ("en", "en-US", "en-GB", "en-orig"):
#                 if lang in lang_map:
#                     track, source_kind = lang_map[lang], kind
#                     break
#             if track:
#                 break
#         if not track:
#             _log("No English captions found on the source platform for this video — "
#                  "subtitles and the AI clip-finder are skipped for it (Whisper stays off).")
#             return {"words": [], "chapters": chapters, "source_kind": None}

#         vtt_url = next((f["url"] for f in track if f.get("ext") == "vtt"), track[0].get("url"))
#         with urllib.request.urlopen(vtt_url, timeout=15) as resp:
#             vtt_text = resp.read().decode("utf-8", errors="ignore")
#         words = _vtt_to_words(vtt_text)
#         _log(f"Pulled {len(words)} caption word(s) [{source_kind}] + {len(chapters)} chapter(s) "
#              f"straight from the source — no Whisper needed.")
#         return {"words": words, "chapters": chapters, "source_kind": source_kind}
#     except Exception as e:
#         _log(f"Caption fetch failed ({e}) — continuing without a transcript for this video.")
#         return None


# def get_transcript_words(source_path, log_cb=None):
#     """The ONE place analyze/render should call for a transcript. Looks up
#     the URL this source came from (if it was fetched at all — see
#     SOURCE_URL_MAP) and returns cached/fresh caption-derived words, or None
#     if there's no caption source. Never touches Whisper."""
#     key = str(source_path)
#     if key in TRANSCRIPT_CACHE:
#         return TRANSCRIPT_CACHE[key]
#     url = SOURCE_URL_MAP.get(key)
#     if not url:
#         TRANSCRIPT_CACHE[key] = None
#         return None
#     data = fetch_ytdlp_transcript(url, log_cb=log_cb)
#     TRANSCRIPT_CACHE[key] = data
#     return data


# # ══════════════════ AI providers — viral clip finder (transcript -> ranked clips) ══════════════════
# # Paste your own FREE API keys below, OR set them as environment variables
# # with the same name (env vars win if both are set). Tried top to bottom —
# # the first provider with a key that actually succeeds is used; a missing
# # key, a rate limit, a timeout, or a bad response all just fall through to
# # the next one automatically, so one exhausted free quota doesn't stop the
# # feature. All of these have a free tier as of writing:
# #
# #   Gemini      key_env=GEMINI_API_KEY      https://aistudio.google.com/apikey
# #               free tier: ~15 req/min, ~1500 req/day, resets daily (midnight Pacific)
# #   Groq        key_env=GROQ_API_KEY        https://console.groq.com/keys
# #               free tier: very fast inference, generous daily token limit, resets daily
# #   OpenRouter  key_env=OPENROUTER_API_KEY  https://openrouter.ai/keys
# #               free tier: use any model ending in ":free", resets daily
# #   Cerebras    key_env=CEREBRAS_API_KEY    https://cloud.cerebras.ai
# #               free tier: fast Llama models, resets daily
# #


# def _ai_key(provider):
#     return os.environ.get(provider["key_env"]) or provider.get("key") or ""


# def _http_post_json(url, headers, body, timeout=90):
#     req = urllib.request.Request(
#         url, data=json.dumps(body).encode("utf-8"),
#         headers={"Content-Type": "application/json", **headers}, method="POST",
#     )
#     with urllib.request.urlopen(req, timeout=timeout) as resp:
#         return json.loads(resp.read().decode("utf-8", errors="ignore"))


# def _call_gemini(provider, prompt):
#     key = _ai_key(provider)
#     if not key:
#         raise RuntimeError("no API key set")
#     url = f"https://generativelanguage.googleapis.com/v1beta/models/{provider['model']}:generateContent?key={key}"
#     body = {"contents": [{"parts": [{"text": prompt}]}],
#             "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4}}
#     data = _http_post_json(url, {}, body)
#     return data["candidates"][0]["content"]["parts"][0]["text"]


# def _call_openai_compatible(provider, prompt, base_url):
#     key = _ai_key(provider)
#     if not key:
#         raise RuntimeError("no API key set")
#     body = {
#         "model": provider["model"],
#         "messages": [
#             {"role": "system", "content": "You reply with ONLY valid JSON. No markdown fences, no commentary."},
#             {"role": "user", "content": prompt},
#         ],
#         "temperature": 0.4,
#         "response_format": {"type": "json_object"},
#     }
#     data = _http_post_json(f"{base_url}/chat/completions", {"Authorization": f"Bearer {key}"}, body)
#     return data["choices"][0]["message"]["content"]


# def _call_ai_provider(provider, prompt):
#     name = provider["name"]
#     if name == "Gemini":
#         return _call_gemini(provider, prompt)
#     if name == "Groq":
#         return _call_openai_compatible(provider, prompt, "https://api.groq.com/openai/v1")
#     if name == "OpenRouter":
#         return _call_openai_compatible(provider, prompt, "https://openrouter.ai/api/v1")
#     if name == "Cerebras":
#         return _call_openai_compatible(provider, prompt, "https://api.cerebras.ai/v1")
#     raise RuntimeError(f"Unknown provider {name}")


# def call_ai_with_fallback(prompt, log_cb=None):
#     def _log(msg):
#         if log_cb:
#             log_cb(msg)
#     last_err = None
#     for provider in AI_PROVIDERS:
#         if not _ai_key(provider):
#             _log(f"{provider['name']}: no API key set — skipping.")
#             continue
#         try:
#             _log(f"{provider['name']}: sending transcript for viral-clip analysis…")
#             text = _call_ai_provider(provider, prompt)
#             _log(f"{provider['name']}: responded.")
#             return text, provider["name"]
#         except urllib.error.HTTPError as e:
#             body = ""
#             try:
#                 body = e.read().decode("utf-8", errors="ignore")[:300]
#             except Exception:
#                 pass
#             last_err = f"{provider['name']} HTTP {e.code}: {body}"
#             _log(f"{provider['name']} failed (HTTP {e.code}) — trying next provider…")
#         except Exception as e:
#             last_err = f"{provider['name']}: {e}"
#             _log(f"{provider['name']} failed ({e}) — trying next provider…")
#     raise RuntimeError(f"All AI providers failed or have no key set. Last error: {last_err}")


# VIRAL_CLIP_PROMPT = """You are an expert short-form video editor who finds viral clips inside long-form video transcripts.

# Video title: {title}

# Transcript with timestamps (JSON array of objects with "start", "end", "text"):
# {transcript_json}

# Your task: Analyze the complete transcript with timestamps.
# Find ALL potential short-form viral clips.
# Do not limit to 20 clips. Return every strong candidate.
# A good clip must have:
# 1. Strong opening hook
# 2. Curiosity gap
# 3. Emotional trigger
# 4. Useful information
# 5. Complete story within the clip
# Avoid:
# - introductions
# - greetings
# - boring explanations
# - incomplete sentences

# For every clip return an object with exactly these fields:
# start_time, end_time, duration, hook_text, hook_type, score (0-100), reason, category, editing_suggestion

# Rank all clips from best to worst (highest score first).

# Reply with ONLY a JSON object of this exact shape: {{"clips": [ {{...}}, {{...}} ]}}
# No markdown fences, no commentary — JSON only."""


# def _words_to_caption_cues(words, max_gap=1.2, max_len=220):
#     """Groups word-level timestamps back into readable lines (start/end/text)
#     — what the AI prompt (and a human) actually wants to read, instead of
#     one JSON object per single word."""
#     cues, cur = [], None
#     for w in words:
#         if cur is None or w["start"] - cur["end"] > max_gap or len(cur["text"]) > max_len:
#             if cur:
#                 cues.append(cur)
#             cur = {"start": w["start"], "end": w["end"], "text": w["text"]}
#         else:
#             cur["end"] = w["end"]
#             cur["text"] += " " + w["text"]
#     if cur:
#         cues.append(cur)
#     return cues


# def find_viral_clips(words, video_title=None, log_cb=None):
#     """The AI hook/clip-finder. Feeds the (caption-derived) transcript to
#     the first working AI_PROVIDERS entry and returns a ranked list of clip
#     dicts. Returns [] (never raises) if there's no transcript or no AI
#     provider is configured/working, so callers just fall back to the
#     existing loudness/silence heuristics instead."""
#     def _log(msg):
#         if log_cb:
#             log_cb(msg)
#     if not words:
#         return []
#     cues = _words_to_caption_cues(words)
#     prompt = VIRAL_CLIP_PROMPT.format(
#         title=video_title or "Untitled",
#         transcript_json=json.dumps(
#             [{"start": round(c["start"], 2), "end": round(c["end"], 2), "text": c["text"]} for c in cues],
#             ensure_ascii=False,
#         ),
#     )
#     try:
#         raw, provider_name = call_ai_with_fallback(prompt, log_cb=log_cb)
#     except Exception as e:
#         _log(f"AI clip-finder unavailable ({e}) — falling back to loudness/silence-based hook detection only.")
#         return []

#     raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
#     try:
#         parsed = json.loads(raw)
#     except json.JSONDecodeError:
#         m = re.search(r"\{.*\}", raw, re.S)
#         if not m:
#             _log("AI response wasn't valid JSON — skipping AI clips for this run.")
#             return []
#         try:
#             parsed = json.loads(m.group(0))
#         except json.JSONDecodeError:
#             _log("AI response wasn't valid JSON even after cleanup — skipping AI clips for this run.")
#             return []

#     clips = parsed.get("clips") if isinstance(parsed, dict) else parsed
#     if not isinstance(clips, list):
#         return []
#     cleaned = []
#     for c in clips:
#         try:
#             start, end = float(c["start_time"]), float(c["end_time"])
#             if end <= start:
#                 continue
#             cleaned.append({
#                 "start_time": round(start, 2), "end_time": round(end, 2),
#                 "duration": round(end - start, 2),
#                 "hook_text": c.get("hook_text", ""), "hook_type": c.get("hook_type", ""),
#                 "score": max(0, min(100, int(c.get("score", 0)))),
#                 "reason": c.get("reason", ""), "category": c.get("category", ""),
#                 "editing_suggestion": c.get("editing_suggestion", ""),
#             })
#         except (KeyError, TypeError, ValueError):
#             continue
#     cleaned.sort(key=lambda c: c["score"], reverse=True)
#     _log(f"AI ({provider_name}) found {len(cleaned)} candidate clip(s).")
#     return cleaned
 
 
# def _chunk_words(words, max_words=4, max_span=1.6):
#     """Groups words into short on-screen caption chunks — the standard
#     'trending shorts' style of 2-5 words on screen at a time, not full
#     sentences."""
#     chunks = []
#     cur = []
#     for w in words:
#         if cur and (len(cur) >= max_words or (w["end"] - cur[0]["start"]) > max_span):
#             chunks.append(cur)
#             cur = []
#         cur.append(w)
#     if cur:
#         chunks.append(cur)
#     return chunks
 
 
# def _ass_time(t):
#     h = int(t // 3600)
#     m = int((t % 3600) // 60)
#     s = t % 60
#     return f"{h}:{m:02d}:{s:05.2f}"
 
 
# def build_ass(words, style_name, video_w, video_h):
#     """Builds a full .ass subtitle document with word-by-word highlight
#     animation for the chosen trending style, sized for the given output
#     resolution so text scales correctly across aspect ratios."""
#     style = CAPTION_STYLES.get(style_name, CAPTION_STYLES["bold_pop"])
#     header = f"""[Script Info]
# ScriptType: v4.00+
# PlayResX: {video_w}
# PlayResY: {video_h}
# ScaledBorderAndShadow: yes
 
# [V4+ Styles]
# Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
# {style["style"]}
 
# [Events]
# Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
# """
#     lines = []
#     for chunk in _chunk_words(words):
#         start, end = chunk[0]["start"], chunk[-1]["end"]
#         if style.get("karaoke_fill"):
#             # \k tags: whole line present, active word wipes to highlight color
#             text = "".join(
#                 r"{\k%d}%s " % (max(1, int(round((w["end"] - w["start"]) * 100))), w["text"])
#                 for w in chunk
#             )
#             lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
#         elif style.get("word_by_word"):
#             # emit one Dialogue event per word so only the current word is
#             # shown in the highlight color while the rest stay default —
#             # gives the punchy "pop" caption look trending on shorts.
#             for w in chunk:
#                 parts = []
#                 for w2 in chunk:
#                     if w2 is w:
#                         scale = r"\fscx115\fscy115" if style.get("pop_scale") else ""
#                         parts.append(r"{\c%s%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], scale, w2["text"]))
#                     elif _is_emphasis_word(w2["text"]):
#                         # numbers / money / power-words get punched up even
#                         # when they're not the "active" word of the moment
#                         parts.append(r"{\c%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], w2["text"]))
#                     else:
#                         parts.append(w2["text"])
#                 text = " ".join(parts)
#                 lines.append(f"Dialogue: 0,{_ass_time(w['start'])},{_ass_time(w['end'])},Default,,0,0,0,,{text}")
#         else:
#             parts = []
#             for w2 in chunk:
#                 if _is_emphasis_word(w2["text"]):
#                     parts.append(r"{\c%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], w2["text"]))
#                 else:
#                     parts.append(w2["text"])
#             text = " ".join(parts)
#             lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
#     return header + "\n".join(lines) + "\n"
 
 
# # ══════════════════════════════ hook-cut / segments ══════════════════════════════
 
# def _read_pcm_mono(video_path, sr=8000):
#     """Decodes a fast, low-res mono PCM stream via ffmpeg for lightweight
#     loudness analysis (used only to pick the auto hook window)."""
#     cmd = [FFMPEG_PATH, "-v", "error", "-i", str(video_path),
#            "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
#     proc = subprocess.run(cmd, capture_output=True, **_no_console_kwargs())
#     raw = proc.stdout
#     count = len(raw) // 2
#     samples = struct.unpack(f"<{count}h", raw[:count * 2])
#     return np.array(samples, dtype=np.float32) / 32768.0, sr
 
 
# def auto_hook_window(video_path, duration, hook_len=6.0):
#     """Finds the loudest `hook_len`-second window in the clip — a cheap but
#     effective proxy for 'most energetic / most likely to hook a viewer'
#     moment, used to auto-suggest where the exported clip should start."""
#     if not _HAS_NUMPY:
#         raise RuntimeError("numpy is not installed — auto hook-cut unavailable")
#     audio, sr = _read_pcm_mono(video_path)
#     if len(audio) == 0:
#         return {"start": 0.0, "end": min(hook_len, duration)}
#     win = int(hook_len * sr)
#     if win >= len(audio):
#         return {"start": 0.0, "end": duration}
#     energy = audio ** 2
#     # cumulative sum for O(1) windowed average lookups
#     cumsum = np.cumsum(np.insert(energy, 0, 0))
#     window_energy = cumsum[win:] - cumsum[:-win]
#     best_start_sample = int(np.argmax(window_energy))
#     start = best_start_sample / sr
#     return {"start": round(start, 2), "end": round(min(start + hook_len, duration), 2)}
 
 
# def detect_silences(video_path, noise_db=-30, min_silence=0.5):
#     """Runs ffmpeg's silencedetect and returns the list of NON-silent
#     {start,end} ranges — i.e. the segments worth keeping. This powers
#     automatic 'jump-cut' editing (removing dead air/pauses), the other
#     trending short-form edit style besides the loudest-window hook-cut."""
#     cmd = [FFMPEG_PATH, "-v", "info", "-i", str(video_path),
#            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
#     result = _run(cmd)
#     log = result.stderr or ""
#     starts, ends = [], []
#     for line in log.splitlines():
#         if "silence_start" in line:
#             starts.append(float(line.split("silence_start:")[1].strip().split(" ")[0]))
#         elif "silence_end" in line:
#             ends.append(float(line.split("silence_end:")[1].strip().split(" ")[0].split("|")[0]))
#     info = _probe(video_path)
#     duration = info["duration"]
#     silences = list(zip(starts, ends[: len(starts)]))
#     keep = []
#     cursor = 0.0
#     for s, e in silences:
#         if s > cursor:
#             keep.append({"start": round(cursor, 2), "end": round(s, 2)})
#         cursor = max(cursor, e)
#     if cursor < duration:
#         keep.append({"start": round(cursor, 2), "end": round(duration, 2)})
#     return [k for k in keep if k["end"] - k["start"] > 0.15]
 
 
# def _build_concat_file(video_path, segments, tmp_dir):
#     """Writes an ffmpeg concat-demuxer list after trimming each segment,
#     so multiple {start,end} ranges can be stitched in a custom, manually
#     chosen order (rearrange / hook-cut)."""
#     parts = []
#     for i, seg in enumerate(segments):
#         out = tmp_dir / f"seg_{i}.mp4"
#         cmd = [
#             FFMPEG_PATH, "-y", "-v", "error",
#             "-ss", str(seg["start"]), "-to", str(seg["end"]),
#             "-i", str(video_path),
#             "-c", "copy", "-avoid_negative_ts", "make_zero",
#             str(out),
#         ]
#         _run(cmd)
#         parts.append(out)
#     list_path = tmp_dir / "concat_list.txt"
#     with open(list_path, "w", encoding="utf-8") as f:
#         for p in parts:
#             f.write(f"file '{p.as_posix()}'\n")
#     return list_path
 
 
# def _measure_loudnorm(path):
#     """First pass of two-pass EBU R128 loudness normalization: measures the
#     input's actual loudness/true-peak/range so the second (real encode)
#     pass can normalize with `linear=true` against real measured values
#     instead of loudnorm's single-pass dynamic estimate — noticeably more
#     accurate and avoids the pumping/gain-riding artifacts single-pass mode
#     can introduce on speech-heavy short-form video."""
#     cmd = [FFMPEG_PATH, "-v", "info", "-i", str(path),
#            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
#            "-f", "null", "-"]
#     result = _run(cmd)
#     log = result.stderr or ""
#     try:
#         start = log.rindex("{")
#         end = log.rindex("}") + 1
#         return json.loads(log[start:end])
#     except (ValueError, json.JSONDecodeError):
#         return None  # falls back to single-pass below
 
 
# # ══════════════════════════════ render pipeline ══════════════════════════════
 
# def _render_one_ratio(job, source_path, ratio_key, track_points, src_w, src_h,
#                        words, tmp_dir, loud_measured=None):
#     cfg = job["config"]
#     ratio = RATIO_PRESETS[ratio_key]
#     is_auto = (ratio_key == "auto")
#     # "auto" keeps the SOURCE's own dimensions — no target size is imposed.
#     target_w, target_h = (src_w, src_h) if is_auto else (ratio["w"], ratio["h"])
#     quality = QUALITY_PRESETS.get(cfg.get("quality", "high"), QUALITY_PRESETS["high"])
 
#     fill_mode = cfg.get("fill_mode", "crop")  # "crop" (default) or "blur_pad"
#     # "auto" never crops or letterboxes — it IS the full original frame —
#     # so blur_pad/crop fill-mode choices simply don't apply to it.
#     if is_auto:
#         fill_mode = "none"
 
#     # ── subtitles: build the .ass first so both fill-mode branches can use it ──
#     ass_path = None
#     if cfg["features"].get("subtitles") and words:
#         ass_content = build_ass(words, cfg.get("caption_style", "bold_pop"), target_w, target_h)
#         ass_path = tmp_dir / f"subs_{ratio_key.replace(':', 'x')}.ass"
#         ass_path.write_text(ass_content, encoding="utf-8")
#         ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
 
#     if is_auto and not ass_path and not cfg.get("manual_crop_keyframes"):
#         # Nothing needs to touch the picture at all: fastest AND highest
#         # possible quality path is a pure stream-copy remux (no re-encode
#         # -> byte-for-byte original video quality; only trimmed/normalized
#         # audio may differ). This is what "zero quality loss" means here.
#         out_name = f"{job['job_id']}_{ratio_key.replace(':', 'x')}.mp4"
#         out_path = EDIT_DIR / out_name
#         if cfg.get("normalize_audio", True):
#             measured = loud_measured
#             audio_filter = (
#                 "loudnorm=I=-14:TP=-1.5:LRA=11:"
#                 f"measured_I={measured.get('input_i', -14)}:"
#                 f"measured_TP={measured.get('input_tp', -1.5)}:"
#                 f"measured_LRA={measured.get('input_lra', 11)}:"
#                 f"measured_thresh={measured.get('input_thresh', -24)}:"
#                 "linear=true:print_format=summary"
#             ) if measured else "loudnorm=I=-14:TP=-1.5:LRA=11"
#             cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path),
#                    "-c:v", "copy", "-af", audio_filter, "-c:a", "aac", "-b:a", "192k",
#                    "-movflags", "+faststart", str(out_path)]
#         else:
#             cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path),
#                    "-c", "copy", "-movflags", "+faststart", str(out_path)]
#         result = _run(cmd)
#         if result.returncode != 0:
#             raise RuntimeError(f"ffmpeg failed for {ratio_key}: {result.stderr[-800:]}")
#         return out_name
 
#     if fill_mode == "blur_pad":
#         # "Advanced" reframe mode: instead of cropping content away, fit the
#         # WHOLE frame inside the target canvas and fill the leftover bars
#         # with a blurred, zoomed copy of the same frame — the look used by
#         # most professional auto-reframe tools when a hard crop would cut
#         # off important content. No face tracking needed for this branch,
#         # since nothing is being cropped out.
#         fc = (
#             f"[0:v]split=2[bg][fg];"
#             f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
#             f"crop={target_w}:{target_h},gblur=sigma=25,eq=brightness=-0.05[bg2];"
#             f"[fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg2];"
#             f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2:format=auto[base]"
#         )
#         if ass_path:
#             fc += f",[base]ass='{ass_escaped}'[vout]"
#             map_v = "[vout]"
#         else:
#             fc += "[vout]"
#             map_v = "[vout]"
#         vf_args = ["-filter_complex", fc, "-map", map_v, "-map", "0:a?"]
#     elif is_auto:
#         # Captions and/or a manual crop keyframe are present, so a re-encode
#         # can't be avoided — but the frame itself is never cropped or
#         # rescaled; it stays at the source's own resolution. Quality is set
#         # one notch above "high" (crf 16, effectively visually lossless)
#         # specifically for this path so adding captions never visibly
#         # degrades the original picture.
#         filters = ["setsar=1"]
#         if ass_path:
#             filters.append(f"ass='{ass_escaped}'")
#         vf_args = ["-vf", ",".join(filters)]
#         quality = {"crf": 16, "preset": quality["preset"]}
#     else:
#         filters = []
#         # ── crop (face-tracked or manual) ──
#         if cfg["features"].get("face_tracking") or cfg.get("manual_crop_keyframes"):
#             crop_w, crop_h = _crop_size_for_ratio(src_w, src_h, target_w, target_h)
#             if cfg.get("manual_crop_keyframes"):
#                 kfs = sorted(cfg["manual_crop_keyframes"], key=lambda k: k["t"])
#             else:
#                 kfs = _build_crop_keyframes(track_points, src_w, src_h, crop_w, crop_h)
#             x_expr = _expr_chain(kfs, "x")
#             y_expr = _expr_chain(kfs, "y")
#             filters.append(f"crop={crop_w}:{crop_h}:'{x_expr}':'{y_expr}'")
#         else:
#             # no face tracking requested -> simple centered crop to the target ratio
#             crop_w, crop_h = _crop_size_for_ratio(src_w, src_h, target_w, target_h)
#             filters.append(f"crop={crop_w}:{crop_h}:(in_w-out_w)/2:(in_h-out_h)/2")
 
#         filters.append(f"scale={target_w}:{target_h}:flags=lanczos")
#         filters.append("setsar=1")
#         if ass_path:
#             filters.append(f"ass='{ass_escaped}'")
#         vf_args = ["-vf", ",".join(filters)]
 
#     out_name = f"{job['job_id']}_{ratio_key.replace(':', 'x')}.mp4"
#     out_path = EDIT_DIR / out_name
 
#     # Loudness normalization (EBU R128, -14 LUFS is the standard target for
#     # social platforms) so exported audio doesn't sound quiet/inconsistent
#     # next to other content in-feed. On "high" quality we do a real
#     # measure-then-normalize TWO-PASS pass for accuracy; cheaper qualities
#     # use fast single-pass so exports stay quick.
#     audio_filters = []
#     if cfg.get("normalize_audio", True):
#         measured = loud_measured
#         if measured:
#             audio_filters = [
#                 "loudnorm=I=-14:TP=-1.5:LRA=11:"
#                 f"measured_I={measured.get('input_i', -14)}:"
#                 f"measured_TP={measured.get('input_tp', -1.5)}:"
#                 f"measured_LRA={measured.get('input_lra', 11)}:"
#                 f"measured_thresh={measured.get('input_thresh', -24)}:"
#                 "linear=true:print_format=summary"
#             ]
#         else:
#             audio_filters = ["loudnorm=I=-14:TP=-1.5:LRA=11"]
 
#     cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path), *vf_args]
#     if audio_filters:
#         cmd += ["-af", ",".join(audio_filters)]
#     cmd += [
#         "-c:v", "libx264", "-crf", str(quality["crf"]), "-preset", quality["preset"],
#         "-c:a", "aac", "-b:a", "192k",
#         "-movflags", "+faststart",
#         str(out_path),
#     ]
#     result = _run(cmd)
#     if result.returncode != 0:
#         raise RuntimeError(f"ffmpeg failed for {ratio_key}: {result.stderr[-800:]}")
#     return out_name
 
 
# def _run_render_job(job_id):
#     job = EDIT_JOBS[job_id]
#     cfg = job["config"]
#     job.setdefault("logs", [])
#     log = lambda msg, **kw: _job_log(job, msg, **kw)
#     tmp_dir = EDIT_DIR / f"tmp_{job_id}"
#     tmp_dir.mkdir(exist_ok=True)
 
#     try:
#         original_source_path = Path(cfg["source_path"])
#         source_path = original_source_path
#         if not source_path.exists():
#             raise RuntimeError("Source video not found")
#         log(f"Starting render of {len(cfg.get('export_ratios') or ['9:16'])} ratio(s)...", stage="segments", percent=2)
 
#         # ── 1. hook-cut / manual rearrange: trim+stitch segments first ──
#         segments = cfg.get("segments")
#         want_jump = cfg["features"].get("jump_cut")
#         want_hook = cfg["features"].get("hook_cut")
#         if not segments and (want_jump or want_hook):
#             log("Auto-detecting hook/silence cuts...", percent=5)
#             info = _probe(source_path)
#             jump_segments = detect_silences(source_path) if want_jump else None
#             if want_hook:
#                 hook = auto_hook_window(source_path, info["duration"], cfg.get("hook_len", 6.0))
#                 # Cold-open on the loudest window, then play the rest of the
#                 # clip — dead-air-trimmed if jump_cut is also on, otherwise
#                 # the full original timeline. Combining both no longer means
#                 # jump_cut silently wins.
#                 rest = jump_segments if jump_segments else [{"start": 0.0, "end": info["duration"]}]
#                 segments = [hook] + rest
#             else:
#                 segments = jump_segments
#         if segments:
#             log(f"Trimming/stitching {len(segments)} segment(s)...", percent=8)
#             list_path = _build_concat_file(source_path, segments, tmp_dir)
#             stitched = tmp_dir / "stitched.mp4"
#             _run([FFMPEG_PATH, "-y", "-v", "error", "-f", "concat", "-safe", "0",
#                   "-i", str(list_path), "-c", "copy", str(stitched)])
#             source_path = stitched
#         job["percent"] = 10
 
#         info = _probe(source_path)
#         src_w, src_h = info["width"], info["height"]
 
#         # ── 2. face tracking (once, reused for every export ratio) ──
#         job["stage"] = "face_tracking"
#         track_points = []
#         if cfg["features"].get("face_tracking"):
#             log("Face-tracking the video...", percent=10)
#             def cb(pct):
#                 job["percent"] = 10 + int(pct * 0.3)
#             track_points, src_w, src_h, advanced = analyze_face_track(
#                 source_path, sample_fps=cfg.get("track_fps", 2.0), progress_cb=cb
#             )
#             job["face_tracking_mode"] = "advanced (mediapipe)" if advanced else "standard (haar cascade)"
#             log(f"Face tracking done ({job['face_tracking_mode']}).", percent=40)
#         job["percent"] = 40
 
#         # ── 3. subtitles (once, reused for every export ratio) — from real
#         # platform captions (yt-dlp), never Whisper. Captions are only used
#         # when NO trim/stitch happened to source_path, since caption
#         # timestamps are relative to the ORIGINAL full video and would be
#         # misaligned against a stitched/trimmed timeline. ──
#         job["stage"] = "subtitles"
#         words = []
#         if cfg["features"].get("subtitles"):
#             if source_path != original_source_path:
#                 log("Skipping captions for this render: hook-cut/jump-cut trimmed the "
#                     "timeline, and platform caption timestamps would no longer line up.", percent=55)
#             else:
#                 log("Fetching captions for subtitles...", percent=45)
#                 transcript = get_transcript_words(original_source_path, log_cb=log)
#                 words = (transcript or {}).get("words") or []
#                 if words:
#                     log(f"Using {len(words)} caption word(s) for subtitles.", percent=55)
#                 else:
#                     log("No captions available for this video — rendering without subtitles.", percent=55)
#         job["percent"] = 60
 
#         # ── 4. render each requested aspect ratio as one final file ──
#         job["stage"] = "render"
#         ratios = cfg.get("export_ratios") or ["9:16"]
#         job["ratios"] = {}
#         n = len(ratios)
 
#         # measured once (not per-ratio, source audio is identical across
#         # ratios) and only for "high" quality, where the extra ffmpeg pass
#         # is worth the accuracy; cheaper qualities use fast single-pass.
#         loud_measured = None
#         if cfg.get("normalize_audio", True) and cfg.get("quality", "high") == "high":
#             log("Measuring loudness for normalization...", percent=58)
#             loud_measured = _measure_loudnorm(source_path)
 
#         for i, ratio_key in enumerate(ratios):
#             if ratio_key not in RATIO_PRESETS:
#                 continue
#             log(f"Rendering {ratio_key} ({i+1}/{n})...", percent=60 + int(i * 40 / max(1, n)))
#             out_name = _render_one_ratio(job, source_path, ratio_key, track_points,
#                                           src_w, src_h, words, tmp_dir, loud_measured)
#             job["ratios"][ratio_key] = out_name
#             job["percent"] = 60 + int((i + 1) * 40 / max(1, n))
#             log(f"{ratio_key} done.")
 
#         job["status"] = "done"
#         job["stage"] = "done"
#         job["percent"] = 100
#         log("Render complete.")
#     except Exception as e:
#         job["status"] = "error"
#         job["error"] = str(e)
#         log(f"Render failed: {e}", stage="error")

#     finally:
#         # scratch files (trimmed segments, stitched source, .ass files) —
#         # keep only the final per-ratio outputs in EDIT_DIR
#         try:
#             for f in tmp_dir.glob("*"):
#                 f.unlink(missing_ok=True)
#             tmp_dir.rmdir()
#         except OSError:
#             pass
 
 
# # ══════════════════════════════ routes ══════════════════════════════
 
# def _run_analyze_job(job_id):
#     job = ANALYZE_JOBS[job_id]
#     cfg = job["config"]
#     source_path = Path(cfg["source_path"])
#     log = lambda msg, **kw: _job_log(job, msg, **kw)

#     try:
#         log("Reading video info (ffprobe)…", percent=2, stage="probe")
#         info = _probe(source_path)
#         result = {"probe": info, "features_available": feature_status()}
#         log(f"Source is {info.get('width')}x{info.get('height')}, "
#             f"{info.get('duration', 0):.1f}s.", percent=8)

#         if cfg.get("face_tracking", True) and _HAS_CV2:
#             log("Face-tracking the video (this scans the whole clip)…", stage="face_tracking", percent=12)
#             try:
#                 points, w, h, advanced = analyze_face_track(source_path, sample_fps=cfg.get("track_fps", 2.0))
#                 result["face_track"] = points
#                 result["face_tracking_mode"] = "advanced" if advanced else "standard"
#                 result["source_size"] = {"w": w, "h": h}
#                 log(f"Face tracking done — {len(points)} sample point(s), "
#                     f"{'advanced (mediapipe)' if advanced else 'standard (haar cascade)'} mode.", percent=40)
#             except Exception as e:
#                 result["face_track"] = []
#                 result["face_tracking_error"] = str(e)
#                 log(f"Face tracking failed: {e} — continuing without it.", percent=40)
#         else:
#             result["face_track"] = []

#         words, ai_clips = [], []
#         if cfg.get("subtitles", True):
#             log("Looking for real captions on the source platform (yt-dlp)…", stage="captions", percent=45)
#             transcript = get_transcript_words(source_path, log_cb=log)
#             if transcript and transcript.get("words"):
#                 words = transcript["words"]
#                 result["chapters"] = transcript.get("chapters") or []
#                 result["caption_source"] = transcript.get("source_kind")
#                 log(f"{len(words)} caption word(s) ready — Whisper was not used.", percent=55)
#             elif transcript is not None:
#                 result["subtitle_note"] = "No English captions on the source for this video."
#                 log("No captions available for this video (Whisper is disabled by design) — "
#                     "subtitles/AI clip-finder skipped, face-track + silence-based editing still work.", percent=55)
#             else:
#                 result["subtitle_note"] = "This video has no linked URL (PC upload/library file) — no caption source."
#                 log("No source URL for this file, so no captions to pull (Whisper is disabled by design).", percent=55)
#             result["words"] = words

#             if words:
#                 log("Sending transcript to AI for viral-clip analysis…", stage="ai_clips", percent=60)
#                 title = SOURCE_TITLE_MAP.get(str(source_path))
#                 ai_clips = find_viral_clips(words, video_title=title, log_cb=log)
#                 result["ai_clips"] = ai_clips
#                 log(f"AI clip-finder returned {len(ai_clips)} ranked clip(s).", percent=78)
#             else:
#                 result["ai_clips"] = []

#         if cfg.get("hook_cut", False) and _HAS_NUMPY:
#             log("Scanning loudness for a cold-open hook window…", stage="hook_cut", percent=85)
#             try:
#                 result["suggested_hook"] = auto_hook_window(source_path, info["duration"], cfg.get("hook_len", 6.0))
#             except Exception as e:
#                 result["hook_cut_error"] = str(e)
#                 log(f"Hook-cut detection failed: {e}")

#         if cfg.get("jump_cut", False):
#             log("Detecting dead-air / silence to auto-trim…", stage="jump_cut", percent=92)
#             try:
#                 result["suggested_segments"] = detect_silences(source_path)
#             except Exception as e:
#                 result["jump_cut_error"] = str(e)
#                 log(f"Silence detection failed: {e}")

#         job["result"] = result
#         job["status"] = "done"
#         log("Analysis complete.", percent=100, stage="done")
#     except Exception as e:
#         job["status"] = "error"
#         job["error"] = str(e)
#         log(f"Analysis failed: {e}", stage="error")


# @editor_bp.route("/api/editor/analyze/start", methods=["POST"])
# def api_editor_analyze_start():
#     """Kicks off analysis (face-track + real captions + AI viral-clip
#     finder + hook/silence detection) as a background job instead of
#     blocking the request, so the frontend can show a live progress bar and
#     a scrolling log console instead of the page just sitting there with no
#     feedback. Poll /api/editor/analyze/progress/<job_id> for updates."""
#     data = request.json or {}
#     source_path = Path(data.get("source_path", ""))
#     if not source_path.exists():
#         return jsonify({"error": "source_path not found"}), 400
#     job_id = uuid.uuid4().hex[:10]
#     ANALYZE_JOBS[job_id] = {
#         "status": "running", "stage": "queued", "percent": 0, "error": None,
#         "logs": [], "result": None,
#         "config": {
#             "source_path": str(source_path),
#             "face_tracking": bool(data.get("face_tracking", True)),
#             "subtitles": bool(data.get("subtitles", True)),
#             "hook_cut": bool(data.get("hook_cut", False)),
#             "jump_cut": bool(data.get("jump_cut", False)),
#             "track_fps": data.get("track_fps", 2.0),
#             "hook_len": data.get("hook_len", 6.0),
#         },
#     }
#     threading.Thread(target=_run_analyze_job, args=(job_id,), daemon=True).start()
#     return jsonify({"job_id": job_id})


# @editor_bp.route("/api/editor/analyze/progress/<job_id>")
# def api_editor_analyze_progress(job_id):
#     job = ANALYZE_JOBS.get(job_id)
#     if not job:
#         return jsonify({"error": "Unknown analyze job"}), 404
#     return jsonify({k: v for k, v in job.items() if k != "config"})
 
 
# @editor_bp.route("/api/editor/upload", methods=["POST"])
# def api_editor_upload():
#     """Lets the frontend upload a local file directly instead of only
#     pointing at a path already on the server (e.g. a downloader.py output)."""
#     f = request.files.get("file")
#     if not f or not f.filename:
#         return jsonify({"error": "No file uploaded"}), 400
#     safe_name = f"{uuid.uuid4().hex[:10]}_{Path(f.filename).name}"
#     dest = EDIT_DIR / "uploads"
#     dest.mkdir(exist_ok=True)
#     path = dest / safe_name
#     f.save(path)
#     return jsonify({"source_path": str(path)})
 
 
# # ══════════════════════════════ fetch-from-link (built-in yt-dlp) ══════════════════════════════
# # A third source alongside "upload from PC" and "pick from downloader" —
# # paste a link right here in the editor. Deliberately skips the full
# # format-picker: it always resolves ONE single, best-available video+audio
# # mp4 (never a video-only stream that would need a silent-audio placeholder,
# # never separate files) and hands that straight to analyze/render.
# #
# # Speed trick mirrors downloader.py: remembers whichever auth mode (cookies
# # file / browser cookies / plain / bypass) last worked and tries that first,
# # so repeat fetches stay fast. If downloader.py is also registered in this
# # process, its already-resolved mode is reused immediately instead of
# # re-discovering it from scratch.
 
# FETCH_JOBS = {}  # fetch_id -> {status, stage, percent, error, source_path, url}
# _FETCH_RESOLVED_MODE = {"mode": None}
# _FETCH_COOKIE_BROWSERS = ["chrome", "edge", "firefox", "brave"]
 
 
# def _fetch_no_console_kwargs():
#     if os.name == "nt":
#         return {"creationflags": subprocess.CREATE_NO_WINDOW}
#     return {}
 
 
# def _fetch_auth_attempts():
#     attempts = []
#     # reuse downloader.py's already-known-good mode first, if it's loaded
#     try:
#         import downloader
#         if downloader._RESOLVED_MODE.get("mode"):
#             attempts.append(downloader._RESOLVED_MODE["mode"])
#     except Exception:
#         pass
#     if _FETCH_RESOLVED_MODE["mode"]:
#         attempts.append(_FETCH_RESOLVED_MODE["mode"])
#     cookies_file = SRC_DIR.parent / "cookies.txt" if SRC_DIR else None
#     rest = []
#     if cookies_file and cookies_file.exists():
#         rest.append("cookies_file")
#     rest.extend(_FETCH_COOKIE_BROWSERS)
#     rest.append("default")
#     rest.append("bypass")
#     for m in rest:
#         if m not in attempts:
#             attempts.append(m)
#     return attempts, cookies_file
 
 
# def _fetch_apply_auth(opts, mode, cookies_file):
#     opts = dict(opts)
#     if mode == "cookies_file" and cookies_file:
#         opts["cookiefile"] = str(cookies_file)
#     elif mode in _FETCH_COOKIE_BROWSERS:
#         opts["cookiesfrombrowser"] = (mode,)
#     elif mode == "bypass":
#         opts["extractor_args"] = {"youtube": {"player_client": ["ios", "android", "mweb"]}}
#     return opts
 
 
# def _fetch_extract(url, extra_opts=None, download=False):
#     base_opts = {
#         "quiet": True, "no_warnings": True, "noplaylist": True,
#         "socket_timeout": 15, "retries": 2, "extractor_retries": 1, "geo_bypass": True,
#     }
#     if FFMPEG_PATH:
#         base_opts["ffmpeg_location"] = str(FFMPEG_PATH)
#     if extra_opts:
#         base_opts.update(extra_opts)
 
#     attempts, cookies_file = _fetch_auth_attempts()
#     last_err = None
#     for mode in attempts:
#         opts = _fetch_apply_auth(base_opts, mode, cookies_file)
#         try:
#             with yt_dlp.YoutubeDL(opts) as ydl:
#                 info = ydl.extract_info(url, download=download)
#             _FETCH_RESOLVED_MODE["mode"] = mode
#             return info, ydl if download else None, None
#         except Exception as e:
#             last_err = e
#             continue
#     return None, None, last_err
 
 
# def _fetch_fmt_duration(sec):
#     if sec is None:
#         return None
#     sec = int(sec)
#     h, rem = divmod(sec, 3600)
#     m, s = divmod(rem, 60)
#     return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
 
 
# def _fetch_fmt_int(n):
#     if n is None:
#         return None
#     try:
#         return f"{int(n):,}"
#     except (TypeError, ValueError):
#         return str(n)
 
 
# def _fetch_fmt_upload_date(d):
#     if not d:
#         return None
#     try:
#         from datetime import datetime as _dt
#         return _dt.strptime(str(d), "%Y%m%d").strftime("%d %b %Y")
#     except ValueError:
#         return str(d)
 
 
# @editor_bp.route("/api/editor/fetch/info", methods=["POST"])
# def api_fetch_info():
#     """Resolves a link (YouTube/Instagram/TikTok/X/Facebook/etc.) and hands
#     back the FULL set of metadata yt-dlp/the source platform already knows
#     about it — title, uploader, view/like counts, upload date, best
#     available resolution — so the UI never has to re-derive any of this by
#     hand, and so a video that started life as a fetched link can show that
#     rich context in the analysis panel too (a plain PC upload naturally
#     won't have this — it only gets the ffprobe technical details)."""
#     if not _HAS_YTDLP:
#         return jsonify({"error": "yt-dlp is not installed on the server — pip install yt-dlp"}), 400
#     data = request.json or {}
#     url = (data.get("url") or "").strip()
#     if not url:
#         return jsonify({"error": "Paste a video URL first"}), 400
#     info, _, err = _fetch_extract(url, extra_opts={"format": "bestvideo+bestaudio/best"})
#     if info is None:
#         return jsonify({"error": f"Could not resolve that link: {err}"}), 400
 
#     best_h = best_w = None
#     for f in info.get("formats", []) or []:
#         if f.get("vcodec") not in (None, "none") and f.get("height"):
#             if not best_h or f["height"] > best_h:
#                 best_h, best_w = f["height"], f.get("width")
 
#     return jsonify({
#         "title": info.get("title"),
#         "uploader": info.get("uploader") or info.get("channel"),
#         "duration": info.get("duration"),
#         "duration_str": _fetch_fmt_duration(info.get("duration")),
#         "thumbnail": info.get("thumbnail"),
#         "view_count": info.get("view_count"),
#         "view_count_str": _fetch_fmt_int(info.get("view_count")),
#         "like_count": info.get("like_count"),
#         "like_count_str": _fetch_fmt_int(info.get("like_count")),
#         "upload_date": info.get("upload_date"),
#         "upload_date_str": _fetch_fmt_upload_date(info.get("upload_date")),
#         "resolution": f"{best_w}x{best_h}" if best_w and best_h else None,
#         "extractor": info.get("extractor_key"),
#         "webpage_url": info.get("webpage_url"),
#     })
 
 
# def _run_fetch_job(fetch_id, url):
#     job = FETCH_JOBS[fetch_id]
#     job["stage"] = "connect"
 
#     def hook(d):
#         if d.get("status") == "downloading":
#             total = d.get("total_bytes") or d.get("total_bytes_estimate")
#             downloaded = d.get("downloaded_bytes") or 0
#             speed = d.get("speed")
#             job.update({
#                 "status": "downloading", "stage": "download",
#                 "percent": round(downloaded * 100 / total, 1) if total else None,
#                 "speed_str": (_fmt_size(speed) + "/s") if speed else None,
#             })
#         elif d.get("status") == "finished":
#             job["status"] = "processing"
#             job["stage"] = "merge"
 
#     # Saved into SRC_DIR (the SAME "downloads" folder downloader.py uses) so
#     # a link fetched here shows up in /api/editor/sources too, and nothing
#     # about the analyze/render pipeline needs to know it came from here
#     # instead of the Downloader tab or a PC upload.
#     extra_opts = {
#         "format": "bestvideo+bestaudio/best",
#         "outtmpl": str(SRC_DIR / f"editorfetch_{fetch_id}_%(title).60s.%(ext)s"),
#         "progress_hooks": [hook],
#         "merge_output_format": "mp4",   # always ONE muxed video+audio file
#     }
 
#     orig_popen = subprocess.Popen
#     def quiet_popen(*args, **kwargs):
#         kwargs.update(_fetch_no_console_kwargs())
#         return orig_popen(*args, **kwargs)
#     subprocess.Popen = quiet_popen
#     try:
#         info, ydl, err = _fetch_extract(url, extra_opts=extra_opts, download=True)
#     finally:
#         subprocess.Popen = orig_popen
 
#     if info is None:
#         job["status"] = "error"
#         job["error"] = f"Fetch failed: {err}"
#         return
#     try:
#         fname = ydl.prepare_filename(info)
#         p = Path(fname)
#         if not p.exists():
#             p2 = p.with_suffix(".mp4")
#             if p2.exists():
#                 p = p2
#         job["source_path"] = str(p.resolve())
#         SOURCE_URL_MAP[job["source_path"]] = url
#         SOURCE_TITLE_MAP[job["source_path"]] = info.get("title")
#         job["status"] = "done"
#         job["stage"] = "done"
#         job["percent"] = 100
#     except Exception as e:
#         job["status"] = "error"
#         job["error"] = f"Could not finalize file: {e}"
 
 
# @editor_bp.route("/api/editor/fetch/start", methods=["POST"])
# def api_fetch_start():
#     if not _HAS_YTDLP:
#         return jsonify({"error": "yt-dlp is not installed on the server — pip install yt-dlp"}), 400
#     data = request.json or {}
#     url = (data.get("url") or "").strip()
#     if not url:
#         return jsonify({"error": "No URL given"}), 400
#     fetch_id = uuid.uuid4().hex[:10]
#     FETCH_JOBS[fetch_id] = {
#         "status": "starting", "stage": "connect", "percent": 0,
#         "speed_str": None, "error": None, "url": url, "source_path": None,
#     }
#     threading.Thread(target=_run_fetch_job, args=(fetch_id, url), daemon=True).start()
#     return jsonify({"fetch_id": fetch_id})
 
 
# @editor_bp.route("/api/editor/fetch/progress/<fetch_id>")
# def api_fetch_progress(fetch_id):
#     job = FETCH_JOBS.get(fetch_id)
#     if not job:
#         return jsonify({"error": "Unknown fetch job"}), 404
#     return jsonify(job)
 
 
# @editor_bp.route("/api/editor/render", methods=["POST"])
# def api_editor_render():
#     data = request.json or {}
#     source_path = data.get("source_path", "")
#     if not source_path or not Path(source_path).exists():
#         return jsonify({"error": "source_path not found"}), 400
 
#     export_ratios = data.get("export_ratios") or ["9:16"]
#     bad = [r for r in export_ratios if r not in RATIO_PRESETS]
#     if bad:
#         return jsonify({"error": f"Unknown ratio(s): {bad}. Valid: {list(RATIO_PRESETS)}"}), 400
 
#     features = {
#         "face_tracking": bool(data.get("face_tracking", True)),
#         "subtitles": bool(data.get("subtitles", True)),
#         "hook_cut": bool(data.get("hook_cut", False)),
#         "jump_cut": bool(data.get("jump_cut", False)),
#     }
#     if features["face_tracking"] and not _HAS_CV2:
#         return jsonify({"error": "Face tracking requested but opencv-python is not installed"}), 400
#     # Subtitles are no longer a hard requirement on anything being installed —
#     # they simply depend on the source having real platform captions
#     # available (see get_transcript_words); if not, rendering proceeds
#     # without subtitles instead of failing the whole request.
#     if features["hook_cut"] and not data.get("segments") and not _HAS_NUMPY:
#         return jsonify({"error": "Auto hook-cut requested but numpy is not installed"}), 400
 
#     job_id = uuid.uuid4().hex[:10]
#     EDIT_JOBS[job_id] = {
#         "job_id": job_id, "status": "starting", "stage": "queued", "percent": 0,
#         "error": None, "ratios": {},
#         "config": {
#             "source_path": source_path,
#             "export_ratios": export_ratios,
#             "features": features,
#             "caption_style": data.get("caption_style", "bold_pop"),
#             "quality": data.get("quality", "high"),
#             "track_fps": data.get("track_fps", 2.0),
#             "whisper_model": data.get("whisper_model", "small"),
#             "language": data.get("language"),
#             "hook_len": data.get("hook_len", 6.0),
#             "manual_crop_keyframes": data.get("manual_crop_keyframes"),  # [{t,x,y}]
#             "segments": data.get("segments"),  # [{start,end}] manual rearrange/trim order
#             "fill_mode": data.get("fill_mode", "crop"),  # "crop" or "blur_pad"
#             "normalize_audio": bool(data.get("normalize_audio", True)),
#         },
#     }
#     threading.Thread(target=_run_render_job, args=(job_id,), daemon=True).start()
#     return jsonify({"job_id": job_id})
 
 
# @editor_bp.route("/api/editor/progress/<job_id>")
# def api_editor_progress(job_id):
#     job = EDIT_JOBS.get(job_id)
#     if not job:
#         return jsonify({"error": "Unknown edit job"}), 404
#     return jsonify({k: v for k, v in job.items() if k != "config"} | {
#         "features": job["config"]["features"], "export_ratios": job["config"]["export_ratios"]
#     })
 
 
# @editor_bp.route("/api/editor/file/<job_id>/<ratio>")
# def api_editor_file(job_id, ratio):
#     job = EDIT_JOBS.get(job_id)
#     if not job or job.get("status") != "done":
#         return "Not ready", 404
#     fname = job["ratios"].get(ratio)
#     if not fname:
#         return "Ratio not found for this job", 404
#     p = EDIT_DIR / fname
#     if not p.exists():
#         return "Not found", 404
#     return send_file(p, as_attachment=True, download_name=fname)
 
 
# # ── manual crop keyframe add/remove (per not-yet-rendered job config) ──
 
# @editor_bp.route("/api/editor/keyframe/add", methods=["POST"])
# def api_keyframe_add():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     kf = {"t": float(data["t"]), "x": int(data["x"]), "y": int(data["y"])}
#     job["config"].setdefault("manual_crop_keyframes", [])
#     job["config"]["manual_crop_keyframes"] = job["config"]["manual_crop_keyframes"] or []
#     job["config"]["manual_crop_keyframes"].append(kf)
#     job["config"]["manual_crop_keyframes"].sort(key=lambda k: k["t"])
#     return jsonify({"manual_crop_keyframes": job["config"]["manual_crop_keyframes"]})
 
 
# @editor_bp.route("/api/editor/keyframe/remove", methods=["POST"])
# def api_keyframe_remove():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     t = float(data["t"])
#     kfs = job["config"].get("manual_crop_keyframes") or []
#     job["config"]["manual_crop_keyframes"] = [k for k in kfs if abs(k["t"] - t) > 1e-6]
#     return jsonify({"manual_crop_keyframes": job["config"]["manual_crop_keyframes"]})
 
 
# # ── hook-cut / rearrange segment add/remove ──
 
# @editor_bp.route("/api/editor/segment/add", methods=["POST"])
# def api_segment_add():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     seg = {"start": float(data["start"]), "end": float(data["end"])}
#     index = data.get("index")
#     segs = job["config"].get("segments") or []
#     if index is None or index >= len(segs):
#         segs.append(seg)
#     else:
#         segs.insert(int(index), seg)
#     job["config"]["segments"] = segs
#     return jsonify({"segments": segs})
 
 
# @editor_bp.route("/api/editor/segment/remove", methods=["POST"])
# def api_segment_remove():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     index = int(data["index"])
#     segs = job["config"].get("segments") or []
#     if 0 <= index < len(segs):
#         segs.pop(index)
#     job["config"]["segments"] = segs
#     return jsonify({"segments": segs})








# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# AutoShortAi — Video Editor module
# ──────────────────────────────────
# Adds AI auto-reframe (face tracking), trending animated captions,
# hook-cut / manual clip reordering, and one-click multi-aspect-ratio
# export — all producing a SINGLE muxed video+audio file per ratio.
 
# Wire it into the main app with:
 
#     from video_editor import editor_bp, init_editor
#     init_editor(BASE, FFMPEG)
#     app.register_blueprint(editor_bp)
 
# Routes exposed (all under /api/editor/...):
#     GET  /api/editor/editor             -> serves the standalone editor.html UI (open this in a tab)
#     POST /api/editor/upload             -> upload a local file, returns a source_path to use below
#     POST /api/editor/analyze            -> face-track + transcript + silence/hook preview (for the UI)
#     GET  /api/editor/styles             -> list of caption style presets
#     GET  /api/editor/ratios             -> list of export aspect-ratio presets
#     GET  /api/editor/features           -> which advanced features are available (installed libs)
#     POST /api/editor/render             -> kicks off a background render job
#     GET  /api/editor/progress/<job_id>  -> live progress
#     GET  /api/editor/file/<job_id>/<ratio> -> serves one finished ratio's file
#     POST /api/editor/keyframe/add       -> add a manual crop keyframe to a job's override track
#     POST /api/editor/keyframe/remove    -> remove a manual crop keyframe
#     POST /api/editor/segment/add        -> add a hook-cut / reorder segment
#     POST /api/editor/segment/remove     -> remove a hook-cut / reorder segment
 
# ADVANCED EXTRAS (on top of the original spec):
#     • fill_mode="blur_pad"   -> reframe by fitting the WHOLE frame + blurred
#       background bars instead of hard-cropping (no content ever cut off).
#     • jump_cut feature       -> auto-detects and removes silence/dead-air,
#       the other half of "trending" short-form editing besides hook-cut.
#     • hook_cut + jump_cut COMBINED -> if both are on, the loudest window is
#       used as the cold-open "hook" and is prepended in front of the
#       dead-air-trimmed rest of the video (previously turning jump_cut on
#       silently dropped hook_cut — now they compose).
#     • normalize_audio        -> EBU R128 loudness normalization (-14 LUFS),
#       on by default, matches the loudness social platforms expect.
#     • caption keyword emphasis -> numbers, money, and punchy "power words"
#       auto-highlight in captions even outside full word-by-word styles.
#     • quality/caption_style inputs are now validated with a safe fallback
#       instead of raising a raw KeyError if the frontend sends a typo'd key.
#     • self-serves its own frontend (editor.html) at GET /api/editor/editor,
#       the same "own file, same process/port" pattern as downloader.py.
 
# DEPENDENCIES (install what you want to use — everything degrades gracefully
# if a library is missing, see FEATURE FLAGS AT IMPORT TIME below):
#     pip install opencv-python            # required for any face tracking
#     pip install mediapipe                # optional, gives the "advanced" tracker
#                                           # (falls back to Haar cascade otherwise)
#     pip install faster-whisper           # required for subtitles (word timestamps)
#     pip install numpy
 
# DESIGN NOTES — how each part of the request maps to code:
#   • "advance level face tracking"   -> _FaceTracker (mediapipe BlazeFace if present,
#                                         else OpenCV Haar cascade), EMA-smoothed
#                                         centroid so the crop doesn't jitter frame
#                                         to frame, multi-face -> largest/most-central
#                                         face wins.
#   • "perfect advance subtitles,
#      trending caption styles"       -> CAPTION_STYLES presets + build_ass() which
#                                         groups words into short on-screen chunks and
#                                         emits karaoke-style \\k tags for word-by-word
#                                         pop/highlight animation, burned in with
#                                         ffmpeg's `ass` filter (not a soft-sub track,
#                                         so it always displays correctly everywhere).
#   • "single high quality video+audio
#      in one file"                   -> every render always outputs one .mp4 with
#                                         both streams muxed, CRF-based high quality
#                                         encode (see QUALITY_PRESETS).
#   • "export in all ratio perfectly" -> RATIO_PRESETS, rendered in a loop, one file
#                                         per ratio, each independently croppped using
#                                         the SAME face track (recomputed crop window
#                                         per target aspect).
#   • "rearrange video manually /
#      hook cut"                      -> job["segments"]: an ordered list of
#                                         {start,end} clip ranges. Auto hook-cut picks
#                                         the loudest window as segment 1 automatically;
#                                         manual mode lets the caller add/remove/reorder
#                                         segments directly (segment/add, segment/remove).
#   • "add and remove option for all" -> every feature is a boolean flag in job config
#                                         (features{}) that can be flipped and re-rendered,
#                                         plus explicit add/remove endpoints for the two
#                                         list-based tracks (crop keyframes, segments).
# """
 
# import os
# import re
# import json
# import uuid
# import struct
# import threading
# import subprocess
# import concurrent.futures
# from pathlib import Path
 
# from flask import Blueprint, request, jsonify, send_file
 
# editor_bp = Blueprint("editor_bp", __name__)
 
# # ══════════════════════════════ embedded frontend ══════════════════════════════
# # The whole editor UI lives right here as one string — no separate .html file
# # to ship or lose track of. Served as-is by GET /api/editor/editor. Its fetch
# # calls use relative paths (same-origin), so it works immediately wherever
# # this blueprint is registered.
# EDITOR_HTML = r"""<!doctype html>
# <html lang="en">
# <head>
# <meta charset="utf-8">
# <meta name="viewport" content="width=device-width, initial-scale=1">
# <title>Reframe — AI video editor</title>
# <link rel="preconnect" href="https://fonts.googleapis.com">
# <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
# <style>
#   /* ── Theme — mirrors the main app's palette exactly (same variable
#      VALUES, just different variable NAMES for this page's own CSS). Default
#      below is "dark-violet", the main app's default. The <html data-theme="…">
#      overrides further down are read straight off the SAME localStorage key
#      the main app writes to (see the sync script at the bottom of <body>),
#      so switching themes on Studio/Downloader also re-themes this page —
#      no manual matching needed ever again. ── */
#   :root{
#     --bg:#0b0d12; --panel:#13151c; --panel-2:#191c25; --border:#262a36;
#     --text:#eef0f6; --text-dim:#8a90a4; --text-faint:#8a90a4;
#     --amber:#6e5bff; --teal:#22d3c4; --red:#ff5a6e;
#     --amber-dim: color-mix(in srgb, var(--amber) 22%, #000);
#     --teal-dim: color-mix(in srgb, var(--teal) 22%, #000);
#     --radius:10px;
#     font-size:15px;
#   }
#   html[data-theme="light"]{
#     --bg:#f4f5f9; --panel:#ffffff; --panel-2:#eef0f6; --border:#dde1ea;
#     --text:#171923; --text-dim:#5b6472; --text-faint:#5b6472;
#     --amber:#5b46e0; --teal:#0891b2; --red:#e0324a;
#   }
#   html[data-theme="midnight-blue"]{
#     --bg:#050a16; --panel:#0c1526; --panel-2:#12213a; --border:#1f3358;
#     --text:#e8f1ff; --text-dim:#7f93b8; --text-faint:#7f93b8;
#     --amber:#3b82f6; --teal:#22d3ee; --red:#ff5a6e;
#   }
#   html[data-theme="forest"]{
#     --bg:#0a130d; --panel:#0f1e14; --panel-2:#16301f; --border:#234a30;
#     --text:#eafbea; --text-dim:#86b399; --text-faint:#86b399;
#     --amber:#22c55e; --teal:#a3e635; --red:#ff5a6e;
#   }
#   html[data-theme="sunset"]{
#     --bg:#1a0e12; --panel:#241419; --panel-2:#331d27; --border:#4a2635;
#     --text:#fdf1ee; --text-dim:#cb9aa3; --text-faint:#cb9aa3;
#     --amber:#fb7185; --teal:#f59e0b; --red:#ff5a6e;
#   }
#   html[data-theme="ocean"]{
#     --bg:#061319; --panel:#0b1f27; --panel-2:#123039; --border:#1e4a56;
#     --text:#e6fbff; --text-dim:#7fb8c4; --text-faint:#7fb8c4;
#     --amber:#06b6d4; --teal:#6366f1; --red:#ff5a6e;
#   }
#   html[data-theme="rose"]{
#     --bg:#170f14; --panel:#221720; --panel-2:#2e1e2b; --border:#4a2f45;
#     --text:#fdeef5; --text-dim:#c79ab0; --text-faint:#c79ab0;
#     --amber:#ec4899; --teal:#f472b6; --red:#ff5a6e;
#   }
#   html[data-theme="amber"]{
#     --bg:#160f05; --panel:#221708; --panel-2:#33230d; --border:#4d3416;
#     --text:#fdf4e3; --text-dim:#c9a877; --text-faint:#c9a877;
#     --amber:#f59e0b; --teal:#eab308; --red:#ff5a6e;
#   }
#   html[data-theme="slate"]{
#     --bg:#101215; --panel:#181b1f; --panel-2:#20242a; --border:#333941;
#     --text:#eceef1; --text-dim:#9aa2ad; --text-faint:#9aa2ad;
#     --amber:#64748b; --teal:#94a3b8; --red:#ff5a6e;
#   }
#   html[data-theme="cyberpunk"]{
#     --bg:#0a0014; --panel:#150124; --panel-2:#210235; --border:#4a0a6b;
#     --text:#f5e6ff; --text-dim:#b083d6; --text-faint:#b083d6;
#     --amber:#e91e9c; --teal:#00f0ff; --red:#ff5a6e;
#   }
#   *{box-sizing:border-box;}
#   body{
#     margin:0; background:var(--bg); color:var(--text);
#     font-family:'Inter',sans-serif; min-height:100vh;
#   }
#   h1,h2,h3,.display{font-family:'Space Grotesk',sans-serif;}
#   .mono{font-family:'JetBrains Mono',monospace;}
#   ::selection{background:var(--amber-dim); color:var(--amber);}
 
#   /* ── top bar ── */
#   .topbar{
#     display:flex; align-items:center; gap:16px; padding:14px 22px;
#     border-bottom:1px solid var(--border); background:var(--panel);
#     position:sticky; top:0; z-index:20;
#   }
#   .brand{display:flex; align-items:center; gap:10px;}
#   .brand .dot{width:10px; height:10px; border-radius:50%; background:var(--amber); box-shadow:0 0 10px var(--amber);}
#   .brand h1{font-size:18px; font-weight:700; margin:0; letter-spacing:.2px;}
#   .brand span{color:var(--text-dim); font-size:12px;}
#   .topbar .spacer{flex:1;}
#   .api-input{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text-dim);
#     border-radius:8px; padding:7px 10px; font-size:12px; font-family:'JetBrains Mono',monospace;
#     width:220px;
#   }
#   .api-input:focus{outline:none; border-color:var(--amber); color:var(--text);}
 
#   /* ── layout ── */
#   .app{display:grid; grid-template-columns:1fr 380px; gap:1px; background:var(--border);}
#   @media (max-width:980px){ .app{grid-template-columns:1fr;} }
#   .col{background:var(--bg); padding:22px; min-width:0;}
 
#   /* ── upload zone ── */
#   .dropzone{
#     border:1.5px dashed var(--border); border-radius:var(--radius);
#     padding:40px 20px; text-align:center; cursor:pointer;
#     transition:border-color .15s, background .15s;
#   }
#   .dropzone:hover, .dropzone.drag{border-color:var(--amber); background:rgba(255,176,32,.04);}
#   .dropzone svg{opacity:.5; margin-bottom:10px;}
#   .dropzone .hint{color:var(--text-dim); font-size:13px; margin-top:6px;}
 
#   /* ── fetch-from-link (ytdlp-style resolver, built into the editor) ── */
#   .fetch-section{border:1px solid var(--border); border-radius:var(--radius); padding:12px; background:var(--panel-2); margin-bottom:14px;}
#   .fetch-row{display:flex; gap:8px; margin-top:8px;}
#   .fetch-input{
#     flex:1; background:var(--panel); border:1px solid var(--border); color:var(--text);
#     border-radius:8px; padding:9px 10px; font-size:13px;
#   }
#   .fetch-input:focus{outline:none; border-color:var(--amber);}
#   .btn-accent{background:var(--amber); color:#1a1200; border:none; font-weight:600;}
#   .btn-accent:hover{filter:brightness(1.08);}
#   .fetch-card{
#     display:flex; gap:10px; align-items:center; margin-top:10px;
#     border:1px solid var(--border); border-radius:8px; padding:10px; background:var(--panel);
#   }
#   .fetch-thumb{width:88px; height:56px; object-fit:cover; border-radius:6px; background:#000; flex-shrink:0;}
#   .fetch-meta{flex:1; min-width:0;}
#   .fetch-title{font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
#   .fetch-sub{font-size:11.5px; color:var(--text-dim); margin-top:2px;}
#   .fetch-status{font-size:12px; color:var(--text-dim); margin-top:8px; min-height:16px;}
 
#   /* ── server library (downloader output + past uploads) ── */
#   .or-divider{
#     display:flex; align-items:center; gap:10px; margin:14px 0;
#     color:var(--text-faint); font-size:12px; text-transform:uppercase; letter-spacing:.5px;
#   }
#   .or-divider::before, .or-divider::after{content:""; flex:1; height:1px; background:var(--border);}
#   .lib-section{border:1px solid var(--border); border-radius:var(--radius); padding:12px; background:var(--panel-2);}
#   .lib-header{display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; font-size:13px; color:var(--text-dim);}
#   .lib-grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:8px; max-height:220px; overflow-y:auto;}
#   .lib-empty{color:var(--text-faint); font-size:12.5px; grid-column:1/-1;}
#   .lib-item{
#     border:1px solid var(--border); border-radius:8px; padding:10px; cursor:pointer;
#     background:var(--panel); transition:border-color .15s, transform .1s;
#   }
#   .lib-item:hover{border-color:var(--amber); transform:translateY(-1px);}
#   .lib-item.active{border-color:var(--teal); background:var(--teal-dim);}
#   .lib-item .lib-name{font-size:12.5px; font-weight:600; word-break:break-word; margin-bottom:4px;}
#   .lib-item .lib-meta{font-size:11px; color:var(--text-dim); display:flex; justify-content:space-between;}
#   .lib-item .lib-tag{
#     display:inline-block; font-size:10px; padding:1px 6px; border-radius:20px;
#     background:var(--amber-dim); color:var(--amber); margin-bottom:4px;
#   }
 
#   /* ── preview ── */
#   .preview-wrap{position:relative; display:flex; justify-content:center; margin-bottom:16px;}
#   .preview-frame{position:relative; background:#000; border-radius:8px; overflow:hidden; max-width:100%;}
#   video{display:block; max-width:100%; max-height:60vh;}
#   .crop-box{
#     position:absolute; border:1.5px solid var(--amber); pointer-events:none;
#     box-shadow:0 0 0 2000px rgba(0,0,0,.45);
#   }
#   .crop-box .corner{
#     position:absolute; width:16px; height:16px; border-color:var(--amber); border-style:solid; border-width:0;
#   }
#   .crop-box .tl{top:-1.5px; left:-1.5px; border-top-width:3px; border-left-width:3px;}
#   .crop-box .tr{top:-1.5px; right:-1.5px; border-top-width:3px; border-right-width:3px;}
#   .crop-box .bl{bottom:-1.5px; left:-1.5px; border-bottom-width:3px; border-left-width:3px;}
#   .crop-box .br{bottom:-1.5px; right:-1.5px; border-bottom-width:3px; border-right-width:3px;}
#   .crop-box{cursor:grab;}
#   .crop-box.dragging{cursor:grabbing;}
#   .crop-readout{
#     position:absolute; top:8px; left:8px; background:rgba(0,0,0,.6);
#     color:var(--amber); font-size:11px; padding:3px 7px; border-radius:5px;
#   }
 
#   .preview-controls{display:flex; align-items:center; gap:10px; justify-content:center; margin-bottom:8px;}
#   .btn-icon{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text);
#     width:34px; height:34px; border-radius:50%; cursor:pointer; display:flex;
#     align-items:center; justify-content:center;
#   }
#   .btn-icon:hover{border-color:var(--amber);}
#   .time{font-size:12px; color:var(--text-dim); min-width:96px; text-align:center;}
 
#   /* ── timeline ── */
#   .timeline-panel{background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:16px;}
#   .timeline-panel h3{margin:0 0 4px; font-size:13px; letter-spacing:.4px; text-transform:uppercase; color:var(--text-dim);}
#   .timeline{
#     position:relative; height:56px; background:var(--panel-2); border-radius:6px;
#     margin-top:10px; overflow:hidden; cursor:pointer;
#   }
#   .tl-face{position:absolute; top:0; height:18px; background:linear-gradient(90deg, transparent, var(--teal-dim)); opacity:.6;}
#   .tl-seg{
#     position:absolute; top:20px; height:18px; background:var(--teal); opacity:.85; border-radius:3px;
#     display:flex; align-items:center; justify-content:center; font-size:10px; color:#04231D; font-weight:600;
#     cursor:grab;
#   }
#   .tl-hook{position:absolute; top:38px; height:14px; background:var(--amber); opacity:.9; border-radius:3px;}
#   .tl-playhead{position:absolute; top:0; bottom:0; width:2px; background:var(--red);}
#   .tl-legend{display:flex; gap:16px; margin-top:8px; font-size:11px; color:var(--text-dim);}
#   .tl-legend span{display:inline-flex; align-items:center; gap:5px;}
#   .swatch{width:9px; height:9px; border-radius:2px; display:inline-block;}
 
#   .seg-list{margin-top:10px; display:flex; flex-direction:column; gap:6px;}
#   .seg-row{
#     display:flex; align-items:center; gap:8px; background:var(--panel-2);
#     border:1px solid var(--border); border-radius:7px; padding:6px 10px; font-size:12px;
#   }
#   .seg-row .mono{color:var(--teal);}
#   .seg-row .spacer{flex:1;}
#   .seg-row button{background:none; border:none; color:var(--text-faint); cursor:pointer; font-size:14px;}
#   .seg-row button:hover{color:var(--red);}
#   .btn-small{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text-dim);
#     border-radius:6px; padding:6px 10px; font-size:12px; cursor:pointer;
#   }
#   .btn-small:hover{border-color:var(--amber); color:var(--amber);}
 
#   /* ── side panel ── */
#   .section{margin-bottom:26px;}
#   .section h3{
#     font-size:12px; letter-spacing:.5px; text-transform:uppercase; color:var(--text-dim);
#     margin:0 0 12px; display:flex; align-items:center; gap:8px;
#   }
#   .section h3 .num{
#     width:18px; height:18px; border-radius:50%; background:var(--panel-2); border:1px solid var(--border);
#     display:flex; align-items:center; justify-content:center; font-size:10px; color:var(--amber);
#   }
 
#   .toggle-row{
#     display:flex; align-items:center; justify-content:space-between;
#     padding:10px 12px; background:var(--panel); border:1px solid var(--border);
#     border-radius:8px; margin-bottom:8px;
#   }
#   .toggle-row .label{font-size:13px;}
#   .toggle-row .desc{font-size:11px; color:var(--text-faint); margin-top:2px;}
#   .toggle-row.disabled{opacity:.4;}
#   .switch{position:relative; width:38px; height:21px; flex-shrink:0;}
#   .switch input{opacity:0; width:0; height:0;}
#   .slider{
#     position:absolute; inset:0; background:var(--panel-2); border:1px solid var(--border);
#     border-radius:20px; cursor:pointer; transition:.15s;
#   }
#   .slider::before{
#     content:''; position:absolute; width:15px; height:15px; left:2px; top:2px;
#     background:var(--text-dim); border-radius:50%; transition:.15s;
#   }
#   .switch input:checked + .slider{background:var(--amber-dim); border-color:var(--amber);}
#   .switch input:checked + .slider::before{transform:translateX(17px); background:var(--amber);}
#   .switch input:disabled + .slider{cursor:not-allowed;}
 
#   .ratio-grid{display:grid; grid-template-columns:1fr 1fr; gap:8px;}
#   .ratio-card{
#     border:1px solid var(--border); background:var(--panel); border-radius:8px;
#     padding:10px; cursor:pointer; text-align:center; position:relative;
#   }
#   .ratio-card.active{border-color:var(--amber); background:rgba(255,176,32,.06);}
#   .ratio-card .shape{margin:0 auto 6px; background:var(--panel-2); border:1px solid var(--border); border-radius:3px;}
#   .ratio-card .name{font-size:12px; font-weight:600;}
#   .ratio-card .lbl{font-size:10px; color:var(--text-faint); margin-top:2px;}
 
#   .style-list{display:flex; flex-direction:column; gap:8px;}
#   .style-card{
#     display:flex; align-items:center; gap:12px; border:1px solid var(--border);
#     background:var(--panel); border-radius:8px; padding:10px 12px; cursor:pointer;
#   }
#   .style-card.active{border-color:var(--teal);}
#   .style-swatch{
#     flex-shrink:0; width:64px; height:36px; border-radius:6px; background:#000;
#     display:flex; align-items:center; justify-content:center; font-size:9px; font-weight:800;
#     letter-spacing:.5px;
#   }
#   .style-card .name{font-size:12.5px; font-weight:600;}
#   .style-card .lbl{font-size:10.5px; color:var(--text-faint);}
 
#   select, .fill-toggle{
#     width:100%; background:var(--panel); border:1px solid var(--border); color:var(--text);
#     padding:9px 10px; border-radius:8px; font-size:13px;
#   }
#   .fill-toggle{display:flex; gap:6px;}
#   .fill-opt{flex:1; text-align:center; padding:8px; border-radius:6px; cursor:pointer; font-size:12px; border:1px solid var(--border);}
#   .fill-opt.active{border-color:var(--amber); color:var(--amber); background:rgba(255,176,32,.06);}
 
#   .render-btn{
#     width:100%; background:var(--amber); color:#1A1204; border:none; border-radius:10px;
#     padding:14px; font-size:14px; font-weight:700; cursor:pointer; font-family:'Space Grotesk',sans-serif;
#     letter-spacing:.3px;
#   }
#   .render-btn:disabled{background:var(--panel-2); color:var(--text-faint); cursor:not-allowed;}
#   .render-btn:not(:disabled):hover{filter:brightness(1.08);}
 
#   .progress-wrap{margin-top:14px;}
#   .progress-track{height:6px; background:var(--panel-2); border-radius:4px; overflow:hidden;}
#   .progress-fill{height:100%; background:linear-gradient(90deg, var(--teal), var(--amber)); width:0%; transition:width .3s;}
#   .progress-label{display:flex; justify-content:space-between; font-size:11px; color:var(--text-dim); margin-top:6px;}
 
#   .results{display:flex; flex-direction:column; gap:8px; margin-top:14px;}
#   .result-row{
#     display:flex; align-items:center; gap:10px; background:var(--panel); border:1px solid var(--border);
#     border-radius:8px; padding:10px 12px;
#   }
#   .result-row .name{font-size:12.5px; font-weight:600; flex:1;}
#   .result-row a{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--teal);
#     text-decoration:none; font-size:11.5px; padding:6px 10px; border-radius:6px;
#   }
#   .result-row a:hover{border-color:var(--teal);}
 
#   .status-line{font-size:11.5px; color:var(--text-faint); margin-top:10px; line-height:1.5;}
#   .status-line b{color:var(--text-dim);}
#   .feature-warn{
#     font-size:11px; color:var(--amber); background:rgba(255,176,32,.08); border:1px solid var(--amber-dim);
#     border-radius:6px; padding:8px 10px; margin-top:8px; display:none;
#   }
# </style>
# </head>
# <body>
# <script>
# // ── Theme sync — this page is served from the SAME Flask app/port as the
# // main Studio (registered as a blueprint), and when opened as an <iframe> on
# // that page it's also same-origin, so it shares localStorage automatically.
# // Reading 'shortsStudioTheme' here — the exact key the main page's theme
# // picker writes to — means this editor always matches whatever theme is
# // active there, with zero manual setup. Runs immediately (before first
# // paint) so there's no flash of the wrong palette, and again whenever the
# // theme changes elsewhere (the 'storage' event fires in this frame whenever
# // another same-origin window/frame updates localStorage).
# (function(){
#   function applyTheme(){
#     const t = localStorage.getItem('shortsStudioTheme') || 'dark-violet';
#     if(t === 'dark-violet') document.documentElement.removeAttribute('data-theme');
#     else document.documentElement.setAttribute('data-theme', t);
#   }
#   applyTheme();
#   window.addEventListener('storage', (e)=>{ if(e.key === 'shortsStudioTheme') applyTheme(); });
# })();
# </script>
 
# <div class="topbar">
#   <div class="brand"><span class="dot"></span><h1>Reframe</h1><span>face-tracked auto edit</span></div>
#   <div class="spacer"></div>
#   <span id="apiBaseHint" class="mono" style="font-size:11px; color:var(--text-faint); margin-right:6px;">same server</span>
#   <button type="button" id="apiBaseToggle" class="btn-small" title="Only needed if this page is served from a different host than the API">⚙</button>
#   <input id="apiBase" class="api-input mono" placeholder="leave blank — uses this same page's server" value="" style="display:none;">
# </div>
 
# <div class="app">
#   <!-- LEFT: preview + timeline -->
#   <div class="col">
#     <div class="fetch-section">
#       <div class="lib-header">
#         <span>🔗 Fetch from a link (YouTube / Instagram / TikTok / X / Facebook…)</span>
#       </div>
#       <div class="fetch-row">
#         <input type="text" id="fetchUrl" class="fetch-input mono" placeholder="Paste a video URL…">
#         <button type="button" class="btn-small btn-accent" id="fetchInfoBtn">Fetch</button>
#       </div>
#       <div id="fetchCard" class="fetch-card" style="display:none;">
#         <img id="fetchThumb" class="fetch-thumb" alt="">
#         <div class="fetch-meta">
#           <div id="fetchTitle" class="fetch-title"></div>
#           <div id="fetchSub" class="fetch-sub"></div>
#         </div>
#         <button type="button" class="btn-small btn-accent" id="fetchUseBtn">⬇ Fetch high-quality &amp; use</button>
#       </div>
#       <div id="fetchStatus" class="fetch-status"></div>
#     </div>
 
#     <div class="or-divider">or drop / choose a video from this PC</div>
#     <div id="dropzone" class="dropzone">
#       <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8A93A3" stroke-width="1.5"><path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v3a2 2 0 002 2h12a2 2 0 002-2v-3"/></svg>
#       <div>Drop a video, or click to choose one</div>
#       <div class="hint">mp4 / mov · analyzed locally before render</div>
#       <input type="file" id="fileInput" accept="video/*" style="display:none">
#     </div>
 
#     <div class="or-divider">or pick a video already fetched by the Downloader</div>
#     <div class="lib-section">
#       <div class="lib-header">
#         <span>📥 From Downloader / server library</span>
#         <button type="button" class="btn-small" id="refreshLibBtn">Refresh</button>
#       </div>
#       <div class="lib-grid" id="libGrid"><span class="lib-empty">Click Refresh to list videos already downloaded via the Downloader tab.</span></div>
#     </div>
 
#     <div id="previewSection" style="display:none;">
#       <div class="preview-wrap">
#         <div class="preview-frame" id="previewFrame">
#           <video id="video" muted></video>
#           <div class="crop-box" id="cropBox" style="display:none;">
#             <div class="corner tl"></div><div class="corner tr"></div><div class="corner bl"></div><div class="corner br"></div>
#             <div class="crop-readout" id="cropReadout">9:16</div>
#           </div>
#         </div>
#       </div>
#       <div class="preview-controls">
#         <div class="btn-icon" id="playBtn">▶</div>
#         <div class="time mono" id="timeLabel">0:00 / 0:00</div>
#       </div>
 
#       <div class="timeline-panel">
#         <h3>Timeline</h3>
#         <div class="timeline" id="timeline">
#           <div class="tl-playhead" id="playhead" style="left:0%"></div>
#         </div>
#         <div class="tl-legend">
#           <span><span class="swatch" style="background:var(--teal-dim)"></span>face track</span>
#           <span><span class="swatch" style="background:var(--teal)"></span>segments</span>
#           <span><span class="swatch" style="background:var(--amber)"></span>suggested hook</span>
#         </div>
#         <div class="seg-list" id="segList"></div>
#         <div style="display:flex; gap:8px; margin-top:10px;">
#           <button class="btn-small" id="addSegBtn">+ add segment at playhead (±3s)</button>
#           <button class="btn-small" id="useHookBtn" style="display:none;">use suggested hook</button>
#           <button class="btn-small" id="useSilenceBtn" style="display:none;">use auto jump-cut segments</button>
#         </div>
#         <div style="display:flex; gap:8px; margin-top:8px;">
#           <button class="btn-small" id="addKfBtn">+ pin crop here (manual keyframe)</button>
#           <button class="btn-small" id="clearKfBtn">clear manual crop</button>
#         </div>
#       </div>
#       <div class="status-line" id="statusLine"></div>
#       <div id="sourceMetaPanel" class="lib-section" style="display:none; margin-top:10px;"></div>
#     </div>
#   </div>
 
#   <!-- RIGHT: controls -->
#   <div class="col">
#     <div class="section">
#       <h3><span class="num">1</span>Auto-reframe</h3>
#       <div class="toggle-row" id="rowFace">
#         <div><div class="label">Face tracking</div><div class="desc">Advanced (mediapipe) if installed, else standard cascade</div></div>
#         <label class="switch"><input type="checkbox" id="chkFace" checked><span class="slider"></span></label>
#       </div>
#       <div class="fill-toggle" style="margin-top:8px;">
#         <div class="fill-opt active" data-fill="crop">Crop to face</div>
#         <div class="fill-opt" data-fill="blur_pad">Fit + blurred bars</div>
#       </div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">2</span>Captions</h3>
#       <div class="toggle-row" id="rowSubs">
#         <div><div class="label">Auto subtitles</div><div class="desc">Word-level, keyword emphasis auto-highlighted</div></div>
#         <label class="switch"><input type="checkbox" id="chkSubs" checked><span class="slider"></span></label>
#       </div>
#       <div class="style-list" id="styleList"></div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">3</span>Hook &amp; pacing</h3>
#       <div class="toggle-row" id="rowHook">
#         <div><div class="label">Hook-cut</div><div class="desc">Open on the loudest / most energetic moment</div></div>
#         <label class="switch"><input type="checkbox" id="chkHook"><span class="slider"></span></label>
#       </div>
#       <div class="toggle-row" id="rowJump">
#         <div><div class="label">Jump-cut silences</div><div class="desc">Auto-remove dead air between lines</div></div>
#         <label class="switch"><input type="checkbox" id="chkJump"><span class="slider"></span></label>
#       </div>
#       <div class="toggle-row">
#         <div><div class="label">Loudness normalize</div><div class="desc">-14 LUFS, matches platform playback volume</div></div>
#         <label class="switch"><input type="checkbox" id="chkNorm" checked><span class="slider"></span></label>
#       </div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">4</span>Export ratios</h3>
#       <div class="ratio-grid" id="ratioGrid"></div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">5</span>Quality</h3>
#       <select id="qualitySel">
#         <option value="high">High (crf 18, slow) — best quality</option>
#         <option value="medium">Medium (crf 21) — balanced</option>
#         <option value="fast">Fast (crf 23) — quick preview</option>
#       </select>
#     </div>
 
#     <button class="render-btn" id="renderBtn" disabled>Analyze a video first</button>
#     <div class="feature-warn" id="featureWarn"></div>
#     <div class="progress-wrap" id="progressWrap" style="display:none;">
#       <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
#       <div class="progress-label"><span id="progressStage">queued</span><span id="progressPct">0%</span></div>
#     </div>
#     <div class="results" id="results"></div>
#   </div>
# </div>
 
# <script>
# const $ = (id) => document.getElementById(id);
# const state = {
#   sourcePath: null, sourceMeta: null, sourceOrigin: 'pc', duration: 0, srcW: 0, srcH: 0, srcFps: null,
#   faceTrack: [], words: [], suggestedHook: null, suggestedSilences: null,
#   segments: [], manualKeyframes: [],
#   // Multiple export ratios can be selected at once now (a Set of ratio
#   // keys, e.g. {'9:16','1:1'}), not just one — "auto" is pre-selected by
#   // default since it's the safest zero-quality-loss choice.
#   selectedRatios: new Set(['auto']),
#   previewRatio: '9:16', // drives the live crop box only, independent of export selection
#   selectedStyle: 'bold_pop', fillMode: 'crop',
#   jobId: null, dragging: false,
# };
 
# // Same-origin by default: this page is served BY the same Flask app/port
# // that Reframe, the Downloader, and everything else run on (they're all
# // registered on one Flask app as blueprints, per RenderDetect's main()),
# // so relative paths already hit the right server. The "⚙" override is
# // only for the rare case this HTML got copied to a different host — if
# // left blank (the default), api() never leaves the current origin, which
# // is what fixes the class of bug where a stray/incorrect API host caused
# // every request (analyze, render, fetch…) to silently fail.
# function apiBase(){ const v = $('apiBase').value.trim().replace(/\/$/, ''); return v; }
# function api(path){ return apiBase() + path; }
# $('apiBaseToggle').onclick = ()=>{
#   const el = $('apiBase');
#   const show = el.style.display === 'none';
#   el.style.display = show ? 'inline-block' : 'none';
#   $('apiBaseHint').style.display = show ? 'none' : 'inline';
# };
 
# const RATIO_META = {
#   '9:16': {w:9,h:16}, '16:9': {w:16,h:9}, '1:1': {w:1,h:1}, '4:5': {w:4,h:5}, '4:3': {w:4,h:3},
# };
# const STYLE_META = {
#   bold_pop:       {bg:'#111', color:'#fff', hi:'#FFD700', sample:'THIS IS'},
#   karaoke_classic:{bg:'#111', color:'#fff', hi:'#FFA500', sample:'FILLS IN'},
#   neon_glow:      {bg:'#111', color:'#fff', hi:'#00F0FF', sample:'GLOWS UP'},
#   minimal_clean:  {bg:'#111', color:'#fff', hi:'#fff', sample:'stays quiet'},
#   creator_bold:   {bg:'#111', color:'#FFFF00', hi:'#FFFF00', sample:'HUGE TEXT'},
# };
 
# // ── boot: load styles / ratios / features from backend ──
# async function loadMeta(){
#   try{
#     const [styles, ratios, features] = await Promise.all([
#       fetch(api('/api/editor/styles')).then(r=>r.json()),
#       fetch(api('/api/editor/ratios')).then(r=>r.json()),
#       fetch(api('/api/editor/features')).then(r=>r.json()),
#     ]);
#     renderStyles(styles);
#     renderRatios(ratios);
#     applyFeatureAvailability(features);
#   }catch(e){
#     $('featureWarn').style.display='block';
#     $('featureWarn').textContent = 'Could not reach the backend at ' + api('') + ' — set the API base URL above and reload.';
#   }
# }
 
# function renderStyles(styles){
#   const list = $('styleList'); list.innerHTML='';
#   Object.entries(styles).forEach(([key, meta])=>{
#     const m = STYLE_META[key] || {bg:'#111', color:'#fff', hi:'#fff', sample:'Sample'};
#     const card = document.createElement('div');
#     card.className = 'style-card' + (key===state.selectedStyle ? ' active':'');
#     card.innerHTML = `<div class="style-swatch" style="background:${m.bg}">
#         <span style="color:${m.color}">${m.sample.split(' ')[0]}</span>&nbsp;<span style="color:${m.hi}">${m.sample.split(' ')[1]||''}</span>
#       </div>
#       <div><div class="name">${meta.label.split(' (')[0]}</div><div class="lbl">${(meta.label.match(/\((.*)\)/)||[,''])[1]}</div></div>`;
#     card.onclick = ()=>{ state.selectedStyle = key; renderStyles(styles); };
#     list.appendChild(card);
#   });
# }
 
# // Export ratios are now MULTI-select — tap any number of cards and every
# // one you pick gets rendered and offered as a download, all in the same
# // render job. "Auto" (native size, no crop, zero quality loss) is its own
# // card too. state.previewRatio (separate from the export selection) just
# // drives which crop box is drawn live over the video for editing.
# function renderRatios(ratios){
#   const grid = $('ratioGrid'); grid.innerHTML='';
#   Object.entries(ratios).forEach(([key, r])=>{
#     const isAuto = key === 'auto';
#     const srcAspect = (state.srcW && state.srcH) ? {w: state.srcW, h: state.srcH} : {w:9, h:16};
#     const meta = isAuto ? srcAspect : (RATIO_META[key] || {w:1,h:1});
#     const scale = 30 / Math.max(meta.w, meta.h);
#     const selected = state.selectedRatios.has(key);
#     const card = document.createElement('div');
#     card.className = 'ratio-card' + (selected ? ' active':'');
#     card.innerHTML = `<div class="shape" style="width:${meta.w*scale}px; height:${meta.h*scale}px;"></div>
#       <div class="name">${selected ? '✓ ' : ''}${key === 'auto' ? 'Auto' : key}</div><div class="lbl">${r.label}</div>`;
#     card.onclick = ()=>{
#       if(state.selectedRatios.has(key)) state.selectedRatios.delete(key);
#       else state.selectedRatios.add(key);
#       if(!state.selectedRatios.size) state.selectedRatios.add(key); // keep at least one selected
#       if(!isAuto){ state.previewRatio = key; drawCropForCurrentTime(); }
#       renderRatios(ratios);
#     };
#     grid.appendChild(card);
#   });
# }
 
# function applyFeatureAvailability(f){
#   toggleRow('rowFace', 'chkFace', f.face_tracking, 'opencv-python not installed on the server');
#   toggleRow('rowSubs', 'chkSubs', f.subtitles, 'faster-whisper not installed on the server');
#   toggleRow('rowHook', 'chkHook', true, '');
#   toggleRow('rowJump', 'chkJump', true, '');
#   if (!f.face_tracking_advanced && f.face_tracking){
#     $('rowFace').querySelector('.desc').textContent = 'Standard tracker active (install mediapipe for advanced mode)';
#   }
# }
# function toggleRow(rowId, chkId, available, msg){
#   if(!available){
#     $(rowId).classList.add('disabled');
#     $(chkId).checked = false;
#     $(chkId).disabled = true;
#     $(rowId).querySelector('.desc').textContent = msg;
#   }
# }
 
# // ── fill mode ──
# document.querySelectorAll('.fill-opt').forEach(el=>{
#   el.onclick = ()=>{
#     document.querySelectorAll('.fill-opt').forEach(o=>o.classList.remove('active'));
#     el.classList.add('active');
#     state.fillMode = el.dataset.fill;
#     drawCropForCurrentTime();
#   };
# });
 
# // ── fetch-from-link (built-in yt-dlp resolver, high quality single mp4) ──
# // Mirrors the Downloader tab's speed trick (cached auth mode) but skips the
# // full format-picker UI on purpose: this always grabs the single best
# // available video+audio quality and merges it into one mp4, then feeds it
# // straight into the exact same analyze/render pipeline as an upload.
# let fetchState = { url: null };
 
# $('fetchInfoBtn').onclick = async ()=>{
#   const url = $('fetchUrl').value.trim();
#   if(!url) return;
#   $('fetchStatus').textContent = 'Resolving link…';
#   $('fetchCard').style.display = 'none';
#   try{
#     const res = await fetch(api('/api/editor/fetch/info'), {
#       method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url})
#     });
#     const data = await res.json();
#     if(data.error){ $('fetchStatus').textContent = data.error; return; }
#     fetchState.url = url;
#     fetchState.meta = data;   // keep the FULL metadata (views/likes/uploader/etc.) for later
#     $('fetchThumb').src = data.thumbnail || '';
#     $('fetchTitle').textContent = data.title || 'Untitled video';
#     $('fetchSub').textContent = [data.uploader, data.duration_str, data.resolution].filter(Boolean).join(' · ');
#     $('fetchCard').style.display = 'flex';
#     $('fetchStatus').textContent = '';
#   }catch(e){
#     $('fetchStatus').textContent = 'Could not reach the server to resolve that link. Make sure the app is running and reload this page.';
#   }
# };
 
# $('fetchUseBtn').onclick = async ()=>{
#   $('fetchUseBtn').disabled = true;
#   $('fetchStatus').textContent = 'Fetching best quality (video + audio, single file)…';
#   try{
#     const res = await fetch(api('/api/editor/fetch/start'), {
#       method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url: fetchState.url})
#     });
#     const data = await res.json();
#     if(data.error){ $('fetchStatus').textContent = data.error; $('fetchUseBtn').disabled = false; return; }
#     pollFetchProgress(data.fetch_id);
#   }catch(e){
#     $('fetchStatus').textContent = 'Fetch failed to start — check the connection and try again.';
#     $('fetchUseBtn').disabled = false;
#   }
# };
 
# async function pollFetchProgress(fetchId){
#   let st;
#   try{
#     const res = await fetch(api('/api/editor/fetch/progress/'+fetchId));
#     st = await res.json();
#   }catch(e){
#     setTimeout(()=>pollFetchProgress(fetchId), 1200); // transient network hiccup, keep polling
#     return;
#   }
#   if(st.status === 'error'){
#     $('fetchStatus').textContent = 'Error: ' + st.error;
#     $('fetchUseBtn').disabled = false;
#     return;
#   }
#   if(st.status === 'done'){
#     $('fetchStatus').textContent = 'Downloaded — analyzing…';
#     $('fetchUseBtn').disabled = false;
#     $('fetchCard').style.display = 'none';
#     $('fetchUrl').value = '';
#     loadSource(st.source_path, fetchState.meta, 'fetch');
#     return;
#   }
#   const pct = st.percent != null ? st.percent + '%' : '';
#   $('fetchStatus').textContent = `${st.stage || 'downloading'}… ${pct} ${st.speed_str || ''}`.trim();
#   setTimeout(()=>pollFetchProgress(fetchId), 900);
# }
 
# // ── server library (downloader output + earlier uploads) ──
# // Lets the user reuse a video already fetched via the Downloader tab (or an
# // earlier upload) without picking it from disk again — same analyze/render
# // pipeline as a fresh upload, just a different source_path origin.
# $('refreshLibBtn').onclick = loadLibrary;
 
# async function loadLibrary(){
#   const grid = $('libGrid');
#   grid.innerHTML = '<span class="lib-empty">Loading…</span>';
#   try{
#     const res = await fetch(api('/api/editor/sources'));
#     const data = await res.json();
#     const items = data.sources || [];
#     if(!items.length){
#       grid.innerHTML = '<span class="lib-empty">Nothing yet — fetch a video in the Downloader tab, or upload one above.</span>';
#       return;
#     }
#     grid.innerHTML = '';
#     items.forEach(it=>{
#       const el = document.createElement('div');
#       el.className = 'lib-item';
#       el.innerHTML = `<span class="lib-tag">${it.origin === 'downloader' ? '📥 downloaded' : '📤 uploaded'}</span>
#         <div class="lib-name">${it.name}</div>
#         <div class="lib-meta"><span>${it.size_str||''}</span><span>${it.modified_str||''}</span></div>`;
#       el.onclick = ()=>{
#         document.querySelectorAll('.lib-item.active').forEach(n=>n.classList.remove('active'));
#         el.classList.add('active');
#         loadSource(it.path, null, it.origin === 'downloader' ? 'downloader' : 'pc');
#       };
#       grid.appendChild(el);
#     });
#   }catch(e){
#     grid.innerHTML = '<span class="lib-empty">Could not reach the server — make sure the app is running, then hit Refresh again.</span>';
#   }
# }
 
# loadLibrary();
 
# // ── upload flow ──
# $('dropzone').onclick = ()=> $('fileInput').click();
# $('fileInput').onchange = (e)=> handleFile(e.target.files[0]);
# ['dragover','dragleave','drop'].forEach(evt=>{
#   $('dropzone').addEventListener(evt, (e)=>{
#     e.preventDefault();
#     $('dropzone').classList.toggle('drag', evt==='dragover');
#     if(evt==='drop' && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
#   });
# });
 
# async function handleFile(file){
#   if(!file) return;
#   $('video').src = URL.createObjectURL(file);
#   $('previewSection').style.display = 'block';
#   $('statusLine').textContent = 'Uploading…';
#   try{
#     const fd = new FormData(); fd.append('file', file);
#     const res = await fetch(api('/api/editor/upload'), {method:'POST', body: fd});
#     const data = await res.json();
#     if(data.error){ showAnalysisError('Upload error: ' + data.error); return; }
#     loadSource(data.source_path, null, 'pc');
#   }catch(e){
#     showAnalysisError('Could not reach the server to upload this file.');
#   }
# }
 
# // ── unified source loader — every origin (fetched link, downloader
# // library, or a file picked from this PC) funnels through here so
# // analysis always runs the same way and the Export panel always ends up
# // enabled the same way, no matter where the video came from. ──
# async function loadSource(sourcePath, meta, origin){
#   state.sourcePath = sourcePath;
#   state.sourceMeta = meta || null;
#   state.sourceOrigin = origin || 'pc';
#   $('video').src = api('/api/editor/preview?path=' + encodeURIComponent(sourcePath));
#   $('previewSection').style.display = 'block';
#   renderSourceMeta();
#   await runAnalysis();
# }
 
# // Shows the rich details panel. For links fetched from YouTube/Instagram/
# // etc. this uses the metadata the SERVER already pulled during Fetch (no
# // need to re-derive title/uploader/views by hand); for PC uploads or
# // library items with no such metadata it just relies on the ffprobe-based
# // technical details that come back from /api/editor/analyze.
# function renderSourceMeta(){
#   const el = $('sourceMetaPanel');
#   if(!el) return;
#   const m = state.sourceMeta || {};
#   const rows = [
#     ['Title', m.title],
#     ['Channel / uploader', m.uploader],
#     ['Views', m.view_count_str],
#     ['Likes', m.like_count_str],
#     ['Uploaded', m.upload_date_str],
#     // Technical rows come from ffmpeg probing (_probe on the server) and are
#     // filled in for EVERY source — PC upload, Downloader-library pick, or a
#     // freshly fetched link — not just ones that happen to have yt-dlp
#     // metadata. This is what makes "same full details" true across origins.
#     ['Resolution', state.srcW && state.srcH ? `${state.srcW}×${state.srcH}` : null],
#     ['Duration', state.duration ? fmtTime(state.duration) : null],
#     ['Frame rate', state.srcFps ? `${state.srcFps} fps` : null],
#   ].filter(([,v]) => v);
#   if(!rows.length){ el.style.display = 'none'; el.innerHTML = ''; return; }
#   const heading = m.title ? 'ℹ️ Fetched video details' : 'ℹ️ Video details';
#   el.style.display = 'block';
#   el.innerHTML = `<div class="lib-header"><span>${heading}</span></div>` +
#     rows.map(([k,v]) => `<div class="status-line"><b>${k}:</b> ${v}</div>`).join('');
# }
 
# function showAnalysisError(msg){
#   $('statusLine').innerHTML = `<span style="color:var(--amber);">${msg}</span> ` +
#     `<button type="button" class="btn-small" id="retryAnalysisBtn" style="margin-left:6px;">Retry analysis</button>`;
#   const btn = $('retryAnalysisBtn');
#   if(btn) btn.onclick = () => runAnalysis();
#   $('renderBtn').disabled = true;
#   $('renderBtn').textContent = 'Analyze a video first';
# }
 
# async function runAnalysis(){
#   if(!state.sourcePath){ showAnalysisError('No video selected yet.'); return; }
#   $('renderBtn').disabled = true;
#   const startedAt = Date.now();
#   const tick = setInterval(()=>{
#     const secs = Math.round((Date.now()-startedAt)/1000);
#     $('statusLine').textContent = `Analyzing — face track + transcript + hook detection… (${secs}s`
#       + (secs > 20 ? ', first-time subtitle setup can take a couple minutes)' : ')');
#   }, 1000);
#   // Belt-and-braces client-side cap on top of the server's own 180s subtitle
#   // timeout: if the server itself never responds for any reason, this stops
#   // the button from staying disabled forever with no explanation.
#   const controller = new AbortController();
#   const abortTimer = setTimeout(()=>controller.abort(), 240000);
#   const body = {
#     source_path: state.sourcePath,
#     face_tracking: $('chkFace').checked && !$('chkFace').disabled,
#     subtitles: $('chkSubs').checked && !$('chkSubs').disabled,
#     hook_cut: true,
#     jump_cut: true,
#   };
#   let data;
#   try{
#     const res = await fetch(api('/api/editor/analyze'), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body), signal: controller.signal});
#     if(!res.ok){ showAnalysisError(`Analysis request failed (server said HTTP ${res.status}).`); return; }
#     data = await res.json();
#   }catch(e){
#     if(e.name === 'AbortError'){
#       showAnalysisError('Analysis took too long and was stopped — the server may still be finishing in the background. Wait a minute, then retry.');
#     } else {
#       showAnalysisError('Could not reach the server to analyze this video — check it is still running, then retry.');
#     }
#     return;
#   }finally{
#     clearInterval(tick);
#     clearTimeout(abortTimer);
#   }
#   if(data.error){ showAnalysisError('Analysis error: ' + data.error); return; }
#   state.duration = data.probe.duration || 0;
#   state.srcW = (data.source_size && data.source_size.w) || data.probe.width;
#   state.srcH = (data.source_size && data.source_size.h) || data.probe.height;
#   state.srcFps = data.probe.fps ? Math.round(data.probe.fps * 100) / 100 : null;
#   renderSourceMeta();
#   state.faceTrack = data.face_track || [];
#   state.words = data.words || [];
#   state.suggestedHook = data.suggested_hook || null;
#   state.suggestedSilences = data.suggested_segments || null;
#   $('useHookBtn').style.display = state.suggestedHook ? 'inline-block' : 'none';
#   $('useSilenceBtn').style.display = state.suggestedSilences ? 'inline-block' : 'none';
#   drawTimeline();
#   // Export becomes available the instant analysis finishes, identically
#   // for an uploaded PC file, a Downloader-library pick, or a fresh link
#   // fetch — there is no special-casing by source here. This is true even
#   // if one sub-feature (e.g. subtitles) timed out above — analyze always
#   // returns, so render is never blocked by one slow/failed piece.
#   $('renderBtn').disabled = false;
#   $('renderBtn').textContent = 'Render exports';
#   const srcTag = state.sourceOrigin === 'fetch' ? 'fetched video' : (state.sourceOrigin === 'downloader' ? 'downloaded video' : 'uploaded video');
#   const warnings = [data.subtitle_error, data.face_tracking_error, data.hook_cut_error, data.jump_cut_error].filter(Boolean);
#   let line = `Analyzed ${srcTag} — <b>${state.faceTrack.length}</b> face samples · <b>${state.words.length}</b> words transcribed · duration <b>${fmtTime(state.duration)}</b> · source <b>${state.srcW}×${state.srcH}</b>`;
#   if(warnings.length) line += ` <span style="color:var(--amber);">(${warnings.join(' · ')})</span>`;
#   $('statusLine').innerHTML = line;
# }
 
# // ── video controls ──
# const video = $('video');
# $('playBtn').onclick = ()=>{ video.paused ? video.play() : video.pause(); };
# video.onplay = ()=> $('playBtn').textContent = '❚❚';
# video.onpause = ()=> $('playBtn').textContent = '▶';
# video.ontimeupdate = ()=>{
#   $('timeLabel').textContent = `${fmtTime(video.currentTime)} / ${fmtTime(video.duration||0)}`;
#   const pct = video.duration ? (video.currentTime/video.duration*100) : 0;
#   $('playhead').style.left = pct + '%';
#   drawCropForCurrentTime();
# };
# function fmtTime(t){ t=Math.max(0,t||0); const m=Math.floor(t/60), s=Math.floor(t%60); return `${m}:${String(s).padStart(2,'0')}`; }
 
# // ── crop overlay: shows the current auto/manual crop box, draggable ──
# function currentCrop(){
#   const kfs = state.manualKeyframes.length ? state.manualKeyframes : state.faceTrack.map(p=>({t:p.t, x:p.cx, y:p.cy}));
#   if(!kfs.length) return {cx:0.5, cy:0.5};
#   const t = video.currentTime;
#   let lo = kfs[0], hi = kfs[kfs.length-1];
#   for(let i=0;i<kfs.length-1;i++){ if(kfs[i].t<=t && kfs[i+1].t>=t){ lo=kfs[i]; hi=kfs[i+1]; break; } }
#   const span = (hi.t - lo.t) || 1;
#   const f = Math.min(1, Math.max(0, (t - lo.t)/span));
#   const cxKey = state.manualKeyframes.length ? 'x' : 'cx';
#   const cyKey = state.manualKeyframes.length ? 'y' : 'cy';
#   return { cx: lo[cxKey] + (hi[cxKey]-lo[cxKey])*f, cy: lo[cyKey] + (hi[cyKey]-lo[cyKey])*f };
# }
 
# function drawCropForCurrentTime(){
#   const box = $('cropBox');
#   if(state.fillMode === 'blur_pad' || !$('chkFace').checked){ box.style.display='none'; return; }
#   if(!state.srcW || !video.videoWidth){ return; }
#   const meta = RATIO_META[state.previewRatio || '9:16'];
#   const targetAR = meta.w/meta.h;
#   const srcAR = video.videoWidth/video.videoHeight;
#   let cropWFrac, cropHFrac;
#   if(srcAR > targetAR){ cropHFrac = 1; cropWFrac = targetAR/srcAR; }
#   else { cropWFrac = 1; cropHFrac = srcAR/targetAR; }
#   const {cx, cy} = currentCrop();
#   let left = cx - cropWFrac/2, top = cy - cropHFrac/2;
#   left = Math.min(1-cropWFrac, Math.max(0, left));
#   top = Math.min(1-cropHFrac, Math.max(0, top));
 
#   const frame = $('previewFrame').getBoundingClientRect();
#   box.style.display = 'block';
#   box.style.left = (left*100)+'%';
#   box.style.top = (top*100)+'%';
#   box.style.width = (cropWFrac*100)+'%';
#   box.style.height = (cropHFrac*100)+'%';
#   $('cropReadout').textContent = state.previewRatio || '9:16';
#   box.dataset.cx = cx; box.dataset.cy = cy;
# }
 
# // dragging the crop box records a manual keyframe candidate (committed via "pin crop here")
# let dragStart = null;
# $('cropBox').addEventListener('mousedown', (e)=>{
#   state.dragging = true; $('cropBox').classList.add('dragging');
#   dragStart = {x:e.clientX, y:e.clientY, left:parseFloat($('cropBox').style.left), top:parseFloat($('cropBox').style.top)};
# });
# window.addEventListener('mousemove', (e)=>{
#   if(!state.dragging) return;
#   const frame = $('previewFrame').getBoundingClientRect();
#   const dx = (e.clientX-dragStart.x)/frame.width*100;
#   const dy = (e.clientY-dragStart.y)/frame.height*100;
#   const box = $('cropBox');
#   const newLeft = Math.min(100-parseFloat(box.style.width), Math.max(0, dragStart.left+dx));
#   const newTop = Math.min(100-parseFloat(box.style.height), Math.max(0, dragStart.top+dy));
#   box.style.left = newLeft+'%'; box.style.top = newTop+'%';
#   box.dataset.cx = (newLeft + parseFloat(box.style.width)/2)/100;
#   box.dataset.cy = (newTop + parseFloat(box.style.height)/2)/100;
# });
# window.addEventListener('mouseup', ()=>{ state.dragging=false; $('cropBox').classList.remove('dragging'); });
 
# $('addKfBtn').onclick = ()=>{
#   const box = $('cropBox');
#   const meta = RATIO_META[state.previewRatio || '9:16'];
#   const srcAR = video.videoWidth/video.videoHeight;
#   const targetAR = meta.w/meta.h;
#   let cropW, cropH;
#   if(srcAR > targetAR){ cropH = state.srcH; cropW = Math.round(cropH*targetAR); }
#   else { cropW = state.srcW; cropH = Math.round(cropW/targetAR); }
#   const cx = parseFloat(box.dataset.cx||0.5), cy = parseFloat(box.dataset.cy||0.5);
#   let x = Math.round(cx*state.srcW - cropW/2), y = Math.round(cy*state.srcH - cropH/2);
#   x = Math.max(0, Math.min(x, state.srcW-cropW)); y = Math.max(0, Math.min(y, state.srcH-cropH));
#   state.manualKeyframes = state.manualKeyframes.filter(k=>Math.abs(k.t-video.currentTime)>0.05);
#   state.manualKeyframes.push({t: video.currentTime, x, y});
#   state.manualKeyframes.sort((a,b)=>a.t-b.t);
#   drawTimeline();
#   $('statusLine').textContent = `Pinned manual crop at ${fmtTime(video.currentTime)} — ${state.manualKeyframes.length} manual keyframe(s) active (overrides auto tracking).`;
# };
# $('clearKfBtn').onclick = ()=>{ state.manualKeyframes=[]; drawTimeline(); drawCropForCurrentTime(); };
 
# // ── timeline: face density, segments, hook suggestion ──
# function drawTimeline(){
#   const tl = $('timeline');
#   [...tl.querySelectorAll('.tl-face,.tl-hook')].forEach(e=>e.remove());
#   if(state.duration){
#     state.faceTrack.forEach(p=>{
#       const el = document.createElement('div');
#       el.className='tl-face';
#       el.style.left = (p.t/state.duration*100)+'%'; el.style.width='2px';
#       tl.appendChild(el);
#     });
#     if(state.suggestedHook){
#       const el = document.createElement('div'); el.className='tl-hook';
#       el.style.left = (state.suggestedHook.start/state.duration*100)+'%';
#       el.style.width = ((state.suggestedHook.end-state.suggestedHook.start)/state.duration*100)+'%';
#       tl.appendChild(el);
#     }
#   }
#   renderSegList();
# }
# function renderSegList(){
#   const list = $('segList'); list.innerHTML='';
#   state.segments.forEach((seg, i)=>{
#     const row = document.createElement('div'); row.className='seg-row';
#     row.innerHTML = `<span class="mono">#${i+1}</span><span class="mono">${fmtTime(seg.start)} → ${fmtTime(seg.end)}</span><span class="spacer"></span><button data-i="${i}">✕</button>`;
#     row.querySelector('button').onclick = ()=>{ state.segments.splice(i,1); renderSegList(); };
#     list.appendChild(row);
#   });
# }
# $('timeline').addEventListener('click', (e)=>{
#   const rect = $('timeline').getBoundingClientRect();
#   const frac = (e.clientX-rect.left)/rect.width;
#   video.currentTime = frac * (video.duration||0);
# });
# $('addSegBtn').onclick = ()=>{
#   const t = video.currentTime;
#   state.segments.push({start: Math.max(0, +(t-3).toFixed(2)), end: Math.min(state.duration, +(t+3).toFixed(2))});
#   renderSegList();
# };
# $('useHookBtn').onclick = ()=>{
#   if(!state.suggestedHook) return;
#   state.segments = [state.suggestedHook, {start:0, end: state.duration}];
#   renderSegList();
#   $('chkHook').checked = true;
# };
# $('useSilenceBtn').onclick = ()=>{
#   if(!state.suggestedSilences) return;
#   state.segments = state.suggestedSilences;
#   renderSegList();
#   $('chkJump').checked = true;
# };
 
# // ── render ──
# $('renderBtn').onclick = async ()=>{
#   if(!state.selectedRatios.size){ $('statusLine').textContent = 'Pick at least one export ratio first.'; return; }
#   $('renderBtn').disabled = true;
#   $('progressWrap').style.display = 'block';
#   $('results').innerHTML = '';
#   const body = {
#     source_path: state.sourcePath,
#     export_ratios: Array.from(state.selectedRatios),
#     face_tracking: $('chkFace').checked && !$('chkFace').disabled,
#     subtitles: $('chkSubs').checked && !$('chkSubs').disabled,
#     hook_cut: $('chkHook').checked,
#     jump_cut: $('chkJump').checked,
#     normalize_audio: $('chkNorm').checked,
#     caption_style: state.selectedStyle,
#     fill_mode: state.fillMode,
#     quality: $('qualitySel').value,
#     manual_crop_keyframes: state.manualKeyframes.length ? state.manualKeyframes : null,
#     segments: state.segments.length ? state.segments : null,
#   };
#   try{
#     const res = await fetch(api('/api/editor/render'), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
#     const data = await res.json();
#     if(data.error){ $('statusLine').textContent = 'Render error: '+data.error; $('renderBtn').disabled=false; return; }
#     state.jobId = data.job_id;
#     pollProgress();
#   }catch(e){
#     $('statusLine').textContent = 'Could not reach the server to start the render — check it is still running and try again.';
#     $('renderBtn').disabled = false;
#   }
# };
 
# async function pollProgress(){
#   let job;
#   try{
#     const res = await fetch(api('/api/editor/progress/'+state.jobId));
#     job = await res.json();
#   }catch(e){
#     setTimeout(pollProgress, 1500); // transient hiccup, keep polling
#     return;
#   }
#   $('progressFill').style.width = (job.percent||0)+'%';
#   $('progressStage').textContent = job.stage || job.status;
#   $('progressPct').textContent = (job.percent||0)+'%';
#   if(job.status === 'error'){
#     $('statusLine').textContent = 'Render failed: '+job.error;
#     $('renderBtn').disabled = false;
#     return;
#   }
#   if(job.status !== 'done'){ setTimeout(pollProgress, 1200); return; }
#   $('renderBtn').disabled = false;
#   const results = $('results');
#   Object.entries(job.ratios||{}).forEach(([ratio, fname])=>{
#     const row = document.createElement('div'); row.className='result-row';
#     const tag = ratio === 'auto' ? 'Auto (original size — lossless)' : `${ratio} export`;
#     row.innerHTML = `<span class="name">${tag}</span><a href="${api('/api/editor/file/'+state.jobId+'/'+ratio)}" download>Download</a>`;
#     results.appendChild(row);
#   });
# }
 
# loadMeta();
# </script>
# </body>
# </html>
# """
 
 
# # Optional, best-effort link to downloader.py's in-memory job table so the
# # editor can show nicer names for downloaded videos ("Video Title.mp4"
# # instead of "<dl_id>_Video Title.mp4"). video_editor.py still works fine
# # standalone if downloader.py isn't registered — this never raises.
# def _downloader_jobs():
#     try:
#         import downloader
#         return downloader.DL_JOBS
#     except Exception:
#         return {}
 
# # ── configured once via init_editor() ──────────────────────────────
# EDIT_DIR = None
# SRC_DIR = None
# FFMPEG_PATH = "ffmpeg"
# FFPROBE_PATH = "ffprobe"
 
# # job_id -> {status, stage, percent, error, ratios: {ratio: filename}, config}
# EDIT_JOBS = {}
 
 
# def init_editor(base_dir, ffmpeg_path=None):
#     """Call once at startup, e.g. init_editor(BASE, FFMPEG)."""
#     global EDIT_DIR, SRC_DIR, FFMPEG_PATH, FFPROBE_PATH
#     base_dir = Path(base_dir)
#     EDIT_DIR = base_dir / "edited"
#     EDIT_DIR.mkdir(exist_ok=True)
#     SRC_DIR = base_dir / "downloads"   # reuse downloader's output as source pool
#     (EDIT_DIR / "uploads").mkdir(exist_ok=True)
#     if ffmpeg_path:
#         FFMPEG_PATH = str(ffmpeg_path)
#         # ffprobe normally sits next to ffmpeg
#         cand = Path(ffmpeg_path).with_name(
#             "ffprobe.exe" if str(ffmpeg_path).lower().endswith(".exe") else "ffprobe"
#         )
#         if cand.exists():
#             FFPROBE_PATH = str(cand)

#     # Pre-warm faster-whisper's model in the background at startup instead of
#     # on the user's first Analyze click — its first run downloads a
#     # multi-hundred-MB model from Hugging Face, which is exactly what made
#     # the very first analyze on a fresh install look "stuck forever". Doing
#     # it here means that download happens once, quietly, while the app is
#     # starting up, and every Analyze click afterward is fast.
#     if _HAS_WHISPER:
#         def _prewarm():
#             try:
#                 _load_whisper("small")
#             except Exception:
#                 pass  # subtitles will just be unavailable until this succeeds later
#         threading.Thread(target=_prewarm, daemon=True).start()
 
 
# def _no_console_kwargs():
#     if os.name == "nt":
#         return {"creationflags": subprocess.CREATE_NO_WINDOW}
#     return {}
 
 
# def _run(cmd):
#     return subprocess.run(cmd, capture_output=True, text=True, **_no_console_kwargs())
 
 
# # ══════════════════════════════ feature availability ══════════════════════════════
 
# try:
#     import cv2
#     _HAS_CV2 = True
# except ImportError:
#     _HAS_CV2 = False
 
# try:
#     import mediapipe as mp
#     _HAS_MEDIAPIPE = True
#     # Importing successfully doesn't mean `mp.solutions.face_detection` still
#     # exists — recent mediapipe releases dropped the legacy `solutions` API
#     # in favor of the newer Tasks API. Probe it once here so feature_status()
#     # (and every caller) reports the truth instead of assuming "advanced" is
#     # available and crashing later inside _FaceTracker.
#     try:
#         mp.solutions.face_detection.FaceDetection
#         _HAS_MEDIAPIPE_SOLUTIONS = True
#     except AttributeError:
#         _HAS_MEDIAPIPE_SOLUTIONS = False
# except ImportError:
#     _HAS_MEDIAPIPE = False
#     _HAS_MEDIAPIPE_SOLUTIONS = False
 
# try:
#     from faster_whisper import WhisperModel
#     _HAS_WHISPER = True
# except ImportError:
#     _HAS_WHISPER = False
 
# try:
#     import numpy as np
#     _HAS_NUMPY = True
# except ImportError:
#     _HAS_NUMPY = False
 
# try:
#     import yt_dlp
#     _HAS_YTDLP = True
# except ImportError:
#     _HAS_YTDLP = False
 
# _WHISPER_MODEL = None  # lazy-loaded singleton
 
 
# def feature_status():
#     return {
#         "face_tracking": _HAS_CV2,
#         "face_tracking_advanced": _HAS_CV2 and _HAS_MEDIAPIPE_SOLUTIONS,
#         "subtitles": _HAS_WHISPER,
#         "hook_cut_auto": _HAS_NUMPY,
#         "fetch_from_link": _HAS_YTDLP,
#     }
 
 
# # ══════════════════════════════ presets ══════════════════════════════
 
# # width x height for each export ratio, chosen at "good enough to look sharp,
# # small enough to encode fast" — bump these if you want 4K exports.
# RATIO_PRESETS = {
#     # "auto" is a sentinel: w/h are resolved at render time from the SOURCE
#     # video's own dimensions (no target size baked in here). It means "keep
#     # the video exactly as shot" — no crop, no letterbox, no rescale — and
#     # whenever nothing else in the pipeline needs to touch the picture
#     # (no captions/color/manual crop requested) it is exported via ffmpeg
#     # stream-copy, i.e. the original video bytes are re-muxed, not
#     # re-encoded, so there is truly zero generational quality loss.
#     "auto":  {"w": None, "h": None, "label": "Auto — original size, no crop, zero quality loss"},
#     "9:16":  {"w": 1080, "h": 1920, "label": "Reels / Shorts / TikTok"},
#     "16:9":  {"w": 1920, "h": 1080, "label": "YouTube / Landscape"},
#     "1:1":   {"w": 1080, "h": 1080, "label": "Square / Feed post"},
#     "4:5":   {"w": 1080, "h": 1350, "label": "Instagram portrait"},
#     "4:3":   {"w": 1440, "h": 1080, "label": "Classic / Facebook"},
# }
 
# QUALITY_PRESETS = {
#     "high":   {"crf": 18, "preset": "slow"},
#     "medium": {"crf": 21, "preset": "medium"},
#     "fast":   {"crf": 23, "preset": "veryfast"},
# }
 
# # Words that get an automatic extra-emphasis color in captions even in
# # styles that don't do full word-by-word highlighting — numbers, money and
# # a short list of "power words" are what trending caption tools punch up.
# _EMPHASIS_PATTERN = None  # compiled lazily, see _is_emphasis_word()
# _POWER_WORDS = {
#     "free", "now", "never", "always", "secret", "insane", "crazy", "huge",
#     "warning", "stop", "wait", "new", "best", "worst", "you", "your",
# }
 
 
# def _is_emphasis_word(word):
#     import re
#     global _EMPHASIS_PATTERN
#     if _EMPHASIS_PATTERN is None:
#         _EMPHASIS_PATTERN = re.compile(r"[\d%$]|^\$")
#     w = word.strip(".,!?").lower()
#     return bool(_EMPHASIS_PATTERN.search(word)) or w in _POWER_WORDS
 
# # Trending caption styles. Each is a full ASS "Style:" line plus a couple of
# # behavior flags consumed by build_ass(). Colors are &HAABBGGRR (ASS order).
# CAPTION_STYLES = {
#     "bold_pop": {
#         "label": "Bold Pop (white + yellow highlight)",
#         "style": "Style: Default,Montserrat Black,84,&H00FFFFFF,&H0000D7FF,&H00101010,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,6,0,2,60,60,140,1",
#         "highlight_color": "&H0000D7FF",   # active word turns yellow/gold
#         "word_by_word": True,
#         "pop_scale": True,
#     },
#     "karaoke_classic": {
#         "label": "Karaoke Fill (progressive color wipe)",
#         "style": "Style: Default,Montserrat SemiBold,72,&H00FFFFFF,&H0000A5FF,&H00202020,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,4,0,2,60,60,150,1",
#         "highlight_color": "&H0000A5FF",
#         "word_by_word": True,
#         "karaoke_fill": True,
#     },
#     "neon_glow": {
#         "label": "Neon Glow (cyan outline pop)",
#         "style": "Style: Default,Montserrat Black,80,&H00FFFFFF,&H00FFF000,&H00902000,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,5,2,2,60,60,140,1",
#         "highlight_color": "&H00FFF000",
#         "word_by_word": True,
#         "pop_scale": True,
#     },
#     "minimal_clean": {
#         "label": "Minimal Clean (small centered white)",
#         "style": "Style: Default,Inter Medium,54,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
#                   "0,0,0,0,100,100,0,0,1,2,1,2,60,60,120,1",
#         "highlight_color": "&H00FFFFFF",
#         "word_by_word": False,
#         "pop_scale": False,
#     },
#     "creator_bold": {
#         "label": "Creator Bold (huge yellow, black outline)",
#         "style": "Style: Default,Montserrat Black,92,&H0000FFFF,&H000000FF,&H00000000,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,8,0,2,50,50,160,1",
#         "highlight_color": "&H000000FF",
#         "word_by_word": True,
#         "pop_scale": True,
#     },
# }
 
 
# @editor_bp.route("/api/editor/styles")
# def api_editor_styles():
#     return jsonify({k: {"label": v["label"]} for k, v in CAPTION_STYLES.items()})
 
 
# @editor_bp.route("/api/editor/ratios")
# def api_editor_ratios():
#     return jsonify(RATIO_PRESETS)
 
 
# @editor_bp.route("/api/editor/editor")
# def api_editor_page():
#     """Serves the built-in editor UI — no separate .html file to manage,
#     it's embedded in this module (EDITOR_HTML below) and rendered straight
#     from memory. Open this URL in a tab; its fetch calls are same-origin."""
#     from flask import Response
#     return Response(EDITOR_HTML, mimetype="text/html")
 
 
# def _fmt_size(n):
#     if not n:
#         return "0 B"
#     n = float(n)
#     for unit in ["B", "KB", "MB", "GB"]:
#         if n < 1024:
#             return f"{n:.1f} {unit}"
#         n /= 1024
#     return f"{n:.1f} TB"
 
 
# @editor_bp.route("/api/editor/sources")
# def api_editor_sources():
#     """Lists videos the editor can open WITHOUT a fresh upload: everything
#     downloader.py has saved to DL_DIR/downloads (so a just-fetched video can
#     be sent straight into the editor), plus anything uploaded here before.
#     Both origins return the exact same shape and both feed the exact same
#     source_path used by /analyze and /render — the editor doesn't care where
#     a file came from."""
#     import datetime
#     jobs = _downloader_jobs()
#     # dl_id -> nicer display title, when downloader.py is registered too
#     title_by_stub = {}
#     for job in jobs.values():
#         fname = job.get("filename")
#         if fname and "_" in fname:
#             stub = fname.split("_", 1)[0]
#             title_by_stub[stub] = fname.split("_", 1)[-1]
 
#     sources = []
#     for folder, origin in ((SRC_DIR, "downloader"), (EDIT_DIR / "uploads", "upload")):
#         if not folder or not folder.exists():
#             continue
#         for p in sorted(folder.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
#             if not p.is_file() or p.suffix.lower() not in (".mp4", ".mov", ".mkv", ".webm", ".m4v"):
#                 continue
#             stub = p.name.split("_", 1)[0]
#             display = title_by_stub.get(stub) or (p.name.split("_", 1)[-1] if "_" in p.name else p.name)
#             stat = p.stat()
#             sources.append({
#                 "name": display,
#                 "path": str(p.resolve()),
#                 "origin": origin,
#                 "size": stat.st_size,
#                 "size_str": _fmt_size(stat.st_size),
#                 "modified": stat.st_mtime,
#                 "modified_str": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%d %b, %H:%M"),
#             })
#     sources.sort(key=lambda s: s["modified"], reverse=True)
#     return jsonify({"sources": sources})
 
 
# def _is_allowed_source(path):
#     """Only ever stream files that live under the editor's own known roots
#     (downloader's downloads folder, this module's uploads folder, or its
#     own rendered-output folder) — never an arbitrary server path."""
#     try:
#         rp = Path(path).resolve()
#     except Exception:
#         return False
#     for root in (SRC_DIR, EDIT_DIR / "uploads", EDIT_DIR):
#         if root and root.exists():
#             try:
#                 rp.relative_to(root.resolve())
#                 return True
#             except ValueError:
#                 continue
#     return False
 
 
# @editor_bp.route("/api/editor/preview")
# def api_editor_preview():
#     """Streams a source (or rendered) video for <video> tag playback/scrub,
#     so picking a file from the library previews instantly without a
#     round-trip upload. Range requests are handled by Flask's send_file."""
#     path = request.args.get("path", "")
#     if not path or not _is_allowed_source(path) or not Path(path).exists():
#         return "Not found or not allowed", 404
#     return send_file(path, conditional=True)
 
 
# @editor_bp.route("/api/editor/features")
# def api_editor_features():
#     return jsonify(feature_status())
 
 
# # ══════════════════════════════ probing ══════════════════════════════
 
# def _probe(path):
#     """Returns {duration, width, height, fps} for any video — PC upload,
#     Downloader-library pick, or freshly fetched link — using ONLY ffmpeg
#     (FFMPEG_PATH), never ffprobe.

#     Why: this whole app is wired up with `FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()`
#     in the main file, which downloads/bundles an ffmpeg BINARY ONLY — it does
#     not ship ffprobe next to it, and a bare "ffprobe" is not guaranteed to be
#     on PATH on a fresh Windows machine (that mismatch is exactly what produced
#     `FileNotFoundError: [WinError 2] The system cannot find the file
#     specified` — the app was calling a program that doesn't exist on disk).
#     `ffmpeg -i <file>` already prints duration/resolution/fps to stderr for
#     every input, so we parse that instead — one binary, guaranteed present,
#     identical accurate metadata for every source, no more ffprobe dependency.
#     """
#     cmd = [FFMPEG_PATH, "-hide_banner", "-i", str(path)]
#     out = _run(cmd)
#     text = (out.stderr or "") + (out.stdout or "")

#     duration = 0.0
#     dm = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
#     if dm:
#         hh, mm, ss = dm.groups()
#         duration = int(hh) * 3600 + int(mm) * 60 + float(ss)

#     width = height = None
#     # e.g. "Stream #0:0: Video: h264 ..., yuv420p, 1080x1920 [SAR 1:1 DAR 9:16], 30 fps, 30 tbr, ..."
#     vm = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", text)
#     if vm:
#         width, height = int(vm.group(1)), int(vm.group(2))

#     fps = 25.0
#     fm = re.search(r"([\d.]+)\s+fps", text)
#     if fm:
#         try:
#             fps = float(fm.group(1))
#         except ValueError:
#             pass

#     return {"duration": duration, "width": width, "height": height, "fps": fps}
 
 
# # ══════════════════════════════ face tracking ══════════════════════════════
 
# class _FaceTracker:
#     """Samples frames, finds a face each time, and produces a smoothed
#     (jitter-free) track of the subject's center point over time.
 
#     Uses mediapipe's BlazeFace model when available ("advanced" mode — much
#     more accurate on angled/small faces), otherwise falls back to OpenCV's
#     bundled Haar cascade so face tracking still works with just opencv-python
#     installed.
#     """
 
#     def __init__(self):
#         # mediapipe importing successfully doesn't guarantee the legacy
#         # `mp.solutions.face_detection` API is present — recent mediapipe
#         # builds moved to a different Tasks API and dropped `.solutions`
#         # entirely, which used to crash every analyze request with
#         # `AttributeError: module 'mediapipe' has no attribute 'solutions'`.
#         # _HAS_MEDIAPIPE_SOLUTIONS was probed once at startup, so this just
#         # picks the mode that's actually usable instead of assuming.
#         self.advanced = _HAS_MEDIAPIPE_SOLUTIONS
#         if self.advanced:
#             self._mp_detector = mp.solutions.face_detection.FaceDetection(
#                 model_selection=1, min_detection_confidence=0.5
#             )
#         elif _HAS_CV2:
#             self._cascade = cv2.CascadeClassifier(
#                 cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#             )
#         else:
#             raise RuntimeError("opencv-python is required for face tracking")
 
#     def _detect(self, frame_bgr):
#         """Returns list of (cx, cy, w, h) in pixel coords, largest first."""
#         h_img, w_img = frame_bgr.shape[:2]
#         faces = []
#         if self.advanced:
#             rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#             result = self._mp_detector.process(rgb)
#             if result.detections:
#                 for d in result.detections:
#                     box = d.location_data.relative_bounding_box
#                     fw, fh = box.width * w_img, box.height * h_img
#                     fx = box.xmin * w_img + fw / 2
#                     fy = box.ymin * h_img + fh / 2
#                     faces.append((fx, fy, fw, fh))
#         else:
#             gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#             dets = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50))
#             for (x, y, w, h) in dets:
#                 faces.append((x + w / 2, y + h / 2, w, h))
#         faces.sort(key=lambda f: f[2] * f[3], reverse=True)  # largest face first
#         return faces
 
#     def track(self, video_path, sample_fps=2.0, progress_cb=None):
#         """Returns a list of {t, cx, cy, w, h} in NORMALIZED (0-1) coords,
#         exponentially smoothed so the crop pans instead of jumping."""
#         cap = cv2.VideoCapture(str(video_path))
#         if not cap.isOpened():
#             raise RuntimeError("Could not open video for face tracking")
#         src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#         w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         step = max(1, int(round(src_fps / sample_fps)))
 
#         raw_points = []
#         last_center = (0.5, 0.5)  # default: frame center when nobody's found yet
#         frame_idx = 0
#         while True:
#             ok = cap.grab()
#             if not ok:
#                 break
#             if frame_idx % step == 0:
#                 ok2, frame = cap.retrieve()
#                 if ok2:
#                     faces = self._detect(frame)
#                     if faces:
#                         fx, fy, fw, fh = faces[0]
#                         cx, cy = fx / w_img, fy / h_img
#                         last_center = (cx, cy)
#                     else:
#                         cx, cy = last_center  # hold last known position
#                     t = frame_idx / src_fps
#                     raw_points.append({"t": t, "cx": cx, "cy": cy})
#                 if progress_cb and total_frames:
#                     progress_cb(min(99, int(frame_idx * 100 / total_frames)))
#             frame_idx += 1
#         cap.release()
 
#         if not raw_points:
#             raw_points = [{"t": 0.0, "cx": 0.5, "cy": 0.5}]
 
#         # EMA smoothing so the crop glides instead of snapping every sample
#         alpha = 0.25
#         smoothed = [raw_points[0]]
#         for p in raw_points[1:]:
#             prev = smoothed[-1]
#             smoothed.append({
#                 "t": p["t"],
#                 "cx": prev["cx"] + alpha * (p["cx"] - prev["cx"]),
#                 "cy": prev["cy"] + alpha * (p["cy"] - prev["cy"]),
#             })
#         return smoothed, (w_img, h_img)
 
 
# def analyze_face_track(video_path, sample_fps=2.0, progress_cb=None):
#     if not _HAS_CV2:
#         raise RuntimeError("opencv-python is not installed — face tracking unavailable")
#     tracker = _FaceTracker()
#     points, (w, h) = tracker.track(video_path, sample_fps=sample_fps, progress_cb=progress_cb)
#     return points, w, h, tracker.advanced
 
 
# # ══════════════════════════════ crop-window math ══════════════════════════════
 
# def _crop_size_for_ratio(src_w, src_h, target_w, target_h):
#     """Largest crop rectangle matching the target aspect that still fits
#     inside the source frame (so we crop, never letterbox/pad)."""
#     target_ar = target_w / target_h
#     src_ar = src_w / src_h
#     if src_ar > target_ar:
#         # source is wider than target -> crop width, keep full height
#         crop_h = src_h
#         crop_w = int(round(crop_h * target_ar))
#     else:
#         # source is taller than target -> crop height, keep full width
#         crop_w = src_w
#         crop_h = int(round(crop_w / target_ar))
#     return crop_w, crop_h
 
 
# def _build_crop_keyframes(track_points, src_w, src_h, crop_w, crop_h):
#     """Turns normalized face-center points into pixel-space crop x/y
#     top-left keyframes, clamped so the crop never leaves the frame."""
#     kfs = []
#     for p in track_points:
#         cx_px = p["cx"] * src_w
#         cy_px = p["cy"] * src_h
#         x = int(cx_px - crop_w / 2)
#         y = int(cy_px - crop_h / 2)
#         x = max(0, min(x, src_w - crop_w))
#         y = max(0, min(y, src_h - crop_h))
#         kfs.append({"t": p["t"], "x": x, "y": y})
#     return kfs
 
 
# def _expr_chain(keyframes, key):
#     """Builds an ffmpeg time-expression string that linearly interpolates
#     between keyframes for either 'x' or 'y' — this drives a moving crop
#     window without needing any external filter graph patching."""
#     if len(keyframes) == 1:
#         return str(keyframes[0][key])
#     expr = str(keyframes[-1][key])
#     for i in range(len(keyframes) - 2, -1, -1):
#         t0, v0 = keyframes[i]["t"], keyframes[i][key]
#         t1, v1 = keyframes[i + 1]["t"], keyframes[i + 1][key]
#         if t1 <= t0:
#             continue
#         # linear ramp between (t0,v0) and (t1,v1), holds v0 before t0
#         ramp = f"({v0}+({v1}-{v0})*(t-{t0})/{(t1 - t0)})"
#         expr = f"if(lt(t,{t1}),{ramp},{expr})"
#     return expr
 
 
# # ══════════════════════════════ subtitles ══════════════════════════════
 
# def _load_whisper(model_size="small"):
#     global _WHISPER_MODEL
#     if _WHISPER_MODEL is None:
#         _WHISPER_MODEL = WhisperModel(model_size, compute_type="int8")
#     return _WHISPER_MODEL
 
 
# def transcribe_words(video_path, model_size="small", language=None):
#     """Returns [{start, end, text}, ...] word-level timestamps."""
#     if not _HAS_WHISPER:
#         raise RuntimeError("faster-whisper is not installed — subtitles unavailable")
#     model = _load_whisper(model_size)
#     segments, _info = model.transcribe(str(video_path), word_timestamps=True, language=language)
#     words = []
#     for seg in segments:
#         for w in (seg.words or []):
#             text = (w.word or "").strip()
#             if text:
#                 words.append({"start": w.start, "end": w.end, "text": text})
#     return words
 
 
# def _chunk_words(words, max_words=4, max_span=1.6):
#     """Groups words into short on-screen caption chunks — the standard
#     'trending shorts' style of 2-5 words on screen at a time, not full
#     sentences."""
#     chunks = []
#     cur = []
#     for w in words:
#         if cur and (len(cur) >= max_words or (w["end"] - cur[0]["start"]) > max_span):
#             chunks.append(cur)
#             cur = []
#         cur.append(w)
#     if cur:
#         chunks.append(cur)
#     return chunks
 
 
# def _ass_time(t):
#     h = int(t // 3600)
#     m = int((t % 3600) // 60)
#     s = t % 60
#     return f"{h}:{m:02d}:{s:05.2f}"
 
 
# def build_ass(words, style_name, video_w, video_h):
#     """Builds a full .ass subtitle document with word-by-word highlight
#     animation for the chosen trending style, sized for the given output
#     resolution so text scales correctly across aspect ratios."""
#     style = CAPTION_STYLES.get(style_name, CAPTION_STYLES["bold_pop"])
#     header = f"""[Script Info]
# ScriptType: v4.00+
# PlayResX: {video_w}
# PlayResY: {video_h}
# ScaledBorderAndShadow: yes
 
# [V4+ Styles]
# Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
# {style["style"]}
 
# [Events]
# Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
# """
#     lines = []
#     for chunk in _chunk_words(words):
#         start, end = chunk[0]["start"], chunk[-1]["end"]
#         if style.get("karaoke_fill"):
#             # \k tags: whole line present, active word wipes to highlight color
#             text = "".join(
#                 r"{\k%d}%s " % (max(1, int(round((w["end"] - w["start"]) * 100))), w["text"])
#                 for w in chunk
#             )
#             lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
#         elif style.get("word_by_word"):
#             # emit one Dialogue event per word so only the current word is
#             # shown in the highlight color while the rest stay default —
#             # gives the punchy "pop" caption look trending on shorts.
#             for w in chunk:
#                 parts = []
#                 for w2 in chunk:
#                     if w2 is w:
#                         scale = r"\fscx115\fscy115" if style.get("pop_scale") else ""
#                         parts.append(r"{\c%s%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], scale, w2["text"]))
#                     elif _is_emphasis_word(w2["text"]):
#                         # numbers / money / power-words get punched up even
#                         # when they're not the "active" word of the moment
#                         parts.append(r"{\c%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], w2["text"]))
#                     else:
#                         parts.append(w2["text"])
#                 text = " ".join(parts)
#                 lines.append(f"Dialogue: 0,{_ass_time(w['start'])},{_ass_time(w['end'])},Default,,0,0,0,,{text}")
#         else:
#             parts = []
#             for w2 in chunk:
#                 if _is_emphasis_word(w2["text"]):
#                     parts.append(r"{\c%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], w2["text"]))
#                 else:
#                     parts.append(w2["text"])
#             text = " ".join(parts)
#             lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
#     return header + "\n".join(lines) + "\n"
 
 
# # ══════════════════════════════ hook-cut / segments ══════════════════════════════
 
# def _read_pcm_mono(video_path, sr=8000):
#     """Decodes a fast, low-res mono PCM stream via ffmpeg for lightweight
#     loudness analysis (used only to pick the auto hook window)."""
#     cmd = [FFMPEG_PATH, "-v", "error", "-i", str(video_path),
#            "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
#     proc = subprocess.run(cmd, capture_output=True, **_no_console_kwargs())
#     raw = proc.stdout
#     count = len(raw) // 2
#     samples = struct.unpack(f"<{count}h", raw[:count * 2])
#     return np.array(samples, dtype=np.float32) / 32768.0, sr
 
 
# def auto_hook_window(video_path, duration, hook_len=6.0):
#     """Finds the loudest `hook_len`-second window in the clip — a cheap but
#     effective proxy for 'most energetic / most likely to hook a viewer'
#     moment, used to auto-suggest where the exported clip should start."""
#     if not _HAS_NUMPY:
#         raise RuntimeError("numpy is not installed — auto hook-cut unavailable")
#     audio, sr = _read_pcm_mono(video_path)
#     if len(audio) == 0:
#         return {"start": 0.0, "end": min(hook_len, duration)}
#     win = int(hook_len * sr)
#     if win >= len(audio):
#         return {"start": 0.0, "end": duration}
#     energy = audio ** 2
#     # cumulative sum for O(1) windowed average lookups
#     cumsum = np.cumsum(np.insert(energy, 0, 0))
#     window_energy = cumsum[win:] - cumsum[:-win]
#     best_start_sample = int(np.argmax(window_energy))
#     start = best_start_sample / sr
#     return {"start": round(start, 2), "end": round(min(start + hook_len, duration), 2)}
 
 
# def detect_silences(video_path, noise_db=-30, min_silence=0.5):
#     """Runs ffmpeg's silencedetect and returns the list of NON-silent
#     {start,end} ranges — i.e. the segments worth keeping. This powers
#     automatic 'jump-cut' editing (removing dead air/pauses), the other
#     trending short-form edit style besides the loudest-window hook-cut."""
#     cmd = [FFMPEG_PATH, "-v", "info", "-i", str(video_path),
#            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
#     result = _run(cmd)
#     log = result.stderr or ""
#     starts, ends = [], []
#     for line in log.splitlines():
#         if "silence_start" in line:
#             starts.append(float(line.split("silence_start:")[1].strip().split(" ")[0]))
#         elif "silence_end" in line:
#             ends.append(float(line.split("silence_end:")[1].strip().split(" ")[0].split("|")[0]))
#     info = _probe(video_path)
#     duration = info["duration"]
#     silences = list(zip(starts, ends[: len(starts)]))
#     keep = []
#     cursor = 0.0
#     for s, e in silences:
#         if s > cursor:
#             keep.append({"start": round(cursor, 2), "end": round(s, 2)})
#         cursor = max(cursor, e)
#     if cursor < duration:
#         keep.append({"start": round(cursor, 2), "end": round(duration, 2)})
#     return [k for k in keep if k["end"] - k["start"] > 0.15]
 
 
# def _build_concat_file(video_path, segments, tmp_dir):
#     """Writes an ffmpeg concat-demuxer list after trimming each segment,
#     so multiple {start,end} ranges can be stitched in a custom, manually
#     chosen order (rearrange / hook-cut)."""
#     parts = []
#     for i, seg in enumerate(segments):
#         out = tmp_dir / f"seg_{i}.mp4"
#         cmd = [
#             FFMPEG_PATH, "-y", "-v", "error",
#             "-ss", str(seg["start"]), "-to", str(seg["end"]),
#             "-i", str(video_path),
#             "-c", "copy", "-avoid_negative_ts", "make_zero",
#             str(out),
#         ]
#         _run(cmd)
#         parts.append(out)
#     list_path = tmp_dir / "concat_list.txt"
#     with open(list_path, "w", encoding="utf-8") as f:
#         for p in parts:
#             f.write(f"file '{p.as_posix()}'\n")
#     return list_path
 
 
# def _measure_loudnorm(path):
#     """First pass of two-pass EBU R128 loudness normalization: measures the
#     input's actual loudness/true-peak/range so the second (real encode)
#     pass can normalize with `linear=true` against real measured values
#     instead of loudnorm's single-pass dynamic estimate — noticeably more
#     accurate and avoids the pumping/gain-riding artifacts single-pass mode
#     can introduce on speech-heavy short-form video."""
#     cmd = [FFMPEG_PATH, "-v", "info", "-i", str(path),
#            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
#            "-f", "null", "-"]
#     result = _run(cmd)
#     log = result.stderr or ""
#     try:
#         start = log.rindex("{")
#         end = log.rindex("}") + 1
#         return json.loads(log[start:end])
#     except (ValueError, json.JSONDecodeError):
#         return None  # falls back to single-pass below
 
 
# # ══════════════════════════════ render pipeline ══════════════════════════════
 
# def _render_one_ratio(job, source_path, ratio_key, track_points, src_w, src_h,
#                        words, tmp_dir, loud_measured=None):
#     cfg = job["config"]
#     ratio = RATIO_PRESETS[ratio_key]
#     is_auto = (ratio_key == "auto")
#     # "auto" keeps the SOURCE's own dimensions — no target size is imposed.
#     target_w, target_h = (src_w, src_h) if is_auto else (ratio["w"], ratio["h"])
#     quality = QUALITY_PRESETS.get(cfg.get("quality", "high"), QUALITY_PRESETS["high"])
 
#     fill_mode = cfg.get("fill_mode", "crop")  # "crop" (default) or "blur_pad"
#     # "auto" never crops or letterboxes — it IS the full original frame —
#     # so blur_pad/crop fill-mode choices simply don't apply to it.
#     if is_auto:
#         fill_mode = "none"
 
#     # ── subtitles: build the .ass first so both fill-mode branches can use it ──
#     ass_path = None
#     if cfg["features"].get("subtitles") and words:
#         ass_content = build_ass(words, cfg.get("caption_style", "bold_pop"), target_w, target_h)
#         ass_path = tmp_dir / f"subs_{ratio_key.replace(':', 'x')}.ass"
#         ass_path.write_text(ass_content, encoding="utf-8")
#         ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
 
#     if is_auto and not ass_path and not cfg.get("manual_crop_keyframes"):
#         # Nothing needs to touch the picture at all: fastest AND highest
#         # possible quality path is a pure stream-copy remux (no re-encode
#         # -> byte-for-byte original video quality; only trimmed/normalized
#         # audio may differ). This is what "zero quality loss" means here.
#         out_name = f"{job['job_id']}_{ratio_key.replace(':', 'x')}.mp4"
#         out_path = EDIT_DIR / out_name
#         if cfg.get("normalize_audio", True):
#             measured = loud_measured
#             audio_filter = (
#                 "loudnorm=I=-14:TP=-1.5:LRA=11:"
#                 f"measured_I={measured.get('input_i', -14)}:"
#                 f"measured_TP={measured.get('input_tp', -1.5)}:"
#                 f"measured_LRA={measured.get('input_lra', 11)}:"
#                 f"measured_thresh={measured.get('input_thresh', -24)}:"
#                 "linear=true:print_format=summary"
#             ) if measured else "loudnorm=I=-14:TP=-1.5:LRA=11"
#             cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path),
#                    "-c:v", "copy", "-af", audio_filter, "-c:a", "aac", "-b:a", "192k",
#                    "-movflags", "+faststart", str(out_path)]
#         else:
#             cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path),
#                    "-c", "copy", "-movflags", "+faststart", str(out_path)]
#         result = _run(cmd)
#         if result.returncode != 0:
#             raise RuntimeError(f"ffmpeg failed for {ratio_key}: {result.stderr[-800:]}")
#         return out_name
 
#     if fill_mode == "blur_pad":
#         # "Advanced" reframe mode: instead of cropping content away, fit the
#         # WHOLE frame inside the target canvas and fill the leftover bars
#         # with a blurred, zoomed copy of the same frame — the look used by
#         # most professional auto-reframe tools when a hard crop would cut
#         # off important content. No face tracking needed for this branch,
#         # since nothing is being cropped out.
#         fc = (
#             f"[0:v]split=2[bg][fg];"
#             f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
#             f"crop={target_w}:{target_h},gblur=sigma=25,eq=brightness=-0.05[bg2];"
#             f"[fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg2];"
#             f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2:format=auto[base]"
#         )
#         if ass_path:
#             fc += f",[base]ass='{ass_escaped}'[vout]"
#             map_v = "[vout]"
#         else:
#             fc += "[vout]"
#             map_v = "[vout]"
#         vf_args = ["-filter_complex", fc, "-map", map_v, "-map", "0:a?"]
#     elif is_auto:
#         # Captions and/or a manual crop keyframe are present, so a re-encode
#         # can't be avoided — but the frame itself is never cropped or
#         # rescaled; it stays at the source's own resolution. Quality is set
#         # one notch above "high" (crf 16, effectively visually lossless)
#         # specifically for this path so adding captions never visibly
#         # degrades the original picture.
#         filters = ["setsar=1"]
#         if ass_path:
#             filters.append(f"ass='{ass_escaped}'")
#         vf_args = ["-vf", ",".join(filters)]
#         quality = {"crf": 16, "preset": quality["preset"]}
#     else:
#         filters = []
#         # ── crop (face-tracked or manual) ──
#         if cfg["features"].get("face_tracking") or cfg.get("manual_crop_keyframes"):
#             crop_w, crop_h = _crop_size_for_ratio(src_w, src_h, target_w, target_h)
#             if cfg.get("manual_crop_keyframes"):
#                 kfs = sorted(cfg["manual_crop_keyframes"], key=lambda k: k["t"])
#             else:
#                 kfs = _build_crop_keyframes(track_points, src_w, src_h, crop_w, crop_h)
#             x_expr = _expr_chain(kfs, "x")
#             y_expr = _expr_chain(kfs, "y")
#             filters.append(f"crop={crop_w}:{crop_h}:'{x_expr}':'{y_expr}'")
#         else:
#             # no face tracking requested -> simple centered crop to the target ratio
#             crop_w, crop_h = _crop_size_for_ratio(src_w, src_h, target_w, target_h)
#             filters.append(f"crop={crop_w}:{crop_h}:(in_w-out_w)/2:(in_h-out_h)/2")
 
#         filters.append(f"scale={target_w}:{target_h}:flags=lanczos")
#         filters.append("setsar=1")
#         if ass_path:
#             filters.append(f"ass='{ass_escaped}'")
#         vf_args = ["-vf", ",".join(filters)]
 
#     out_name = f"{job['job_id']}_{ratio_key.replace(':', 'x')}.mp4"
#     out_path = EDIT_DIR / out_name
 
#     # Loudness normalization (EBU R128, -14 LUFS is the standard target for
#     # social platforms) so exported audio doesn't sound quiet/inconsistent
#     # next to other content in-feed. On "high" quality we do a real
#     # measure-then-normalize TWO-PASS pass for accuracy; cheaper qualities
#     # use fast single-pass so exports stay quick.
#     audio_filters = []
#     if cfg.get("normalize_audio", True):
#         measured = loud_measured
#         if measured:
#             audio_filters = [
#                 "loudnorm=I=-14:TP=-1.5:LRA=11:"
#                 f"measured_I={measured.get('input_i', -14)}:"
#                 f"measured_TP={measured.get('input_tp', -1.5)}:"
#                 f"measured_LRA={measured.get('input_lra', 11)}:"
#                 f"measured_thresh={measured.get('input_thresh', -24)}:"
#                 "linear=true:print_format=summary"
#             ]
#         else:
#             audio_filters = ["loudnorm=I=-14:TP=-1.5:LRA=11"]
 
#     cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path), *vf_args]
#     if audio_filters:
#         cmd += ["-af", ",".join(audio_filters)]
#     cmd += [
#         "-c:v", "libx264", "-crf", str(quality["crf"]), "-preset", quality["preset"],
#         "-c:a", "aac", "-b:a", "192k",
#         "-movflags", "+faststart",
#         str(out_path),
#     ]
#     result = _run(cmd)
#     if result.returncode != 0:
#         raise RuntimeError(f"ffmpeg failed for {ratio_key}: {result.stderr[-800:]}")
#     return out_name
 
 
# def _run_render_job(job_id):
#     job = EDIT_JOBS[job_id]
#     cfg = job["config"]
#     tmp_dir = EDIT_DIR / f"tmp_{job_id}"
#     tmp_dir.mkdir(exist_ok=True)
 
#     try:
#         source_path = Path(cfg["source_path"])
#         if not source_path.exists():
#             raise RuntimeError("Source video not found")
 
#         # ── 1. hook-cut / manual rearrange: trim+stitch segments first ──
#         job["stage"] = "segments"
#         segments = cfg.get("segments")
#         want_jump = cfg["features"].get("jump_cut")
#         want_hook = cfg["features"].get("hook_cut")
#         if not segments and (want_jump or want_hook):
#             info = _probe(source_path)
#             jump_segments = detect_silences(source_path) if want_jump else None
#             if want_hook:
#                 hook = auto_hook_window(source_path, info["duration"], cfg.get("hook_len", 6.0))
#                 # Cold-open on the loudest window, then play the rest of the
#                 # clip — dead-air-trimmed if jump_cut is also on, otherwise
#                 # the full original timeline. Combining both no longer means
#                 # jump_cut silently wins.
#                 rest = jump_segments if jump_segments else [{"start": 0.0, "end": info["duration"]}]
#                 segments = [hook] + rest
#             else:
#                 segments = jump_segments
#         if segments:
#             list_path = _build_concat_file(source_path, segments, tmp_dir)
#             stitched = tmp_dir / "stitched.mp4"
#             _run([FFMPEG_PATH, "-y", "-v", "error", "-f", "concat", "-safe", "0",
#                   "-i", str(list_path), "-c", "copy", str(stitched)])
#             source_path = stitched
#         job["percent"] = 10
 
#         info = _probe(source_path)
#         src_w, src_h = info["width"], info["height"]
 
#         # ── 2. face tracking (once, reused for every export ratio) ──
#         job["stage"] = "face_tracking"
#         track_points = []
#         if cfg["features"].get("face_tracking"):
#             def cb(pct):
#                 job["percent"] = 10 + int(pct * 0.3)
#             track_points, src_w, src_h, advanced = analyze_face_track(
#                 source_path, sample_fps=cfg.get("track_fps", 2.0), progress_cb=cb
#             )
#             job["face_tracking_mode"] = "advanced (mediapipe)" if advanced else "standard (haar cascade)"
#         job["percent"] = 40
 
#         # ── 3. subtitles (once, reused for every export ratio) ──
#         job["stage"] = "subtitles"
#         words = []
#         if cfg["features"].get("subtitles"):
#             words = transcribe_words(source_path, model_size=cfg.get("whisper_model", "small"),
#                                       language=cfg.get("language"))
#         job["percent"] = 60
 
#         # ── 4. render each requested aspect ratio as one final file ──
#         job["stage"] = "render"
#         ratios = cfg.get("export_ratios") or ["9:16"]
#         job["ratios"] = {}
#         n = len(ratios)
 
#         # measured once (not per-ratio, source audio is identical across
#         # ratios) and only for "high" quality, where the extra ffmpeg pass
#         # is worth the accuracy; cheaper qualities use fast single-pass.
#         loud_measured = None
#         if cfg.get("normalize_audio", True) and cfg.get("quality", "high") == "high":
#             loud_measured = _measure_loudnorm(source_path)
 
#         for i, ratio_key in enumerate(ratios):
#             if ratio_key not in RATIO_PRESETS:
#                 continue
#             out_name = _render_one_ratio(job, source_path, ratio_key, track_points,
#                                           src_w, src_h, words, tmp_dir, loud_measured)
#             job["ratios"][ratio_key] = out_name
#             job["percent"] = 60 + int((i + 1) * 40 / max(1, n))
 
#         job["status"] = "done"
#         job["stage"] = "done"
#         job["percent"] = 100
#     except Exception as e:
#         job["status"] = "error"
#         job["error"] = str(e)
#     finally:
#         # scratch files (trimmed segments, stitched source, .ass files) —
#         # keep only the final per-ratio outputs in EDIT_DIR
#         try:
#             for f in tmp_dir.glob("*"):
#                 f.unlink(missing_ok=True)
#             tmp_dir.rmdir()
#         except OSError:
#             pass
 
 
# # ══════════════════════════════ routes ══════════════════════════════
 
# @editor_bp.route("/api/editor/analyze", methods=["POST"])
# def api_editor_analyze():
#     """Runs face tracking + transcription WITHOUT rendering, so a frontend
#     can show a preview / let the user manually adjust crop keyframes or
#     hook segments before committing to a full render.

#     Every sub-feature below is individually wrapped in try/except (and, for
#     subtitles specifically, a hard time limit). Before this, a single slow
#     or failing piece — most commonly faster-whisper's first-run model
#     download hanging on a slow/no-token connection — silently hung the
#     ENTIRE request forever: no response, no error, nothing in the server
#     log, and the "Analyze a video first" button on the frontend stayed
#     disabled with no way to know why. Now the endpoint always returns
#     within a bounded time with whatever it managed to compute, plus an
#     "*_error" field per feature that failed, so the button reliably
#     re-enables and any failure is visible instead of an infinite hang.
#     """
#     data = request.json or {}
#     source_path = Path(data.get("source_path", ""))
#     if not source_path.exists():
#         return jsonify({"error": "source_path not found"}), 400

#     try:
#         info = _probe(source_path)
#     except Exception as e:
#         return jsonify({"error": f"Could not read this video (probe failed): {e}"}), 500

#     result = {"probe": info, "features_available": feature_status()}

#     if data.get("face_tracking", True) and _HAS_CV2:
#         try:
#             points, w, h, advanced = analyze_face_track(source_path, sample_fps=data.get("track_fps", 2.0))
#             result["face_track"] = points
#             result["face_tracking_mode"] = "advanced" if advanced else "standard"
#             result["source_size"] = {"w": w, "h": h}
#         except Exception as e:
#             result["face_track"] = []
#             result["face_tracking_error"] = str(e)

#     if data.get("subtitles", True) and _HAS_WHISPER:
#         # faster-whisper downloads its model from Hugging Face on first use,
#         # which can take minutes (or hang) on a slow/rate-limited/offline
#         # connection. Bound it in a worker thread so a stuck download can
#         # never hang analyze forever — if it doesn't finish in time we skip
#         # captions for THIS request and move on; the model keeps downloading
#         # in the background, so the NEXT analyze on any video will be fast
#         # and will pick subtitles back up automatically.
#         try:
#             with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
#                 future = ex.submit(transcribe_words, source_path, data.get("whisper_model", "small"))
#                 words = future.result(timeout=180)
#             result["words"] = words
#         except concurrent.futures.TimeoutError:
#             result["words"] = []
#             result["subtitle_error"] = (
#                 "Subtitles timed out (likely still downloading the whisper model in the "
#                 "background on first run) — continuing without captions. Try Analyze again "
#                 "in a minute or two."
#             )
#         except Exception as e:
#             result["words"] = []
#             result["subtitle_error"] = str(e)

#     if data.get("hook_cut", False) and _HAS_NUMPY:
#         try:
#             result["suggested_hook"] = auto_hook_window(source_path, info["duration"], data.get("hook_len", 6.0))
#         except Exception as e:
#             result["hook_cut_error"] = str(e)

#     if data.get("jump_cut", False):
#         try:
#             result["suggested_segments"] = detect_silences(source_path)
#         except Exception as e:
#             result["jump_cut_error"] = str(e)

#     return jsonify(result)
 
 
# @editor_bp.route("/api/editor/upload", methods=["POST"])
# def api_editor_upload():
#     """Lets the frontend upload a local file directly instead of only
#     pointing at a path already on the server (e.g. a downloader.py output)."""
#     f = request.files.get("file")
#     if not f or not f.filename:
#         return jsonify({"error": "No file uploaded"}), 400
#     safe_name = f"{uuid.uuid4().hex[:10]}_{Path(f.filename).name}"
#     dest = EDIT_DIR / "uploads"
#     dest.mkdir(exist_ok=True)
#     path = dest / safe_name
#     f.save(path)
#     return jsonify({"source_path": str(path)})
 
 
# # ══════════════════════════════ fetch-from-link (built-in yt-dlp) ══════════════════════════════
# # A third source alongside "upload from PC" and "pick from downloader" —
# # paste a link right here in the editor. Deliberately skips the full
# # format-picker: it always resolves ONE single, best-available video+audio
# # mp4 (never a video-only stream that would need a silent-audio placeholder,
# # never separate files) and hands that straight to analyze/render.
# #
# # Speed trick mirrors downloader.py: remembers whichever auth mode (cookies
# # file / browser cookies / plain / bypass) last worked and tries that first,
# # so repeat fetches stay fast. If downloader.py is also registered in this
# # process, its already-resolved mode is reused immediately instead of
# # re-discovering it from scratch.
 
# FETCH_JOBS = {}  # fetch_id -> {status, stage, percent, error, source_path, url}
# _FETCH_RESOLVED_MODE = {"mode": None}
# _FETCH_COOKIE_BROWSERS = ["chrome", "edge", "firefox", "brave"]
 
 
# def _fetch_no_console_kwargs():
#     if os.name == "nt":
#         return {"creationflags": subprocess.CREATE_NO_WINDOW}
#     return {}
 
 
# def _fetch_auth_attempts():
#     attempts = []
#     # reuse downloader.py's already-known-good mode first, if it's loaded
#     try:
#         import downloader
#         if downloader._RESOLVED_MODE.get("mode"):
#             attempts.append(downloader._RESOLVED_MODE["mode"])
#     except Exception:
#         pass
#     if _FETCH_RESOLVED_MODE["mode"]:
#         attempts.append(_FETCH_RESOLVED_MODE["mode"])
#     cookies_file = SRC_DIR.parent / "cookies.txt" if SRC_DIR else None
#     rest = []
#     if cookies_file and cookies_file.exists():
#         rest.append("cookies_file")
#     rest.extend(_FETCH_COOKIE_BROWSERS)
#     rest.append("default")
#     rest.append("bypass")
#     for m in rest:
#         if m not in attempts:
#             attempts.append(m)
#     return attempts, cookies_file
 
 
# def _fetch_apply_auth(opts, mode, cookies_file):
#     opts = dict(opts)
#     if mode == "cookies_file" and cookies_file:
#         opts["cookiefile"] = str(cookies_file)
#     elif mode in _FETCH_COOKIE_BROWSERS:
#         opts["cookiesfrombrowser"] = (mode,)
#     elif mode == "bypass":
#         opts["extractor_args"] = {"youtube": {"player_client": ["ios", "android", "mweb"]}}
#     return opts
 
 
# def _fetch_extract(url, extra_opts=None, download=False):
#     base_opts = {
#         "quiet": True, "no_warnings": True, "noplaylist": True,
#         "socket_timeout": 15, "retries": 2, "extractor_retries": 1, "geo_bypass": True,
#     }
#     if FFMPEG_PATH:
#         base_opts["ffmpeg_location"] = str(FFMPEG_PATH)
#     if extra_opts:
#         base_opts.update(extra_opts)
 
#     attempts, cookies_file = _fetch_auth_attempts()
#     last_err = None
#     for mode in attempts:
#         opts = _fetch_apply_auth(base_opts, mode, cookies_file)
#         try:
#             with yt_dlp.YoutubeDL(opts) as ydl:
#                 info = ydl.extract_info(url, download=download)
#             _FETCH_RESOLVED_MODE["mode"] = mode
#             return info, ydl if download else None, None
#         except Exception as e:
#             last_err = e
#             continue
#     return None, None, last_err
 
 
# def _fetch_fmt_duration(sec):
#     if sec is None:
#         return None
#     sec = int(sec)
#     h, rem = divmod(sec, 3600)
#     m, s = divmod(rem, 60)
#     return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
 
 
# def _fetch_fmt_int(n):
#     if n is None:
#         return None
#     try:
#         return f"{int(n):,}"
#     except (TypeError, ValueError):
#         return str(n)
 
 
# def _fetch_fmt_upload_date(d):
#     if not d:
#         return None
#     try:
#         from datetime import datetime as _dt
#         return _dt.strptime(str(d), "%Y%m%d").strftime("%d %b %Y")
#     except ValueError:
#         return str(d)
 
 
# @editor_bp.route("/api/editor/fetch/info", methods=["POST"])
# def api_fetch_info():
#     """Resolves a link (YouTube/Instagram/TikTok/X/Facebook/etc.) and hands
#     back the FULL set of metadata yt-dlp/the source platform already knows
#     about it — title, uploader, view/like counts, upload date, best
#     available resolution — so the UI never has to re-derive any of this by
#     hand, and so a video that started life as a fetched link can show that
#     rich context in the analysis panel too (a plain PC upload naturally
#     won't have this — it only gets the ffprobe technical details)."""
#     if not _HAS_YTDLP:
#         return jsonify({"error": "yt-dlp is not installed on the server — pip install yt-dlp"}), 400
#     data = request.json or {}
#     url = (data.get("url") or "").strip()
#     if not url:
#         return jsonify({"error": "Paste a video URL first"}), 400
#     info, _, err = _fetch_extract(url, extra_opts={"format": "bestvideo+bestaudio/best"})
#     if info is None:
#         return jsonify({"error": f"Could not resolve that link: {err}"}), 400
 
#     best_h = best_w = None
#     for f in info.get("formats", []) or []:
#         if f.get("vcodec") not in (None, "none") and f.get("height"):
#             if not best_h or f["height"] > best_h:
#                 best_h, best_w = f["height"], f.get("width")
 
#     return jsonify({
#         "title": info.get("title"),
#         "uploader": info.get("uploader") or info.get("channel"),
#         "duration": info.get("duration"),
#         "duration_str": _fetch_fmt_duration(info.get("duration")),
#         "thumbnail": info.get("thumbnail"),
#         "view_count": info.get("view_count"),
#         "view_count_str": _fetch_fmt_int(info.get("view_count")),
#         "like_count": info.get("like_count"),
#         "like_count_str": _fetch_fmt_int(info.get("like_count")),
#         "upload_date": info.get("upload_date"),
#         "upload_date_str": _fetch_fmt_upload_date(info.get("upload_date")),
#         "resolution": f"{best_w}x{best_h}" if best_w and best_h else None,
#         "extractor": info.get("extractor_key"),
#         "webpage_url": info.get("webpage_url"),
#     })
 
 
# def _run_fetch_job(fetch_id, url):
#     job = FETCH_JOBS[fetch_id]
#     job["stage"] = "connect"
 
#     def hook(d):
#         if d.get("status") == "downloading":
#             total = d.get("total_bytes") or d.get("total_bytes_estimate")
#             downloaded = d.get("downloaded_bytes") or 0
#             speed = d.get("speed")
#             job.update({
#                 "status": "downloading", "stage": "download",
#                 "percent": round(downloaded * 100 / total, 1) if total else None,
#                 "speed_str": (_fmt_size(speed) + "/s") if speed else None,
#             })
#         elif d.get("status") == "finished":
#             job["status"] = "processing"
#             job["stage"] = "merge"
 
#     # Saved into SRC_DIR (the SAME "downloads" folder downloader.py uses) so
#     # a link fetched here shows up in /api/editor/sources too, and nothing
#     # about the analyze/render pipeline needs to know it came from here
#     # instead of the Downloader tab or a PC upload.
#     extra_opts = {
#         "format": "bestvideo+bestaudio/best",
#         "outtmpl": str(SRC_DIR / f"editorfetch_{fetch_id}_%(title).60s.%(ext)s"),
#         "progress_hooks": [hook],
#         "merge_output_format": "mp4",   # always ONE muxed video+audio file
#     }
 
#     orig_popen = subprocess.Popen
#     def quiet_popen(*args, **kwargs):
#         kwargs.update(_fetch_no_console_kwargs())
#         return orig_popen(*args, **kwargs)
#     subprocess.Popen = quiet_popen
#     try:
#         info, ydl, err = _fetch_extract(url, extra_opts=extra_opts, download=True)
#     finally:
#         subprocess.Popen = orig_popen
 
#     if info is None:
#         job["status"] = "error"
#         job["error"] = f"Fetch failed: {err}"
#         return
#     try:
#         fname = ydl.prepare_filename(info)
#         p = Path(fname)
#         if not p.exists():
#             p2 = p.with_suffix(".mp4")
#             if p2.exists():
#                 p = p2
#         job["source_path"] = str(p.resolve())
#         job["status"] = "done"
#         job["stage"] = "done"
#         job["percent"] = 100
#     except Exception as e:
#         job["status"] = "error"
#         job["error"] = f"Could not finalize file: {e}"
 
 
# @editor_bp.route("/api/editor/fetch/start", methods=["POST"])
# def api_fetch_start():
#     if not _HAS_YTDLP:
#         return jsonify({"error": "yt-dlp is not installed on the server — pip install yt-dlp"}), 400
#     data = request.json or {}
#     url = (data.get("url") or "").strip()
#     if not url:
#         return jsonify({"error": "No URL given"}), 400
#     fetch_id = uuid.uuid4().hex[:10]
#     FETCH_JOBS[fetch_id] = {
#         "status": "starting", "stage": "connect", "percent": 0,
#         "speed_str": None, "error": None, "url": url, "source_path": None,
#     }
#     threading.Thread(target=_run_fetch_job, args=(fetch_id, url), daemon=True).start()
#     return jsonify({"fetch_id": fetch_id})
 
 
# @editor_bp.route("/api/editor/fetch/progress/<fetch_id>")
# def api_fetch_progress(fetch_id):
#     job = FETCH_JOBS.get(fetch_id)
#     if not job:
#         return jsonify({"error": "Unknown fetch job"}), 404
#     return jsonify(job)
 
 
# @editor_bp.route("/api/editor/render", methods=["POST"])
# def api_editor_render():
#     data = request.json or {}
#     source_path = data.get("source_path", "")
#     if not source_path or not Path(source_path).exists():
#         return jsonify({"error": "source_path not found"}), 400
 
#     export_ratios = data.get("export_ratios") or ["9:16"]
#     bad = [r for r in export_ratios if r not in RATIO_PRESETS]
#     if bad:
#         return jsonify({"error": f"Unknown ratio(s): {bad}. Valid: {list(RATIO_PRESETS)}"}), 400
 
#     features = {
#         "face_tracking": bool(data.get("face_tracking", True)),
#         "subtitles": bool(data.get("subtitles", True)),
#         "hook_cut": bool(data.get("hook_cut", False)),
#         "jump_cut": bool(data.get("jump_cut", False)),
#     }
#     if features["face_tracking"] and not _HAS_CV2:
#         return jsonify({"error": "Face tracking requested but opencv-python is not installed"}), 400
#     if features["subtitles"] and not _HAS_WHISPER:
#         return jsonify({"error": "Subtitles requested but faster-whisper is not installed"}), 400
#     if features["hook_cut"] and not data.get("segments") and not _HAS_NUMPY:
#         return jsonify({"error": "Auto hook-cut requested but numpy is not installed"}), 400
 
#     job_id = uuid.uuid4().hex[:10]
#     EDIT_JOBS[job_id] = {
#         "job_id": job_id, "status": "starting", "stage": "queued", "percent": 0,
#         "error": None, "ratios": {},
#         "config": {
#             "source_path": source_path,
#             "export_ratios": export_ratios,
#             "features": features,
#             "caption_style": data.get("caption_style", "bold_pop"),
#             "quality": data.get("quality", "high"),
#             "track_fps": data.get("track_fps", 2.0),
#             "whisper_model": data.get("whisper_model", "small"),
#             "language": data.get("language"),
#             "hook_len": data.get("hook_len", 6.0),
#             "manual_crop_keyframes": data.get("manual_crop_keyframes"),  # [{t,x,y}]
#             "segments": data.get("segments"),  # [{start,end}] manual rearrange/trim order
#             "fill_mode": data.get("fill_mode", "crop"),  # "crop" or "blur_pad"
#             "normalize_audio": bool(data.get("normalize_audio", True)),
#         },
#     }
#     threading.Thread(target=_run_render_job, args=(job_id,), daemon=True).start()
#     return jsonify({"job_id": job_id})
 
 
# @editor_bp.route("/api/editor/progress/<job_id>")
# def api_editor_progress(job_id):
#     job = EDIT_JOBS.get(job_id)
#     if not job:
#         return jsonify({"error": "Unknown edit job"}), 404
#     return jsonify({k: v for k, v in job.items() if k != "config"} | {
#         "features": job["config"]["features"], "export_ratios": job["config"]["export_ratios"]
#     })
 
 
# @editor_bp.route("/api/editor/file/<job_id>/<ratio>")
# def api_editor_file(job_id, ratio):
#     job = EDIT_JOBS.get(job_id)
#     if not job or job.get("status") != "done":
#         return "Not ready", 404
#     fname = job["ratios"].get(ratio)
#     if not fname:
#         return "Ratio not found for this job", 404
#     p = EDIT_DIR / fname
#     if not p.exists():
#         return "Not found", 404
#     return send_file(p, as_attachment=True, download_name=fname)
 
 
# # ── manual crop keyframe add/remove (per not-yet-rendered job config) ──
 
# @editor_bp.route("/api/editor/keyframe/add", methods=["POST"])
# def api_keyframe_add():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     kf = {"t": float(data["t"]), "x": int(data["x"]), "y": int(data["y"])}
#     job["config"].setdefault("manual_crop_keyframes", [])
#     job["config"]["manual_crop_keyframes"] = job["config"]["manual_crop_keyframes"] or []
#     job["config"]["manual_crop_keyframes"].append(kf)
#     job["config"]["manual_crop_keyframes"].sort(key=lambda k: k["t"])
#     return jsonify({"manual_crop_keyframes": job["config"]["manual_crop_keyframes"]})
 
 
# @editor_bp.route("/api/editor/keyframe/remove", methods=["POST"])
# def api_keyframe_remove():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     t = float(data["t"])
#     kfs = job["config"].get("manual_crop_keyframes") or []
#     job["config"]["manual_crop_keyframes"] = [k for k in kfs if abs(k["t"] - t) > 1e-6]
#     return jsonify({"manual_crop_keyframes": job["config"]["manual_crop_keyframes"]})
 
 
# # ── hook-cut / rearrange segment add/remove ──
 
# @editor_bp.route("/api/editor/segment/add", methods=["POST"])
# def api_segment_add():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     seg = {"start": float(data["start"]), "end": float(data["end"])}
#     index = data.get("index")
#     segs = job["config"].get("segments") or []
#     if index is None or index >= len(segs):
#         segs.append(seg)
#     else:
#         segs.insert(int(index), seg)
#     job["config"]["segments"] = segs
#     return jsonify({"segments": segs})
 
 
# @editor_bp.route("/api/editor/segment/remove", methods=["POST"])
# def api_segment_remove():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     index = int(data["index"])
#     segs = job["config"].get("segments") or []
#     if 0 <= index < len(segs):
#         segs.pop(index)
#     job["config"]["segments"] = segs
#     return jsonify({"segments": segs})










# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# AutoShortAi — Video Editor module
# ──────────────────────────────────
# Adds AI auto-reframe (face tracking), trending animated captions,
# hook-cut / manual clip reordering, and one-click multi-aspect-ratio
# export — all producing a SINGLE muxed video+audio file per ratio.
 
# Wire it into the main app with:
 
#     from video_editor import editor_bp, init_editor
#     init_editor(BASE, FFMPEG)
#     app.register_blueprint(editor_bp)
 
# Routes exposed (all under /api/editor/...):
#     GET  /api/editor/editor             -> serves the standalone editor.html UI (open this in a tab)
#     POST /api/editor/upload             -> upload a local file, returns a source_path to use below
#     POST /api/editor/analyze            -> face-track + transcript + silence/hook preview (for the UI)
#     GET  /api/editor/styles             -> list of caption style presets
#     GET  /api/editor/ratios             -> list of export aspect-ratio presets
#     GET  /api/editor/features           -> which advanced features are available (installed libs)
#     POST /api/editor/render             -> kicks off a background render job
#     GET  /api/editor/progress/<job_id>  -> live progress
#     GET  /api/editor/file/<job_id>/<ratio> -> serves one finished ratio's file
#     POST /api/editor/keyframe/add       -> add a manual crop keyframe to a job's override track
#     POST /api/editor/keyframe/remove    -> remove a manual crop keyframe
#     POST /api/editor/segment/add        -> add a hook-cut / reorder segment
#     POST /api/editor/segment/remove     -> remove a hook-cut / reorder segment
 
# ADVANCED EXTRAS (on top of the original spec):
#     • fill_mode="blur_pad"   -> reframe by fitting the WHOLE frame + blurred
#       background bars instead of hard-cropping (no content ever cut off).
#     • jump_cut feature       -> auto-detects and removes silence/dead-air,
#       the other half of "trending" short-form editing besides hook-cut.
#     • hook_cut + jump_cut COMBINED -> if both are on, the loudest window is
#       used as the cold-open "hook" and is prepended in front of the
#       dead-air-trimmed rest of the video (previously turning jump_cut on
#       silently dropped hook_cut — now they compose).
#     • normalize_audio        -> EBU R128 loudness normalization (-14 LUFS),
#       on by default, matches the loudness social platforms expect.
#     • caption keyword emphasis -> numbers, money, and punchy "power words"
#       auto-highlight in captions even outside full word-by-word styles.
#     • quality/caption_style inputs are now validated with a safe fallback
#       instead of raising a raw KeyError if the frontend sends a typo'd key.
#     • self-serves its own frontend (editor.html) at GET /api/editor/editor,
#       the same "own file, same process/port" pattern as downloader.py.
 
# DEPENDENCIES (install what you want to use — everything degrades gracefully
# if a library is missing, see FEATURE FLAGS AT IMPORT TIME below):
#     pip install opencv-python            # required for any face tracking
#     pip install mediapipe                # optional, gives the "advanced" tracker
#                                           # (falls back to Haar cascade otherwise)
#     pip install faster-whisper           # required for subtitles (word timestamps)
#     pip install numpy
 
# DESIGN NOTES — how each part of the request maps to code:
#   • "advance level face tracking"   -> _FaceTracker (mediapipe BlazeFace if present,
#                                         else OpenCV Haar cascade), EMA-smoothed
#                                         centroid so the crop doesn't jitter frame
#                                         to frame, multi-face -> largest/most-central
#                                         face wins.
#   • "perfect advance subtitles,
#      trending caption styles"       -> CAPTION_STYLES presets + build_ass() which
#                                         groups words into short on-screen chunks and
#                                         emits karaoke-style \\k tags for word-by-word
#                                         pop/highlight animation, burned in with
#                                         ffmpeg's `ass` filter (not a soft-sub track,
#                                         so it always displays correctly everywhere).
#   • "single high quality video+audio
#      in one file"                   -> every render always outputs one .mp4 with
#                                         both streams muxed, CRF-based high quality
#                                         encode (see QUALITY_PRESETS).
#   • "export in all ratio perfectly" -> RATIO_PRESETS, rendered in a loop, one file
#                                         per ratio, each independently croppped using
#                                         the SAME face track (recomputed crop window
#                                         per target aspect).
#   • "rearrange video manually /
#      hook cut"                      -> job["segments"]: an ordered list of
#                                         {start,end} clip ranges. Auto hook-cut picks
#                                         the loudest window as segment 1 automatically;
#                                         manual mode lets the caller add/remove/reorder
#                                         segments directly (segment/add, segment/remove).
#   • "add and remove option for all" -> every feature is a boolean flag in job config
#                                         (features{}) that can be flipped and re-rendered,
#                                         plus explicit add/remove endpoints for the two
#                                         list-based tracks (crop keyframes, segments).
# """
 
# import os
# import json
# import uuid
# import struct
# import threading
# import subprocess
# from pathlib import Path
 
# from flask import Blueprint, request, jsonify, send_file
 
# editor_bp = Blueprint("editor_bp", __name__)
 
# # ══════════════════════════════ embedded frontend ══════════════════════════════
# # The whole editor UI lives right here as one string — no separate .html file
# # to ship or lose track of. Served as-is by GET /api/editor/editor. Its fetch
# # calls use relative paths (same-origin), so it works immediately wherever
# # this blueprint is registered.
# EDITOR_HTML = r"""<!doctype html>
# <html lang="en">
# <head>
# <meta charset="utf-8">
# <meta name="viewport" content="width=device-width, initial-scale=1">
# <title>Reframe — AI video editor</title>
# <link rel="preconnect" href="https://fonts.googleapis.com">
# <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
# <style>
#   /* ── Theme — mirrors the main app's palette exactly (same variable
#      VALUES, just different variable NAMES for this page's own CSS). Default
#      below is "dark-violet", the main app's default. The <html data-theme="…">
#      overrides further down are read straight off the SAME localStorage key
#      the main app writes to (see the sync script at the bottom of <body>),
#      so switching themes on Studio/Downloader also re-themes this page —
#      no manual matching needed ever again. ── */
#   :root{
#     --bg:#0b0d12; --panel:#13151c; --panel-2:#191c25; --border:#262a36;
#     --text:#eef0f6; --text-dim:#8a90a4; --text-faint:#8a90a4;
#     --amber:#6e5bff; --teal:#22d3c4; --red:#ff5a6e;
#     --amber-dim: color-mix(in srgb, var(--amber) 22%, #000);
#     --teal-dim: color-mix(in srgb, var(--teal) 22%, #000);
#     --radius:10px;
#     font-size:15px;
#   }
#   html[data-theme="light"]{
#     --bg:#f4f5f9; --panel:#ffffff; --panel-2:#eef0f6; --border:#dde1ea;
#     --text:#171923; --text-dim:#5b6472; --text-faint:#5b6472;
#     --amber:#5b46e0; --teal:#0891b2; --red:#e0324a;
#   }
#   html[data-theme="midnight-blue"]{
#     --bg:#050a16; --panel:#0c1526; --panel-2:#12213a; --border:#1f3358;
#     --text:#e8f1ff; --text-dim:#7f93b8; --text-faint:#7f93b8;
#     --amber:#3b82f6; --teal:#22d3ee; --red:#ff5a6e;
#   }
#   html[data-theme="forest"]{
#     --bg:#0a130d; --panel:#0f1e14; --panel-2:#16301f; --border:#234a30;
#     --text:#eafbea; --text-dim:#86b399; --text-faint:#86b399;
#     --amber:#22c55e; --teal:#a3e635; --red:#ff5a6e;
#   }
#   html[data-theme="sunset"]{
#     --bg:#1a0e12; --panel:#241419; --panel-2:#331d27; --border:#4a2635;
#     --text:#fdf1ee; --text-dim:#cb9aa3; --text-faint:#cb9aa3;
#     --amber:#fb7185; --teal:#f59e0b; --red:#ff5a6e;
#   }
#   html[data-theme="ocean"]{
#     --bg:#061319; --panel:#0b1f27; --panel-2:#123039; --border:#1e4a56;
#     --text:#e6fbff; --text-dim:#7fb8c4; --text-faint:#7fb8c4;
#     --amber:#06b6d4; --teal:#6366f1; --red:#ff5a6e;
#   }
#   html[data-theme="rose"]{
#     --bg:#170f14; --panel:#221720; --panel-2:#2e1e2b; --border:#4a2f45;
#     --text:#fdeef5; --text-dim:#c79ab0; --text-faint:#c79ab0;
#     --amber:#ec4899; --teal:#f472b6; --red:#ff5a6e;
#   }
#   html[data-theme="amber"]{
#     --bg:#160f05; --panel:#221708; --panel-2:#33230d; --border:#4d3416;
#     --text:#fdf4e3; --text-dim:#c9a877; --text-faint:#c9a877;
#     --amber:#f59e0b; --teal:#eab308; --red:#ff5a6e;
#   }
#   html[data-theme="slate"]{
#     --bg:#101215; --panel:#181b1f; --panel-2:#20242a; --border:#333941;
#     --text:#eceef1; --text-dim:#9aa2ad; --text-faint:#9aa2ad;
#     --amber:#64748b; --teal:#94a3b8; --red:#ff5a6e;
#   }
#   html[data-theme="cyberpunk"]{
#     --bg:#0a0014; --panel:#150124; --panel-2:#210235; --border:#4a0a6b;
#     --text:#f5e6ff; --text-dim:#b083d6; --text-faint:#b083d6;
#     --amber:#e91e9c; --teal:#00f0ff; --red:#ff5a6e;
#   }
#   *{box-sizing:border-box;}
#   body{
#     margin:0; background:var(--bg); color:var(--text);
#     font-family:'Inter',sans-serif; min-height:100vh;
#   }
#   h1,h2,h3,.display{font-family:'Space Grotesk',sans-serif;}
#   .mono{font-family:'JetBrains Mono',monospace;}
#   ::selection{background:var(--amber-dim); color:var(--amber);}
 
#   /* ── top bar ── */
#   .topbar{
#     display:flex; align-items:center; gap:16px; padding:14px 22px;
#     border-bottom:1px solid var(--border); background:var(--panel);
#     position:sticky; top:0; z-index:20;
#   }
#   .brand{display:flex; align-items:center; gap:10px;}
#   .brand .dot{width:10px; height:10px; border-radius:50%; background:var(--amber); box-shadow:0 0 10px var(--amber);}
#   .brand h1{font-size:18px; font-weight:700; margin:0; letter-spacing:.2px;}
#   .brand span{color:var(--text-dim); font-size:12px;}
#   .topbar .spacer{flex:1;}
#   .api-input{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text-dim);
#     border-radius:8px; padding:7px 10px; font-size:12px; font-family:'JetBrains Mono',monospace;
#     width:220px;
#   }
#   .api-input:focus{outline:none; border-color:var(--amber); color:var(--text);}
 
#   /* ── layout ── */
#   .app{display:grid; grid-template-columns:1fr 380px; gap:1px; background:var(--border);}
#   @media (max-width:980px){ .app{grid-template-columns:1fr;} }
#   .col{background:var(--bg); padding:22px; min-width:0;}
 
#   /* ── upload zone ── */
#   .dropzone{
#     border:1.5px dashed var(--border); border-radius:var(--radius);
#     padding:40px 20px; text-align:center; cursor:pointer;
#     transition:border-color .15s, background .15s;
#   }
#   .dropzone:hover, .dropzone.drag{border-color:var(--amber); background:rgba(255,176,32,.04);}
#   .dropzone svg{opacity:.5; margin-bottom:10px;}
#   .dropzone .hint{color:var(--text-dim); font-size:13px; margin-top:6px;}
 
#   /* ── fetch-from-link (ytdlp-style resolver, built into the editor) ── */
#   .fetch-section{border:1px solid var(--border); border-radius:var(--radius); padding:12px; background:var(--panel-2); margin-bottom:14px;}
#   .fetch-row{display:flex; gap:8px; margin-top:8px;}
#   .fetch-input{
#     flex:1; background:var(--panel); border:1px solid var(--border); color:var(--text);
#     border-radius:8px; padding:9px 10px; font-size:13px;
#   }
#   .fetch-input:focus{outline:none; border-color:var(--amber);}
#   .btn-accent{background:var(--amber); color:#1a1200; border:none; font-weight:600;}
#   .btn-accent:hover{filter:brightness(1.08);}
#   .fetch-card{
#     display:flex; gap:10px; align-items:center; margin-top:10px;
#     border:1px solid var(--border); border-radius:8px; padding:10px; background:var(--panel);
#   }
#   .fetch-thumb{width:88px; height:56px; object-fit:cover; border-radius:6px; background:#000; flex-shrink:0;}
#   .fetch-meta{flex:1; min-width:0;}
#   .fetch-title{font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
#   .fetch-sub{font-size:11.5px; color:var(--text-dim); margin-top:2px;}
#   .fetch-status{font-size:12px; color:var(--text-dim); margin-top:8px; min-height:16px;}
 
#   /* ── server library (downloader output + past uploads) ── */
#   .or-divider{
#     display:flex; align-items:center; gap:10px; margin:14px 0;
#     color:var(--text-faint); font-size:12px; text-transform:uppercase; letter-spacing:.5px;
#   }
#   .or-divider::before, .or-divider::after{content:""; flex:1; height:1px; background:var(--border);}
#   .lib-section{border:1px solid var(--border); border-radius:var(--radius); padding:12px; background:var(--panel-2);}
#   .lib-header{display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; font-size:13px; color:var(--text-dim);}
#   .lib-grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:8px; max-height:220px; overflow-y:auto;}
#   .lib-empty{color:var(--text-faint); font-size:12.5px; grid-column:1/-1;}
#   .lib-item{
#     border:1px solid var(--border); border-radius:8px; padding:10px; cursor:pointer;
#     background:var(--panel); transition:border-color .15s, transform .1s;
#   }
#   .lib-item:hover{border-color:var(--amber); transform:translateY(-1px);}
#   .lib-item.active{border-color:var(--teal); background:var(--teal-dim);}
#   .lib-item .lib-name{font-size:12.5px; font-weight:600; word-break:break-word; margin-bottom:4px;}
#   .lib-item .lib-meta{font-size:11px; color:var(--text-dim); display:flex; justify-content:space-between;}
#   .lib-item .lib-tag{
#     display:inline-block; font-size:10px; padding:1px 6px; border-radius:20px;
#     background:var(--amber-dim); color:var(--amber); margin-bottom:4px;
#   }
 
#   /* ── preview ── */
#   .preview-wrap{position:relative; display:flex; justify-content:center; margin-bottom:16px;}
#   .preview-frame{position:relative; background:#000; border-radius:8px; overflow:hidden; max-width:100%;}
#   video{display:block; max-width:100%; max-height:60vh;}
#   .crop-box{
#     position:absolute; border:1.5px solid var(--amber); pointer-events:none;
#     box-shadow:0 0 0 2000px rgba(0,0,0,.45);
#   }
#   .crop-box .corner{
#     position:absolute; width:16px; height:16px; border-color:var(--amber); border-style:solid; border-width:0;
#   }
#   .crop-box .tl{top:-1.5px; left:-1.5px; border-top-width:3px; border-left-width:3px;}
#   .crop-box .tr{top:-1.5px; right:-1.5px; border-top-width:3px; border-right-width:3px;}
#   .crop-box .bl{bottom:-1.5px; left:-1.5px; border-bottom-width:3px; border-left-width:3px;}
#   .crop-box .br{bottom:-1.5px; right:-1.5px; border-bottom-width:3px; border-right-width:3px;}
#   .crop-box{cursor:grab;}
#   .crop-box.dragging{cursor:grabbing;}
#   .crop-readout{
#     position:absolute; top:8px; left:8px; background:rgba(0,0,0,.6);
#     color:var(--amber); font-size:11px; padding:3px 7px; border-radius:5px;
#   }
 
#   .preview-controls{display:flex; align-items:center; gap:10px; justify-content:center; margin-bottom:8px;}
#   .btn-icon{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text);
#     width:34px; height:34px; border-radius:50%; cursor:pointer; display:flex;
#     align-items:center; justify-content:center;
#   }
#   .btn-icon:hover{border-color:var(--amber);}
#   .time{font-size:12px; color:var(--text-dim); min-width:96px; text-align:center;}
 
#   /* ── timeline ── */
#   .timeline-panel{background:var(--panel); border:1px solid var(--border); border-radius:var(--radius); padding:16px;}
#   .timeline-panel h3{margin:0 0 4px; font-size:13px; letter-spacing:.4px; text-transform:uppercase; color:var(--text-dim);}
#   .timeline{
#     position:relative; height:56px; background:var(--panel-2); border-radius:6px;
#     margin-top:10px; overflow:hidden; cursor:pointer;
#   }
#   .tl-face{position:absolute; top:0; height:18px; background:linear-gradient(90deg, transparent, var(--teal-dim)); opacity:.6;}
#   .tl-seg{
#     position:absolute; top:20px; height:18px; background:var(--teal); opacity:.85; border-radius:3px;
#     display:flex; align-items:center; justify-content:center; font-size:10px; color:#04231D; font-weight:600;
#     cursor:grab;
#   }
#   .tl-hook{position:absolute; top:38px; height:14px; background:var(--amber); opacity:.9; border-radius:3px;}
#   .tl-playhead{position:absolute; top:0; bottom:0; width:2px; background:var(--red);}
#   .tl-legend{display:flex; gap:16px; margin-top:8px; font-size:11px; color:var(--text-dim);}
#   .tl-legend span{display:inline-flex; align-items:center; gap:5px;}
#   .swatch{width:9px; height:9px; border-radius:2px; display:inline-block;}
 
#   .seg-list{margin-top:10px; display:flex; flex-direction:column; gap:6px;}
#   .seg-row{
#     display:flex; align-items:center; gap:8px; background:var(--panel-2);
#     border:1px solid var(--border); border-radius:7px; padding:6px 10px; font-size:12px;
#   }
#   .seg-row .mono{color:var(--teal);}
#   .seg-row .spacer{flex:1;}
#   .seg-row button{background:none; border:none; color:var(--text-faint); cursor:pointer; font-size:14px;}
#   .seg-row button:hover{color:var(--red);}
#   .btn-small{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--text-dim);
#     border-radius:6px; padding:6px 10px; font-size:12px; cursor:pointer;
#   }
#   .btn-small:hover{border-color:var(--amber); color:var(--amber);}
 
#   /* ── side panel ── */
#   .section{margin-bottom:26px;}
#   .section h3{
#     font-size:12px; letter-spacing:.5px; text-transform:uppercase; color:var(--text-dim);
#     margin:0 0 12px; display:flex; align-items:center; gap:8px;
#   }
#   .section h3 .num{
#     width:18px; height:18px; border-radius:50%; background:var(--panel-2); border:1px solid var(--border);
#     display:flex; align-items:center; justify-content:center; font-size:10px; color:var(--amber);
#   }
 
#   .toggle-row{
#     display:flex; align-items:center; justify-content:space-between;
#     padding:10px 12px; background:var(--panel); border:1px solid var(--border);
#     border-radius:8px; margin-bottom:8px;
#   }
#   .toggle-row .label{font-size:13px;}
#   .toggle-row .desc{font-size:11px; color:var(--text-faint); margin-top:2px;}
#   .toggle-row.disabled{opacity:.4;}
#   .switch{position:relative; width:38px; height:21px; flex-shrink:0;}
#   .switch input{opacity:0; width:0; height:0;}
#   .slider{
#     position:absolute; inset:0; background:var(--panel-2); border:1px solid var(--border);
#     border-radius:20px; cursor:pointer; transition:.15s;
#   }
#   .slider::before{
#     content:''; position:absolute; width:15px; height:15px; left:2px; top:2px;
#     background:var(--text-dim); border-radius:50%; transition:.15s;
#   }
#   .switch input:checked + .slider{background:var(--amber-dim); border-color:var(--amber);}
#   .switch input:checked + .slider::before{transform:translateX(17px); background:var(--amber);}
#   .switch input:disabled + .slider{cursor:not-allowed;}
 
#   .ratio-grid{display:grid; grid-template-columns:1fr 1fr; gap:8px;}
#   .ratio-card{
#     border:1px solid var(--border); background:var(--panel); border-radius:8px;
#     padding:10px; cursor:pointer; text-align:center; position:relative;
#   }
#   .ratio-card.active{border-color:var(--amber); background:rgba(255,176,32,.06);}
#   .ratio-card .shape{margin:0 auto 6px; background:var(--panel-2); border:1px solid var(--border); border-radius:3px;}
#   .ratio-card .name{font-size:12px; font-weight:600;}
#   .ratio-card .lbl{font-size:10px; color:var(--text-faint); margin-top:2px;}
 
#   .style-list{display:flex; flex-direction:column; gap:8px;}
#   .style-card{
#     display:flex; align-items:center; gap:12px; border:1px solid var(--border);
#     background:var(--panel); border-radius:8px; padding:10px 12px; cursor:pointer;
#   }
#   .style-card.active{border-color:var(--teal);}
#   .style-swatch{
#     flex-shrink:0; width:64px; height:36px; border-radius:6px; background:#000;
#     display:flex; align-items:center; justify-content:center; font-size:9px; font-weight:800;
#     letter-spacing:.5px;
#   }
#   .style-card .name{font-size:12.5px; font-weight:600;}
#   .style-card .lbl{font-size:10.5px; color:var(--text-faint);}
 
#   select, .fill-toggle{
#     width:100%; background:var(--panel); border:1px solid var(--border); color:var(--text);
#     padding:9px 10px; border-radius:8px; font-size:13px;
#   }
#   .fill-toggle{display:flex; gap:6px;}
#   .fill-opt{flex:1; text-align:center; padding:8px; border-radius:6px; cursor:pointer; font-size:12px; border:1px solid var(--border);}
#   .fill-opt.active{border-color:var(--amber); color:var(--amber); background:rgba(255,176,32,.06);}
 
#   .render-btn{
#     width:100%; background:var(--amber); color:#1A1204; border:none; border-radius:10px;
#     padding:14px; font-size:14px; font-weight:700; cursor:pointer; font-family:'Space Grotesk',sans-serif;
#     letter-spacing:.3px;
#   }
#   .render-btn:disabled{background:var(--panel-2); color:var(--text-faint); cursor:not-allowed;}
#   .render-btn:not(:disabled):hover{filter:brightness(1.08);}
 
#   .progress-wrap{margin-top:14px;}
#   .progress-track{height:6px; background:var(--panel-2); border-radius:4px; overflow:hidden;}
#   .progress-fill{height:100%; background:linear-gradient(90deg, var(--teal), var(--amber)); width:0%; transition:width .3s;}
#   .progress-label{display:flex; justify-content:space-between; font-size:11px; color:var(--text-dim); margin-top:6px;}
 
#   .results{display:flex; flex-direction:column; gap:8px; margin-top:14px;}
#   .result-row{
#     display:flex; align-items:center; gap:10px; background:var(--panel); border:1px solid var(--border);
#     border-radius:8px; padding:10px 12px;
#   }
#   .result-row .name{font-size:12.5px; font-weight:600; flex:1;}
#   .result-row a{
#     background:var(--panel-2); border:1px solid var(--border); color:var(--teal);
#     text-decoration:none; font-size:11.5px; padding:6px 10px; border-radius:6px;
#   }
#   .result-row a:hover{border-color:var(--teal);}
 
#   .status-line{font-size:11.5px; color:var(--text-faint); margin-top:10px; line-height:1.5;}
#   .status-line b{color:var(--text-dim);}
#   .feature-warn{
#     font-size:11px; color:var(--amber); background:rgba(255,176,32,.08); border:1px solid var(--amber-dim);
#     border-radius:6px; padding:8px 10px; margin-top:8px; display:none;
#   }
# </style>
# </head>
# <body>
# <script>
# // ── Theme sync — this page is served from the SAME Flask app/port as the
# // main Studio (registered as a blueprint), and when opened as an <iframe> on
# // that page it's also same-origin, so it shares localStorage automatically.
# // Reading 'shortsStudioTheme' here — the exact key the main page's theme
# // picker writes to — means this editor always matches whatever theme is
# // active there, with zero manual setup. Runs immediately (before first
# // paint) so there's no flash of the wrong palette, and again whenever the
# // theme changes elsewhere (the 'storage' event fires in this frame whenever
# // another same-origin window/frame updates localStorage).
# (function(){
#   function applyTheme(){
#     const t = localStorage.getItem('shortsStudioTheme') || 'dark-violet';
#     if(t === 'dark-violet') document.documentElement.removeAttribute('data-theme');
#     else document.documentElement.setAttribute('data-theme', t);
#   }
#   applyTheme();
#   window.addEventListener('storage', (e)=>{ if(e.key === 'shortsStudioTheme') applyTheme(); });
# })();
# </script>
 
# <div class="topbar">
#   <div class="brand"><span class="dot"></span><h1>Reframe</h1><span>face-tracked auto edit</span></div>
#   <div class="spacer"></div>
#   <span id="apiBaseHint" class="mono" style="font-size:11px; color:var(--text-faint); margin-right:6px;">same server</span>
#   <button type="button" id="apiBaseToggle" class="btn-small" title="Only needed if this page is served from a different host than the API">⚙</button>
#   <input id="apiBase" class="api-input mono" placeholder="leave blank — uses this same page's server" value="" style="display:none;">
# </div>
 
# <div class="app">
#   <!-- LEFT: preview + timeline -->
#   <div class="col">
#     <div class="fetch-section">
#       <div class="lib-header">
#         <span>🔗 Fetch from a link (YouTube / Instagram / TikTok / X / Facebook…)</span>
#       </div>
#       <div class="fetch-row">
#         <input type="text" id="fetchUrl" class="fetch-input mono" placeholder="Paste a video URL…">
#         <button type="button" class="btn-small btn-accent" id="fetchInfoBtn">Fetch</button>
#       </div>
#       <div id="fetchCard" class="fetch-card" style="display:none;">
#         <img id="fetchThumb" class="fetch-thumb" alt="">
#         <div class="fetch-meta">
#           <div id="fetchTitle" class="fetch-title"></div>
#           <div id="fetchSub" class="fetch-sub"></div>
#         </div>
#         <button type="button" class="btn-small btn-accent" id="fetchUseBtn">⬇ Fetch high-quality &amp; use</button>
#       </div>
#       <div id="fetchStatus" class="fetch-status"></div>
#     </div>
 
#     <div class="or-divider">or drop / choose a video from this PC</div>
#     <div id="dropzone" class="dropzone">
#       <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#8A93A3" stroke-width="1.5"><path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v3a2 2 0 002 2h12a2 2 0 002-2v-3"/></svg>
#       <div>Drop a video, or click to choose one</div>
#       <div class="hint">mp4 / mov · analyzed locally before render</div>
#       <input type="file" id="fileInput" accept="video/*" style="display:none">
#     </div>
 
#     <div class="or-divider">or pick a video already fetched by the Downloader</div>
#     <div class="lib-section">
#       <div class="lib-header">
#         <span>📥 From Downloader / server library</span>
#         <button type="button" class="btn-small" id="refreshLibBtn">Refresh</button>
#       </div>
#       <div class="lib-grid" id="libGrid"><span class="lib-empty">Click Refresh to list videos already downloaded via the Downloader tab.</span></div>
#     </div>
 
#     <div id="previewSection" style="display:none;">
#       <div class="preview-wrap">
#         <div class="preview-frame" id="previewFrame">
#           <video id="video" muted></video>
#           <div class="crop-box" id="cropBox" style="display:none;">
#             <div class="corner tl"></div><div class="corner tr"></div><div class="corner bl"></div><div class="corner br"></div>
#             <div class="crop-readout" id="cropReadout">9:16</div>
#           </div>
#         </div>
#       </div>
#       <div class="preview-controls">
#         <div class="btn-icon" id="playBtn">▶</div>
#         <div class="time mono" id="timeLabel">0:00 / 0:00</div>
#       </div>
 
#       <div class="timeline-panel">
#         <h3>Timeline</h3>
#         <div class="timeline" id="timeline">
#           <div class="tl-playhead" id="playhead" style="left:0%"></div>
#         </div>
#         <div class="tl-legend">
#           <span><span class="swatch" style="background:var(--teal-dim)"></span>face track</span>
#           <span><span class="swatch" style="background:var(--teal)"></span>segments</span>
#           <span><span class="swatch" style="background:var(--amber)"></span>suggested hook</span>
#         </div>
#         <div class="seg-list" id="segList"></div>
#         <div style="display:flex; gap:8px; margin-top:10px;">
#           <button class="btn-small" id="addSegBtn">+ add segment at playhead (±3s)</button>
#           <button class="btn-small" id="useHookBtn" style="display:none;">use suggested hook</button>
#           <button class="btn-small" id="useSilenceBtn" style="display:none;">use auto jump-cut segments</button>
#         </div>
#         <div style="display:flex; gap:8px; margin-top:8px;">
#           <button class="btn-small" id="addKfBtn">+ pin crop here (manual keyframe)</button>
#           <button class="btn-small" id="clearKfBtn">clear manual crop</button>
#         </div>
#       </div>
#       <div class="status-line" id="statusLine"></div>
#       <div id="sourceMetaPanel" class="lib-section" style="display:none; margin-top:10px;"></div>
#     </div>
#   </div>
 
#   <!-- RIGHT: controls -->
#   <div class="col">
#     <div class="section">
#       <h3><span class="num">1</span>Auto-reframe</h3>
#       <div class="toggle-row" id="rowFace">
#         <div><div class="label">Face tracking</div><div class="desc">Advanced (mediapipe) if installed, else standard cascade</div></div>
#         <label class="switch"><input type="checkbox" id="chkFace" checked><span class="slider"></span></label>
#       </div>
#       <div class="fill-toggle" style="margin-top:8px;">
#         <div class="fill-opt active" data-fill="crop">Crop to face</div>
#         <div class="fill-opt" data-fill="blur_pad">Fit + blurred bars</div>
#       </div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">2</span>Captions</h3>
#       <div class="toggle-row" id="rowSubs">
#         <div><div class="label">Auto subtitles</div><div class="desc">Word-level, keyword emphasis auto-highlighted</div></div>
#         <label class="switch"><input type="checkbox" id="chkSubs" checked><span class="slider"></span></label>
#       </div>
#       <div class="style-list" id="styleList"></div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">3</span>Hook &amp; pacing</h3>
#       <div class="toggle-row" id="rowHook">
#         <div><div class="label">Hook-cut</div><div class="desc">Open on the loudest / most energetic moment</div></div>
#         <label class="switch"><input type="checkbox" id="chkHook"><span class="slider"></span></label>
#       </div>
#       <div class="toggle-row" id="rowJump">
#         <div><div class="label">Jump-cut silences</div><div class="desc">Auto-remove dead air between lines</div></div>
#         <label class="switch"><input type="checkbox" id="chkJump"><span class="slider"></span></label>
#       </div>
#       <div class="toggle-row">
#         <div><div class="label">Loudness normalize</div><div class="desc">-14 LUFS, matches platform playback volume</div></div>
#         <label class="switch"><input type="checkbox" id="chkNorm" checked><span class="slider"></span></label>
#       </div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">4</span>Export ratios</h3>
#       <div class="ratio-grid" id="ratioGrid"></div>
#     </div>
 
#     <div class="section">
#       <h3><span class="num">5</span>Quality</h3>
#       <select id="qualitySel">
#         <option value="high">High (crf 18, slow) — best quality</option>
#         <option value="medium">Medium (crf 21) — balanced</option>
#         <option value="fast">Fast (crf 23) — quick preview</option>
#       </select>
#     </div>
 
#     <button class="render-btn" id="renderBtn" disabled>Analyze a video first</button>
#     <div class="feature-warn" id="featureWarn"></div>
#     <div class="progress-wrap" id="progressWrap" style="display:none;">
#       <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
#       <div class="progress-label"><span id="progressStage">queued</span><span id="progressPct">0%</span></div>
#     </div>
#     <div class="results" id="results"></div>
#   </div>
# </div>
 
# <script>
# const $ = (id) => document.getElementById(id);
# const state = {
#   sourcePath: null, sourceMeta: null, sourceOrigin: 'pc', duration: 0, srcW: 0, srcH: 0,
#   faceTrack: [], words: [], suggestedHook: null, suggestedSilences: null,
#   segments: [], manualKeyframes: [],
#   // Multiple export ratios can be selected at once now (a Set of ratio
#   // keys, e.g. {'9:16','1:1'}), not just one — "auto" is pre-selected by
#   // default since it's the safest zero-quality-loss choice.
#   selectedRatios: new Set(['auto']),
#   previewRatio: '9:16', // drives the live crop box only, independent of export selection
#   selectedStyle: 'bold_pop', fillMode: 'crop',
#   jobId: null, dragging: false,
# };
 
# // Same-origin by default: this page is served BY the same Flask app/port
# // that Reframe, the Downloader, and everything else run on (they're all
# // registered on one Flask app as blueprints, per RenderDetect's main()),
# // so relative paths already hit the right server. The "⚙" override is
# // only for the rare case this HTML got copied to a different host — if
# // left blank (the default), api() never leaves the current origin, which
# // is what fixes the class of bug where a stray/incorrect API host caused
# // every request (analyze, render, fetch…) to silently fail.
# function apiBase(){ const v = $('apiBase').value.trim().replace(/\/$/, ''); return v; }
# function api(path){ return apiBase() + path; }
# $('apiBaseToggle').onclick = ()=>{
#   const el = $('apiBase');
#   const show = el.style.display === 'none';
#   el.style.display = show ? 'inline-block' : 'none';
#   $('apiBaseHint').style.display = show ? 'none' : 'inline';
# };
 
# const RATIO_META = {
#   '9:16': {w:9,h:16}, '16:9': {w:16,h:9}, '1:1': {w:1,h:1}, '4:5': {w:4,h:5}, '4:3': {w:4,h:3},
# };
# const STYLE_META = {
#   bold_pop:       {bg:'#111', color:'#fff', hi:'#FFD700', sample:'THIS IS'},
#   karaoke_classic:{bg:'#111', color:'#fff', hi:'#FFA500', sample:'FILLS IN'},
#   neon_glow:      {bg:'#111', color:'#fff', hi:'#00F0FF', sample:'GLOWS UP'},
#   minimal_clean:  {bg:'#111', color:'#fff', hi:'#fff', sample:'stays quiet'},
#   creator_bold:   {bg:'#111', color:'#FFFF00', hi:'#FFFF00', sample:'HUGE TEXT'},
# };
 
# // ── boot: load styles / ratios / features from backend ──
# async function loadMeta(){
#   try{
#     const [styles, ratios, features] = await Promise.all([
#       fetch(api('/api/editor/styles')).then(r=>r.json()),
#       fetch(api('/api/editor/ratios')).then(r=>r.json()),
#       fetch(api('/api/editor/features')).then(r=>r.json()),
#     ]);
#     renderStyles(styles);
#     renderRatios(ratios);
#     applyFeatureAvailability(features);
#   }catch(e){
#     $('featureWarn').style.display='block';
#     $('featureWarn').textContent = 'Could not reach the backend at ' + api('') + ' — set the API base URL above and reload.';
#   }
# }
 
# function renderStyles(styles){
#   const list = $('styleList'); list.innerHTML='';
#   Object.entries(styles).forEach(([key, meta])=>{
#     const m = STYLE_META[key] || {bg:'#111', color:'#fff', hi:'#fff', sample:'Sample'};
#     const card = document.createElement('div');
#     card.className = 'style-card' + (key===state.selectedStyle ? ' active':'');
#     card.innerHTML = `<div class="style-swatch" style="background:${m.bg}">
#         <span style="color:${m.color}">${m.sample.split(' ')[0]}</span>&nbsp;<span style="color:${m.hi}">${m.sample.split(' ')[1]||''}</span>
#       </div>
#       <div><div class="name">${meta.label.split(' (')[0]}</div><div class="lbl">${(meta.label.match(/\((.*)\)/)||[,''])[1]}</div></div>`;
#     card.onclick = ()=>{ state.selectedStyle = key; renderStyles(styles); };
#     list.appendChild(card);
#   });
# }
 
# // Export ratios are now MULTI-select — tap any number of cards and every
# // one you pick gets rendered and offered as a download, all in the same
# // render job. "Auto" (native size, no crop, zero quality loss) is its own
# // card too. state.previewRatio (separate from the export selection) just
# // drives which crop box is drawn live over the video for editing.
# function renderRatios(ratios){
#   const grid = $('ratioGrid'); grid.innerHTML='';
#   Object.entries(ratios).forEach(([key, r])=>{
#     const isAuto = key === 'auto';
#     const srcAspect = (state.srcW && state.srcH) ? {w: state.srcW, h: state.srcH} : {w:9, h:16};
#     const meta = isAuto ? srcAspect : (RATIO_META[key] || {w:1,h:1});
#     const scale = 30 / Math.max(meta.w, meta.h);
#     const selected = state.selectedRatios.has(key);
#     const card = document.createElement('div');
#     card.className = 'ratio-card' + (selected ? ' active':'');
#     card.innerHTML = `<div class="shape" style="width:${meta.w*scale}px; height:${meta.h*scale}px;"></div>
#       <div class="name">${selected ? '✓ ' : ''}${key === 'auto' ? 'Auto' : key}</div><div class="lbl">${r.label}</div>`;
#     card.onclick = ()=>{
#       if(state.selectedRatios.has(key)) state.selectedRatios.delete(key);
#       else state.selectedRatios.add(key);
#       if(!state.selectedRatios.size) state.selectedRatios.add(key); // keep at least one selected
#       if(!isAuto){ state.previewRatio = key; drawCropForCurrentTime(); }
#       renderRatios(ratios);
#     };
#     grid.appendChild(card);
#   });
# }
 
# function applyFeatureAvailability(f){
#   toggleRow('rowFace', 'chkFace', f.face_tracking, 'opencv-python not installed on the server');
#   toggleRow('rowSubs', 'chkSubs', f.subtitles, 'faster-whisper not installed on the server');
#   toggleRow('rowHook', 'chkHook', true, '');
#   toggleRow('rowJump', 'chkJump', true, '');
#   if (!f.face_tracking_advanced && f.face_tracking){
#     $('rowFace').querySelector('.desc').textContent = 'Standard tracker active (install mediapipe for advanced mode)';
#   }
# }
# function toggleRow(rowId, chkId, available, msg){
#   if(!available){
#     $(rowId).classList.add('disabled');
#     $(chkId).checked = false;
#     $(chkId).disabled = true;
#     $(rowId).querySelector('.desc').textContent = msg;
#   }
# }
 
# // ── fill mode ──
# document.querySelectorAll('.fill-opt').forEach(el=>{
#   el.onclick = ()=>{
#     document.querySelectorAll('.fill-opt').forEach(o=>o.classList.remove('active'));
#     el.classList.add('active');
#     state.fillMode = el.dataset.fill;
#     drawCropForCurrentTime();
#   };
# });
 
# // ── fetch-from-link (built-in yt-dlp resolver, high quality single mp4) ──
# // Mirrors the Downloader tab's speed trick (cached auth mode) but skips the
# // full format-picker UI on purpose: this always grabs the single best
# // available video+audio quality and merges it into one mp4, then feeds it
# // straight into the exact same analyze/render pipeline as an upload.
# let fetchState = { url: null };
 
# $('fetchInfoBtn').onclick = async ()=>{
#   const url = $('fetchUrl').value.trim();
#   if(!url) return;
#   $('fetchStatus').textContent = 'Resolving link…';
#   $('fetchCard').style.display = 'none';
#   try{
#     const res = await fetch(api('/api/editor/fetch/info'), {
#       method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url})
#     });
#     const data = await res.json();
#     if(data.error){ $('fetchStatus').textContent = data.error; return; }
#     fetchState.url = url;
#     fetchState.meta = data;   // keep the FULL metadata (views/likes/uploader/etc.) for later
#     $('fetchThumb').src = data.thumbnail || '';
#     $('fetchTitle').textContent = data.title || 'Untitled video';
#     $('fetchSub').textContent = [data.uploader, data.duration_str, data.resolution].filter(Boolean).join(' · ');
#     $('fetchCard').style.display = 'flex';
#     $('fetchStatus').textContent = '';
#   }catch(e){
#     $('fetchStatus').textContent = 'Could not reach the server to resolve that link. Make sure the app is running and reload this page.';
#   }
# };
 
# $('fetchUseBtn').onclick = async ()=>{
#   $('fetchUseBtn').disabled = true;
#   $('fetchStatus').textContent = 'Fetching best quality (video + audio, single file)…';
#   try{
#     const res = await fetch(api('/api/editor/fetch/start'), {
#       method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({url: fetchState.url})
#     });
#     const data = await res.json();
#     if(data.error){ $('fetchStatus').textContent = data.error; $('fetchUseBtn').disabled = false; return; }
#     pollFetchProgress(data.fetch_id);
#   }catch(e){
#     $('fetchStatus').textContent = 'Fetch failed to start — check the connection and try again.';
#     $('fetchUseBtn').disabled = false;
#   }
# };
 
# async function pollFetchProgress(fetchId){
#   let st;
#   try{
#     const res = await fetch(api('/api/editor/fetch/progress/'+fetchId));
#     st = await res.json();
#   }catch(e){
#     setTimeout(()=>pollFetchProgress(fetchId), 1200); // transient network hiccup, keep polling
#     return;
#   }
#   if(st.status === 'error'){
#     $('fetchStatus').textContent = 'Error: ' + st.error;
#     $('fetchUseBtn').disabled = false;
#     return;
#   }
#   if(st.status === 'done'){
#     $('fetchStatus').textContent = 'Downloaded — analyzing…';
#     $('fetchUseBtn').disabled = false;
#     $('fetchCard').style.display = 'none';
#     $('fetchUrl').value = '';
#     loadSource(st.source_path, fetchState.meta, 'fetch');
#     return;
#   }
#   const pct = st.percent != null ? st.percent + '%' : '';
#   $('fetchStatus').textContent = `${st.stage || 'downloading'}… ${pct} ${st.speed_str || ''}`.trim();
#   setTimeout(()=>pollFetchProgress(fetchId), 900);
# }
 
# // ── server library (downloader output + earlier uploads) ──
# // Lets the user reuse a video already fetched via the Downloader tab (or an
# // earlier upload) without picking it from disk again — same analyze/render
# // pipeline as a fresh upload, just a different source_path origin.
# $('refreshLibBtn').onclick = loadLibrary;
 
# async function loadLibrary(){
#   const grid = $('libGrid');
#   grid.innerHTML = '<span class="lib-empty">Loading…</span>';
#   try{
#     const res = await fetch(api('/api/editor/sources'));
#     const data = await res.json();
#     const items = data.sources || [];
#     if(!items.length){
#       grid.innerHTML = '<span class="lib-empty">Nothing yet — fetch a video in the Downloader tab, or upload one above.</span>';
#       return;
#     }
#     grid.innerHTML = '';
#     items.forEach(it=>{
#       const el = document.createElement('div');
#       el.className = 'lib-item';
#       el.innerHTML = `<span class="lib-tag">${it.origin === 'downloader' ? '📥 downloaded' : '📤 uploaded'}</span>
#         <div class="lib-name">${it.name}</div>
#         <div class="lib-meta"><span>${it.size_str||''}</span><span>${it.modified_str||''}</span></div>`;
#       el.onclick = ()=>{
#         document.querySelectorAll('.lib-item.active').forEach(n=>n.classList.remove('active'));
#         el.classList.add('active');
#         loadSource(it.path, null, it.origin === 'downloader' ? 'downloader' : 'pc');
#       };
#       grid.appendChild(el);
#     });
#   }catch(e){
#     grid.innerHTML = '<span class="lib-empty">Could not reach the server — make sure the app is running, then hit Refresh again.</span>';
#   }
# }
 
# loadLibrary();
 
# // ── upload flow ──
# $('dropzone').onclick = ()=> $('fileInput').click();
# $('fileInput').onchange = (e)=> handleFile(e.target.files[0]);
# ['dragover','dragleave','drop'].forEach(evt=>{
#   $('dropzone').addEventListener(evt, (e)=>{
#     e.preventDefault();
#     $('dropzone').classList.toggle('drag', evt==='dragover');
#     if(evt==='drop' && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
#   });
# });
 
# async function handleFile(file){
#   if(!file) return;
#   $('video').src = URL.createObjectURL(file);
#   $('previewSection').style.display = 'block';
#   $('statusLine').textContent = 'Uploading…';
#   try{
#     const fd = new FormData(); fd.append('file', file);
#     const res = await fetch(api('/api/editor/upload'), {method:'POST', body: fd});
#     const data = await res.json();
#     if(data.error){ showAnalysisError('Upload error: ' + data.error); return; }
#     loadSource(data.source_path, null, 'pc');
#   }catch(e){
#     showAnalysisError('Could not reach the server to upload this file.');
#   }
# }
 
# // ── unified source loader — every origin (fetched link, downloader
# // library, or a file picked from this PC) funnels through here so
# // analysis always runs the same way and the Export panel always ends up
# // enabled the same way, no matter where the video came from. ──
# async function loadSource(sourcePath, meta, origin){
#   state.sourcePath = sourcePath;
#   state.sourceMeta = meta || null;
#   state.sourceOrigin = origin || 'pc';
#   $('video').src = api('/api/editor/preview?path=' + encodeURIComponent(sourcePath));
#   $('previewSection').style.display = 'block';
#   renderSourceMeta();
#   await runAnalysis();
# }
 
# // Shows the rich details panel. For links fetched from YouTube/Instagram/
# // etc. this uses the metadata the SERVER already pulled during Fetch (no
# // need to re-derive title/uploader/views by hand); for PC uploads or
# // library items with no such metadata it just relies on the ffprobe-based
# // technical details that come back from /api/editor/analyze.
# function renderSourceMeta(){
#   const el = $('sourceMetaPanel');
#   if(!el) return;
#   const m = state.sourceMeta;
#   if(!m){ el.style.display = 'none'; el.innerHTML=''; return; }
#   const rows = [
#     ['Title', m.title],
#     ['Channel / uploader', m.uploader],
#     ['Views', m.view_count_str],
#     ['Likes', m.like_count_str],
#     ['Uploaded', m.upload_date_str],
#     ['Source resolution', m.resolution],
#   ].filter(([,v]) => v);
#   if(!rows.length){ el.style.display='none'; return; }
#   el.style.display = 'block';
#   el.innerHTML = '<div class="lib-header"><span>ℹ️ Fetched video details</span></div>' +
#     rows.map(([k,v]) => `<div class="status-line"><b>${k}:</b> ${v}</div>`).join('');
# }
 
# function showAnalysisError(msg){
#   $('statusLine').innerHTML = `<span style="color:var(--amber);">${msg}</span> ` +
#     `<button type="button" class="btn-small" id="retryAnalysisBtn" style="margin-left:6px;">Retry analysis</button>`;
#   const btn = $('retryAnalysisBtn');
#   if(btn) btn.onclick = () => runAnalysis();
#   $('renderBtn').disabled = true;
#   $('renderBtn').textContent = 'Analyze a video first';
# }
 
# async function runAnalysis(){
#   if(!state.sourcePath){ showAnalysisError('No video selected yet.'); return; }
#   $('statusLine').textContent = 'Analyzing — face track + transcript + hook detection…';
#   $('renderBtn').disabled = true;
#   const body = {
#     source_path: state.sourcePath,
#     face_tracking: $('chkFace').checked && !$('chkFace').disabled,
#     subtitles: $('chkSubs').checked && !$('chkSubs').disabled,
#     hook_cut: true,
#     jump_cut: true,
#   };
#   let data;
#   try{
#     const res = await fetch(api('/api/editor/analyze'), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
#     if(!res.ok){ showAnalysisError(`Analysis request failed (server said HTTP ${res.status}).`); return; }
#     data = await res.json();
#   }catch(e){
#     showAnalysisError('Could not reach the server to analyze this video — check it is still running, then retry.');
#     return;
#   }
#   if(data.error){ showAnalysisError('Analysis error: ' + data.error); return; }
#   state.duration = data.probe.duration || 0;
#   state.srcW = (data.source_size && data.source_size.w) || data.probe.width;
#   state.srcH = (data.source_size && data.source_size.h) || data.probe.height;
#   state.faceTrack = data.face_track || [];
#   state.words = data.words || [];
#   state.suggestedHook = data.suggested_hook || null;
#   state.suggestedSilences = data.suggested_segments || null;
#   $('useHookBtn').style.display = state.suggestedHook ? 'inline-block' : 'none';
#   $('useSilenceBtn').style.display = state.suggestedSilences ? 'inline-block' : 'none';
#   drawTimeline();
#   // Export becomes available the instant analysis finishes, identically
#   // for an uploaded PC file, a Downloader-library pick, or a fresh link
#   // fetch — there is no special-casing by source here.
#   $('renderBtn').disabled = false;
#   $('renderBtn').textContent = 'Render exports';
#   const srcTag = state.sourceOrigin === 'fetch' ? 'fetched video' : (state.sourceOrigin === 'downloader' ? 'downloaded video' : 'uploaded video');
#   $('statusLine').innerHTML = `Analyzed ${srcTag} — <b>${state.faceTrack.length}</b> face samples · <b>${state.words.length}</b> words transcribed · duration <b>${fmtTime(state.duration)}</b> · source <b>${state.srcW}×${state.srcH}</b>`;
# }
 
# // ── video controls ──
# const video = $('video');
# $('playBtn').onclick = ()=>{ video.paused ? video.play() : video.pause(); };
# video.onplay = ()=> $('playBtn').textContent = '❚❚';
# video.onpause = ()=> $('playBtn').textContent = '▶';
# video.ontimeupdate = ()=>{
#   $('timeLabel').textContent = `${fmtTime(video.currentTime)} / ${fmtTime(video.duration||0)}`;
#   const pct = video.duration ? (video.currentTime/video.duration*100) : 0;
#   $('playhead').style.left = pct + '%';
#   drawCropForCurrentTime();
# };
# function fmtTime(t){ t=Math.max(0,t||0); const m=Math.floor(t/60), s=Math.floor(t%60); return `${m}:${String(s).padStart(2,'0')}`; }
 
# // ── crop overlay: shows the current auto/manual crop box, draggable ──
# function currentCrop(){
#   const kfs = state.manualKeyframes.length ? state.manualKeyframes : state.faceTrack.map(p=>({t:p.t, x:p.cx, y:p.cy}));
#   if(!kfs.length) return {cx:0.5, cy:0.5};
#   const t = video.currentTime;
#   let lo = kfs[0], hi = kfs[kfs.length-1];
#   for(let i=0;i<kfs.length-1;i++){ if(kfs[i].t<=t && kfs[i+1].t>=t){ lo=kfs[i]; hi=kfs[i+1]; break; } }
#   const span = (hi.t - lo.t) || 1;
#   const f = Math.min(1, Math.max(0, (t - lo.t)/span));
#   const cxKey = state.manualKeyframes.length ? 'x' : 'cx';
#   const cyKey = state.manualKeyframes.length ? 'y' : 'cy';
#   return { cx: lo[cxKey] + (hi[cxKey]-lo[cxKey])*f, cy: lo[cyKey] + (hi[cyKey]-lo[cyKey])*f };
# }
 
# function drawCropForCurrentTime(){
#   const box = $('cropBox');
#   if(state.fillMode === 'blur_pad' || !$('chkFace').checked){ box.style.display='none'; return; }
#   if(!state.srcW || !video.videoWidth){ return; }
#   const meta = RATIO_META[state.previewRatio || '9:16'];
#   const targetAR = meta.w/meta.h;
#   const srcAR = video.videoWidth/video.videoHeight;
#   let cropWFrac, cropHFrac;
#   if(srcAR > targetAR){ cropHFrac = 1; cropWFrac = targetAR/srcAR; }
#   else { cropWFrac = 1; cropHFrac = srcAR/targetAR; }
#   const {cx, cy} = currentCrop();
#   let left = cx - cropWFrac/2, top = cy - cropHFrac/2;
#   left = Math.min(1-cropWFrac, Math.max(0, left));
#   top = Math.min(1-cropHFrac, Math.max(0, top));
 
#   const frame = $('previewFrame').getBoundingClientRect();
#   box.style.display = 'block';
#   box.style.left = (left*100)+'%';
#   box.style.top = (top*100)+'%';
#   box.style.width = (cropWFrac*100)+'%';
#   box.style.height = (cropHFrac*100)+'%';
#   $('cropReadout').textContent = state.previewRatio || '9:16';
#   box.dataset.cx = cx; box.dataset.cy = cy;
# }
 
# // dragging the crop box records a manual keyframe candidate (committed via "pin crop here")
# let dragStart = null;
# $('cropBox').addEventListener('mousedown', (e)=>{
#   state.dragging = true; $('cropBox').classList.add('dragging');
#   dragStart = {x:e.clientX, y:e.clientY, left:parseFloat($('cropBox').style.left), top:parseFloat($('cropBox').style.top)};
# });
# window.addEventListener('mousemove', (e)=>{
#   if(!state.dragging) return;
#   const frame = $('previewFrame').getBoundingClientRect();
#   const dx = (e.clientX-dragStart.x)/frame.width*100;
#   const dy = (e.clientY-dragStart.y)/frame.height*100;
#   const box = $('cropBox');
#   const newLeft = Math.min(100-parseFloat(box.style.width), Math.max(0, dragStart.left+dx));
#   const newTop = Math.min(100-parseFloat(box.style.height), Math.max(0, dragStart.top+dy));
#   box.style.left = newLeft+'%'; box.style.top = newTop+'%';
#   box.dataset.cx = (newLeft + parseFloat(box.style.width)/2)/100;
#   box.dataset.cy = (newTop + parseFloat(box.style.height)/2)/100;
# });
# window.addEventListener('mouseup', ()=>{ state.dragging=false; $('cropBox').classList.remove('dragging'); });
 
# $('addKfBtn').onclick = ()=>{
#   const box = $('cropBox');
#   const meta = RATIO_META[state.previewRatio || '9:16'];
#   const srcAR = video.videoWidth/video.videoHeight;
#   const targetAR = meta.w/meta.h;
#   let cropW, cropH;
#   if(srcAR > targetAR){ cropH = state.srcH; cropW = Math.round(cropH*targetAR); }
#   else { cropW = state.srcW; cropH = Math.round(cropW/targetAR); }
#   const cx = parseFloat(box.dataset.cx||0.5), cy = parseFloat(box.dataset.cy||0.5);
#   let x = Math.round(cx*state.srcW - cropW/2), y = Math.round(cy*state.srcH - cropH/2);
#   x = Math.max(0, Math.min(x, state.srcW-cropW)); y = Math.max(0, Math.min(y, state.srcH-cropH));
#   state.manualKeyframes = state.manualKeyframes.filter(k=>Math.abs(k.t-video.currentTime)>0.05);
#   state.manualKeyframes.push({t: video.currentTime, x, y});
#   state.manualKeyframes.sort((a,b)=>a.t-b.t);
#   drawTimeline();
#   $('statusLine').textContent = `Pinned manual crop at ${fmtTime(video.currentTime)} — ${state.manualKeyframes.length} manual keyframe(s) active (overrides auto tracking).`;
# };
# $('clearKfBtn').onclick = ()=>{ state.manualKeyframes=[]; drawTimeline(); drawCropForCurrentTime(); };
 
# // ── timeline: face density, segments, hook suggestion ──
# function drawTimeline(){
#   const tl = $('timeline');
#   [...tl.querySelectorAll('.tl-face,.tl-hook')].forEach(e=>e.remove());
#   if(state.duration){
#     state.faceTrack.forEach(p=>{
#       const el = document.createElement('div');
#       el.className='tl-face';
#       el.style.left = (p.t/state.duration*100)+'%'; el.style.width='2px';
#       tl.appendChild(el);
#     });
#     if(state.suggestedHook){
#       const el = document.createElement('div'); el.className='tl-hook';
#       el.style.left = (state.suggestedHook.start/state.duration*100)+'%';
#       el.style.width = ((state.suggestedHook.end-state.suggestedHook.start)/state.duration*100)+'%';
#       tl.appendChild(el);
#     }
#   }
#   renderSegList();
# }
# function renderSegList(){
#   const list = $('segList'); list.innerHTML='';
#   state.segments.forEach((seg, i)=>{
#     const row = document.createElement('div'); row.className='seg-row';
#     row.innerHTML = `<span class="mono">#${i+1}</span><span class="mono">${fmtTime(seg.start)} → ${fmtTime(seg.end)}</span><span class="spacer"></span><button data-i="${i}">✕</button>`;
#     row.querySelector('button').onclick = ()=>{ state.segments.splice(i,1); renderSegList(); };
#     list.appendChild(row);
#   });
# }
# $('timeline').addEventListener('click', (e)=>{
#   const rect = $('timeline').getBoundingClientRect();
#   const frac = (e.clientX-rect.left)/rect.width;
#   video.currentTime = frac * (video.duration||0);
# });
# $('addSegBtn').onclick = ()=>{
#   const t = video.currentTime;
#   state.segments.push({start: Math.max(0, +(t-3).toFixed(2)), end: Math.min(state.duration, +(t+3).toFixed(2))});
#   renderSegList();
# };
# $('useHookBtn').onclick = ()=>{
#   if(!state.suggestedHook) return;
#   state.segments = [state.suggestedHook, {start:0, end: state.duration}];
#   renderSegList();
#   $('chkHook').checked = true;
# };
# $('useSilenceBtn').onclick = ()=>{
#   if(!state.suggestedSilences) return;
#   state.segments = state.suggestedSilences;
#   renderSegList();
#   $('chkJump').checked = true;
# };
 
# // ── render ──
# $('renderBtn').onclick = async ()=>{
#   if(!state.selectedRatios.size){ $('statusLine').textContent = 'Pick at least one export ratio first.'; return; }
#   $('renderBtn').disabled = true;
#   $('progressWrap').style.display = 'block';
#   $('results').innerHTML = '';
#   const body = {
#     source_path: state.sourcePath,
#     export_ratios: Array.from(state.selectedRatios),
#     face_tracking: $('chkFace').checked && !$('chkFace').disabled,
#     subtitles: $('chkSubs').checked && !$('chkSubs').disabled,
#     hook_cut: $('chkHook').checked,
#     jump_cut: $('chkJump').checked,
#     normalize_audio: $('chkNorm').checked,
#     caption_style: state.selectedStyle,
#     fill_mode: state.fillMode,
#     quality: $('qualitySel').value,
#     manual_crop_keyframes: state.manualKeyframes.length ? state.manualKeyframes : null,
#     segments: state.segments.length ? state.segments : null,
#   };
#   try{
#     const res = await fetch(api('/api/editor/render'), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
#     const data = await res.json();
#     if(data.error){ $('statusLine').textContent = 'Render error: '+data.error; $('renderBtn').disabled=false; return; }
#     state.jobId = data.job_id;
#     pollProgress();
#   }catch(e){
#     $('statusLine').textContent = 'Could not reach the server to start the render — check it is still running and try again.';
#     $('renderBtn').disabled = false;
#   }
# };
 
# async function pollProgress(){
#   let job;
#   try{
#     const res = await fetch(api('/api/editor/progress/'+state.jobId));
#     job = await res.json();
#   }catch(e){
#     setTimeout(pollProgress, 1500); // transient hiccup, keep polling
#     return;
#   }
#   $('progressFill').style.width = (job.percent||0)+'%';
#   $('progressStage').textContent = job.stage || job.status;
#   $('progressPct').textContent = (job.percent||0)+'%';
#   if(job.status === 'error'){
#     $('statusLine').textContent = 'Render failed: '+job.error;
#     $('renderBtn').disabled = false;
#     return;
#   }
#   if(job.status !== 'done'){ setTimeout(pollProgress, 1200); return; }
#   $('renderBtn').disabled = false;
#   const results = $('results');
#   Object.entries(job.ratios||{}).forEach(([ratio, fname])=>{
#     const row = document.createElement('div'); row.className='result-row';
#     const tag = ratio === 'auto' ? 'Auto (original size — lossless)' : `${ratio} export`;
#     row.innerHTML = `<span class="name">${tag}</span><a href="${api('/api/editor/file/'+state.jobId+'/'+ratio)}" download>Download</a>`;
#     results.appendChild(row);
#   });
# }
 
# loadMeta();
# </script>
# </body>
# </html>
# """
 
 
# # Optional, best-effort link to downloader.py's in-memory job table so the
# # editor can show nicer names for downloaded videos ("Video Title.mp4"
# # instead of "<dl_id>_Video Title.mp4"). video_editor.py still works fine
# # standalone if downloader.py isn't registered — this never raises.
# def _downloader_jobs():
#     try:
#         import downloader
#         return downloader.DL_JOBS
#     except Exception:
#         return {}
 
# # ── configured once via init_editor() ──────────────────────────────
# EDIT_DIR = None
# SRC_DIR = None
# FFMPEG_PATH = "ffmpeg"
# FFPROBE_PATH = "ffprobe"
 
# # job_id -> {status, stage, percent, error, ratios: {ratio: filename}, config}
# EDIT_JOBS = {}
 
 
# def init_editor(base_dir, ffmpeg_path=None):
#     """Call once at startup, e.g. init_editor(BASE, FFMPEG)."""
#     global EDIT_DIR, SRC_DIR, FFMPEG_PATH, FFPROBE_PATH
#     base_dir = Path(base_dir)
#     EDIT_DIR = base_dir / "edited"
#     EDIT_DIR.mkdir(exist_ok=True)
#     SRC_DIR = base_dir / "downloads"   # reuse downloader's output as source pool
#     (EDIT_DIR / "uploads").mkdir(exist_ok=True)
#     if ffmpeg_path:
#         FFMPEG_PATH = str(ffmpeg_path)
#         # ffprobe normally sits next to ffmpeg
#         cand = Path(ffmpeg_path).with_name(
#             "ffprobe.exe" if str(ffmpeg_path).lower().endswith(".exe") else "ffprobe"
#         )
#         if cand.exists():
#             FFPROBE_PATH = str(cand)
 
 
# def _no_console_kwargs():
#     if os.name == "nt":
#         return {"creationflags": subprocess.CREATE_NO_WINDOW}
#     return {}
 
 
# def _run(cmd):
#     return subprocess.run(cmd, capture_output=True, text=True, **_no_console_kwargs())
 
 
# # ══════════════════════════════ feature availability ══════════════════════════════
 
# try:
#     import cv2
#     _HAS_CV2 = True
# except ImportError:
#     _HAS_CV2 = False
 
# try:
#     import mediapipe as mp
#     _HAS_MEDIAPIPE = True
# except ImportError:
#     _HAS_MEDIAPIPE = False
 
# try:
#     from faster_whisper import WhisperModel
#     _HAS_WHISPER = True
# except ImportError:
#     _HAS_WHISPER = False
 
# try:
#     import numpy as np
#     _HAS_NUMPY = True
# except ImportError:
#     _HAS_NUMPY = False
 
# try:
#     import yt_dlp
#     _HAS_YTDLP = True
# except ImportError:
#     _HAS_YTDLP = False
 
# _WHISPER_MODEL = None  # lazy-loaded singleton
 
 
# def feature_status():
#     return {
#         "face_tracking": _HAS_CV2,
#         "face_tracking_advanced": _HAS_CV2 and _HAS_MEDIAPIPE,
#         "subtitles": _HAS_WHISPER,
#         "hook_cut_auto": _HAS_NUMPY,
#         "fetch_from_link": _HAS_YTDLP,
#     }
 
 
# # ══════════════════════════════ presets ══════════════════════════════
 
# # width x height for each export ratio, chosen at "good enough to look sharp,
# # small enough to encode fast" — bump these if you want 4K exports.
# RATIO_PRESETS = {
#     # "auto" is a sentinel: w/h are resolved at render time from the SOURCE
#     # video's own dimensions (no target size baked in here). It means "keep
#     # the video exactly as shot" — no crop, no letterbox, no rescale — and
#     # whenever nothing else in the pipeline needs to touch the picture
#     # (no captions/color/manual crop requested) it is exported via ffmpeg
#     # stream-copy, i.e. the original video bytes are re-muxed, not
#     # re-encoded, so there is truly zero generational quality loss.
#     "auto":  {"w": None, "h": None, "label": "Auto — original size, no crop, zero quality loss"},
#     "9:16":  {"w": 1080, "h": 1920, "label": "Reels / Shorts / TikTok"},
#     "16:9":  {"w": 1920, "h": 1080, "label": "YouTube / Landscape"},
#     "1:1":   {"w": 1080, "h": 1080, "label": "Square / Feed post"},
#     "4:5":   {"w": 1080, "h": 1350, "label": "Instagram portrait"},
#     "4:3":   {"w": 1440, "h": 1080, "label": "Classic / Facebook"},
# }
 
# QUALITY_PRESETS = {
#     "high":   {"crf": 18, "preset": "slow"},
#     "medium": {"crf": 21, "preset": "medium"},
#     "fast":   {"crf": 23, "preset": "veryfast"},
# }
 
# # Words that get an automatic extra-emphasis color in captions even in
# # styles that don't do full word-by-word highlighting — numbers, money and
# # a short list of "power words" are what trending caption tools punch up.
# _EMPHASIS_PATTERN = None  # compiled lazily, see _is_emphasis_word()
# _POWER_WORDS = {
#     "free", "now", "never", "always", "secret", "insane", "crazy", "huge",
#     "warning", "stop", "wait", "new", "best", "worst", "you", "your",
# }
 
 
# def _is_emphasis_word(word):
#     import re
#     global _EMPHASIS_PATTERN
#     if _EMPHASIS_PATTERN is None:
#         _EMPHASIS_PATTERN = re.compile(r"[\d%$]|^\$")
#     w = word.strip(".,!?").lower()
#     return bool(_EMPHASIS_PATTERN.search(word)) or w in _POWER_WORDS
 
# # Trending caption styles. Each is a full ASS "Style:" line plus a couple of
# # behavior flags consumed by build_ass(). Colors are &HAABBGGRR (ASS order).
# CAPTION_STYLES = {
#     "bold_pop": {
#         "label": "Bold Pop (white + yellow highlight)",
#         "style": "Style: Default,Montserrat Black,84,&H00FFFFFF,&H0000D7FF,&H00101010,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,6,0,2,60,60,140,1",
#         "highlight_color": "&H0000D7FF",   # active word turns yellow/gold
#         "word_by_word": True,
#         "pop_scale": True,
#     },
#     "karaoke_classic": {
#         "label": "Karaoke Fill (progressive color wipe)",
#         "style": "Style: Default,Montserrat SemiBold,72,&H00FFFFFF,&H0000A5FF,&H00202020,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,4,0,2,60,60,150,1",
#         "highlight_color": "&H0000A5FF",
#         "word_by_word": True,
#         "karaoke_fill": True,
#     },
#     "neon_glow": {
#         "label": "Neon Glow (cyan outline pop)",
#         "style": "Style: Default,Montserrat Black,80,&H00FFFFFF,&H00FFF000,&H00902000,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,5,2,2,60,60,140,1",
#         "highlight_color": "&H00FFF000",
#         "word_by_word": True,
#         "pop_scale": True,
#     },
#     "minimal_clean": {
#         "label": "Minimal Clean (small centered white)",
#         "style": "Style: Default,Inter Medium,54,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
#                   "0,0,0,0,100,100,0,0,1,2,1,2,60,60,120,1",
#         "highlight_color": "&H00FFFFFF",
#         "word_by_word": False,
#         "pop_scale": False,
#     },
#     "creator_bold": {
#         "label": "Creator Bold (huge yellow, black outline)",
#         "style": "Style: Default,Montserrat Black,92,&H0000FFFF,&H000000FF,&H00000000,&H00000000,"
#                   "-1,0,0,0,100,100,0,0,1,8,0,2,50,50,160,1",
#         "highlight_color": "&H000000FF",
#         "word_by_word": True,
#         "pop_scale": True,
#     },
# }
 
 
# @editor_bp.route("/api/editor/styles")
# def api_editor_styles():
#     return jsonify({k: {"label": v["label"]} for k, v in CAPTION_STYLES.items()})
 
 
# @editor_bp.route("/api/editor/ratios")
# def api_editor_ratios():
#     return jsonify(RATIO_PRESETS)
 
 
# @editor_bp.route("/api/editor/editor")
# def api_editor_page():
#     """Serves the built-in editor UI — no separate .html file to manage,
#     it's embedded in this module (EDITOR_HTML below) and rendered straight
#     from memory. Open this URL in a tab; its fetch calls are same-origin."""
#     from flask import Response
#     return Response(EDITOR_HTML, mimetype="text/html")
 
 
# def _fmt_size(n):
#     if not n:
#         return "0 B"
#     n = float(n)
#     for unit in ["B", "KB", "MB", "GB"]:
#         if n < 1024:
#             return f"{n:.1f} {unit}"
#         n /= 1024
#     return f"{n:.1f} TB"
 
 
# @editor_bp.route("/api/editor/sources")
# def api_editor_sources():
#     """Lists videos the editor can open WITHOUT a fresh upload: everything
#     downloader.py has saved to DL_DIR/downloads (so a just-fetched video can
#     be sent straight into the editor), plus anything uploaded here before.
#     Both origins return the exact same shape and both feed the exact same
#     source_path used by /analyze and /render — the editor doesn't care where
#     a file came from."""
#     import datetime
#     jobs = _downloader_jobs()
#     # dl_id -> nicer display title, when downloader.py is registered too
#     title_by_stub = {}
#     for job in jobs.values():
#         fname = job.get("filename")
#         if fname and "_" in fname:
#             stub = fname.split("_", 1)[0]
#             title_by_stub[stub] = fname.split("_", 1)[-1]
 
#     sources = []
#     for folder, origin in ((SRC_DIR, "downloader"), (EDIT_DIR / "uploads", "upload")):
#         if not folder or not folder.exists():
#             continue
#         for p in sorted(folder.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
#             if not p.is_file() or p.suffix.lower() not in (".mp4", ".mov", ".mkv", ".webm", ".m4v"):
#                 continue
#             stub = p.name.split("_", 1)[0]
#             display = title_by_stub.get(stub) or (p.name.split("_", 1)[-1] if "_" in p.name else p.name)
#             stat = p.stat()
#             sources.append({
#                 "name": display,
#                 "path": str(p.resolve()),
#                 "origin": origin,
#                 "size": stat.st_size,
#                 "size_str": _fmt_size(stat.st_size),
#                 "modified": stat.st_mtime,
#                 "modified_str": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%d %b, %H:%M"),
#             })
#     sources.sort(key=lambda s: s["modified"], reverse=True)
#     return jsonify({"sources": sources})
 
 
# def _is_allowed_source(path):
#     """Only ever stream files that live under the editor's own known roots
#     (downloader's downloads folder, this module's uploads folder, or its
#     own rendered-output folder) — never an arbitrary server path."""
#     try:
#         rp = Path(path).resolve()
#     except Exception:
#         return False
#     for root in (SRC_DIR, EDIT_DIR / "uploads", EDIT_DIR):
#         if root and root.exists():
#             try:
#                 rp.relative_to(root.resolve())
#                 return True
#             except ValueError:
#                 continue
#     return False
 
 
# @editor_bp.route("/api/editor/preview")
# def api_editor_preview():
#     """Streams a source (or rendered) video for <video> tag playback/scrub,
#     so picking a file from the library previews instantly without a
#     round-trip upload. Range requests are handled by Flask's send_file."""
#     path = request.args.get("path", "")
#     if not path or not _is_allowed_source(path) or not Path(path).exists():
#         return "Not found or not allowed", 404
#     return send_file(path, conditional=True)
 
 
# @editor_bp.route("/api/editor/features")
# def api_editor_features():
#     return jsonify(feature_status())
 
 
# # ══════════════════════════════ probing ══════════════════════════════
 
# def _probe(path):
#     """Returns {duration, width, height, fps} via ffprobe."""
#     cmd = [
#         FFPROBE_PATH, "-v", "error", "-select_streams", "v:0",
#         "-show_entries", "stream=width,height,avg_frame_rate,duration",
#         "-show_entries", "format=duration",
#         "-of", "json", str(path),
#     ]
#     out = _run(cmd)
#     data = json.loads(out.stdout or "{}")
#     stream = (data.get("streams") or [{}])[0]
#     w = stream.get("width")
#     h = stream.get("height")
#     fps_raw = stream.get("avg_frame_rate") or "25/1"
#     try:
#         num, den = fps_raw.split("/")
#         fps = float(num) / float(den) if float(den) else 25.0
#     except Exception:
#         fps = 25.0
#     duration = stream.get("duration") or (data.get("format") or {}).get("duration")
#     try:
#         duration = float(duration)
#     except (TypeError, ValueError):
#         duration = 0.0
#     return {"duration": duration, "width": w, "height": h, "fps": fps}
 
 
# # ══════════════════════════════ face tracking ══════════════════════════════
 
# class _FaceTracker:
#     """Samples frames, finds a face each time, and produces a smoothed
#     (jitter-free) track of the subject's center point over time.
 
#     Uses mediapipe's BlazeFace model when available ("advanced" mode — much
#     more accurate on angled/small faces), otherwise falls back to OpenCV's
#     bundled Haar cascade so face tracking still works with just opencv-python
#     installed.
#     """
 
#     def __init__(self):
#         self.advanced = _HAS_MEDIAPIPE
#         if self.advanced:
#             self._mp_detector = mp.solutions.face_detection.FaceDetection(
#                 model_selection=1, min_detection_confidence=0.5
#             )
#         elif _HAS_CV2:
#             self._cascade = cv2.CascadeClassifier(
#                 cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#             )
#         else:
#             raise RuntimeError("opencv-python is required for face tracking")
 
#     def _detect(self, frame_bgr):
#         """Returns list of (cx, cy, w, h) in pixel coords, largest first."""
#         h_img, w_img = frame_bgr.shape[:2]
#         faces = []
#         if self.advanced:
#             rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#             result = self._mp_detector.process(rgb)
#             if result.detections:
#                 for d in result.detections:
#                     box = d.location_data.relative_bounding_box
#                     fw, fh = box.width * w_img, box.height * h_img
#                     fx = box.xmin * w_img + fw / 2
#                     fy = box.ymin * h_img + fh / 2
#                     faces.append((fx, fy, fw, fh))
#         else:
#             gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#             dets = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50))
#             for (x, y, w, h) in dets:
#                 faces.append((x + w / 2, y + h / 2, w, h))
#         faces.sort(key=lambda f: f[2] * f[3], reverse=True)  # largest face first
#         return faces
 
#     def track(self, video_path, sample_fps=2.0, progress_cb=None):
#         """Returns a list of {t, cx, cy, w, h} in NORMALIZED (0-1) coords,
#         exponentially smoothed so the crop pans instead of jumping."""
#         cap = cv2.VideoCapture(str(video_path))
#         if not cap.isOpened():
#             raise RuntimeError("Could not open video for face tracking")
#         src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#         w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         step = max(1, int(round(src_fps / sample_fps)))
 
#         raw_points = []
#         last_center = (0.5, 0.5)  # default: frame center when nobody's found yet
#         frame_idx = 0
#         while True:
#             ok = cap.grab()
#             if not ok:
#                 break
#             if frame_idx % step == 0:
#                 ok2, frame = cap.retrieve()
#                 if ok2:
#                     faces = self._detect(frame)
#                     if faces:
#                         fx, fy, fw, fh = faces[0]
#                         cx, cy = fx / w_img, fy / h_img
#                         last_center = (cx, cy)
#                     else:
#                         cx, cy = last_center  # hold last known position
#                     t = frame_idx / src_fps
#                     raw_points.append({"t": t, "cx": cx, "cy": cy})
#                 if progress_cb and total_frames:
#                     progress_cb(min(99, int(frame_idx * 100 / total_frames)))
#             frame_idx += 1
#         cap.release()
 
#         if not raw_points:
#             raw_points = [{"t": 0.0, "cx": 0.5, "cy": 0.5}]
 
#         # EMA smoothing so the crop glides instead of snapping every sample
#         alpha = 0.25
#         smoothed = [raw_points[0]]
#         for p in raw_points[1:]:
#             prev = smoothed[-1]
#             smoothed.append({
#                 "t": p["t"],
#                 "cx": prev["cx"] + alpha * (p["cx"] - prev["cx"]),
#                 "cy": prev["cy"] + alpha * (p["cy"] - prev["cy"]),
#             })
#         return smoothed, (w_img, h_img)
 
 
# def analyze_face_track(video_path, sample_fps=2.0, progress_cb=None):
#     if not _HAS_CV2:
#         raise RuntimeError("opencv-python is not installed — face tracking unavailable")
#     tracker = _FaceTracker()
#     points, (w, h) = tracker.track(video_path, sample_fps=sample_fps, progress_cb=progress_cb)
#     return points, w, h, tracker.advanced
 
 
# # ══════════════════════════════ crop-window math ══════════════════════════════
 
# def _crop_size_for_ratio(src_w, src_h, target_w, target_h):
#     """Largest crop rectangle matching the target aspect that still fits
#     inside the source frame (so we crop, never letterbox/pad)."""
#     target_ar = target_w / target_h
#     src_ar = src_w / src_h
#     if src_ar > target_ar:
#         # source is wider than target -> crop width, keep full height
#         crop_h = src_h
#         crop_w = int(round(crop_h * target_ar))
#     else:
#         # source is taller than target -> crop height, keep full width
#         crop_w = src_w
#         crop_h = int(round(crop_w / target_ar))
#     return crop_w, crop_h
 
 
# def _build_crop_keyframes(track_points, src_w, src_h, crop_w, crop_h):
#     """Turns normalized face-center points into pixel-space crop x/y
#     top-left keyframes, clamped so the crop never leaves the frame."""
#     kfs = []
#     for p in track_points:
#         cx_px = p["cx"] * src_w
#         cy_px = p["cy"] * src_h
#         x = int(cx_px - crop_w / 2)
#         y = int(cy_px - crop_h / 2)
#         x = max(0, min(x, src_w - crop_w))
#         y = max(0, min(y, src_h - crop_h))
#         kfs.append({"t": p["t"], "x": x, "y": y})
#     return kfs
 
 
# def _expr_chain(keyframes, key):
#     """Builds an ffmpeg time-expression string that linearly interpolates
#     between keyframes for either 'x' or 'y' — this drives a moving crop
#     window without needing any external filter graph patching."""
#     if len(keyframes) == 1:
#         return str(keyframes[0][key])
#     expr = str(keyframes[-1][key])
#     for i in range(len(keyframes) - 2, -1, -1):
#         t0, v0 = keyframes[i]["t"], keyframes[i][key]
#         t1, v1 = keyframes[i + 1]["t"], keyframes[i + 1][key]
#         if t1 <= t0:
#             continue
#         # linear ramp between (t0,v0) and (t1,v1), holds v0 before t0
#         ramp = f"({v0}+({v1}-{v0})*(t-{t0})/{(t1 - t0)})"
#         expr = f"if(lt(t,{t1}),{ramp},{expr})"
#     return expr
 
 
# # ══════════════════════════════ subtitles ══════════════════════════════
 
# def _load_whisper(model_size="small"):
#     global _WHISPER_MODEL
#     if _WHISPER_MODEL is None:
#         _WHISPER_MODEL = WhisperModel(model_size, compute_type="int8")
#     return _WHISPER_MODEL
 
 
# def transcribe_words(video_path, model_size="small", language=None):
#     """Returns [{start, end, text}, ...] word-level timestamps."""
#     if not _HAS_WHISPER:
#         raise RuntimeError("faster-whisper is not installed — subtitles unavailable")
#     model = _load_whisper(model_size)
#     segments, _info = model.transcribe(str(video_path), word_timestamps=True, language=language)
#     words = []
#     for seg in segments:
#         for w in (seg.words or []):
#             text = (w.word or "").strip()
#             if text:
#                 words.append({"start": w.start, "end": w.end, "text": text})
#     return words
 
 
# def _chunk_words(words, max_words=4, max_span=1.6):
#     """Groups words into short on-screen caption chunks — the standard
#     'trending shorts' style of 2-5 words on screen at a time, not full
#     sentences."""
#     chunks = []
#     cur = []
#     for w in words:
#         if cur and (len(cur) >= max_words or (w["end"] - cur[0]["start"]) > max_span):
#             chunks.append(cur)
#             cur = []
#         cur.append(w)
#     if cur:
#         chunks.append(cur)
#     return chunks
 
 
# def _ass_time(t):
#     h = int(t // 3600)
#     m = int((t % 3600) // 60)
#     s = t % 60
#     return f"{h}:{m:02d}:{s:05.2f}"
 
 
# def build_ass(words, style_name, video_w, video_h):
#     """Builds a full .ass subtitle document with word-by-word highlight
#     animation for the chosen trending style, sized for the given output
#     resolution so text scales correctly across aspect ratios."""
#     style = CAPTION_STYLES.get(style_name, CAPTION_STYLES["bold_pop"])
#     header = f"""[Script Info]
# ScriptType: v4.00+
# PlayResX: {video_w}
# PlayResY: {video_h}
# ScaledBorderAndShadow: yes
 
# [V4+ Styles]
# Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
# {style["style"]}
 
# [Events]
# Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
# """
#     lines = []
#     for chunk in _chunk_words(words):
#         start, end = chunk[0]["start"], chunk[-1]["end"]
#         if style.get("karaoke_fill"):
#             # \k tags: whole line present, active word wipes to highlight color
#             text = "".join(
#                 r"{\k%d}%s " % (max(1, int(round((w["end"] - w["start"]) * 100))), w["text"])
#                 for w in chunk
#             )
#             lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
#         elif style.get("word_by_word"):
#             # emit one Dialogue event per word so only the current word is
#             # shown in the highlight color while the rest stay default —
#             # gives the punchy "pop" caption look trending on shorts.
#             for w in chunk:
#                 parts = []
#                 for w2 in chunk:
#                     if w2 is w:
#                         scale = r"\fscx115\fscy115" if style.get("pop_scale") else ""
#                         parts.append(r"{\c%s%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], scale, w2["text"]))
#                     elif _is_emphasis_word(w2["text"]):
#                         # numbers / money / power-words get punched up even
#                         # when they're not the "active" word of the moment
#                         parts.append(r"{\c%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], w2["text"]))
#                     else:
#                         parts.append(w2["text"])
#                 text = " ".join(parts)
#                 lines.append(f"Dialogue: 0,{_ass_time(w['start'])},{_ass_time(w['end'])},Default,,0,0,0,,{text}")
#         else:
#             parts = []
#             for w2 in chunk:
#                 if _is_emphasis_word(w2["text"]):
#                     parts.append(r"{\c%s}%s{\c&HFFFFFF&}" % (style["highlight_color"], w2["text"]))
#                 else:
#                     parts.append(w2["text"])
#             text = " ".join(parts)
#             lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}")
#     return header + "\n".join(lines) + "\n"
 
 
# # ══════════════════════════════ hook-cut / segments ══════════════════════════════
 
# def _read_pcm_mono(video_path, sr=8000):
#     """Decodes a fast, low-res mono PCM stream via ffmpeg for lightweight
#     loudness analysis (used only to pick the auto hook window)."""
#     cmd = [FFMPEG_PATH, "-v", "error", "-i", str(video_path),
#            "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
#     proc = subprocess.run(cmd, capture_output=True, **_no_console_kwargs())
#     raw = proc.stdout
#     count = len(raw) // 2
#     samples = struct.unpack(f"<{count}h", raw[:count * 2])
#     return np.array(samples, dtype=np.float32) / 32768.0, sr
 
 
# def auto_hook_window(video_path, duration, hook_len=6.0):
#     """Finds the loudest `hook_len`-second window in the clip — a cheap but
#     effective proxy for 'most energetic / most likely to hook a viewer'
#     moment, used to auto-suggest where the exported clip should start."""
#     if not _HAS_NUMPY:
#         raise RuntimeError("numpy is not installed — auto hook-cut unavailable")
#     audio, sr = _read_pcm_mono(video_path)
#     if len(audio) == 0:
#         return {"start": 0.0, "end": min(hook_len, duration)}
#     win = int(hook_len * sr)
#     if win >= len(audio):
#         return {"start": 0.0, "end": duration}
#     energy = audio ** 2
#     # cumulative sum for O(1) windowed average lookups
#     cumsum = np.cumsum(np.insert(energy, 0, 0))
#     window_energy = cumsum[win:] - cumsum[:-win]
#     best_start_sample = int(np.argmax(window_energy))
#     start = best_start_sample / sr
#     return {"start": round(start, 2), "end": round(min(start + hook_len, duration), 2)}
 
 
# def detect_silences(video_path, noise_db=-30, min_silence=0.5):
#     """Runs ffmpeg's silencedetect and returns the list of NON-silent
#     {start,end} ranges — i.e. the segments worth keeping. This powers
#     automatic 'jump-cut' editing (removing dead air/pauses), the other
#     trending short-form edit style besides the loudest-window hook-cut."""
#     cmd = [FFMPEG_PATH, "-v", "info", "-i", str(video_path),
#            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
#     result = _run(cmd)
#     log = result.stderr or ""
#     starts, ends = [], []
#     for line in log.splitlines():
#         if "silence_start" in line:
#             starts.append(float(line.split("silence_start:")[1].strip().split(" ")[0]))
#         elif "silence_end" in line:
#             ends.append(float(line.split("silence_end:")[1].strip().split(" ")[0].split("|")[0]))
#     info = _probe(video_path)
#     duration = info["duration"]
#     silences = list(zip(starts, ends[: len(starts)]))
#     keep = []
#     cursor = 0.0
#     for s, e in silences:
#         if s > cursor:
#             keep.append({"start": round(cursor, 2), "end": round(s, 2)})
#         cursor = max(cursor, e)
#     if cursor < duration:
#         keep.append({"start": round(cursor, 2), "end": round(duration, 2)})
#     return [k for k in keep if k["end"] - k["start"] > 0.15]
 
 
# def _build_concat_file(video_path, segments, tmp_dir):
#     """Writes an ffmpeg concat-demuxer list after trimming each segment,
#     so multiple {start,end} ranges can be stitched in a custom, manually
#     chosen order (rearrange / hook-cut)."""
#     parts = []
#     for i, seg in enumerate(segments):
#         out = tmp_dir / f"seg_{i}.mp4"
#         cmd = [
#             FFMPEG_PATH, "-y", "-v", "error",
#             "-ss", str(seg["start"]), "-to", str(seg["end"]),
#             "-i", str(video_path),
#             "-c", "copy", "-avoid_negative_ts", "make_zero",
#             str(out),
#         ]
#         _run(cmd)
#         parts.append(out)
#     list_path = tmp_dir / "concat_list.txt"
#     with open(list_path, "w", encoding="utf-8") as f:
#         for p in parts:
#             f.write(f"file '{p.as_posix()}'\n")
#     return list_path
 
 
# def _measure_loudnorm(path):
#     """First pass of two-pass EBU R128 loudness normalization: measures the
#     input's actual loudness/true-peak/range so the second (real encode)
#     pass can normalize with `linear=true` against real measured values
#     instead of loudnorm's single-pass dynamic estimate — noticeably more
#     accurate and avoids the pumping/gain-riding artifacts single-pass mode
#     can introduce on speech-heavy short-form video."""
#     cmd = [FFMPEG_PATH, "-v", "info", "-i", str(path),
#            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
#            "-f", "null", "-"]
#     result = _run(cmd)
#     log = result.stderr or ""
#     try:
#         start = log.rindex("{")
#         end = log.rindex("}") + 1
#         return json.loads(log[start:end])
#     except (ValueError, json.JSONDecodeError):
#         return None  # falls back to single-pass below
 
 
# # ══════════════════════════════ render pipeline ══════════════════════════════
 
# def _render_one_ratio(job, source_path, ratio_key, track_points, src_w, src_h,
#                        words, tmp_dir, loud_measured=None):
#     cfg = job["config"]
#     ratio = RATIO_PRESETS[ratio_key]
#     is_auto = (ratio_key == "auto")
#     # "auto" keeps the SOURCE's own dimensions — no target size is imposed.
#     target_w, target_h = (src_w, src_h) if is_auto else (ratio["w"], ratio["h"])
#     quality = QUALITY_PRESETS.get(cfg.get("quality", "high"), QUALITY_PRESETS["high"])
 
#     fill_mode = cfg.get("fill_mode", "crop")  # "crop" (default) or "blur_pad"
#     # "auto" never crops or letterboxes — it IS the full original frame —
#     # so blur_pad/crop fill-mode choices simply don't apply to it.
#     if is_auto:
#         fill_mode = "none"
 
#     # ── subtitles: build the .ass first so both fill-mode branches can use it ──
#     ass_path = None
#     if cfg["features"].get("subtitles") and words:
#         ass_content = build_ass(words, cfg.get("caption_style", "bold_pop"), target_w, target_h)
#         ass_path = tmp_dir / f"subs_{ratio_key.replace(':', 'x')}.ass"
#         ass_path.write_text(ass_content, encoding="utf-8")
#         ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
 
#     if is_auto and not ass_path and not cfg.get("manual_crop_keyframes"):
#         # Nothing needs to touch the picture at all: fastest AND highest
#         # possible quality path is a pure stream-copy remux (no re-encode
#         # -> byte-for-byte original video quality; only trimmed/normalized
#         # audio may differ). This is what "zero quality loss" means here.
#         out_name = f"{job['job_id']}_{ratio_key.replace(':', 'x')}.mp4"
#         out_path = EDIT_DIR / out_name
#         if cfg.get("normalize_audio", True):
#             measured = loud_measured
#             audio_filter = (
#                 "loudnorm=I=-14:TP=-1.5:LRA=11:"
#                 f"measured_I={measured.get('input_i', -14)}:"
#                 f"measured_TP={measured.get('input_tp', -1.5)}:"
#                 f"measured_LRA={measured.get('input_lra', 11)}:"
#                 f"measured_thresh={measured.get('input_thresh', -24)}:"
#                 "linear=true:print_format=summary"
#             ) if measured else "loudnorm=I=-14:TP=-1.5:LRA=11"
#             cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path),
#                    "-c:v", "copy", "-af", audio_filter, "-c:a", "aac", "-b:a", "192k",
#                    "-movflags", "+faststart", str(out_path)]
#         else:
#             cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path),
#                    "-c", "copy", "-movflags", "+faststart", str(out_path)]
#         result = _run(cmd)
#         if result.returncode != 0:
#             raise RuntimeError(f"ffmpeg failed for {ratio_key}: {result.stderr[-800:]}")
#         return out_name
 
#     if fill_mode == "blur_pad":
#         # "Advanced" reframe mode: instead of cropping content away, fit the
#         # WHOLE frame inside the target canvas and fill the leftover bars
#         # with a blurred, zoomed copy of the same frame — the look used by
#         # most professional auto-reframe tools when a hard crop would cut
#         # off important content. No face tracking needed for this branch,
#         # since nothing is being cropped out.
#         fc = (
#             f"[0:v]split=2[bg][fg];"
#             f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
#             f"crop={target_w}:{target_h},gblur=sigma=25,eq=brightness=-0.05[bg2];"
#             f"[fg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg2];"
#             f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2:format=auto[base]"
#         )
#         if ass_path:
#             fc += f",[base]ass='{ass_escaped}'[vout]"
#             map_v = "[vout]"
#         else:
#             fc += "[vout]"
#             map_v = "[vout]"
#         vf_args = ["-filter_complex", fc, "-map", map_v, "-map", "0:a?"]
#     elif is_auto:
#         # Captions and/or a manual crop keyframe are present, so a re-encode
#         # can't be avoided — but the frame itself is never cropped or
#         # rescaled; it stays at the source's own resolution. Quality is set
#         # one notch above "high" (crf 16, effectively visually lossless)
#         # specifically for this path so adding captions never visibly
#         # degrades the original picture.
#         filters = ["setsar=1"]
#         if ass_path:
#             filters.append(f"ass='{ass_escaped}'")
#         vf_args = ["-vf", ",".join(filters)]
#         quality = {"crf": 16, "preset": quality["preset"]}
#     else:
#         filters = []
#         # ── crop (face-tracked or manual) ──
#         if cfg["features"].get("face_tracking") or cfg.get("manual_crop_keyframes"):
#             crop_w, crop_h = _crop_size_for_ratio(src_w, src_h, target_w, target_h)
#             if cfg.get("manual_crop_keyframes"):
#                 kfs = sorted(cfg["manual_crop_keyframes"], key=lambda k: k["t"])
#             else:
#                 kfs = _build_crop_keyframes(track_points, src_w, src_h, crop_w, crop_h)
#             x_expr = _expr_chain(kfs, "x")
#             y_expr = _expr_chain(kfs, "y")
#             filters.append(f"crop={crop_w}:{crop_h}:'{x_expr}':'{y_expr}'")
#         else:
#             # no face tracking requested -> simple centered crop to the target ratio
#             crop_w, crop_h = _crop_size_for_ratio(src_w, src_h, target_w, target_h)
#             filters.append(f"crop={crop_w}:{crop_h}:(in_w-out_w)/2:(in_h-out_h)/2")
 
#         filters.append(f"scale={target_w}:{target_h}:flags=lanczos")
#         filters.append("setsar=1")
#         if ass_path:
#             filters.append(f"ass='{ass_escaped}'")
#         vf_args = ["-vf", ",".join(filters)]
 
#     out_name = f"{job['job_id']}_{ratio_key.replace(':', 'x')}.mp4"
#     out_path = EDIT_DIR / out_name
 
#     # Loudness normalization (EBU R128, -14 LUFS is the standard target for
#     # social platforms) so exported audio doesn't sound quiet/inconsistent
#     # next to other content in-feed. On "high" quality we do a real
#     # measure-then-normalize TWO-PASS pass for accuracy; cheaper qualities
#     # use fast single-pass so exports stay quick.
#     audio_filters = []
#     if cfg.get("normalize_audio", True):
#         measured = loud_measured
#         if measured:
#             audio_filters = [
#                 "loudnorm=I=-14:TP=-1.5:LRA=11:"
#                 f"measured_I={measured.get('input_i', -14)}:"
#                 f"measured_TP={measured.get('input_tp', -1.5)}:"
#                 f"measured_LRA={measured.get('input_lra', 11)}:"
#                 f"measured_thresh={measured.get('input_thresh', -24)}:"
#                 "linear=true:print_format=summary"
#             ]
#         else:
#             audio_filters = ["loudnorm=I=-14:TP=-1.5:LRA=11"]
 
#     cmd = [FFMPEG_PATH, "-y", "-v", "error", "-i", str(source_path), *vf_args]
#     if audio_filters:
#         cmd += ["-af", ",".join(audio_filters)]
#     cmd += [
#         "-c:v", "libx264", "-crf", str(quality["crf"]), "-preset", quality["preset"],
#         "-c:a", "aac", "-b:a", "192k",
#         "-movflags", "+faststart",
#         str(out_path),
#     ]
#     result = _run(cmd)
#     if result.returncode != 0:
#         raise RuntimeError(f"ffmpeg failed for {ratio_key}: {result.stderr[-800:]}")
#     return out_name
 
 
# def _run_render_job(job_id):
#     job = EDIT_JOBS[job_id]
#     cfg = job["config"]
#     tmp_dir = EDIT_DIR / f"tmp_{job_id}"
#     tmp_dir.mkdir(exist_ok=True)
 
#     try:
#         source_path = Path(cfg["source_path"])
#         if not source_path.exists():
#             raise RuntimeError("Source video not found")
 
#         # ── 1. hook-cut / manual rearrange: trim+stitch segments first ──
#         job["stage"] = "segments"
#         segments = cfg.get("segments")
#         want_jump = cfg["features"].get("jump_cut")
#         want_hook = cfg["features"].get("hook_cut")
#         if not segments and (want_jump or want_hook):
#             info = _probe(source_path)
#             jump_segments = detect_silences(source_path) if want_jump else None
#             if want_hook:
#                 hook = auto_hook_window(source_path, info["duration"], cfg.get("hook_len", 6.0))
#                 # Cold-open on the loudest window, then play the rest of the
#                 # clip — dead-air-trimmed if jump_cut is also on, otherwise
#                 # the full original timeline. Combining both no longer means
#                 # jump_cut silently wins.
#                 rest = jump_segments if jump_segments else [{"start": 0.0, "end": info["duration"]}]
#                 segments = [hook] + rest
#             else:
#                 segments = jump_segments
#         if segments:
#             list_path = _build_concat_file(source_path, segments, tmp_dir)
#             stitched = tmp_dir / "stitched.mp4"
#             _run([FFMPEG_PATH, "-y", "-v", "error", "-f", "concat", "-safe", "0",
#                   "-i", str(list_path), "-c", "copy", str(stitched)])
#             source_path = stitched
#         job["percent"] = 10
 
#         info = _probe(source_path)
#         src_w, src_h = info["width"], info["height"]
 
#         # ── 2. face tracking (once, reused for every export ratio) ──
#         job["stage"] = "face_tracking"
#         track_points = []
#         if cfg["features"].get("face_tracking"):
#             def cb(pct):
#                 job["percent"] = 10 + int(pct * 0.3)
#             track_points, src_w, src_h, advanced = analyze_face_track(
#                 source_path, sample_fps=cfg.get("track_fps", 2.0), progress_cb=cb
#             )
#             job["face_tracking_mode"] = "advanced (mediapipe)" if advanced else "standard (haar cascade)"
#         job["percent"] = 40
 
#         # ── 3. subtitles (once, reused for every export ratio) ──
#         job["stage"] = "subtitles"
#         words = []
#         if cfg["features"].get("subtitles"):
#             words = transcribe_words(source_path, model_size=cfg.get("whisper_model", "small"),
#                                       language=cfg.get("language"))
#         job["percent"] = 60
 
#         # ── 4. render each requested aspect ratio as one final file ──
#         job["stage"] = "render"
#         ratios = cfg.get("export_ratios") or ["9:16"]
#         job["ratios"] = {}
#         n = len(ratios)
 
#         # measured once (not per-ratio, source audio is identical across
#         # ratios) and only for "high" quality, where the extra ffmpeg pass
#         # is worth the accuracy; cheaper qualities use fast single-pass.
#         loud_measured = None
#         if cfg.get("normalize_audio", True) and cfg.get("quality", "high") == "high":
#             loud_measured = _measure_loudnorm(source_path)
 
#         for i, ratio_key in enumerate(ratios):
#             if ratio_key not in RATIO_PRESETS:
#                 continue
#             out_name = _render_one_ratio(job, source_path, ratio_key, track_points,
#                                           src_w, src_h, words, tmp_dir, loud_measured)
#             job["ratios"][ratio_key] = out_name
#             job["percent"] = 60 + int((i + 1) * 40 / max(1, n))
 
#         job["status"] = "done"
#         job["stage"] = "done"
#         job["percent"] = 100
#     except Exception as e:
#         job["status"] = "error"
#         job["error"] = str(e)
#     finally:
#         # scratch files (trimmed segments, stitched source, .ass files) —
#         # keep only the final per-ratio outputs in EDIT_DIR
#         try:
#             for f in tmp_dir.glob("*"):
#                 f.unlink(missing_ok=True)
#             tmp_dir.rmdir()
#         except OSError:
#             pass
 
 
# # ══════════════════════════════ routes ══════════════════════════════
 
# @editor_bp.route("/api/editor/analyze", methods=["POST"])
# def api_editor_analyze():
#     """Runs face tracking + transcription WITHOUT rendering, so a frontend
#     can show a preview / let the user manually adjust crop keyframes or
#     hook segments before committing to a full render."""
#     data = request.json or {}
#     source_path = Path(data.get("source_path", ""))
#     if not source_path.exists():
#         return jsonify({"error": "source_path not found"}), 400
 
#     info = _probe(source_path)
#     result = {"probe": info, "features_available": feature_status()}
 
#     if data.get("face_tracking", True) and _HAS_CV2:
#         points, w, h, advanced = analyze_face_track(source_path, sample_fps=data.get("track_fps", 2.0))
#         result["face_track"] = points
#         result["face_tracking_mode"] = "advanced" if advanced else "standard"
#         result["source_size"] = {"w": w, "h": h}
 
#     if data.get("subtitles", True) and _HAS_WHISPER:
#         words = transcribe_words(source_path, model_size=data.get("whisper_model", "small"))
#         result["words"] = words
 
#     if data.get("hook_cut", False) and _HAS_NUMPY:
#         result["suggested_hook"] = auto_hook_window(source_path, info["duration"], data.get("hook_len", 6.0))
 
#     if data.get("jump_cut", False):
#         result["suggested_segments"] = detect_silences(source_path)
 
#     return jsonify(result)
 
 
# @editor_bp.route("/api/editor/upload", methods=["POST"])
# def api_editor_upload():
#     """Lets the frontend upload a local file directly instead of only
#     pointing at a path already on the server (e.g. a downloader.py output)."""
#     f = request.files.get("file")
#     if not f or not f.filename:
#         return jsonify({"error": "No file uploaded"}), 400
#     safe_name = f"{uuid.uuid4().hex[:10]}_{Path(f.filename).name}"
#     dest = EDIT_DIR / "uploads"
#     dest.mkdir(exist_ok=True)
#     path = dest / safe_name
#     f.save(path)
#     return jsonify({"source_path": str(path)})
 
 
# # ══════════════════════════════ fetch-from-link (built-in yt-dlp) ══════════════════════════════
# # A third source alongside "upload from PC" and "pick from downloader" —
# # paste a link right here in the editor. Deliberately skips the full
# # format-picker: it always resolves ONE single, best-available video+audio
# # mp4 (never a video-only stream that would need a silent-audio placeholder,
# # never separate files) and hands that straight to analyze/render.
# #
# # Speed trick mirrors downloader.py: remembers whichever auth mode (cookies
# # file / browser cookies / plain / bypass) last worked and tries that first,
# # so repeat fetches stay fast. If downloader.py is also registered in this
# # process, its already-resolved mode is reused immediately instead of
# # re-discovering it from scratch.
 
# FETCH_JOBS = {}  # fetch_id -> {status, stage, percent, error, source_path, url}
# _FETCH_RESOLVED_MODE = {"mode": None}
# _FETCH_COOKIE_BROWSERS = ["chrome", "edge", "firefox", "brave"]
 
 
# def _fetch_no_console_kwargs():
#     if os.name == "nt":
#         return {"creationflags": subprocess.CREATE_NO_WINDOW}
#     return {}
 
 
# def _fetch_auth_attempts():
#     attempts = []
#     # reuse downloader.py's already-known-good mode first, if it's loaded
#     try:
#         import downloader
#         if downloader._RESOLVED_MODE.get("mode"):
#             attempts.append(downloader._RESOLVED_MODE["mode"])
#     except Exception:
#         pass
#     if _FETCH_RESOLVED_MODE["mode"]:
#         attempts.append(_FETCH_RESOLVED_MODE["mode"])
#     cookies_file = SRC_DIR.parent / "cookies.txt" if SRC_DIR else None
#     rest = []
#     if cookies_file and cookies_file.exists():
#         rest.append("cookies_file")
#     rest.extend(_FETCH_COOKIE_BROWSERS)
#     rest.append("default")
#     rest.append("bypass")
#     for m in rest:
#         if m not in attempts:
#             attempts.append(m)
#     return attempts, cookies_file
 
 
# def _fetch_apply_auth(opts, mode, cookies_file):
#     opts = dict(opts)
#     if mode == "cookies_file" and cookies_file:
#         opts["cookiefile"] = str(cookies_file)
#     elif mode in _FETCH_COOKIE_BROWSERS:
#         opts["cookiesfrombrowser"] = (mode,)
#     elif mode == "bypass":
#         opts["extractor_args"] = {"youtube": {"player_client": ["ios", "android", "mweb"]}}
#     return opts
 
 
# def _fetch_extract(url, extra_opts=None, download=False):
#     base_opts = {
#         "quiet": True, "no_warnings": True, "noplaylist": True,
#         "socket_timeout": 15, "retries": 2, "extractor_retries": 1, "geo_bypass": True,
#     }
#     if FFMPEG_PATH:
#         base_opts["ffmpeg_location"] = str(FFMPEG_PATH)
#     if extra_opts:
#         base_opts.update(extra_opts)
 
#     attempts, cookies_file = _fetch_auth_attempts()
#     last_err = None
#     for mode in attempts:
#         opts = _fetch_apply_auth(base_opts, mode, cookies_file)
#         try:
#             with yt_dlp.YoutubeDL(opts) as ydl:
#                 info = ydl.extract_info(url, download=download)
#             _FETCH_RESOLVED_MODE["mode"] = mode
#             return info, ydl if download else None, None
#         except Exception as e:
#             last_err = e
#             continue
#     return None, None, last_err
 
 
# def _fetch_fmt_duration(sec):
#     if sec is None:
#         return None
#     sec = int(sec)
#     h, rem = divmod(sec, 3600)
#     m, s = divmod(rem, 60)
#     return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
 
 
# def _fetch_fmt_int(n):
#     if n is None:
#         return None
#     try:
#         return f"{int(n):,}"
#     except (TypeError, ValueError):
#         return str(n)
 
 
# def _fetch_fmt_upload_date(d):
#     if not d:
#         return None
#     try:
#         from datetime import datetime as _dt
#         return _dt.strptime(str(d), "%Y%m%d").strftime("%d %b %Y")
#     except ValueError:
#         return str(d)
 
 
# @editor_bp.route("/api/editor/fetch/info", methods=["POST"])
# def api_fetch_info():
#     """Resolves a link (YouTube/Instagram/TikTok/X/Facebook/etc.) and hands
#     back the FULL set of metadata yt-dlp/the source platform already knows
#     about it — title, uploader, view/like counts, upload date, best
#     available resolution — so the UI never has to re-derive any of this by
#     hand, and so a video that started life as a fetched link can show that
#     rich context in the analysis panel too (a plain PC upload naturally
#     won't have this — it only gets the ffprobe technical details)."""
#     if not _HAS_YTDLP:
#         return jsonify({"error": "yt-dlp is not installed on the server — pip install yt-dlp"}), 400
#     data = request.json or {}
#     url = (data.get("url") or "").strip()
#     if not url:
#         return jsonify({"error": "Paste a video URL first"}), 400
#     info, _, err = _fetch_extract(url, extra_opts={"format": "bestvideo+bestaudio/best"})
#     if info is None:
#         return jsonify({"error": f"Could not resolve that link: {err}"}), 400
 
#     best_h = best_w = None
#     for f in info.get("formats", []) or []:
#         if f.get("vcodec") not in (None, "none") and f.get("height"):
#             if not best_h or f["height"] > best_h:
#                 best_h, best_w = f["height"], f.get("width")
 
#     return jsonify({
#         "title": info.get("title"),
#         "uploader": info.get("uploader") or info.get("channel"),
#         "duration": info.get("duration"),
#         "duration_str": _fetch_fmt_duration(info.get("duration")),
#         "thumbnail": info.get("thumbnail"),
#         "view_count": info.get("view_count"),
#         "view_count_str": _fetch_fmt_int(info.get("view_count")),
#         "like_count": info.get("like_count"),
#         "like_count_str": _fetch_fmt_int(info.get("like_count")),
#         "upload_date": info.get("upload_date"),
#         "upload_date_str": _fetch_fmt_upload_date(info.get("upload_date")),
#         "resolution": f"{best_w}x{best_h}" if best_w and best_h else None,
#         "extractor": info.get("extractor_key"),
#         "webpage_url": info.get("webpage_url"),
#     })
 
 
# def _run_fetch_job(fetch_id, url):
#     job = FETCH_JOBS[fetch_id]
#     job["stage"] = "connect"
 
#     def hook(d):
#         if d.get("status") == "downloading":
#             total = d.get("total_bytes") or d.get("total_bytes_estimate")
#             downloaded = d.get("downloaded_bytes") or 0
#             speed = d.get("speed")
#             job.update({
#                 "status": "downloading", "stage": "download",
#                 "percent": round(downloaded * 100 / total, 1) if total else None,
#                 "speed_str": (_fmt_size(speed) + "/s") if speed else None,
#             })
#         elif d.get("status") == "finished":
#             job["status"] = "processing"
#             job["stage"] = "merge"
 
#     # Saved into SRC_DIR (the SAME "downloads" folder downloader.py uses) so
#     # a link fetched here shows up in /api/editor/sources too, and nothing
#     # about the analyze/render pipeline needs to know it came from here
#     # instead of the Downloader tab or a PC upload.
#     extra_opts = {
#         "format": "bestvideo+bestaudio/best",
#         "outtmpl": str(SRC_DIR / f"editorfetch_{fetch_id}_%(title).60s.%(ext)s"),
#         "progress_hooks": [hook],
#         "merge_output_format": "mp4",   # always ONE muxed video+audio file
#     }
 
#     orig_popen = subprocess.Popen
#     def quiet_popen(*args, **kwargs):
#         kwargs.update(_fetch_no_console_kwargs())
#         return orig_popen(*args, **kwargs)
#     subprocess.Popen = quiet_popen
#     try:
#         info, ydl, err = _fetch_extract(url, extra_opts=extra_opts, download=True)
#     finally:
#         subprocess.Popen = orig_popen
 
#     if info is None:
#         job["status"] = "error"
#         job["error"] = f"Fetch failed: {err}"
#         return
#     try:
#         fname = ydl.prepare_filename(info)
#         p = Path(fname)
#         if not p.exists():
#             p2 = p.with_suffix(".mp4")
#             if p2.exists():
#                 p = p2
#         job["source_path"] = str(p.resolve())
#         job["status"] = "done"
#         job["stage"] = "done"
#         job["percent"] = 100
#     except Exception as e:
#         job["status"] = "error"
#         job["error"] = f"Could not finalize file: {e}"
 
 
# @editor_bp.route("/api/editor/fetch/start", methods=["POST"])
# def api_fetch_start():
#     if not _HAS_YTDLP:
#         return jsonify({"error": "yt-dlp is not installed on the server — pip install yt-dlp"}), 400
#     data = request.json or {}
#     url = (data.get("url") or "").strip()
#     if not url:
#         return jsonify({"error": "No URL given"}), 400
#     fetch_id = uuid.uuid4().hex[:10]
#     FETCH_JOBS[fetch_id] = {
#         "status": "starting", "stage": "connect", "percent": 0,
#         "speed_str": None, "error": None, "url": url, "source_path": None,
#     }
#     threading.Thread(target=_run_fetch_job, args=(fetch_id, url), daemon=True).start()
#     return jsonify({"fetch_id": fetch_id})
 
 
# @editor_bp.route("/api/editor/fetch/progress/<fetch_id>")
# def api_fetch_progress(fetch_id):
#     job = FETCH_JOBS.get(fetch_id)
#     if not job:
#         return jsonify({"error": "Unknown fetch job"}), 404
#     return jsonify(job)
 
 
# @editor_bp.route("/api/editor/render", methods=["POST"])
# def api_editor_render():
#     data = request.json or {}
#     source_path = data.get("source_path", "")
#     if not source_path or not Path(source_path).exists():
#         return jsonify({"error": "source_path not found"}), 400
 
#     export_ratios = data.get("export_ratios") or ["9:16"]
#     bad = [r for r in export_ratios if r not in RATIO_PRESETS]
#     if bad:
#         return jsonify({"error": f"Unknown ratio(s): {bad}. Valid: {list(RATIO_PRESETS)}"}), 400
 
#     features = {
#         "face_tracking": bool(data.get("face_tracking", True)),
#         "subtitles": bool(data.get("subtitles", True)),
#         "hook_cut": bool(data.get("hook_cut", False)),
#         "jump_cut": bool(data.get("jump_cut", False)),
#     }
#     if features["face_tracking"] and not _HAS_CV2:
#         return jsonify({"error": "Face tracking requested but opencv-python is not installed"}), 400
#     if features["subtitles"] and not _HAS_WHISPER:
#         return jsonify({"error": "Subtitles requested but faster-whisper is not installed"}), 400
#     if features["hook_cut"] and not data.get("segments") and not _HAS_NUMPY:
#         return jsonify({"error": "Auto hook-cut requested but numpy is not installed"}), 400
 
#     job_id = uuid.uuid4().hex[:10]
#     EDIT_JOBS[job_id] = {
#         "job_id": job_id, "status": "starting", "stage": "queued", "percent": 0,
#         "error": None, "ratios": {},
#         "config": {
#             "source_path": source_path,
#             "export_ratios": export_ratios,
#             "features": features,
#             "caption_style": data.get("caption_style", "bold_pop"),
#             "quality": data.get("quality", "high"),
#             "track_fps": data.get("track_fps", 2.0),
#             "whisper_model": data.get("whisper_model", "small"),
#             "language": data.get("language"),
#             "hook_len": data.get("hook_len", 6.0),
#             "manual_crop_keyframes": data.get("manual_crop_keyframes"),  # [{t,x,y}]
#             "segments": data.get("segments"),  # [{start,end}] manual rearrange/trim order
#             "fill_mode": data.get("fill_mode", "crop"),  # "crop" or "blur_pad"
#             "normalize_audio": bool(data.get("normalize_audio", True)),
#         },
#     }
#     threading.Thread(target=_run_render_job, args=(job_id,), daemon=True).start()
#     return jsonify({"job_id": job_id})
 
 
# @editor_bp.route("/api/editor/progress/<job_id>")
# def api_editor_progress(job_id):
#     job = EDIT_JOBS.get(job_id)
#     if not job:
#         return jsonify({"error": "Unknown edit job"}), 404
#     return jsonify({k: v for k, v in job.items() if k != "config"} | {
#         "features": job["config"]["features"], "export_ratios": job["config"]["export_ratios"]
#     })
 
 
# @editor_bp.route("/api/editor/file/<job_id>/<ratio>")
# def api_editor_file(job_id, ratio):
#     job = EDIT_JOBS.get(job_id)
#     if not job or job.get("status") != "done":
#         return "Not ready", 404
#     fname = job["ratios"].get(ratio)
#     if not fname:
#         return "Ratio not found for this job", 404
#     p = EDIT_DIR / fname
#     if not p.exists():
#         return "Not found", 404
#     return send_file(p, as_attachment=True, download_name=fname)
 
 
# # ── manual crop keyframe add/remove (per not-yet-rendered job config) ──
 
# @editor_bp.route("/api/editor/keyframe/add", methods=["POST"])
# def api_keyframe_add():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     kf = {"t": float(data["t"]), "x": int(data["x"]), "y": int(data["y"])}
#     job["config"].setdefault("manual_crop_keyframes", [])
#     job["config"]["manual_crop_keyframes"] = job["config"]["manual_crop_keyframes"] or []
#     job["config"]["manual_crop_keyframes"].append(kf)
#     job["config"]["manual_crop_keyframes"].sort(key=lambda k: k["t"])
#     return jsonify({"manual_crop_keyframes": job["config"]["manual_crop_keyframes"]})
 
 
# @editor_bp.route("/api/editor/keyframe/remove", methods=["POST"])
# def api_keyframe_remove():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     t = float(data["t"])
#     kfs = job["config"].get("manual_crop_keyframes") or []
#     job["config"]["manual_crop_keyframes"] = [k for k in kfs if abs(k["t"] - t) > 1e-6]
#     return jsonify({"manual_crop_keyframes": job["config"]["manual_crop_keyframes"]})
 
 
# # ── hook-cut / rearrange segment add/remove ──
 
# @editor_bp.route("/api/editor/segment/add", methods=["POST"])
# def api_segment_add():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     seg = {"start": float(data["start"]), "end": float(data["end"])}
#     index = data.get("index")
#     segs = job["config"].get("segments") or []
#     if index is None or index >= len(segs):
#         segs.append(seg)
#     else:
#         segs.insert(int(index), seg)
#     job["config"]["segments"] = segs
#     return jsonify({"segments": segs})
 
 
# @editor_bp.route("/api/editor/segment/remove", methods=["POST"])
# def api_segment_remove():
#     data = request.json or {}
#     job = EDIT_JOBS.get(data.get("job_id"))
#     if not job:
#         return jsonify({"error": "Unknown job"}), 404
#     index = int(data["index"])
#     segs = job["config"].get("segments") or []
#     if 0 <= index < len(segs):
#         segs.pop(index)
#     job["config"]["segments"] = segs
#     return jsonify({"segments": segs})
