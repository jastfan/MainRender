#!/usr/bin/env python3
# word_captions.py — YouTube ke internal json3 timedtext format se REAL
# word-by-word timestamps nikalta hai (guess/estimate nahi, actual data).
# pip install yt-dlp

import re
import json
import os
import yt_dlp
import tempfile

def get_word_level_captions(url, lang="en", cookie_file_path=None):
    """Returns [{"word": str, "start": float_sec, "end": float_sec}, ...]"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'noplaylist': True,
        # FIX 1: format resolve fail ho (n-challenge) toh bhi crash na ho,
        # sirf subtitles chahiye actual video download nahi karna
        'ignoreerrors': 'only_download',
    }
    # FIX 2: cookies optional support — bot-detection / empty response se bachne ke liye
    if cookie_file_path:
        ydl_opts['cookiefile'] = cookie_file_path

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        # FIX 3: ignoreerrors ke saath info None bhi aa sakta hai
        if not info:
            raise RuntimeError("Video info extract nahi ho paya (blocked ya invalid URL).")

        bucket = info.get('subtitles') or {}
        if not bucket:
            bucket = info.get('automatic_captions') or {}
        if not bucket:
            raise RuntimeError("Is video me koi captions nahi hain.")

        if lang not in bucket:
            lang = "en" if "en" in bucket else next(iter(bucket))
        tracks = bucket[lang]

        # json3-native track dhundo, warna kisi bhi track ka URL le ke fmt=json3 force karo
        base_url = None
        for t in tracks:
            if t.get('ext') == 'json3':
                base_url = t['url']
                break
        if not base_url:
            base_url = tracks[0]['url']
            if 'fmt=' in base_url:
                base_url = re.sub(r'fmt=[^&]+', 'fmt=json3', base_url)
            else:
                base_url += ('&' if '?' in base_url else '?') + 'fmt=json3'

        # FIX 4: same ydl session reuse karo (cookies/headers wahi rahenge),
        # nayi YoutubeDL instance banane ki zarurat nahi
        raw_data = ydl.urlopen(base_url).read().decode('utf-8')

        # FIX 5: empty response ko clearly handle karo
        if not raw_data.strip():
            raise RuntimeError("YouTube se khali response mila. Fresh cookies try karo.")

        data = json.loads(raw_data)

    words = []
    for event in data.get('events', []):
        t_start = event.get('tStartMs')
        if t_start is None:
            continue
        for seg in (event.get('segs') or []):
            text = seg.get('utf8', '')
            if not text.strip():
                continue
            offset = seg.get('tOffsetMs', 0)
            words.append({"word": text.strip(), "start": (t_start + offset) / 1000.0})

    # end time = agle word ka start (last word ke liye +0.4s default)
    for i in range(len(words) - 1):
        words[i]["end"] = words[i + 1]["start"]
    if words:
        words[-1]["end"] = words[-1]["start"] + 0.4

    # SAFETY SPLIT: kuch caption tracks (especially translated ones) ek
    # "word" ke andar poori line bhej dete hain, single word nahi. Aise
    # multi-word entries ko unke apne hi time-slot ke andar real words me
    # proportionally split karo (length ke hisaab se time baant ke) — taaki
    # downstream ko hamesha genuine single-word entries milein.
    split_words = []
    for w in words:
        parts = w["word"].split()
        if len(parts) <= 1:
            split_words.append(w)
            continue
        seg_start, seg_end = w["start"], w["end"]
        seg_dur = max(seg_end - seg_start, 0.001)
        weights = [len(p) + 1 for p in parts]
        total = sum(weights)
        cursor = seg_start
        for i, p in enumerate(parts):
            share = (weights[i] / total) * seg_dur
            w_start = cursor
            w_end = seg_end if i == len(parts) - 1 else min(seg_end, cursor + share)
            cursor = w_end
            if w_end <= w_start:
                continue
            split_words.append({"word": p, "start": w_start, "end": w_end})

    return split_words


def save_words_to_file(words, url, out_path=None):
    """Found word-level captions ko JSON file mein save karta hai (same route/folder)."""
    if out_path is None:
        # video ID se filename banao taaki har video ki alag file bane
        video_id = url.strip().split("v=")[-1].split("&")[0].split("/")[-1].split("?")[0]
        video_id = re.sub(r'[^a-zA-Z0-9_-]', '', video_id) or "captions"
        out_path = f"{video_id}_word_captions.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    return os.path.abspath(out_path)


if __name__ == "__main__":
    url = input("YouTube URL: ").strip()
    # cookies.txt ka path optional hai — agar bot-detection lage toh isse pass karo
    cookies_path = "www.youtube.com_cookies.txt"  # ya None rakh do agar zarurat nahi
    try:
        words = get_word_level_captions(url, cookie_file_path=cookies_path)
        print(f"\n✅ {len(words)} words with timing found:\n")
        for w in words[:40]:
            print(f"[{w['start']:.2f}s - {w['end']:.2f}s]  {w['word']}")
        if len(words) > 40:
            print(f"... aur {len(words)-40} words")

        # FIX: found data ko usi route (script ke folder) mein JSON file ke roop me save karo
        saved_path = save_words_to_file(words, url)
        print(f"\n💾 Poora data save ho gaya: {saved_path}")

    except Exception as e:
        print(f"\n❌ Error encountered: {e}")