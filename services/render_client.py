"""
render_client.py — platform_hub ka HTTP client jo render_service
(RenderDetect.py) ko call karta hai. Khud koi video processing nahi karta,
sirf HTTP requests bhejta hai aur result wapas deta hai.
"""
import time
import requests

from config import Config

BASE_URL = Config.RENDER_SERVICE_URL.rstrip("/")
DEFAULT_TIMEOUT = 30


class RenderServiceError(Exception):
    pass


def _post(path, json_body):
    try:
        resp = requests.post(f"{BASE_URL}{path}", json=json_body, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise RenderServiceError(f"Could not reach render_service at {BASE_URL}: {e}")
    if resp.status_code >= 400:
        raise RenderServiceError(f"render_service {path} failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


def _get(path):
    try:
        resp = requests.get(f"{BASE_URL}{path}", timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise RenderServiceError(f"Could not reach render_service at {BASE_URL}: {e}")
    if resp.status_code >= 400:
        raise RenderServiceError(f"render_service {path} failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


# ---- Polling: "kya kuch naya bana hai?" — no URL ever sent from here ----
# def list_all_jobs():
#     return _get("/api/jobs/list")["jobs"]


# def poll_cut_status(job_id, poll_interval=5, max_wait=1800):
#     waited = 0
#     while waited < max_wait:
#         status = _get(f"/api/cut_status/{job_id}")
#         if status.get("error"):
#             raise RenderServiceError(f"Cut job {job_id} failed: {status['error']}")
#         if status.get("done"):
#             return status
#         time.sleep(poll_interval)
#         waited += poll_interval
#     raise RenderServiceError(f"Cut job {job_id} timed out after {max_wait}s")

def list_all_jobs(user_id):
    return _get(f"/api/jobs/list?user_id={user_id}")["jobs"]


def poll_cut_status_once(job_id, user_id):
    """Non-blocking — ek baar status check karta hai, sleep/wait nahi karta.
    (Purana poll_cut_status blocking tha — scheduler ke andar wo hi hang ka source tha.)"""
    return _get(f"/api/cut_status/{job_id}?user_id={user_id}")

# ---- Export a single approved clip to its final rendered file ----
def start_export(clip_id, settings=None):
    data = _post("/api/export", {"clip_id": clip_id, "settings": settings or {}})
    return data["export_id"]


def poll_export_status(export_id, poll_interval=5, max_wait=1800):
    waited = 0
    while waited < max_wait:
        status = _get(f"/api/export_status/{export_id}")
        if status.get("error"):
            raise RenderServiceError(f"Export job {export_id} failed: {status['error']}")
        if status.get("status") == "done":
            return status
        time.sleep(poll_interval)
        waited += poll_interval
    raise RenderServiceError(f"Export job {export_id} timed out after {max_wait}s")


def download_file_stream(fname):
    import io
    try:
        resp = requests.get(f"{BASE_URL}/api/download/{fname}", stream=True, timeout=120)
    except requests.RequestException as e:
        raise RenderServiceError(f"Could not download {fname}: {e}")
    if resp.status_code >= 400:
        raise RenderServiceError(f"download failed ({resp.status_code}) for {fname}")
    resp.raw.decode_content = True
    # resp.raw (urllib3's streaming body) is NOT seekable, but
    # storage_service.upload_video_stream() needs to seek() to measure the
    # file size before uploading. Buffer it into a seekable BytesIO once here,
    # so callers can just pass this straight through.
    return io.BytesIO(resp.content)


def submit_url(user_id, url, mode="auto", clip_len=30):
    data = _post("/api/fetch_and_cut", {"user_id": user_id, "url": url, "mode": mode, "clip_len": clip_len})
    return data["job_id"]


def cleanup_clip(fname):
    try:
        requests.post(f"{BASE_URL}/api/cleanup_clip", json={"fname": fname}, timeout=10)
    except requests.RequestException:
        pass