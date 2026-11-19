"""A downloaded or posted clip carries its chosen thumbnail inside it: as the first frame (what phones, galleries,
TikTok and Instagram show) and as cover art (what Finder and players show).

Runs against the code directly, with its own empty data folder and no network."""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-coverclip-")
os.environ.update(DATABASE_URL="", PEAKCUT_ACCOUNTS="0", PEAKCUT_PUBLIC_URL="http://127.0.0.1:8765")
sys.path.insert(0, ".")
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from studio import accounts, auth, config, coverclip, ctx, db, jobs, posting, server, thumbs  # noqa: E402

db.init()
fails = []
client = TestClient(server.app, base_url="http://127.0.0.1:8765", follow_redirects=False)


def check(name, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"   got {got!r}"))
    ok or fails.append(name)


def probe(path, entries):
    return subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True).stdout.strip()


uid = auth.local_user().id
jid = "20260101-000000-coverclip1"
d = config.user_dir(uid) / "jobs" / jid
(d / "clips").mkdir(parents=True)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30:duration=4",
                "-f", "lavfi", "-i", "sine=frequency=500:duration=4", "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "clips" / "clip01.mp4")], check=True)
Image.new("RGB", (1080, 1920), (250, 20, 200)).save(d / "clips" / "clip01-ai123-v.jpg")       # a bright pink cover
clip = {"id": "clip01", "start": 10.0, "end": 14.0, "status": "done", "version": 1, "file": "clip01.mp4", "cover": "clip01.jpg",
        "title": "T", "hook": "H", "ai_thumbs": {"status": "done", "chosen": None,
                                                  "items": [{"id": "123", "fmt": "v", "file": "clip01-ai123-v.jpg", "t": 11.0}]}}
job = {"id": jid, "user": uid, "kind": "video", "creator": None, "dir": str(d), "created": time.time(), "source": {"type": "url", "url": "x"},
       "options": dict(jobs.DEFAULTS), "status": "done", "title": "T", "duration": 60, "clips": [clip]}
with jobs._lock:
    jobs._jobs[jid] = job
    jobs._by_user.setdefault(uid, {})[jid] = job
jobs.save(job)

with ctx.as_user(uid):
    check("with no cover chosen, the plain clip is handed out", coverclip.path_for(job, clip).name, "clip01.mp4")
    clip["ai_thumbs"]["chosen"] = "123"
    out = coverclip.path_for(job, clip)
    check("with a cover chosen, a file with the cover inside is made", out.name.startswith("clip01-wc-"), True)
    first = d / "first.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-map", "0:v:0", "-frames:v", "1", str(first)], check=True)
    px = np.asarray(Image.open(first).convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
    check("its first frame is the cover", px[0] > 200 and px[1] < 70 and px[2] > 160, True)
    check("the cover is also embedded as cover art", "1" in probe(out, "stream=index:stream_disposition=attached_pic").split("\n")[-1], True)
    check("it still has the sound", "aac" in probe(out, "stream=codec_name"), True)
    dur = float(probe(out, "format=duration"))
    check("it is only 0.1 s longer", 4.05 <= dur <= 4.2, True)
    mtime = out.stat().st_mtime
    check("a second download reuses it (instant)", (coverclip.path_for(job, clip), out.stat().st_mtime), (out, mtime))
    clip["ai_thumbs"]["chosen"] = None
    clip["thumbs"] = {"status": "done", "chosen": 1, "variants": [{"files": {"v": "clip01-t1-v1.jpg"}, "t": 11.0}]}
    Image.new("RGB", (1080, 1920), (20, 200, 40)).save(d / "clips" / "clip01-t1-v1.jpg")
    out2 = coverclip.path_for(job, clip)
    check("a normal thumbnail chosen as cover works the same way", out2.name.startswith("clip01-wc-clip01-t1-v1"), True)
    check("the file for the old cover is cleaned up", out.exists(), False)

    # ---- downloading
    r = client.get(f"/media/{jid}/clip01.mp4?download=1")
    check("downloading gives the clip with its cover inside, under the clip's own name",
          (r.status_code, r.headers.get("content-disposition", "").endswith('clip01.mp4"'), len(r.content) == out2.stat().st_size),
          (200, True, True))
    r = client.get(f"/media/{jid}/clip01.mp4")
    check("watching in the page still plays the plain clip", len(r.content) == (d / "clips" / "clip01.mp4").stat().st_size, True)

    # ---- posting
    got = {}

    def fake_tiktok(acc, path, caption, privacy="public", cover_ms=None):
        got.update(path=Path(path).name, ms=cover_ms)
        return "id1", "https://www.tiktok.com/@x/video/1"
    p = {"id": "post1", "platform": "tiktok", "job": jid, "clip": "clip01", "account": "tt1", "account_name": "@x", "creator": None,
         "tries": 0, "title": "t", "status": "posting", "due_at": 0, "user": uid}
    posting._posts.append(p)
    with patch.object(accounts, "get", return_value={"id": "tt1", "platform": "tiktok", "name": "@x"}), \
            patch.object(accounts, "tiktok_upload", side_effect=fake_tiktok):
        posting._post(p)
    check("TikTok gets the clip with the cover as its first frame, and is told to use frame 0",
          (got.get("path", "").startswith("clip01-wc-"), got.get("ms")), (True, 0))

    # ---- cleanup
    thumbs.remove_files(job, "clip01")
    check("deleting the clip's thumbnails removes the with-cover file too", list((d / "clips").glob("clip01-wc-*")), [])

shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
print(f"{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
