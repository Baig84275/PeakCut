"""AI-designed thumbnails: the plan the text AI writes, the design sent to the image model (with the creator's
brand), exact 9:16 and 16:9 pictures, the daily spending limit, choosing a cover, posting with it, cleanup.

Runs against the code directly, with its own empty data folder. The text AI and the image model are replaced."""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-aithumb-")
os.environ.update(DATABASE_URL="", PEAKCUT_ACCOUNTS="0", PEAKCUT_AI_PROVIDER="openai", PEAKCUT_AI_MODELS="test-model",
                  PEAKCUT_AI_KEY="sk-chat-key-123456", PEAKCUT_AI_KEY_FILE="", PEAKCUT_AI_BASE_URL="", PEAKCUT_PUBLIC_URL="http://127.0.0.1:8765")
sys.path.insert(0, ".")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from studio import aithumb, auth, brain, config, creators, ctx, db, imagegen, jobs, posting, server, thumbs  # noqa: E402

db.init()
fails = []
client = TestClient(server.app, base_url="http://127.0.0.1:8765", follow_redirects=False)
ORIGIN = {"Origin": "http://127.0.0.1:8765"}


def check(name, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"   got {got!r}"))
    ok or fails.append(name)


work = tempfile.mkdtemp(prefix="pc-aithumb-w-")
uid = auth.local_user().id
BRIEF = {"frame": 2, "line1": "CAR", "line2": "LOCKED?!", "scene": "A man tries the door of a locked yellow car at night",
         "arrow_to": "the yellow car", "icon": "a padlock", "expression": "shocked", "face": [0.83, 0.0, 0.17, 0.28]}


def picture(w, h):
    out = io.BytesIO()
    Image.new("RGB", (w, h), (200, 40, 160)).save(out, "JPEG")
    return out.getvalue()


try:
    src = os.path.join(work, "v.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25:duration=8",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", src], check=True)
    with ctx.as_user(uid):
        # ---- brand
        check("a bad colour style falls back to automatic", creators._clean({"brand_style": "neon-rainbow"})["brand_style"], "auto")
        check("the badge name is kept short and plain", creators._clean({"brand_name": "Ducky Bhai <script>"})["brand_name"], "Ducky Bhai script")
        check("GTA VI is written GTA 6 on the thumbnail", aithumb.game_label("Grand Theft Auto VI"), "GTA 6")
        check("GTA games get the neon Vice City colours automatically", aithumb.style_for("auto", "Grand Theft Auto VI"), "vice")
        check("other games get fire colours automatically", aithumb.style_for("auto", "Fortnite"), "fire")

        # ---- the written design
        brand = {"name": "DUCKY BHAI", "game": "GTA 6", "style": "vice"}
        p = aithumb.prompt(BRIEF, brand, "v", True)
        check("the headline is spelled out exactly", '"CAR"' in p and '"LOCKED?!"' in p, True)
        check("the badge name and the game are in it", '"DUCKY BHAI"' in p and '"GTA 6"' in p, True)
        check("the creator's colours are in it", "Vice City sunset" in p, True)
        check("it asks to keep the streamer the same person", "same person" in p, True)
        check("it keeps text off the edges that get cropped", "9%" in p, True)
        check("it asks for no official logos and nothing offensive", "not any official logo" in p and "offensive" in p, True)
        check("the arrow and icon are in it", "the yellow car" in p and "a padlock" in p, True)
        p2 = aithumb.prompt(dict(BRIEF, arrow_to="", icon=""), {"name": "", "game": "", "style": "fire"}, "h", False)
        check("without a face, the game moment fills the picture", "whole picture" in p2 and "Image 1 is the streamer" not in p2, True)
        check("no badge or game line when there is none", "play-button" not in p2 and "Top corner" not in p2, True)
        check("16:9 keeps text off the top and bottom edges", "top and bottom 8%" in p2, True)

        # ---- the brief the text AI writes is cleaned
        b = brain._tidy_brief({"frame": 9, "line1": "kya hua visit www.x.com", "line2": "भाई!", "scene": "x", "expression": "nervous"}, 6)
        check("a frame out of range is dropped", b["frame"], None)
        check("links come out of the headline", "X.COM" not in b["line1"], True)
        check("Hindi becomes Roman letters", b["line2"].isascii(), True)
        check("an unknown expression becomes shocked", b["expression"], "shocked")

        # ---- crops are exact
        check("9:16 is exactly 1080x1920", aithumb.crop_to(Image.new("RGB", (1024, 1536)), "v").size, (1080, 1920))
        check("16:9 is exactly 1280x720", aithumb.crop_to(Image.new("RGB", (1536, 1024)), "h").size, (1280, 720))

        # ---- a clip
        jid = "20260101-000000-aithumb001"
        d = config.user_dir(uid) / "jobs" / jid
        (d / "clips").mkdir(parents=True)
        shutil.copy(src, d / "clips" / "clip01.src.mp4")
        shutil.copy(src, d / "clips" / "clip01.mp4")
        Image.new("RGB", (1080, 1920)).save(d / "clips" / "clip01.jpg")
        clip = {"id": "clip01", "start": 10.0, "end": 16.0, "src": str(d / "clips" / "clip01.src.mp4"), "src_offset": 9.0,
                "status": "done", "version": 1, "file": "clip01.mp4", "cover": "clip01.jpg", "hook": "CAR LOCKED", "title": "Car locked",
                "layout": "blur", "captions": True}
        job = {"id": jid, "user": uid, "kind": "video", "creator": None, "dir": str(d), "created": time.time(),
               "source": {"type": "url", "url": "x"}, "options": dict(jobs.DEFAULTS), "status": "done", "title": "GTA 6 stream",
               "uploader": "Ducky Bhai", "game": "Grand Theft Auto VI", "width": 1280, "height": 720, "duration": 60, "clips": [clip]}
        with jobs._lock:
            jobs._jobs[jid] = job
            jobs._by_user.setdefault(uid, {})[jid] = job
        jobs.save(job)
        check("without a creator, the badge uses the stream's channel name", aithumb.brand_for(job), {"name": "DUCKY BHAI", "game": "GTA 6", "style": "vice"})

        # ---- off by default
        try:
            aithumb.request(job, clip, "v", True)
            check("AI thumbnails need AI pictures switched on", False)
        except imagegen.NotAllowed as e:
            check("AI thumbnails need AI pictures switched on", "Settings" in str(e), True)
        imagegen.save({"provider": "openai", "use_chat_key": True, "design_quality": "high", "daily_usd": 0.3})
        check("the design quality is high by default (sharp text)", imagegen.current()["design_quality"], "high")

        # ---- making one
        sent = {}

        def fake_design(images, prompt, size):
            sent.update(images=[n for n, _b in images], prompt=prompt, size=size,
                        face=Image.open(io.BytesIO(images[0][1])).size if images else None)
            return picture(1024, 1536)
        with patch.object(thumbs, "_q"):
            aithumb.request(job, clip, "v", True)
        thumbs._pending.clear()
        check("it is queued", clip["ai_thumbs"]["status"], "queued")
        spent = imagegen.spent_today()
        with patch.object(brain, "thumb_brief", return_value=dict(BRIEF)), patch.object(imagegen, "design", side_effect=fake_design):
            aithumb.make(jid, "clip01", "v")
        st = clip["ai_thumbs"]
        check("one AI thumbnail is made", (st["status"], len(st["items"])), ("done", 1))
        item = st["items"][0]
        check("it is exactly 9:16", Image.open(d / "clips" / item["file"]).size, (1080, 1920))
        check("the streamer's face and the game moment were sent", sent["images"], ["streamer.png", "scene.png"])
        check("the model was asked for its portrait size", sent["size"], "1024x1536")
        check("the headline is kept with it", item["headline"], "CAR LOCKED?!")
        check("its cost is counted at the design quality", round(imagegen.spent_today() - spent, 3), imagegen.price(None, "high"))

        # ---- the daily limit
        try:
            aithumb.request(job, clip, "v", True)
            check("the daily limit stops the next one", False)
        except imagegen.NotAllowed as e:
            check("the daily limit stops the next one", "limit" in str(e), True)
        imagegen.save({"daily_usd": 50})

        # ---- choosing, posting, deleting
        aithumb.choose(job, clip, item["id"])
        check("choosing it makes it the cover", (clip["cover"], clip["frame_cover"]), (item["file"], "clip01.jpg"))
        pic, ms = posting.cover_for(job, clip)
        check("posts use the chosen AI thumbnail", pic.name, item["file"])
        clip["thumbs"] = {"status": "done", "gen": 1, "chosen": None, "variants": [
            {"t": 11.0, "words": "X", "accent": "red", "focus": [0.5, 0.5], "base": "b.jpg", "files": {"v": "clip01-t1-v1.jpg", "h": "clip01-t1-h1.jpg"}}]}
        thumbs.choose(job, clip, 1)
        check("choosing a normal thumbnail unchooses the AI one (one cover at a time)", clip["ai_thumbs"]["chosen"], None)
        aithumb.choose(job, clip, None)
        with patch.object(brain, "thumb_brief", return_value=dict(BRIEF, face=None)), patch.object(imagegen, "design", side_effect=lambda i, p, s: picture(1536, 1024)):
            aithumb.make(jid, "clip01", "h")
        h_item = clip["ai_thumbs"]["items"][-1]
        check("a 16:9 one is exactly 1280x720", Image.open(d / "clips" / h_item["file"]).size, (1280, 720))
        try:
            aithumb.choose(job, clip, h_item["id"])
            check("a 16:9 picture can't be the (vertical) clip's cover", False)
        except ValueError:
            check("a 16:9 picture can't be the (vertical) clip's cover", True)
        aithumb.delete(job, clip, h_item["id"])
        check("deleting one removes its file", (d / "clips" / h_item["file"]).exists(), False)

        # ---- failures
        with patch.object(brain, "thumb_brief", return_value=dict(BRIEF)), \
                patch.object(imagegen, "design", side_effect=RuntimeError("The AI picture service refused this picture (its safety rules).")):
            aithumb.make(jid, "clip01", "v")
        check("a refusal is shown on the clip, nothing else changes", ("safety" in clip["ai_thumbs"]["error"], len(clip["ai_thumbs"]["items"])), (True, 1))
        c2 = {"ai_thumbs": {"status": "making", "items": []}}
        aithumb.interrupted(c2)
        check("a restart part way says so", c2["ai_thumbs"]["status"], "error")
        thumbs.remove_files(job, "clip01")
        check("deleting the clip removes its AI thumbnails too", list((d / "clips").glob("clip01-ai*")), [])

    # ---- the routes
    r = client.post(f"/api/jobs/{jid}/clips/nope/aithumbs", json={"fmt": "v"}, headers=ORIGIN)
    check("a clip that doesn't exist is 404", r.status_code, 404)
    r = client.post(f"/api/jobs/{jid}/clips/clip01/aithumbs", json={"fmt": "square"}, headers=ORIGIN)
    check("only 9:16 or 16:9", r.status_code, 400)
    r = client.post(f"/api/jobs/{jid}/clips/clip01/aithumbs/choose", json={"id": "../x"}, headers=ORIGIN)
    check("a made-up id is refused", r.status_code, 400)
    imagegen.save({"provider": "off"})
    r = client.post(f"/api/jobs/{jid}/clips/clip01/aithumbs", json={"fmt": "v"}, headers=ORIGIN)
    check("with AI pictures off, it says so", (r.status_code, "Settings" in r.text), (403, True))
    r = client.post(f"/api/jobs/{jid}/clips/clip01/aithumbs", json={"fmt": "v"}, headers={"Origin": "https://evil.example"})
    check("another website can't spend your money", r.status_code, 403)
finally:
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)

print(f"{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
