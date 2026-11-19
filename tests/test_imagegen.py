"""The paid AI look for thumbnails: off by default, its own settings (the chat AI's are never touched), the key
never shown, a daily spending limit, plain messages when the service refuses, and the words drawn by PeakCut.

Runs against the code directly, with its own empty data folder. The image service is replaced: nothing is paid for."""
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-imagegen-")
os.environ.update(DATABASE_URL="", PEAKCUT_ACCOUNTS="0", PEAKCUT_AI_PROVIDER="openai", PEAKCUT_AI_MODELS="test-model",
                  PEAKCUT_AI_KEY="sk-chat-key-123456", PEAKCUT_AI_KEY_FILE="", PEAKCUT_AI_BASE_URL="", PEAKCUT_PUBLIC_URL="http://127.0.0.1:8765")
sys.path.insert(0, ".")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from studio import auth, config, ctx, db, imagegen, jobs, server, thumbs  # noqa: E402

db.init()
fails = []
BASE = "http://127.0.0.1:8765"
ORIGIN = {"Origin": BASE}
client = TestClient(server.app, base_url=BASE, follow_redirects=False)


def check(name, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"   got {got!r}"))
    ok or fails.append(name)


class Reply:
    def __init__(self, code, data=None, text=None):
        self.status_code, self._d = code, data or {}
        self.text = text if text is not None else json.dumps(self._d)
        self.content = self.text.encode()

    def json(self):
        return self._d


def picture(w, h, colour=(200, 40, 40)):
    out = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(out, "JPEG")
    return base64.b64encode(out.getvalue()).decode()


work = tempfile.mkdtemp(prefix="pc-imagegen-work-")
uid = auth.local_user().id
try:
    with ctx.as_user(uid):
        # ---- off by default, and its own settings
        check("AI pictures are off until switched on", (imagegen.ready(), imagegen.current()["provider"]), (False, "off"))
        before = (config.DATA_DIR / "ai.json").read_text() if (config.DATA_DIR / "ai.json").exists() else None
        imagegen.save({"provider": "openai", "model": "gpt-image-2", "quality": "medium", "daily_usd": 0.1})
        after = (config.DATA_DIR / "ai.json").read_text() if (config.DATA_DIR / "ai.json").exists() else None
        check("the chat AI's settings are never touched", after, before)
        check("with an OpenAI chat key, that key is used", (imagegen.ready(), imagegen.key()), (True, "sk-chat-key-123456"))
        imagegen.save({"use_chat_key": False, "key": "sk-picture-key-987654"})
        pub = imagegen.public()
        check("its own key can be saved", imagegen.key(), "sk-picture-key-987654")
        check("the key is never shown, only its last four characters", ("sk-picture" in json.dumps(pub), pub["key_hint"]), (False, "7654"))
        check("the settings file is readable only by its owner", oct((config.DATA_DIR / "ai_image.json").stat().st_mode)[-3:] in ("600",), True)
        imagegen.save({"quality": "low"})
        check("saving without a key keeps the saved key", imagegen.key(), "sk-picture-key-987654")
        for bad, why in [({"provider": "dall-e-server"}, "provider"), ({"model": "x y/../"}, "model"), ({"quality": "ultra"}, "quality"),
                         ({"daily_usd": -1}, "negative limit"), ({"daily_usd": "lots"}, "limit that isn't a number"), ({"key": "sk with spaces"}, "key")]:
            try:
                imagegen.save(bad)
                check(f"a bad {why} is refused", False)
            except ValueError:
                check(f"a bad {why} is refused", True)
        check("a known model's price is used", imagegen.price("gpt-image-2", "medium"), 0.041)
        check("a dated version of a model costs the same", imagegen.price("gpt-image-2-2026-04-21", "high"), 0.165)
        check("an unknown model is assumed to cost more, not less", imagegen.price("mystery-image", "medium") >= 0.05, True)

        # ---- who may, and the daily limit
        imagegen.save({"quality": "medium", "daily_usd": 0.1, "everyone": False})
        try:
            imagegen.check(is_admin=False)
            check("only the admin may by default", False)
        except imagegen.NotAllowed:
            check("only the admin may by default", True)
        imagegen.check(is_admin=True)
        check("the admin may", True)
        imagegen.record(0.041, "1024x1536")
        imagegen.record(0.041, "1024x1536")
        try:
            imagegen.check(is_admin=True)
            check("the daily limit stops a third picture", False)
        except imagegen.NotAllowed as e:
            check("the daily limit stops a third picture", "limit" in str(e), True)
        check("what was spent today is counted", round(imagegen.spent_today(), 3), 0.082)
        check("the usage page counts AI pictures", imagegen.summary()["images"], 2)
        imagegen.save({"daily_usd": 5})

        # ---- one picture: the service's answers
        seen = []

        def fake_post(url, headers=None, data=None, files=None, timeout=None):
            seen.append({"url": url, "data": dict(data or {}), "files": files, "auth": (headers or {}).get("Authorization")})
            if len(seen) == 1:
                return Reply(400, {"error": {"message": "Unknown parameter: 'input_fidelity'."}})
            return Reply(200, {"data": [{"b64_json": picture(1024, 1536)}]})
        with patch.object(imagegen.httpx, "post", side_effect=fake_post):
            out = imagegen.repaint(b"\x89PNG fake", "1024x1536")
        check("a model that doesn't know a setting is asked again without it", ("input_fidelity" in seen[0]["data"], "input_fidelity" in seen[1]["data"]), (True, False))
        check("the picture comes back", Image.open(io.BytesIO(out)).size, (1024, 1536))
        check("it goes to the image editing address with the picture key", (seen[1]["url"].endswith("/images/edits"), seen[1]["auth"]), (True, "Bearer sk-picture-key-987654"))
        check("the prompt asks to keep faces and add no text", "identity" in seen[1]["data"]["prompt"] and "text" in seen[1]["data"]["prompt"], True)
        for code, body, word in [(401, {"error": {"message": "bad key"}}, "key"), (429, {"error": {"message": "quota"}}, "credit"),
                                 (400, {"error": {"message": "Your request was rejected by the safety system"}}, "safety")]:
            with patch.object(imagegen.httpx, "post", return_value=Reply(code, body)):
                try:
                    imagegen.repaint(b"x", "1024x1536")
                    check(f"HTTP {code} gives a plain message", False)
                except RuntimeError as e:
                    check(f"HTTP {code} gives a plain message about the {word}", word in str(e) and "sk-" not in str(e), True)
        with patch.object(imagegen.httpx, "get", return_value=Reply(200, {"id": "gpt-image-2"})):
            check("checking the key costs nothing (no picture is made)", imagegen.test()["ok"], True)

        # ---- the AI look on a real thumbnail
        src = os.path.join(work, "v.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25:duration=6",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", src], check=True)
        jid = "20260101-000000-imagegen01"
        d = config.user_dir(uid) / "jobs" / jid
        (d / "clips").mkdir(parents=True)
        shutil.copy(src, d / "clips" / "clip01.src.mp4")
        shutil.copy(src, d / "clips" / "clip01.mp4")
        clip = {"id": "clip01", "start": 10.0, "end": 15.0, "src": str(d / "clips" / "clip01.src.mp4"), "src_offset": 9.0, "status": "done",
                "version": 1, "file": "clip01.mp4", "cover": "clip01.jpg", "hook": "BIG MOMENT HERE", "layout": "blur"}
        job = {"id": jid, "user": uid, "kind": "video", "creator": None, "dir": str(d), "created": time.time(), "source": {"type": "url", "url": "x"},
               "options": dict(jobs.DEFAULTS), "status": "done", "title": "T", "width": 1280, "height": 720, "duration": 60, "clips": [clip]}
        with jobs._lock:
            jobs._jobs[jid] = job
            jobs._by_user.setdefault(uid, {})[jid] = job
        jobs.save(job)
        with patch.object(config, "ai_ready", return_value=False):
            thumbs.make(jid, "clip01")
        thumbs.choose(job, clip, 1)
        gen = clip["thumbs"]["gen"]
        plain_v = clip["thumbs"]["variants"][0]["files"]["v"]
        with patch.object(thumbs, "_q"):
            thumbs.request_look(job, clip, 1, "v", True)
            check("an AI look is queued", clip["thumbs"]["restyling"], {"n": 1, "fmt": "v"})
            try:
                thumbs.request_look(job, clip, 2, "h", True)
                check("only one AI look at a time per clip", False)
            except thumbs.Busy:
                check("only one AI look at a time per clip", True)
        thumbs._pending.clear()
        sizes = []

        def fake_repaint(png, size):
            sizes.append((Image.open(io.BytesIO(png)).size, size))
            return base64.b64decode(picture(1024, 1536, (10, 200, 90)))
        spent = imagegen.spent_today()
        with patch.object(imagegen, "repaint", side_effect=fake_repaint):
            thumbs.make_look(jid, "clip01", 1, "v", gen)
        v = clip["thumbs"]["variants"][0]
        folder = d / "clips"
        check("the picture sent is the model's portrait size", sizes[0], ((1024, 1536), "1024x1536"))
        check("the AI look is kept for this shape", (v.get("looks") or {}).get("v", "").startswith("clip01-t"), True)
        check("the finished picture is 1080x1920 again", Image.open(folder / v["files"]["v"]).size, (1080, 1920))
        check("its middle is the repainted green, with the words on top", Image.open(folder / v["files"]["v"]).getpixel((540, 200))[1] > 150, True)
        check("the chosen cover follows the AI look", clip["cover"], v["files"]["v"])
        check("the plain picture it replaced is removed", (folder / plain_v).exists(), False)
        check("it is counted with its rough cost", round(imagegen.spent_today() - spent, 3), imagegen.price())
        check("the busy mark is cleared", clip["thumbs"].get("restyling"), None)
        # new words keep the AI look
        thumbs.edit(job, clip, 1, words="new words")
        v = clip["thumbs"]["variants"][0]
        check("changing the words keeps the AI look", Image.open(folder / v["files"]["v"]).getpixel((540, 200))[1] > 150, True)
        # undo
        look = v["looks"]["v"]
        thumbs.drop_look(job, clip, 1, "v")
        v = clip["thumbs"]["variants"][0]
        check("Undo goes back to the normal picture", ((v.get("looks") or {}).get("v"), (folder / look).exists()), (None, False))
        # a new set made meanwhile: the old AI look is thrown away, not charged twice
        with patch.object(imagegen, "repaint", side_effect=fake_repaint):
            thumbs.make_look(jid, "clip01", 1, "h", gen - 1)
        check("an AI look for a set that was replaced gives a plain message", "replaced" in (clip["thumbs"].get("look_error") or ""), True)
        # the service refuses
        with patch.object(imagegen, "repaint", side_effect=RuntimeError("The AI picture service refused this picture (its safety rules).")):
            thumbs.make_look(jid, "clip01", 2, "h", gen)
        check("a refusal is shown on the thumbnail, nothing changes", ("safety" in clip["thumbs"]["look_error"], (clip["thumbs"]["variants"][1].get("looks") or {}).get("h")), (True, None))
        # deleting the clip removes the AI pictures too
        thumbs.remove_files(job, "clip01")
        check("deleting the clip removes its AI pictures too", sorted(p.name for p in folder.glob("clip01-t*")), [])

    # ---- the routes
    imagegen.save({"provider": "off"})
    r = client.post(f"/api/jobs/{jid}/clips/clip01/thumbs/1/look", json={"fmt": "v"}, headers=ORIGIN)
    check("with AI pictures off, asking for one is refused with a reason", (r.status_code, "Settings" in r.text), (403, True))
    st = client.get("/api/status").json()
    check("the page is told AI pictures are off", st["ai_image"], {"on": False})
    r = client.put("/api/ai/image", json={"provider": "openai", "use_chat_key": True}, headers=ORIGIN)
    check("the owner can switch them on", (r.status_code, r.json()["ready"]), (200, True))
    check("the page is then told the price", client.get("/api/status").json()["ai_image"]["on"], True)
    r = client.get("/api/ai/image")
    check("the settings never include a key", "sk-" in r.text, False)
    r = client.put("/api/ai/image", json={"provider": "openai"}, headers={"Origin": "https://evil.example"})
    check("another website can't change them", r.status_code, 403)
    check("the usage page includes AI pictures", "images" in client.get("/api/usage").json(), True)
finally:
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)

print(f"{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
