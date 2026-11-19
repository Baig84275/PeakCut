"""Thumbnails for clips: frames without captions, three ideas, both sizes, choosing, changing the words, and
everything that can go wrong around them (a clip changed or deleted meanwhile, no AI, an AI that answers badly).

Runs against the code directly, with its own empty data folder and no network: the AI is replaced."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-thumbs-")
os.environ.update(DATABASE_URL="", PEAKCUT_ACCOUNTS="0", PEAKCUT_AI_PROVIDER="openai", PEAKCUT_AI_MODELS="test-model",
                  PEAKCUT_AI_KEY="sk-test", PEAKCUT_AI_KEY_FILE="", PEAKCUT_AI_BASE_URL="", PEAKCUT_PUBLIC_URL="http://127.0.0.1:8765")
sys.path.insert(0, ".")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from studio import ai, auth, brain, config, ctx, db, jobs, server, thumbs  # noqa: E402

db.init()
fails = []
BASE = "http://127.0.0.1:8765"
ORIGIN = {"Origin": BASE}
client = TestClient(server.app, base_url=BASE, follow_redirects=False)


def check(name, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"   got {got!r}"))
    ok or fails.append(name)


work = tempfile.mkdtemp(prefix="pc-thumbs-work-")


def video(name, size="1280x720", secs=8):
    path = os.path.join(work, name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25:duration={secs}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", path], check=True)
    return path


uid = auth.local_user().id if hasattr(auth, "local_user") else "u_local"
n_job = [0]


def new_job(src, w=1280, h=720, **clip_extra):
    """A finished video with one finished clip, registered the way jobs.load does it."""
    n_job[0] += 1
    jid = f"20260101-000000-test{n_job[0]:06d}"
    d = config.user_dir(uid) / "jobs" / jid
    (d / "clips").mkdir(parents=True)
    shutil.copy(src, d / "clips" / "clip01.src.mp4")
    shutil.copy(src, d / "clips" / "clip01.mp4")
    Image.new("RGB", (1080, 1920), (20, 20, 20)).save(d / "clips" / "clip01.jpg")
    clip = {"id": "clip01", "start": 100.0, "end": 106.0, "src": str(d / "clips" / "clip01.src.mp4"), "src_offset": 99.0,
            "status": "done", "version": 1, "file": "clip01.mp4", "cover": "clip01.jpg", "hook": "HE DID NOT SEE IT",
            "title": "Big moment", "highlight_words": ["boom", "what"], "layout": "blur", "captions": True,
            "fixed_words": [{"w": "boom", "s": 101.0, "e": 101.4}, {"w": "what", "s": 103.0, "e": 103.3}], **clip_extra}
    job = {"id": jid, "user": uid, "kind": "video", "creator": None, "dir": str(d), "created": time.time(), "source": {"type": "url", "url": "x"},
           "options": dict(jobs.DEFAULTS), "status": "done", "title": "Test video", "width": w, "height": h, "duration": 600,
           "clips": [clip]}
    with jobs._lock:
        jobs._jobs[jid] = job
        jobs._by_user.setdefault(uid, {})[jid] = job
    jobs.save(job)
    return job, clip


def files(job):
    return sorted(p.name for p in (jobs.Path(job["dir"]) / "clips").glob("clip01-t*.jpg"))


try:
    wide = video("wide.mp4")
    tall = video("tall.mp4", "720x1280")

    with ctx.as_user(uid):
        # ---- words: Roman letters, capitals, no links, at most four words
        check("Hindi becomes Roman letters", thumbs.clean_words("भाई क्या हुआ"), lambda s: s and s.isupper() and all(ord(c) < 0x250 for c in s))
        check("links and handles are removed", thumbs.clean_words("watch www.spam.com @someone NOW"), "WATCH NOW")
        check("at most four words", len(thumbs.clean_words("one two three four five six").split()), 4)
        check("emoji are removed", thumbs.clean_words("CRAZY 😂 WIN"), "CRAZY WIN")
        check("a script the fonts can't draw gives no words", thumbs.clean_words("これは何"), "")
        check("without an AI the hook gives the words", thumbs.fallback_words({"hook": "big clutch moment", "highlight_words": ["wow"]})[0], "BIG CLUTCH MOMENT")
        check("there are always three sets of words", len(thumbs.fallback_words({})), 3)

        # ---- the AI's answer is cleaned
        tidy = brain._tidy_thumbs([{"frame": 9, "words": "visit http://x.com NOW", "accent": "PURPLE", "focus_x": 400},
                                   {"frame": "2", "words": "ok", "accent": "red"}, "junk"], 6)
        check("a frame number out of range is dropped", tidy[0]["frame"], None)
        check("a link in the words is removed", "x.com" not in tidy[0]["words"], True)
        check("focus is kept inside the picture", tidy[0]["focus_x"], 100.0)
        check("a frame number as text is read", tidy[1]["frame"], 2)
        check("junk items are skipped", len(tidy), 2)
        check("an answer wrapped in an object is read", len(brain._tidy_thumbs({"items": [{"frame": 1, "words": "a b", "accent": "red"}]}, 3)), 1)

        # ---- frames come from the clip's own video, without captions
        job, clip = new_job(wide)
        path, off, burned = thumbs.source_for(job, clip)
        check("frames come from the clip's own piece of video", (path.name, off, burned), ("clip01.src.mp4", 99.0, False))
        ts = thumbs.times_for(clip, False)
        check("three to ten moments are looked at, all inside the clip", 3 <= len(ts) <= 10 and all(100 <= t <= 106 for t in ts), True)
        os.remove(clip["src"])
        path, off, burned = thumbs.source_for(job, clip)
        check("without the piece, the finished clip is used and known to have captions", (path.name, burned), ("clip01.mp4", True))
        quiet = thumbs.times_for(clip, True)
        check("...and only moments with no word on screen are used", all(not (100.9 <= t <= 101.5 or 102.9 <= t <= 103.4) for t in quiet), True)
        shutil.copy(wide, clip["src"])

        # ---- a whole set without an AI
        with patch.object(config, "ai_ready", return_value=False):
            thumbs.make(job["id"], "clip01")
        st = clip["thumbs"]
        check("without an AI a set is still made", (st["status"], st["by"], len(st["variants"])), ("done", "auto", 3))
        v = st["variants"][0]
        folder = jobs.Path(job["dir"]) / "clips"
        check("the 9:16 picture is 1080x1920", Image.open(folder / v["files"]["v"]).size, (1080, 1920))
        check("the 16:9 picture is 1280x720", Image.open(folder / v["files"]["h"]).size, (1280, 720))
        check("the clean frame is kept for redrawing", (folder / v["base"]).is_file(), True)
        check("the words came from the hook", v["words"], "HE DID NOT SEE")

        # ---- with an AI: its frames, words and colours are used, its junk is not
        ideas = [{"frame": 2, "words": "NO WAY", "accent": "red", "focus_x": 30.0, "focus_y": 40.0, "subject_w": 20.0},
                 {"frame": 2, "words": "", "accent": "nope", "focus_x": 50.0, "focus_y": 50.0, "subject_w": 50.0},
                 {"frame": None, "words": "INSANE", "accent": "cyan", "focus_x": 50.0, "focus_y": 50.0, "subject_w": 80.0}]
        with patch.object(brain, "thumb_ideas", return_value=ideas) as asked:
            thumbs.make(job["id"], "clip01")
        st = clip["thumbs"]
        check("the AI was shown one picture of numbered frames", asked.call_count == 1 and isinstance(asked.call_args[0][0], bytes), True)
        check("its ideas are used", (st["by"], st["variants"][0]["words"], st["variants"][0]["accent"]), ("ai", "NO WAY", "red"))
        check("empty words fall back to the clip's own", st["variants"][1]["words"] != "", True)
        check("an unknown colour falls back to one of the choices", st["variants"][1]["accent"] in thumbs.ACCENTS, True)
        check("a second set replaces the first: only its files are left", all("-t2-" in f for f in files(job)), True)
        check("each set has its own file names (no stale browser copies)", st["gen"], 2)

        # ---- an AI that fails or can't see pictures: the local choice still makes a set
        with patch.object(brain, "thumb_ideas", side_effect=RuntimeError("The AI did not answer")):
            thumbs.make(job["id"], "clip01")
        check("an AI error still gives thumbnails", (clip["thumbs"]["status"], clip["thumbs"]["by"]), ("done", "auto"))

        # ---- choosing and going back
        thumbs.choose(job, clip, 2)
        check("choosing one makes it the clip's cover", (clip["cover"], clip["thumbs"]["chosen"]), (clip["thumbs"]["variants"][1]["files"]["v"], 2))
        check("the plain cover is remembered", clip["frame_cover"], "clip01.jpg")
        thumbs.choose(job, clip, None)
        check("going back restores the plain cover", (clip["cover"], clip["thumbs"]["chosen"]), ("clip01.jpg", None))
        try:
            thumbs.choose(job, clip, 7)
            check("a thumbnail that doesn't exist can't be chosen", False)
        except ValueError:
            check("a thumbnail that doesn't exist can't be chosen", True)

        # ---- changing the words redraws from the clean frame
        thumbs.choose(job, clip, 1)
        old = dict(clip["thumbs"]["variants"][0]["files"])
        thumbs.edit(job, clip, 1, words="totally new words here please", accent="green")
        v = clip["thumbs"]["variants"][0]
        check("new words are cleaned and used", (v["words"], v["accent"]), ("TOTALLY NEW WORDS HERE", "green"))
        check("the redrawn picture has a new name", v["files"]["v"] != old["v"] and (folder / v["files"]["v"]).is_file(), True)
        check("the old pictures are removed", (folder / old["v"]).exists() or (folder / old["h"]).exists(), False)
        check("the chosen cover follows the redraw", clip["cover"], v["files"]["v"])
        try:
            thumbs.edit(job, clip, 1, accent="purple")
            check("an unknown colour is refused", False)
        except ValueError:
            check("an unknown colour is refused", True)

        # ---- the clip is made again: the chosen cover stays, a moved clip marks the set stale
        clip["cover"] = "clip01.jpg"
        clip["start"], clip["end"] = 200.0, 210.0
        thumbs.clip_changed(clip)
        check("after a re-make the chosen thumbnail is the cover again", clip["cover"], clip["thumbs"]["variants"][0]["files"]["v"])
        check("thumbnails from outside the new times are marked stale", clip["thumbs"]["stale"], True)
        clip["start"], clip["end"] = 100.0, 106.0

        # ---- the clip changes while its thumbnails are being made: the work is thrown away
        before = files(job)

        real = thumbs._ideas

        def racing(j, c, fr):
            out = real(j, c, fr)
            c["version"] += 1
            return out
        with patch.object(thumbs, "_ideas", side_effect=racing), patch.object(config, "ai_ready", return_value=False):
            thumbs.make(job["id"], "clip01")
        check("a clip changed meanwhile gets a clear message", ("changed" in (clip["thumbs"]["error"] or ""), clip["thumbs"]["status"]), (True, "error"))
        check("...and none of the new pictures are kept", files(job), before)

        # ---- the clip is deleted while its thumbnails are being made
        job2, clip2 = new_job(wide)

        def deleting(j, c, fr):
            out = real(j, c, fr)
            j["clips"] = []
            return out
        with patch.object(thumbs, "_ideas", side_effect=deleting), patch.object(config, "ai_ready", return_value=False):
            thumbs.make(job2["id"], "clip01")
        check("a clip deleted meanwhile leaves no pictures behind", files(job2), [])

        # ---- queueing: only finished clips, never twice
        job3, clip3 = new_job(wide, status="rendering")
        try:
            thumbs.request(job3, clip3)
            check("a clip still being made can't get thumbnails yet", False)
        except thumbs.Busy:
            check("a clip still being made can't get thumbnails yet", True)
        clip3["status"] = "done"
        with patch.object(thumbs, "_q"):
            thumbs.request(job3, clip3)
            try:
                thumbs.request(job3, clip3)
                check("pressing twice doesn't queue it twice", False)
            except thumbs.Busy:
                check("pressing twice doesn't queue it twice", True)
        thumbs._pending.clear()

        # ---- a restart part way: a plain message, and it can be pressed again
        c4 = {"thumbs": {"status": "making", "variants": []}}
        thumbs.interrupted(c4)
        check("a restart part way says so", (c4["thumbs"]["status"], "restart" in c4["thumbs"]["error"]), ("error", True))

        # ---- a tall phone video gives a filled 16:9 picture
        job5, clip5 = new_job(tall, 720, 1280)
        with patch.object(config, "ai_ready", return_value=False):
            thumbs.make(job5["id"], "clip01")
        v5 = clip5["thumbs"]["variants"][0]
        check("a tall video still gives both sizes", (Image.open(jobs.Path(job5["dir"]) / "clips" / v5["files"]["h"]).size,
                                                      Image.open(jobs.Path(job5["dir"]) / "clips" / v5["files"]["v"]).size), ((1280, 720), (1080, 1920)))

        # ---- the source video is gone completely
        job6, clip6 = new_job(wide)
        os.remove(clip6["src"])
        os.remove(jobs.Path(job6["dir"]) / "clips" / "clip01.mp4")
        thumbs.make(job6["id"], "clip01")
        check("no video at all gives a plain error, not a crash", (clip6["thumbs"]["status"], "gone" in clip6["thumbs"]["error"]), ("error", True))

        # ---- deleting a clip removes every thumbnail, and only that clip's
        job7, clip7 = new_job(wide)
        with patch.object(config, "ai_ready", return_value=False):
            thumbs.make(job7["id"], "clip01")
        stray = jobs.Path(job7["dir"]) / "clips" / "clip010-t1-v1.jpg"
        stray.write_bytes(b"x")
        thumbs.remove_files(job7, "clip01")
        check("deleting a clip removes its thumbnails", files(job7), [])
        check("...but not another clip's with a similar name", stray.exists(), True)

        # ---- ai.see can answer in JSON, through each kind of service
        sent = {}

        class Reply:
            status_code = 200

            def __init__(self, data):
                self._d = data
                self.text = json.dumps(data)

            def json(self):
                return self._d

        def fake_post(url, json=None, headers=None, timeout=None):
            sent["url"], sent["body"] = url, json
            if url.endswith("/chat/completions"):
                return Reply({"choices": [{"message": {"content": 'Sure! {"items": [{"frame": 1, "words": "A B", "accent": "red"}]}'}}],
                              "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
            return Reply({"candidates": [{"content": {"parts": [{"text": '[{"frame": 2, "words": "C D", "accent": "cyan"}]'}]}}],
                          "usageMetadata": {"promptTokenCount": 7, "totalTokenCount": 9}})
        with patch.object(ai.httpx, "post", side_effect=fake_post):
            out, a, b = ai.see("pick", b"\xff\xd8jpeg", schema=brain.THUMB_SCHEMA)
            check("an OpenAI-style answer with text around the JSON is read", (out[0]["frame"], a, b), (1, 10, 5))
            check("...and the shape was asked for in the message", "JSON Schema" in json.dumps(sent["body"]), True)
            with patch.object(ai, "provider", return_value="gemini"):
                out, a, b = ai.see("pick", b"\xff\xd8jpeg", schema=brain.THUMB_SCHEMA, model="gemini-x")
                check("Gemini is given the shape directly", sent["body"].get("generationConfig", {}).get("responseMimeType"), "application/json")
                check("...and its answer is read", (out[0]["words"], sent["url"].endswith("gemini-x:generateContent")), ("C D", True))
            text, _a, _b = ai.see("plain", b"\xff\xd8jpeg")
            check("without a shape, the plain answer still comes back as text (the webcam finder)", isinstance(text, str), True)

        # ---- the routes: only finished clips, someone else's never
        jobA, clipA = new_job(wide)
        r = client.post(f"/api/jobs/{jobA['id']}/clips/nope/thumbs", headers=ORIGIN)
        check("a clip that doesn't exist is 404", r.status_code, 404)
        r = client.post(f"/api/jobs/20260101-000000-zzzzzzzz/clips/clip01/thumbs", headers=ORIGIN)
        check("a video that doesn't exist is 404", r.status_code, 404)
        clipA["status"] = "rendering"
        r = client.post(f"/api/jobs/{jobA['id']}/clips/clip01/thumbs", headers=ORIGIN)
        check("a clip still being made is 409", r.status_code, 409)
        clipA["status"] = "done"
        with patch.object(thumbs, "_q"):
            r = client.post(f"/api/jobs/{jobA['id']}/clips/clip01/thumbs", headers=ORIGIN)
            check("a finished clip is queued", (r.status_code, clipA["thumbs"]["status"]), (200, "queued"))
            r = client.post(f"/api/jobs/{jobA['id']}/clips/clip01/thumbs", headers=ORIGIN)
            check("pressing again while it's queued is 409", r.status_code, 409)
            r = client.post("/api/thumbs/bulk", json={"clips": [[jobA["id"], "clip01"], [job7["id"], "clip01"], ["bad id!", "x"]]}, headers=ORIGIN)
            check("many at once: the busy one is skipped with its reason", (r.json()["queued"], len(r.json()["skipped"])), (1, 1))
        thumbs._pending.clear()
        r = client.post(f"/api/jobs/{job7['id']}/clips/clip01/thumbs/choose", json={"n": "2"}, headers=ORIGIN)
        check("choosing needs a number", r.status_code, 400)
        r = client.put(f"/api/jobs/{job7['id']}/clips/clip01/thumbs/1", json={"words": "x" * 200}, headers=ORIGIN)
        check("very long words are refused", r.status_code, 400)
        r = client.post("/api/jobs/x/clips/clip01/thumbs", headers={"Origin": "https://evil.example"})
        check("another website can't start thumbnails", r.status_code, 403)
finally:
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)

print(f"{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
