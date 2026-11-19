"""One-click Connect for YouTube, Instagram and TikTok, and posting to each. The platforms are replaced by
stand-ins answering the way their documentation says, so nothing leaves this machine."""
import json, os, sys, tempfile, time
from urllib.parse import parse_qs, urlparse
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-connect-")
os.environ.update(DATABASE_URL="", PEAKCUT_ACCOUNTS="1", PEAKCUT_PUBLIC_URL="https://clips.example.com", PEAKCUT_VERIFY_EMAIL="0",
                  GOOGLE_CLIENT_ID="", GOOGLE_CLIENT_SECRET="", GOOGLE_CLIENT_FILE="", PEAKCUT_IG_APP_ID="", PEAKCUT_IG_APP_SECRET="",
                  PEAKCUT_TIKTOK_CLIENT_KEY="", PEAKCUT_TIKTOK_CLIENT_SECRET="")
sys.path.insert(0, ".")
from pathlib import Path
import sqlalchemy as sa
from fastapi.testclient import TestClient
from studio import accounts, auth, config, ctx, db, server
db.init()
fails = []
def check(n, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   got {got!r}")); ok or fails.append(n)

BASE = "https://clips.example.com"
ORIGIN = {"Origin": BASE}
def person(email, admin=False):
    u, _ = auth._create(email, email.split("@")[0], password_hash="x")
    if admin:
        with db.engine.begin() as c:
            c.execute(db.users.update().where(db.users.c.id == u.id).values(is_admin=True))
    cl = TestClient(server.app, base_url=BASE, follow_redirects=False)
    cl.cookies.set(auth.COOKIE, auth.start_session(u.id))
    return u, cl
boss, a = person("boss@example.com", admin=True)
user, b = person("user@example.com")

# ---- nothing is set up: the buttons say so, and only an admin may set it up
st = b.get("/api/status").json()
check("nothing can be connected before setup", st["connect"], {"youtube": False, "instagram": False, "tiktok": False})
check("a normal user cannot see the setup", b.get("/api/apps").status_code, 404)
check("...or change it", b.put("/api/apps", json={"platform": "tiktok", "client_id": "abcd", "client_secret": "x"}, headers=ORIGIN).status_code, 404)
r = b.get("/oauth/instagram/start")
check("Connect before setup comes back with a reason", (r.status_code, "connect_error" in r.headers.get("location", "")), (303, True))
check("setup from another website is refused", a.put("/api/apps", json={"platform": "tiktok", "client_id": "abcd1234", "client_secret": "s"},
                                                    headers={"Origin": "https://evil.example"}).status_code, 403)
check("a bad id is refused", a.put("/api/apps", json={"platform": "tiktok", "client_id": "a b<", "client_secret": "s"}, headers=ORIGIN).status_code, 400)
for pf, cid in (("youtube", "g-client-123.apps"), ("instagram", "1234567890"), ("tiktok", "awtiktokkey123")):
    r = a.put("/api/apps", json={"platform": pf, "client_id": cid, "client_secret": f"{pf}-secret-value"}, headers=ORIGIN)
    check(f"{pf} app saved", (r.status_code, r.json()[pf]["ready"]), (200, True))
apps = a.get("/api/apps").json()
check("secrets are never sent back", any("secret-value" in json.dumps(v) for v in apps.values()), False)
check("each platform shows the address to register", apps["tiktok"]["redirect"], BASE + "/oauth/tiktok/callback")
r = a.put("/api/apps", json={"platform": "tiktok", "client_id": "", "client_secret": "tiktok-secret-value2"}, headers=ORIGIN)
check("a new secret alone keeps the saved id", accounts.app_keys("tiktok"), ("awtiktokkey123", "tiktok-secret-value2"))
check("everyone can connect now", b.get("/api/status").json()["connect"], {"youtube": True, "instagram": True, "tiktok": True})

# ---- the platforms, as stand-ins
class R:
    def __init__(self, code, data=None, headers=None, text=None):
        self.status_code, self._d, self.headers = code, data, headers or {}
        self.text = text if text is not None else json.dumps(data)
    def json(self):
        if self._d is None:
            raise ValueError
        return self._d
calls, uploads, tt = [], [], {"audited": False, "status": 0}
def post(url, data=None, json=None, headers=None, params=None, timeout=None, content=None):
    calls.append(("POST", url, data or json, headers or {}))
    if url == "https://oauth2.googleapis.com/token":
        return R(200, {"access_token": "yt-access", "refresh_token": "yt-refresh", "expires_in": 3600})
    if url == "https://api.instagram.com/oauth/access_token":
        if data["code"] != "igcode":
            return R(400, {"error_message": "bad code"})
        return R(200, {"data": [{"access_token": "ig-short", "user_id": "17841", "permissions": "instagram_business_basic,instagram_business_content_publish"}]})
    if url.endswith("/oauth/token/"):
        if data.get("grant_type") == "refresh_token":
            return R(200, {"access_token": "tt-access-2", "refresh_token": "tt-refresh-2", "expires_in": 7200, "open_id": "oid", "scope": "user.info.basic,video.publish"})
        return R(200, {"access_token": "tt-access", "refresh_token": "tt-refresh", "expires_in": 7200, "open_id": "oid", "scope": "user.info.basic,video.publish"})
    if url.endswith("/creator_info/query/"):
        return R(200, {"data": {"creator_username": "streamerclips", "creator_nickname": "Streamer Clips",
                                "privacy_level_options": ["PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"]}, "error": {"code": "ok"}})
    if url.endswith("/post/publish/video/init/"):
        lv = json["post_info"]["privacy_level"]
        tt["init"] = json
        if lv != "SELF_ONLY" and not tt["audited"]:
            return R(403, {"error": {"code": "unaudited_client_can_only_post_to_private_accounts", "message": "private only"}})
        return R(200, {"data": {"publish_id": "v_pub_1", "upload_url": "https://open-upload.tiktokapis.com/video/?upload_id=1"}, "error": {"code": "ok"}})
    if url.endswith("/post/publish/status/fetch/"):
        tt["status"] += 1
        if tt["status"] < 2:
            return R(200, {"data": {"status": "PROCESSING_UPLOAD"}, "error": {"code": "ok"}})
        return R(200, {"data": {"status": "PUBLISH_COMPLETE", "publicaly_available_post_id": [7300] if tt["audited"] else []}, "error": {"code": "ok"}})
    if url.endswith("/17841/media"):
        return R(200, {"id": "container1"})
    if url.endswith("/17841/media_publish"):
        return R(200, {"id": "media1"})
    return R(404, {})
def get(url, params=None, headers=None, timeout=None):
    calls.append(("GET", url, params, headers or {}))
    if url.endswith("/youtube/v3/channels"):
        return R(200, {"items": [{"id": "UC123", "snippet": {"title": "My Channel"}}]})
    if url == "https://graph.instagram.com/access_token":
        return R(200, {"access_token": "ig-long", "token_type": "bearer", "expires_in": 5183944})
    if url == "https://graph.instagram.com/me":
        return R(200, {"user_id": "17841", "username": "clipper", "account_type": "MEDIA_CREATOR"})
    if url.endswith("/container1"):
        return R(200, {"status_code": "FINISHED"})
    if url.endswith("/media1"):
        return R(200, {"permalink": "https://www.instagram.com/reel/abc/"})
    return R(404, {})
def put(url, content=None, headers=None, timeout=None):
    uploads.append((url, len(content), headers))
    return R(201, None, text="")
accounts.httpx.post, accounts.httpx.get, accounts.httpx.put = post, get, put
accounts.time.sleep = lambda s: None

# ---- connecting: the button goes to the platform's own page, and back
def connect(cl, pf, code="code1"):
    r = cl.get(f"/oauth/{pf}/start")
    loc = urlparse(r.headers["location"])
    state = parse_qs(loc.query)["state"][0]
    back = cl.get(f"/oauth/{pf}/callback", params={"code": code, "state": state})
    return loc, back
loc, back = connect(b, "youtube")
check("YouTube: Google's page, asking to upload", (loc.netloc, "youtube.upload" in parse_qs(loc.query)["scope"][0]), ("accounts.google.com", True))
check("YouTube: back on Connected apps, connected", (back.status_code, back.headers["location"].startswith("/app/settings/apps?connected=youtube")), (303, True))
loc, back = connect(b, "instagram", "igcode#_")
q = parse_qs(loc.query)
check("Instagram: Instagram's own page, with publishing permission", (loc.netloc, q["client_id"][0], "instagram_business_content_publish" in q["scope"][0]),
      ("www.instagram.com", "1234567890", True))
check("Instagram: connected, with the #_ Instagram adds to the code removed", "connected=instagram" in back.headers["location"])
loc, back = connect(b, "tiktok")
q = parse_qs(loc.query)
check("TikTok: TikTok's own page, asking to post", (loc.netloc, q["client_key"][0], q["scope"][0]), ("www.tiktok.com", "awtiktokkey123", "user.info.basic,video.publish"))
check("TikTok: connected", "connected=tiktok" in back.headers["location"])
accs = {x["platform"]: x for x in b.get("/api/accounts").json()}
check("all three accounts are listed with their names", (accs["youtube"]["name"], accs["instagram"]["name"], accs["tiktok"]["name"]),
      ("My Channel", "@clipper", "@streamerclips"))
check("tokens never reach the page", any(t in b.get("/api/accounts").text for t in ("ig-long", "tt-access", "yt-refresh")), False)
check("someone else's accounts are not mine", a.get("/api/accounts").json(), [])
r = b.get("/oauth/tiktok/callback", params={"code": "x", "state": "made-up"})
check("a made-up return is refused", "connect_error" in r.headers["location"])
r = b.get("/oauth/tiktok/callback", params={"error": "access_denied"})
check("cancelling on the platform says so", "cancelled" in r.headers["location"])
loc = urlparse(a.get("/oauth/tiktok/start").headers["location"])
r = b.get("/oauth/tiktok/callback", params={"code": "c", "state": parse_qs(loc.query)["state"][0]})
check("one person's sign-in cannot be finished by another", "connect_error" in r.headers["location"])
check("an unknown platform does not exist", b.get("/oauth/myspace/start").status_code, 404)

# ---- the one-hour links Instagram fetches clips from
with ctx.as_user(user.id):
    jd = config.user_dir(user.id) / "jobs" / "job-12345678" / "clips"
jd.mkdir(parents=True)
(jd.parent / "job.json").write_text(json.dumps({"id": "job-12345678", "user": user.id, "dir": str(jd.parent), "clips": [], "status": "done",
                                                "creator": None, "created": time.time(), "options": {}, "source": {"type": "file"}}), encoding="utf-8")
(jd / "clip01.mp4").write_bytes(b"\x00" * 6_000)
from studio import jobs
jobs.load_user(user.id)
with ctx.as_user(user.id):
    link = accounts.clip_link("job-12345678", "clip01.mp4")
anon = TestClient(server.app, base_url=BASE)
path = urlparse(link).path
check("a signed link serves the clip to anyone, no sign-in", (anon.get(path).status_code, len(anon.get(path).content)), (200, 6000))
bad = path.replace(".mp4", "x.mp4")
check("a changed link does not", anon.get(bad).status_code, 404)
check("a made-up link does not", anon.get("/pub/abc.def.mp4").status_code, 404)
with ctx.as_user(user.id):
    old = accounts.clip_link("job-12345678", "clip01.mp4", seconds=-5)
check("an expired link does not", anon.get(urlparse(old).path).status_code, 404)

# ---- posting
with ctx.as_user(user.id):
    ig, ttacc = accounts.get(accs["instagram"]["id"]), accounts.get(accs["tiktok"]["id"])
    calls.clear()
    rid, url = accounts.instagram_upload(ig, jd / "clip01.mp4", "My caption", "job-12345678")
    made = next(c for c in calls if c[1].endswith("/17841/media"))
    check("Instagram: a Reel from the signed link", (made[2]["media_type"], made[2]["video_url"].startswith(BASE + "/pub/")), ("REELS", True))
    check("Instagram: published, with its link", (rid, url), ("media1", "https://www.instagram.com/reel/abc/"))

    uploads.clear(); tt.update(audited=False, status=0)
    rid, url = accounts.tiktok_upload(ttacc, jd / "clip01.mp4", "My caption #fyp", "public")
    check("TikTok: an app not yet reviewed posts privately instead of failing", (tt["init"]["post_info"]["privacy_level"], url), ("SELF_ONLY", ""))
    check("TikTok: the whole file in one piece, with the right range", (len(uploads), uploads[0][2]["Content-Range"]), (1, "bytes 0-5999/6000"))
    check("TikTok: the account says why posts are private", "private" in accounts.get(ttacc["id"]).get("detail", ""))
    tt.update(audited=True, status=0)
    rid, url = accounts.tiktok_upload(ttacc, jd / "clip01.mp4", "My caption", "public")
    check("TikTok: once reviewed, public, with the video's link", (tt["init"]["post_info"]["privacy_level"], url),
          ("PUBLIC_TO_EVERYONE", "https://www.tiktok.com/@streamerclips/video/7300"))
    check("TikTok: big files go in pieces, the last taking the rest", accounts._tt_chunks(150_000_123), (10_000_000, 15))
    acc = accounts.get(ttacc["id"]); acc["expires"] = 0; accounts._put(acc)
    accounts._tt_token(accounts.get(ttacc["id"]))
    check("TikTok: an old sign-in is renewed, and the new refresh token kept", accounts.get(ttacc["id"])["refresh_token"], "tt-refresh-2")

# ---- thumbnails go along with a post
from PIL import Image
from studio import posting
Image.new("RGB", (1080, 1920), (200, 30, 30)).save(jd / "clip01-t1-v1.jpg")
Image.new("RGB", (1080, 1920), (30, 200, 30)).save(jd / "clip01-t1-v2.jpg")
clipT = {"id": "clip01", "start": 100.0, "end": 130.0, "status": "done", "file": "clip01.mp4", "cover": "clip01.jpg",
         "thumbs": {"status": "done", "gen": 1, "chosen": 2, "variants": [
             {"t": 101.0, "files": {"v": "clip01-t1-v1.jpg", "h": "clip01-t1-h1.jpg"}},
             {"t": 112.5, "files": {"v": "clip01-t1-v2.jpg", "h": "clip01-t1-h2.jpg"}}]}}
jobT = jobs._jobs["job-12345678"]
jobT["clips"] = [clipT]
pic, ms = posting.cover_for(jobT, clipT)
check("a post uses the chosen thumbnail and its moment in the clip", (pic.name, ms), ("clip01-t1-v2.jpg", 12500))
clipT["thumbs"]["chosen"] = None
check("with none chosen, the first one is used", posting.cover_for(jobT, clipT)[0].name, "clip01-t1-v1.jpg")
check("a clip without thumbnails posts as before", posting.cover_for(jobT, {"start": 0, "end": 5}), (None, None))
clipT["thumbs"]["chosen"] = 2
with ctx.as_user(user.id):
    link = accounts.clip_link("job-12345678", "clip01-t1-v2.jpg")
    check("a thumbnail gets its own signed .jpg address", link.endswith(".jpg") and link.startswith(BASE + "/pub/"), True)
    r = anon.get(urlparse(link).path)
    check("...which serves the picture to the platform, no sign-in", (r.status_code, r.headers["content-type"]), (200, "image/jpeg"))
    check("...but never as a video", anon.get(urlparse(link).path[:-4] + ".mp4").status_code, 404)
    vid = accounts.clip_link("job-12345678", "clip01.mp4")
    check("...and a video address can't be read as a picture", anon.get(urlparse(vid).path[:-4] + ".jpg").status_code, 404)

    calls.clear()
    accounts.instagram_upload(ig, jd / "clip01.mp4", "My caption", "job-12345678", jd / "clip01-t1-v2.jpg", 12500)
    made = next(c for c in calls if c[1].endswith("/17841/media"))
    check("Instagram: the thumbnail goes along as the Reel's cover", made[2].get("cover_url", "").startswith(BASE + "/pub/"), True)
    old_url = config.PUBLIC_URL
    config.PUBLIC_URL = "http://127.0.0.1:8765"
    accounts.PUBLIC_URL = config.PUBLIC_URL
    calls.clear()
    try:
        accounts.instagram_upload(ig, jd / "clip01.mp4", "My caption", "job-12345678", jd / "clip01-t1-v2.jpg", 12500)
    except Exception:  # noqa: BLE001 - the stand-in has no resumable upload; only the first request matters here
        pass
    made = next((c for c in calls if c[1].endswith("/17841/media")), None)
    check("Instagram from a computer it can't reach: the cover is the thumbnail's frame", (made[2].get("thumb_offset"), "cover_url" in made[2]), ("12500", False))
    config.PUBLIC_URL = old_url

    tt.update(audited=True, status=0)
    accounts.tiktok_upload(ttacc, jd / "clip01.mp4", "My caption", "public", 12500)
    check("TikTok: the cover is the thumbnail's frame (TikTok takes no picture)", tt["init"]["post_info"]["video_cover_timestamp_ms"], 12500)
    accounts.tiktok_upload(ttacc, jd / "clip01.mp4", "My caption", "public")
    check("TikTok: without thumbnails, the frame at one second as before", tt["init"]["post_info"]["video_cover_timestamp_ms"], 1000)

    yt = accounts.get(accs["youtube"]["id"])
    big = jd / "big.jpg"
    import os as _os
    Image.frombytes("RGB", (1600, 1600), _os.urandom(1600 * 1600 * 3)).save(big, "JPEG", quality=100)
    sent = []
    real_post = accounts.httpx.post

    def yt_post(url, **kw):
        if "thumbnails/set" in url:
            sent.append((kw.get("params"), len(kw.get("content") or b""), kw.get("headers", {}).get("Content-Type")))
            return R(403 if len(sent) > 1 else 200, {})
        return real_post(url, **kw)
    accounts.httpx.post = yt_post
    accounts.youtube_thumbnail(yt, "vid123", big)
    check("YouTube: the thumbnail is set on the new video, under YouTube's 2 MB limit", (sent[0][0], sent[0][1] <= 2_000_000, sent[0][2]),
          ({"videoId": "vid123"}, True, "image/jpeg"))
    try:
        accounts.youtube_thumbnail(yt, "vid123", jd / "clip01-t1-v2.jpg")
        check("YouTube: a channel not allowed custom thumbnails gets a plain reason", False)
    except RuntimeError as e:
        check("YouTube: a channel not allowed custom thumbnails gets a plain reason", "verified" in str(e), True)
    accounts.httpx.post = real_post

    # the whole post: a refused thumbnail is a note, never a failed post
    real_up, real_th = accounts.youtube_upload, accounts.youtube_thumbnail
    accounts.youtube_upload = lambda *a, **k: ("vid9", "https://www.youtube.com/shorts/vid9")
    accounts.youtube_thumbnail = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("YouTube didn't take the custom thumbnail: verify"))
    p = {"id": "post-thumb-1", "platform": "youtube", "job": "job-12345678", "clip": "clip01", "account": yt["id"], "account_name": yt["name"], "creator": None, "tries": 0,
         "title": "t", "status": "posting", "due_at": 0, "user": user.id}
    posting._posts.append(p)
    posting._post(p)
    check("YouTube: the post counts even when the thumbnail is refused, with a note", (p["status"], "thumbnail" in p.get("note", "")), ("posted", True))
    # thumbnails still being made: the post waits for them, a few times at most
    clipT["thumbs"]["status"] = "making"
    p2 = dict(p, id="post-thumb-2", status="posting", note="", thumb_waits=0)
    posting._posts.append(p2)
    posting._post(p2)
    check("a post waits for thumbnails still being made", (p2["status"], p2["thumb_waits"], p2["due_at"] > time.time() + 60), ("scheduled", 1, True))
    p2.update(status="posting", thumb_waits=5)
    posting._post(p2)
    check("...but not forever", p2["status"], "posted")
    accounts.youtube_upload, accounts.youtube_thumbnail = real_up, real_th

print("ALL PASSED" if not fails else f"{len(fails)} failed")
import shutil; db.engine.dispose(); shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
