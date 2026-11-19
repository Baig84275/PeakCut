"""Pasting a channel link: which links count as a channel, what the list shows, and which videos can't be picked.

Runs against the code directly, with its own empty data folder and no network: yt-dlp and picture fetching are replaced."""
import os
import sys
import tempfile
from unittest.mock import patch

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-channels-")
os.environ.update(DATABASE_URL="", PEAKCUT_ACCOUNTS="0", PEAKCUT_PUBLIC_URL="http://127.0.0.1:8765")
sys.path.insert(0, ".")
from fastapi.testclient import TestClient  # noqa: E402

from studio import channels, db, media, server  # noqa: E402

db.init()
fails = []
BASE = "http://127.0.0.1:8765"
client = TestClient(server.app, base_url=BASE, follow_redirects=False)


def check(name, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"   got {got!r}"))
    ok or fails.append(name)


# ---- which links are channels
for url, want in [("https://www.youtube.com/@CarryisLive", ("youtube", "https://www.youtube.com/@CarryisLive")),
                  ("youtube.com/@CarryisLive/streams?si=abc", ("youtube", "https://www.youtube.com/@CarryisLive")),
                  ("https://m.youtube.com/channel/UCabcdefghijklmnopqrstuv/videos", ("youtube", "https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv")),
                  ("https://www.youtube.com/c/SomeName/live", ("youtube", "https://www.youtube.com/c/SomeName")),
                  ("https://www.twitch.tv/Shroud/videos", ("twitch", "https://www.twitch.tv/shroud")),
                  ("https://kick.com/xqc", ("kick", "https://kick.com/xqc"))]:
    check(f"a channel link is recognised: {url}", channels.parse(url)[:2], want)
for url in ["https://www.youtube.com/watch?v=irNhWMG560k", "https://www.youtube.com/playlist?list=PL123",
            "https://www.twitch.tv/videos/123456789", "https://www.twitch.tv/directory", "https://example.com/@someone"]:
    try:
        channels.parse(url)
        check(f"not a channel: {url}", False)
    except channels.NotAChannel:
        check(f"not a channel: {url}", True)
try:
    channels.list_videos("https://kick.com/xqc")
    check("Kick channels say plainly they can't be listed yet", False)
except channels.NotAChannel as e:
    check("Kick channels say plainly they can't be listed yet", "Kick" in str(e), True)


# ---- what one page of the list says about each video
class FakeYDL:
    calls = []
    entries = [
        {"id": "AAAAAAAAAAA", "title": "Live now", "live_status": "is_live"},
        {"id": "BBBBBBBBBBB", "title": "Soon", "live_status": "is_upcoming"},
        {"id": "CCCCCCCCCCC", "title": "Old stream <b>", "live_status": "was_live", "duration": 7200, "timestamp": 1700000000,
         "view_count": 1500, "thumbnails": [{"url": "https://i.ytimg.com/vi/CCCCCCCCCCC/hq720.jpg"}]},
        {"id": "DDDDDDDDDDD", "title": "Members", "duration": 600, "availability": "subscriber_only"},
        {"id": "EEEEEEEEEEE", "title": "Teaser", "duration": 30},
        {"id": "bad id!", "title": "Broken"},
    ]

    def __init__(self, opts):
        FakeYDL.calls.append(opts)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        FakeYDL.last_url = url
        return {"uploader": "Some Channel", "entries": list(self.entries)}


with patch.object(media, "_ydl", side_effect=lambda opts: FakeYDL(opts)):
    out = channels.list_videos("https://www.youtube.com/@Someone", "streams", 1)
    by = {i["id"]: i for i in out["items"]}
    check("the list is read from the channel's streams page", FakeYDL.last_url, "https://www.youtube.com/@Someone/streams")
    check("only titles are read, nothing downloaded", (FakeYDL.calls[-1]["extract_flat"], FakeYDL.calls[-1]["noplaylist"]), (True, False))
    check("a broken entry is left out", "bad id!" in by, False)
    check("a stream that's live now can't be picked here", ("live", bool(by["AAAAAAAAAAA"]["reason"])), (by["AAAAAAAAAAA"]["state"], True))
    check("an upcoming stream can't be picked", bool(by["BBBBBBBBBBB"]["reason"]), True)
    check("a members-only video can't be picked", "members" in by["DDDDDDDDDDD"]["reason"], True)
    check("a teaser under a minute can't be picked", "short" in by["EEEEEEEEEEE"]["reason"], True)
    old = by["CCCCCCCCCCC"]
    check("a past stream can be picked", (old["state"], old["reason"]), ("was_live", ""))
    check("its link is a plain watch link", old["url"], "https://www.youtube.com/watch?v=CCCCCCCCCCC")
    check("its title can't break the page's prompts", "<" in old["title"], False)
    check("its picture is fetched through PeakCut", old["thumb"], "/api/channel-thumb/youtube/CCCCCCCCCCC")
    check("length, date and views are passed on", (old["duration"], old["date"], old["views"]), (7200.0, 1700000000, 1500))
    n = len(FakeYDL.calls)
    channels.list_videos("https://www.youtube.com/@Someone/videos?si=x", "streams", 1)
    check("the same list asked again soon is not fetched again", len(FakeYDL.calls), n)
    channels.list_videos("https://www.youtube.com/@Someone", "videos", 2)
    check("page 2 of Videos asks for the next 24", (FakeYDL.calls[-1]["playliststart"], FakeYDL.calls[-1]["playlistend"], FakeYDL.last_url),
          (25, 48, "https://www.youtube.com/@Someone/videos"))
    out = channels.list_videos("https://www.twitch.tv/someone", "streams", 1)
    check("Twitch past broadcasts come from its archive list", FakeYDL.last_url, "https://www.twitch.tv/someone/videos?filter=archives&sort=time")

# a channel with no streams page
with patch.object(media, "_ydl", side_effect=Exception("This channel does not have a streams tab")):
    try:
        channels.list_videos("https://www.youtube.com/@NoStreams", "streams", 1)
        check("a channel without past streams says so", False)
    except channels.NotAChannel as e:
        check("a channel without past streams says so", "Videos" in str(e), True)
# YouTube refusing the server
with patch.object(media, "_ydl", side_effect=Exception("Sign in to confirm you're not a bot")):
    r = client.get("/api/channel-videos", params={"url": "https://www.youtube.com/@Blocked2", "tab": "streams"})
    check("YouTube refusing the server is a plain 503", r.status_code, 503)
media._yt_refused_until = 0

# ---- the routes
check("a link to another site is refused", client.get("/api/channel-videos", params={"url": "https://evil.example/@x"}).status_code, 400)
check("a video link is not a channel", client.get("/api/channel-videos", params={"url": "https://www.youtube.com/watch?v=irNhWMG560k"}).status_code, 422)

# ---- the picture proxy only ever asks the sites' own picture servers
check("a YouTube picture address is built, never taken from outside", channels.thumb_address("youtube", "CCCCCCCCCCC"), "https://i.ytimg.com/vi/CCCCCCCCCCC/mqdefault.jpg")
check("a malformed id gets nothing", channels.thumb_address("youtube", "../../etc"), None)
channels._thumbs[("twitch", "123")] = "https://evil.example/x.jpg"
check("a Twitch picture on a foreign server is refused", channels.thumb_address("twitch", "123"), None)
channels._thumbs[("twitch", "124")] = "https://static-cdn.jtvnw.net/cf_vods/x/thumb.jpg"
check("a Twitch picture on Twitch's own server is allowed", channels.thumb_address("twitch", "124"), "https://static-cdn.jtvnw.net/cf_vods/x/thumb.jpg")
check("the picture route refuses other sites", client.get("/api/channel-thumb/evil/abc").status_code, 404)


class Pic:
    def __init__(self, kind, size):
        self.status_code, self.headers, self.content = 200, {"content-type": kind}, b"x" * size


import httpx  # noqa: E402
with patch.object(httpx, "get", return_value=Pic("text/html", 100)):
    check("something that isn't a picture is not passed on", client.get("/api/channel-thumb/youtube/ZZZZZZZZZZZ").status_code, 404)
with patch.object(httpx, "get", return_value=Pic("image/jpeg", 400_000)):
    check("a picture that's too big is not passed on", client.get("/api/channel-thumb/youtube/YYYYYYYYYYY").status_code, 404)
with patch.object(httpx, "get", return_value=Pic("image/jpeg", 5000)):
    r = client.get("/api/channel-thumb/youtube/XXXXXXXXXXX")
    check("a normal picture is passed on", (r.status_code, r.headers["content-type"]), (200, "image/jpeg"))

print(f"{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
