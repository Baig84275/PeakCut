"""A channel's past live streams and videos, so a whole channel link can be pasted and the videos to clip ticked.

Only titles, lengths, dates and picture addresses are read (yt-dlp's quick list mode): nothing is downloaded.
What it finds is public, so one short-lived list is shared by everyone who asks for the same channel."""
import re
import threading
import time
from urllib.parse import urlparse

from . import brain, config, media

PAGE = 24
MAX_PAGES = 5
CACHE_SEC = 600
_cache: dict = {}
_lock = threading.Lock()
_thumbs: dict = {}          # (platform, id) -> picture address from the list, for the picture proxy

TABS = ("streams", "videos")
_YT = re.compile(r"^https?://(?:www\.|m\.)?youtube\.com/(@[\w.\-]{1,100}|channel/UC[\w\-]{10,40}|c/[\w.\-]{1,100}|user/[\w.\-]{1,100})"
                 r"(?:/(?:live|videos|streams|featured|shorts|playlists|community|about))?/?(?:[?#].*)?$", re.I)
_TW = re.compile(r"^https?://(?:www\.|m\.)?twitch\.tv/([A-Za-z0-9_]{2,25})(?:/(?:videos|about|schedule))?/?(?:[?#].*)?$", re.I)
_KICK = re.compile(r"^https?://(?:www\.)?kick\.com/([\w\-]{2,40})/?(?:[?#].*)?$", re.I)
_TW_RESERVED = {"videos", "directory", "settings", "p", "downloads", "jobs", "search", "subscriptions", "inventory"}


class NotAChannel(ValueError):
    pass


def parse(url: str):
    """(platform, the channel's base address, its name) for a channel link. Raises NotAChannel otherwise."""
    url = (url or "").strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    if "list=" in url and "youtube" in url.lower():
        raise NotAChannel("That's a playlist. Paste the channel's link instead, or one video at a time.")
    m = _YT.match(url)
    if m:
        return "youtube", f"https://www.youtube.com/{m.group(1)}", m.group(1).split("/")[-1].lstrip("@")
    m = _TW.match(url)
    if m and m.group(1).lower() not in _TW_RESERVED:
        return "twitch", f"https://www.twitch.tv/{m.group(1).lower()}", m.group(1)
    m = _KICK.match(url)
    if m:
        return "kick", f"https://kick.com/{m.group(1).lower()}", m.group(1)
    raise NotAChannel("That isn't a channel link. Paste a link like youtube.com/@name or twitch.tv/name.")


def is_channel(url: str) -> bool:
    try:
        parse(url)
        return True
    except NotAChannel:
        return False


def _list_url(platform, base, tab):
    if platform == "youtube":
        return f"{base}/{'streams' if tab == 'streams' else 'videos'}"
    if platform == "twitch":
        return f"{base}/videos?filter={'archives' if tab == 'streams' else 'all'}&sort=time"
    raise NotAChannel("Listing a Kick channel's videos isn't supported yet. Paste the link of one of their videos instead.")


def _state(e) -> str:
    s = e.get("live_status")
    if s in ("is_live", "is_upcoming", "was_live", "post_live"):
        return {"is_live": "live", "is_upcoming": "upcoming", "was_live": "was_live", "post_live": "processing"}[s]
    return "video"


def _item(platform, e, tab) -> dict | None:
    vid = str(e.get("id") or "")
    if platform == "youtube":
        if not re.fullmatch(r"[\w\-]{11}", vid):
            return None
        url = f"https://www.youtube.com/watch?v={vid}"
    else:
        num = vid.lstrip("v")
        if not num.isdigit():
            return None
        vid, url = num, f"https://www.twitch.tv/videos/{num}"
    dur = float(e.get("duration") or 0)
    state = _state(e)
    if platform == "twitch" and tab == "streams":
        state = "was_live"
    avail = str(e.get("availability") or "")
    pic = ""
    for t in reversed(e.get("thumbnails") or []):
        if t.get("url"):
            pic = str(t["url"])
            break
    if pic:
        _thumbs[(platform, vid)] = pic
    reason = ""
    if state == "live":
        reason = "Live right now. Record it from a creator instead."
    elif state == "upcoming":
        reason = "Hasn't started yet."
    elif state == "processing":
        reason = "Still being processed by the site. Try again later."
    elif avail in ("subscriber_only", "premium_only", "needs_auth"):
        reason = "Only for members, so it can't be fetched."
    elif avail == "private":
        reason = "Private."
    elif dur and dur > config.MAX_VIDEO_HOURS * 3600:
        reason = f"Longer than {config.MAX_VIDEO_HOURS:g} hours. Paste its link and choose a part under Options."
    elif dur and dur < 60:
        reason = "Under a minute: too short to clip."
    return {"id": vid, "url": url, "title": brain._safe(e.get("title") or "Untitled", 200),
            "duration": dur, "date": int(e.get("timestamp") or 0) or None, "views": int(e.get("view_count") or 0) or None,
            "state": state, "reason": reason, "thumb": f"/api/channel-thumb/{platform}/{vid}"}


def list_videos(url: str, tab: str = "streams", page: int = 1) -> dict:
    """One page of a channel's past streams (tab "streams") or videos (tab "videos")."""
    platform, base, name = parse(url)
    tab = tab if tab in TABS else "streams"
    page = max(1, min(MAX_PAGES, int(page or 1)))
    key = (base.lower(), tab, page)
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_SEC:
            return hit[1]
    src = _list_url(platform, base, tab)
    opts = {"noplaylist": False, "extract_flat": True, "playliststart": (page - 1) * PAGE + 1,
            "playlistend": page * PAGE, "extractor_args": {"youtubetab": {"approximate_date": [""]}}}
    try:
        with media._youtube_guard(src), media._ydl(opts) as ydl:
            info = ydl.extract_info(src, download=False)
    except media.YouTubeBlocked:
        raise
    except Exception as e:  # noqa: BLE001
        text = str(e)
        if tab == "streams" and re.search(r"does not have a .?streams|This channel does not have|404", text, re.I):
            raise NotAChannel("This channel has no past live streams. Look under Videos instead.") from None
        raise NotAChannel("Couldn't open that channel. Check the link is right and the channel is public.") from None
    entries = [x for x in (info or {}).get("entries") or [] if isinstance(x, dict)]
    items = [it for it in (_item(platform, x, tab) for x in entries) if it]
    out = {"platform": platform, "channel": brain._safe(info.get("uploader") or info.get("channel") or name, 100),
           "base": base, "tab": tab, "page": page, "items": items,
           "more": len(entries) >= PAGE and page < MAX_PAGES}
    with _lock:
        if len(_cache) > 200:
            for k in sorted(_cache, key=lambda k: _cache[k][0])[:50]:
                _cache.pop(k, None)
        _cache[key] = (time.time(), out)
    return out


# ---------------------------------------------------------------- pictures for the list
_pic_cache: dict = {}
PIC_HOSTS = ("i.ytimg.com", "static-cdn.jtvnw.net", "vod-secure.twitch.tv", "d2nvs31859zcd8.cloudfront.net")


def thumb_address(platform: str, vid: str) -> str | None:
    if platform == "youtube" and re.fullmatch(r"[\w\-]{11}", vid):
        return f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg"
    if platform == "twitch" and vid.isdigit():
        url = _thumbs.get(("twitch", vid))
        host = (urlparse(url).hostname or "").lower() if url else ""
        if url and url.startswith("https://") and any(host == h or host.endswith("." + h) for h in PIC_HOSTS):
            return url
    return None


def thumb(platform: str, vid: str) -> tuple[bytes, str] | None:
    """The small picture of one listed video, fetched by the server (the page may only show its own pictures)."""
    import httpx
    url = thumb_address(platform, vid)
    if not url:
        return None
    hit = _pic_cache.get(url)
    if hit and time.time() - hit[0] < 86400:
        return hit[1], hit[2]
    try:
        r = httpx.get(url, timeout=10, follow_redirects=False)
    except httpx.HTTPError:
        return None
    kind = r.headers.get("content-type", "").split(";")[0].strip()
    if r.status_code != 200 or kind not in ("image/jpeg", "image/png", "image/webp") or len(r.content) > 300_000:
        return None
    if len(_pic_cache) > 500:
        _pic_cache.clear()
    _pic_cache[url] = (time.time(), r.content, kind)
    return r.content, kind
