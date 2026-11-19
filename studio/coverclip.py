"""A clip's video with its chosen thumbnail inside it, for downloads and posts.

A video file has no "cover" of its own: phones, galleries, Finder, TikTok and Instagram show its first frame.
So when a clip has a chosen thumbnail (a normal one or an AI-designed one), the file that is downloaded or posted
gets it twice:
  - as the first 0.1 s of the video, so whatever shows the first frame shows the thumbnail;
  - as embedded cover art, which Finder, QuickTime and media players show.
The rest of the clip is unchanged; the sound starts 0.1 s later so it stays in step. The file is made once per
cover and kept next to the clip, so a second download is instant."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .media import NO_WINDOW

LEAD = 0.1          # seconds the thumbnail is shown at the start


def chosen_cover(job, clip) -> Path | None:
    """The picture chosen as this clip's cover (9:16), or None when it is the plain frame."""
    folder = Path(job["dir"]) / "clips"
    a = clip.get("ai_thumbs") or {}
    item = next((i for i in a.get("items") or [] if i["id"] == a.get("chosen")), None)
    if item and item.get("fmt") == "v":
        p = folder / Path(item["file"]).name
        if p.is_file():
            return p
    t = clip.get("thumbs") or {}
    if t.get("chosen") and t.get("variants"):
        v = t["variants"][t["chosen"] - 1]
        p = folder / Path(str(v["files"].get("v", ""))).name
        if p.is_file():
            return p
    return None


def _fps(path: Path) -> str:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=30,
                             creationflags=NO_WINDOW).stdout.strip()
        n, d = out.split("/")
        fps = float(n) / float(d)
        return out if 10 <= fps <= 120 else "30"
    except Exception:  # noqa: BLE001
        return "30"


def path_for(job, clip) -> Path:
    """The file to hand out for this clip: with its cover inside when one is chosen, otherwise the clip itself."""
    folder = Path(job["dir"]) / "clips"
    plain = folder / Path(str(clip["file"])).name
    cover = chosen_cover(job, clip)
    if cover is None:
        return plain
    out = folder / f"{clip['id']}-wc-{cover.stem}-v{clip.get('version', 1)}.mp4"
    if out.is_file() and out.stat().st_mtime >= max(plain.stat().st_mtime, cover.stat().st_mtime):
        return out
    work = Path(tempfile.mkdtemp(prefix="pkc"))
    try:
        fps = _fps(plain)
        tmp = work / "out.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-loop", "1", "-framerate", fps, "-t", f"{LEAD}", "-i", str(cover),
               "-i", str(plain), "-i", str(cover),
               "-filter_complex",
               f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps={fps},format=yuv420p[c];"
               f"[1:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps={fps},format=yuv420p[v];"
               f"[c][v]concat=n=2:v=1:a=0[ov];[1:a]adelay={int(LEAD * 1000)}:all=1[oa]",
               "-map", "[ov]", "-map", "[oa]", "-map", "2:v",
               "-c:v:0", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-c:v:1", "mjpeg", "-disposition:v:1", "attached_pic",
               "-movflags", "+faststart", str(tmp)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, creationflags=NO_WINDOW)
        if p.returncode != 0 or not tmp.is_file():
            return plain                     # the plain clip is still a good download
        for old in folder.glob(f"{clip['id']}-wc-*.mp4"):
            old.unlink(missing_ok=True)
        shutil.move(str(tmp), str(out))
        return out
    except Exception:  # noqa: BLE001
        return plain
    finally:
        shutil.rmtree(work, ignore_errors=True)


def remove_files(job, cid):
    for f in (Path(job["dir"]) / "clips").glob(f"{cid}-wc-*.mp4"):
        if re.fullmatch(rf"{re.escape(cid)}-wc-[\w\-]+\.mp4", f.name):
            f.unlink(missing_ok=True)
