"""Thumbnails for finished clips: a 9:16 cover (1080x1920) for Shorts, Reels and TikTok, and a 16:9 picture
(1280x720) for an ordinary YouTube upload.

How one set is made:
  1. A handful of frames are taken from the clip's own piece of video, before captions were burned in.
  2. Each frame is scored here (sharp, well lit, colourful, a face-like share of skin tones).
  3. The best six go to the AI as one numbered picture. It picks three frames and writes two to four
     big words for each, and a colour. Without an AI, or when it cannot look at pictures, the scores
     pick the frames and the words come from the clip's own hook.
  4. Each idea is drawn here with Pillow, in both shapes. The clean frame is kept, so changing the words
     later needs neither the video nor the AI.

The work runs on its own thread, so a slow AI answer never holds up making clips."""
import io
import json
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from . import ai, brain, config, ctx, jobs, roman, safe, usage
from .media import NO_WINDOW

SIZES = {"v": (1080, 1920), "h": (1280, 720)}
VARIANTS = 3
FONT = Path(__file__).resolve().parent / "fonts" / "Anton-Regular.ttf"
# Accent colours the AI may choose from: bright enough to read on any frame next to white text.
ACCENTS = {"yellow": (255, 214, 0), "red": (255, 59, 48), "cyan": (0, 229, 255), "green": (57, 255, 120),
           "orange": (255, 140, 0), "pink": (255, 64, 160), "white": (255, 255, 255)}
ACCENT_ORDER = list(ACCENTS)
MAX_WAITING = 60          # thumbnails one person may have waiting at once
EDIT_SLOTS = threading.BoundedSemaphore(2)

_q: "queue.Queue[tuple]" = queue.Queue()
_pending: set = set()
_plock = threading.Lock()
_started = False


class Busy(Exception):
    pass


# ---------------------------------------------------------------- the clip's state
def state(clip) -> dict:
    return clip.get("thumbs") or {"status": "none", "variants": [], "chosen": None, "gen": 0}


def _snapshot(clip) -> dict:
    return {"start": clip.get("start"), "end": clip.get("end"), "version": clip.get("version")}


def waiting_for(uid) -> int:
    with _plock:
        return sum(1 for (u, _j, _c) in _pending if u == uid)


def request(job, clip, by="button") -> None:
    """Queue one clip for thumbnails. Raises Busy with a plain message when it cannot be done right now."""
    if clip.get("status") != "done" or not clip.get("file"):
        raise Busy("This clip isn't finished yet. Make thumbnails once it is ready.")
    st = state(clip)
    if st.get("status") in ("queued", "making"):
        raise Busy("Thumbnails for this clip are already being made.")
    key = (job["user"], job["id"], clip["id"])
    with _plock:
        if key in _pending:
            raise Busy("Thumbnails for this clip are already being made.")
        if sum(1 for (u, _j, _c) in _pending if u == job["user"]) >= MAX_WAITING:
            raise Busy("Lots of thumbnails are already waiting. Try again when some have finished.")
        _pending.add(key)
    with jobs._lock:
        clip["thumbs"] = {**st, "status": "queued", "error": None, "by": by}
    jobs.save(job)
    _q.put((*key, "make", None))


def request_look(job, clip, n, fmt, is_admin) -> None:
    """Queue the paid AI look for one thumbnail in one shape. Raises Busy or imagegen.NotAllowed."""
    from . import imagegen
    st = state(clip)
    if st.get("status") != "done" or not 1 <= n <= len(st.get("variants") or []):
        raise Busy("These thumbnails aren't ready yet.")
    if fmt not in SIZES:
        raise ValueError("Choose the 9:16 or the 16:9 picture.")
    if st.get("restyling"):
        raise Busy("An AI look is already being made for this clip. Wait for it to finish.")
    imagegen.check(is_admin)
    key = (job["user"], job["id"], clip["id"])
    with _plock:
        if key in _pending:
            raise Busy("Thumbnails for this clip are being made right now. Wait for them to finish.")
        _pending.add(key)
    with jobs._lock:
        st["restyling"] = {"n": n, "fmt": fmt}
        st.pop("look_error", None)
        clip["thumbs"] = st
    jobs.save(job)
    _q.put((*key, "look", (n, fmt, st.get("gen"))))


# ---------------------------------------------------------------- frames
def source_for(job, clip):
    """(video file, seconds to subtract from clip times, whether captions are burned into it)."""
    src = clip.get("src")
    if src and Path(src).is_file():
        return Path(src), float(clip.get("src_offset") or 0.0), False
    sp = job.get("source_path")
    if sp and Path(sp).is_file():
        return Path(sp), 0.0, False
    f = Path(job["dir"]) / "clips" / Path(str(clip.get("file") or "")).name
    if clip.get("file") and f.is_file():
        return f, float(clip["start"]), bool(clip.get("captions", True))
    raise RuntimeError("The video for this clip is gone, so there is nothing to take a picture from.")


def _grab(path: Path, t: float, out: Path) -> bool:
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, t):.2f}", "-i", str(path), "-frames:v", "1",
           "-vf", "scale='min(1920,iw)':-2", "-pix_fmt", "yuvj420p", "-q:v", "2", str(out)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.is_file() and out.stat().st_size > 1000


def _quiet_times(words, start, end) -> list:
    """Moments inside the clip with no word on screen: a frame there has no caption burned in."""
    spoken = sorted((float(w["s"]), float(w["e"])) for w in words or [] if start <= float(w.get("s", -1)) <= end)
    gaps, t = [], start + 0.3
    for s, e in spoken:
        if s - t > 0.8:
            gaps.append((t + s) / 2)
        t = max(t, e + 0.4)
    if end - 0.3 - t > 0.8:
        gaps.append((t + end - 0.3) / 2)
    return gaps


def times_for(clip, burned: bool) -> list:
    """Which moments of the clip to look at (absolute seconds in the video)."""
    s, e = float(clip["start"]), float(clip["end"])
    dur = max(0.5, e - s)
    if burned:
        quiet = _quiet_times(clip.get("fixed_words") or [], s, e)
        if quiet:
            return quiet[:8]
    n = int(max(3, min(8, round(dur / 2))))
    lo, hi = s + min(0.3, dur / 4), e - min(0.3, dur / 4)
    ts = [lo + (hi - lo) * (i + 0.5) / n for i in range(n)]
    punch = {w.lower() for w in clip.get("highlight_words") or []}
    for w in clip.get("fixed_words") or []:
        word = re.sub(r"[^\w]", "", str(w.get("w", ""))).lower()
        if word in punch and s <= float(w["s"]) <= e and all(abs(float(w["e"]) - t) > dur / 12 for t in ts):
            ts.append(min(e - 0.1, float(w["e"]) + 0.15))   # just after the punch word: the reaction
    return sorted(ts)[:10]


def score(img) -> float:
    """How good a frame looks as a thumbnail, without understanding it: sharp, well lit, colourful, with a
    fair share of skin tones (usually a face). Black, blurry and washed-out frames come last."""
    small = img.convert("RGB").resize((320, max(1, int(320 * img.height / img.width))))
    a = np.asarray(small, dtype=np.float32)
    grey = a.mean(axis=2)
    lap = np.abs(grey[1:-1, 1:-1] * 4 - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:])
    sharp = min(1.0, float(lap.mean()) / 18.0)
    light = float(grey.mean()) / 255.0
    exposure = 1.0 - min(1.0, abs(light - 0.5) * 2.2)
    if light < 0.08 or light > 0.95:
        return 0.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    colour = min(1.0, float(np.sqrt(np.var(r - g) + np.var(0.5 * (r + g) - b))) / 70.0)
    skin = ((r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (np.abs(r - g) > 15)).mean()
    face = min(1.0, float(skin) / 0.12)
    return round(0.35 * sharp + 0.2 * exposure + 0.2 * colour + 0.25 * face, 4)


def _differs(a, b) -> bool:
    """Two frames that are clearly not the same shot."""
    x = np.asarray(a.convert("L").resize((48, 27)), dtype=np.float32)
    y = np.asarray(b.convert("L").resize((48, 27)), dtype=np.float32)
    return float(np.abs(x - y).mean()) > 12


def contact_sheet(frames: list) -> bytes:
    """Up to six frames as one numbered picture (3 x 2), small enough for any AI that can look at pictures."""
    from PIL import Image, ImageDraw, ImageFont
    tw, th = 512, 288
    cols = 3
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    font = ImageFont.truetype(str(FONT), 44)
    for i, fr in enumerate(frames):
        im = fr["img"].convert("RGB").copy()
        im.thumbnail((tw, th))
        x, y = (i % cols) * tw + (tw - im.width) // 2, (i // cols) * th + (th - im.height) // 2
        sheet.paste(im, (x, y))
        d = ImageDraw.Draw(sheet)
        d.rectangle([(i % cols) * tw, (i // cols) * th, (i % cols) * tw + 58, (i // cols) * th + 58], fill=(0, 0, 0))
        d.text(((i % cols) * tw + 16, (i // cols) * th + 2), str(i + 1), font=font, fill=(255, 214, 0))
    out = io.BytesIO()
    sheet.save(out, "JPEG", quality=80)
    return out.getvalue()


# ---------------------------------------------------------------- words
def clean_words(text) -> str:
    """Big thumbnail words: Roman letters, capitals, no links or handles, two to four words."""
    s = roman.to_roman(brain._text(text, 60))
    s = re.sub(r"[^\w\s!?'&%.-]", "", s, flags=re.UNICODE).upper()
    words = [w for w in s.split() if w.strip(".-'")]
    s = " ".join(words[:4])
    return s if s and roman.drawable(s) else ""


def fallback_words(clip) -> list:
    """Three sets of words without an AI: the hook, the punch words, and the hook's opening."""
    hook = clean_words(clip.get("hook") or "")
    punch = clean_words(" ".join((clip.get("highlight_words") or [])[:3]))
    short = " ".join(hook.split()[:2])
    out = [w for w in (hook, punch, short) if w]
    while len(out) < VARIANTS:
        out.append(out[0] if out else "")
    return out[:VARIANTS]


# ---------------------------------------------------------------- drawing
def _cover(img, w, h, fx=0.5, fy=0.42, zoom=1.0):
    """Fill w x h from the frame, cropped around the focus point (0..1 of the frame)."""
    from PIL import Image
    scale = max(w / img.width, h / img.height) * zoom
    scale = min(scale, max(w / img.width, h / img.height) * 2.5)     # never blow a small frame up past 2.5x
    nw, nh = max(w, int(img.width * scale)), max(h, int(img.height * scale))
    big = img.resize((nw, nh), Image.LANCZOS)
    x = int(min(max(0, fx * nw - w / 2), nw - w))
    y = int(min(max(0, fy * nh - h / 2), nh - h))
    return big.crop((x, y, x + w, y + h))


def _blur_fill(img, w, h, fx=0.5, side="right"):
    """A tall or square frame in a wide picture: the frame sharp on one side, the rest a blurred copy."""
    from PIL import Image, ImageFilter
    bg = _cover(img, w, h, fx).filter(ImageFilter.GaussianBlur(28))
    fg = img.copy()
    fg.thumbnail((int(w * 0.55), h), Image.LANCZOS)
    x = w - fg.width - 40 if side == "right" else 40
    bg.paste(fg, (max(0, x), (h - fg.height) // 2))
    return bg


def _lines(words: str, font_path, size, max_w):
    from PIL import ImageFont
    f = ImageFont.truetype(str(font_path), size)
    ws = words.split()
    if not ws:
        return [], f
    if len(ws) == 1 or f.getlength(words) <= max_w:
        return [words], f
    cut = min(range(1, len(ws)), key=lambda i: max(f.getlength(" ".join(ws[:i])), f.getlength(" ".join(ws[i:]))))
    return [" ".join(ws[:cut]), " ".join(ws[cut:])], f


def _fit(words, max_w, max_h, start, least):
    """The biggest size at which the words fit the box, on one line or two balanced ones."""
    size = start
    while True:
        lines, f = _lines(words, FONT, size, max_w)
        widest = max((f.getlength(l) for l in lines), default=0)
        tall = size * 1.02 * len(lines)
        if (widest <= max_w and tall <= max_h) or size <= least:
            return lines, f, size
        size = max(least, int(size * 0.92))


def _shade(img, top, bottom):
    """A soft dark band behind the words, strongest in the middle, so they read on bright frames too."""
    from PIL import Image
    top, bottom = max(0, int(top)), min(img.height, int(bottom))
    if bottom - top < 4:
        return
    h = bottom - top
    ramp = np.abs(np.linspace(-1.0, 1.0, h, dtype=np.float32))
    alpha = ((1.0 - ramp ** 2) * 120).astype(np.uint8)
    mask = Image.fromarray(np.repeat(alpha[:, None], img.width, axis=1), "L")
    band = Image.new("RGB", (img.width, h), (0, 0, 0))
    img.paste(band, (0, top), mask)


def _draw_words(img, words, accent, box, align="center"):
    """White words with a thick black outline; the last word in the accent colour."""
    from PIL import ImageDraw
    if not words:
        return img
    x0, y0, x1, y1 = box
    lines, f, size = _fit(words, x1 - x0, y1 - y0, start=int((y1 - y0) / 1.6), least=48)
    line_h = int(size * 1.02)
    y = y0 + ((y1 - y0) - line_h * len(lines)) // 2
    _shade(img, y - size // 2, y + line_h * len(lines) + size // 2)
    d = ImageDraw.Draw(img)
    stroke = max(4, size // 11)
    last = len(lines) - 1
    for i, line in enumerate(lines):
        parts = line.split()
        widths = [f.getlength(p) for p in parts]
        space = f.getlength(" ")
        total = sum(widths) + space * (len(parts) - 1)
        x = x0 if align == "left" else x1 - total if align == "right" else x0 + ((x1 - x0) - total) / 2
        for j, p in enumerate(parts):
            colour = accent if (i == last and j == len(parts) - 1 and len(words.split()) > 1) else (255, 255, 255)
            d.text((x, y), p, font=f, fill=colour, stroke_width=stroke, stroke_fill=(0, 0, 0))
            x += widths[j] + space
        y += line_h
    return img


def _around(img, fx, fy, part_w, aspect):
    """A crop of the frame centred on the focus point, part_w of its width, in the given width/height shape."""
    cw = max(32, min(img.width, int(img.width * part_w)))
    ch = int(cw / aspect)
    if ch > img.height:
        ch = img.height
        cw = int(ch * aspect)
    x = int(min(max(0, fx * img.width - cw / 2), img.width - cw))
    y = int(min(max(0, fy * img.height - ch / 2), img.height - ch))
    return img.crop((x, y, x + cw, y + ch))


def compose(frame, words: str, accent: str, focus=(0.5, 0.42), fmt="v", facecam=None, subject=0.5):
    """One finished thumbnail from a clean frame. fmt "v" is 1080x1920, "h" is 1280x720.

    focus is where the main subject (usually a face) is, 0..1 of the frame; subject is how much of the
    frame's width it takes up. A small subject in a wide frame (a webcam in the corner of a game) gets the
    split picture: the subject big on top, the whole scene below."""
    out, box, align = layout(frame, focus, fmt, facecam, subject)
    return words_on(out, words, accent, box, align)


def words_on(img, words, accent, box, align):
    """The words drawn on a laid-out picture (the plain one, or one the AI repainted)."""
    return _draw_words(img.convert("RGB").copy(), words, ACCENTS.get(accent) or ACCENTS["yellow"], box, align)


def layout(frame, focus=(0.5, 0.42), fmt="v", facecam=None, subject=0.5):
    """The picture without words, and where the words go: (image, text box, alignment)."""
    from PIL import Image, ImageEnhance, ImageFilter
    w, h = SIZES[fmt]
    img = frame.convert("RGB")
    fx, fy = (min(1.0, max(0.0, float(v))) for v in focus)
    subject = min(1.0, max(0.05, float(subject or 0.5)))
    tall_source = img.height >= img.width
    if fmt == "v":
        split = not tall_source and (facecam or subject <= 0.45)
        if split:
            top_h = int(h * 0.5)
            if facecam:
                cx, cy, cw, ch = facecam
                face = img.crop((int(cx), int(cy), int(cx + cw), int(cy + ch)))
                top = _cover(face, w, top_h)
            else:
                top = _cover(_around(img, fx, fy, min(1.0, max(0.22, subject * 1.7)), w / top_h), w, top_h)
            scene = _cover(img, w, h - top_h, 0.5 if fx < 0.3 or fx > 0.7 else fx, 0.5)
            out = Image.new("RGB", (w, h))
            out.paste(top, (0, 0))
            out.paste(scene, (0, top_h))
            text_box = (90, top_h + 120, w - 90, min(1400, top_h + 560))
        else:
            out = _cover(img, w, h, fx, fy)
            # where the subject lands in the picture, so the words never sit on the face
            scale = max(w / img.width, h / img.height)
            sy = fy * img.height * scale - max(0, min(fy * img.height * scale - h / 2, img.height * scale - h))
            text_box = (90, 300, w - 90, max(560, min(820, int(sy) - 260))) if sy > h * 0.55 else \
                (90, max(1000, min(1400 - 240, int(sy) + 340)), w - 90, 1400)
    else:
        if tall_source:
            side = "right" if fx < 0.5 else "left"
            out = _blur_fill(img, w, h, fx, side)
            text_box = (64, 64, int(w * 0.48), h - 64) if side == "right" else (int(w * 0.52), 64, w - 64, h - 120)
        else:
            out = _cover(img, w, h, fx, fy, zoom=1.12)
            # words on the side away from the subject; never in the bottom-right corner (the video length badge)
            text_box = (int(w * 0.5), 64, w - 64, h - 140) if fx < 0.5 else (64, 64, int(w * 0.52), h - 64)
    out = ImageEnhance.Contrast(ImageEnhance.Color(out).enhance(1.18)).enhance(1.06)
    out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=3))
    align = "center" if fmt == "v" else ("left" if text_box[0] < w / 2 else "right")
    return out, text_box, align


def _save(img, path: Path):
    img.save(path, "JPEG", quality=90, optimize=True)


# ---------------------------------------------------------------- the AI
def _ideas_by_ai(job, clip, frames):
    """Ask the AI for three ideas. Returns (ideas, how) where how is "ai", or None when the AI could not help."""
    if not config.ai_ready():
        return None
    sheet = contact_sheet(frames)
    words = " ".join(str(w.get("w", "")) for w in clip.get("fixed_words") or [])
    try:
        out = brain.thumb_ideas(sheet, len(frames), title=clip.get("title") or job.get("title") or "",
                                hook=clip.get("hook") or "", transcript=words, game=job.get("game"))
    except Exception:  # noqa: BLE001 - the local scores still make thumbnails
        traceback.print_exc()
        return None
    return out or None


def _ideas(job, clip, frames):
    """Three (frame index, words, accent, focus) ideas, from the AI when it can, otherwise from the scores."""
    ranked = sorted(range(len(frames)), key=lambda i: -frames[i]["score"])
    picked = []
    for i in ranked:                                   # three different-looking frames where the clip allows it
        if all(_differs(frames[i]["img"], frames[j]["img"]) for j in picked):
            picked.append(i)
        if len(picked) == VARIANTS:
            break
    for i in ranked:
        if len(picked) == VARIANTS:
            break
        if i not in picked:
            picked.append(i)
    while len(picked) < VARIANTS:
        picked.append(ranked[0])
    local = fallback_words(clip)
    ai_out = _ideas_by_ai(job, clip, frames)
    ideas, used = [], set()
    if ai_out:
        for k, idea in enumerate(ai_out[:VARIANTS]):
            fi = idea.get("frame")
            fi = fi - 1 if isinstance(fi, int) and 1 <= fi <= len(frames) else None
            if fi is None or (fi in used and len(frames) > VARIANTS):
                fi = next((i for i in picked if i not in used), picked[0])
            used.add(fi)
            accent = idea.get("accent") if idea.get("accent") in ACCENTS else ACCENT_ORDER[k % len(ACCENT_ORDER)]
            ideas.append({"frame": fi, "words": clean_words(idea.get("words")) or local[k], "accent": accent,
                          "focus": [idea.get("focus_x", 50) / 100.0, idea.get("focus_y", 42) / 100.0],
                          "subject": idea.get("subject_w", 50) / 100.0})
    how = "ai" if ideas else "auto"
    for k in range(len(ideas), VARIANTS):
        ideas.append({"frame": picked[k], "words": local[k], "accent": ("yellow", "cyan", "red")[k % 3], "focus": [0.5, 0.42],
                      "subject": 0.6})
    return ideas, how


# ---------------------------------------------------------------- making a set
def _files(cid, gen, n):
    return {"b": f"{cid}-t{gen}-b{n}.jpg", "v": f"{cid}-t{gen}-v{n}.jpg", "h": f"{cid}-t{gen}-h{n}.jpg"}


def _remove(folder: Path, cid: str, keep_gen=None):
    for f in folder.glob(f"{cid}-t*.jpg"):
        m = re.fullmatch(rf"{re.escape(cid)}-t(\d+)-[a-z]{{1,2}}\d+(?:x\d+)?\.jpg", f.name)
        if m and (keep_gen is None or int(m.group(1)) != keep_gen):
            f.unlink(missing_ok=True)


def make(jid: str, cid: str):
    """Make one clip's thumbnails. Runs on the thumbnail thread, as the clip's owner."""
    from PIL import Image
    job = jobs.get(jid)
    clip = next((c for c in (job or {}).get("clips", []) if c["id"] == cid), None)
    if not clip:
        return
    st = state(clip)
    snap = _snapshot(clip)
    gen = int(st.get("gen") or 0) + 1
    with jobs._lock:
        clip["thumbs"] = {**st, "status": "making", "error": None}
    jobs.save(job)
    work = Path(tempfile.mkdtemp(prefix="pkt"))
    try:
        path, off, burned = source_for(job, clip)
        frames = []
        for k, t in enumerate(times_for(clip, burned)):
            out = work / f"f{k}.jpg"
            if _grab(path, t - off, out):
                img = Image.open(out)
                img.load()
                frames.append({"t": round(t, 2), "img": img, "score": score(img)})
        if not frames:
            raise RuntimeError("Could not take any pictures from this clip's video.")
        frames = sorted(sorted(frames, key=lambda f: -f["score"])[:6], key=lambda f: f["t"])
        ideas, how = _ideas(job, clip, frames)
        facecam = job.get("facecam") if clip.get("layout") == "gameplay" else None
        made = []
        for n, idea in enumerate(ideas, 1):
            fr = frames[idea["frame"]]
            names = _files(cid, gen, n)
            _save(fr["img"].convert("RGB"), work / names["b"])
            for fmt in ("v", "h"):
                cam = facecam if (fmt == "v" and n == 2 and facecam and len(facecam) == 4 and facecam[2] >= 120) else None
                _save(compose(fr["img"], idea["words"], idea["accent"], idea["focus"], fmt, cam, idea["subject"]), work / names[fmt])
            made.append({"t": fr["t"], "words": idea["words"], "accent": idea["accent"], "focus": idea["focus"],
                         "subject": idea["subject"],
                         "kind": "react" if (facecam and n == 2 and facecam[2] >= 120) else "frame",
                         "base": names["b"], "files": {"v": names["v"], "h": names["h"]}})
        low = max(fr["img"].height for fr in frames) < 600
        _finish(jid, cid, snap, gen, made, how, work, note="Low source quality: thumbnails may look soft." if low else "")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job = jobs.get(jid)
        clip = next((c for c in (job or {}).get("clips", []) if c["id"] == cid), None)
        if clip:
            with jobs._lock:
                clip["thumbs"] = {**state(clip), "status": "error", "error": safe.friendly(e, 300)}
            jobs.save(job)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _finish(jid, cid, snap, gen, made, how, work: Path, note=""):
    """Move the new set into place, unless the clip was changed or deleted while it was being made."""
    job = jobs.get(jid)
    clip = next((c for c in (job or {}).get("clips", []) if c["id"] == cid), None)
    if not clip:
        return                                        # the clip or video was deleted meanwhile: nothing to keep
    folder = Path(job["dir"]) / "clips"
    with jobs._lock:
        if _snapshot(clip) != snap:
            clip["thumbs"] = {**state(clip), "status": "error",
                              "error": "The clip changed while its thumbnails were being made. Press Make thumbnails again."}
            changed = True
        else:
            changed = False
            for v in made:
                for name in (v["base"], *v["files"].values()):
                    shutil.move(str(work / name), str(folder / name))
            old = state(clip)
            if old.get("chosen") is not None and clip.get("frame_cover"):
                clip["cover"] = clip["frame_cover"]       # the old choice is gone with the old set
            clip["thumbs"] = {"status": "done", "error": None, "gen": gen, "by": how, "note": note,
                              "made_for": snap, "stale": False, "chosen": None, "variants": made, "made_at": time.time()}
    jobs.save(job)
    if not changed:
        _remove(folder, cid, keep_gen=gen)


def choose(job, clip, n):
    """Use variant n (1..3) as the clip's cover, or go back to the plain frame with None."""
    st = state(clip)
    if st.get("status") != "done":
        raise Busy("These thumbnails aren't ready yet.")
    with jobs._lock:
        if n is None:
            if clip.get("frame_cover"):
                clip["cover"] = clip.pop("frame_cover")
            st["chosen"] = None
        else:
            if not 1 <= n <= len(st["variants"]):
                raise ValueError("No such thumbnail.")
            if not clip.get("frame_cover"):
                clip["frame_cover"] = clip.get("cover")
            clip["cover"] = st["variants"][n - 1]["files"]["v"]
            st["chosen"] = n
        clip["thumbs"] = st
    jobs.save(job)


def edit(job, clip, n, words=None, accent=None):
    """New words or colour on one variant, drawn again from its clean frame."""
    from PIL import Image
    st = state(clip)
    if st.get("status") != "done" or not 1 <= n <= len(st.get("variants") or []):
        raise Busy("These thumbnails aren't ready yet.")
    if not EDIT_SLOTS.acquire(blocking=False):
        raise Busy("Busy drawing other thumbnails. Try again in a moment.")
    try:
        v = dict(st["variants"][n - 1])
        if words is not None:
            v["words"] = clean_words(words)
        if accent is not None:
            if accent not in ACCENTS:
                raise ValueError("That colour isn't one of the choices.")
            v["accent"] = accent
        folder = Path(job["dir"]) / "clips"
        base = Image.open(folder / v["base"])
        base.load()
        stamp = int(time.time() * 1000) % 100000
        facecam = job.get("facecam") if v.get("kind") == "react" else None
        files = {}
        for fmt in ("v", "h"):
            name = f"{clip['id']}-t{st['gen']}-{fmt}{n}x{stamp}.jpg"
            _save(_draw(folder, base, v, fmt, facecam), folder / name)
            files[fmt] = name
        old = v["files"]
        v["files"] = files
        with jobs._lock:
            st["variants"][n - 1] = v
            if st.get("chosen") == n:
                clip["cover"] = files["v"]
            clip["thumbs"] = st
        jobs.save(job)
        for name in old.values():
            (folder / name).unlink(missing_ok=True)
        return v
    finally:
        EDIT_SLOTS.release()


def _draw(folder, base, v, fmt, facecam):
    """One variant in one shape: on its AI look when it has one, otherwise on the plain picture."""
    from PIL import Image
    out, box, align = layout(base, v["focus"], fmt, facecam if fmt == "v" else None, v.get("subject", 0.5))
    look = (v.get("looks") or {}).get(fmt)
    if look and (folder / look).is_file():
        out = Image.open(folder / look)
        out.load()
    return words_on(out, v["words"], v["accent"], box, align)


API_SIZES = {"v": ("1024x1536", 1024, 1536), "h": ("1536x1024", 1536, 1024)}


def _to_api_shape(img, fmt):
    """The laid-out picture placed on the model's canvas (2:3 or 3:2), the spare edges a blurred copy of it."""
    from PIL import Image, ImageFilter
    _size, aw, ah = API_SIZES[fmt]
    bg = _cover(img, aw, ah).filter(ImageFilter.GaussianBlur(30))
    fg = img.copy()
    fg.thumbnail((aw, ah), Image.LANCZOS)
    bg.paste(fg, ((aw - fg.width) // 2, (ah - fg.height) // 2))
    return bg, ((aw - fg.width) // 2, (ah - fg.height) // 2, fg.width, fg.height)


def make_look(jid, cid, n, fmt, gen):
    """The paid AI look for one variant in one shape. Runs on the thumbnail thread, as the clip's owner."""
    from PIL import Image
    from . import imagegen, safe
    job = jobs.get(jid)
    clip = next((c for c in (job or {}).get("clips", []) if c["id"] == cid), None)
    if not clip:
        return
    st = state(clip)
    try:
        if st.get("gen") != gen or not 1 <= n <= len(st.get("variants") or []):
            raise RuntimeError("These thumbnails were replaced meanwhile. Try again on the new ones.")
        folder = Path(job["dir"]) / "clips"
        v = st["variants"][n - 1]
        base = Image.open(folder / v["base"])
        base.load()
        facecam = job.get("facecam") if v.get("kind") == "react" else None
        plain, _box, _align = layout(base, v["focus"], fmt, facecam if fmt == "v" else None, v.get("subject", 0.5))
        canvas, (px, py, pw, ph) = _to_api_shape(plain, fmt)
        got = imagegen.repaint(imagegen.to_png(canvas), API_SIZES[fmt][0])
        imagegen.record(imagegen.price(), API_SIZES[fmt][0], creator=job.get("creator"), place=clip.get("title") or job.get("title"))
        art = Image.open(io.BytesIO(got)).convert("RGB")
        sx, sy = art.width / canvas.width, art.height / canvas.height       # the service may answer at another size
        art = art.crop((int(px * sx), int(py * sy), int((px + pw) * sx), int((py + ph) * sy))).resize(SIZES[fmt], Image.LANCZOS)
        stamp = int(time.time() * 1000) % 100000
        look = f"{cid}-t{gen}-a{fmt}{n}x{stamp}.jpg"
        _save(art, folder / look)
        with jobs._lock:
            st = state(clip)
            if st.get("gen") != gen or clip not in job["clips"]:
                (folder / look).unlink(missing_ok=True)
                return
            v = dict(st["variants"][n - 1])
            old_look = (v.get("looks") or {}).get(fmt)
            v["looks"] = {**(v.get("looks") or {}), fmt: look}
            name = f"{cid}-t{gen}-{fmt}{n}x{stamp}.jpg"
            _save(_draw(folder, base, v, fmt, facecam), folder / name)
            old_file = v["files"][fmt]
            v["files"] = {**v["files"], fmt: name}
            st["variants"][n - 1] = v
            st.pop("restyling", None)
            if st.get("chosen") == n and fmt == "v":
                clip["cover"] = name
            clip["thumbs"] = st
        jobs.save(job)
        for f in (old_file, old_look):
            if f:
                (folder / f).unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        with jobs._lock:
            st = state(clip)
            st["restyling"] = None
            st["look_error"] = safe.friendly(e, 300)
            clip["thumbs"] = st
        jobs.save(job)


def drop_look(job, clip, n, fmt):
    """Back to the plain picture for one variant in one shape."""
    from PIL import Image
    st = state(clip)
    if st.get("status") != "done" or not 1 <= n <= len(st.get("variants") or []) or fmt not in SIZES:
        raise Busy("These thumbnails aren't ready yet.")
    folder = Path(job["dir"]) / "clips"
    v = dict(st["variants"][n - 1])
    look = (v.get("looks") or {}).pop(fmt, None)
    if not look:
        return
    base = Image.open(folder / v["base"])
    base.load()
    stamp = int(time.time() * 1000) % 100000
    name = f"{clip['id']}-t{st['gen']}-{fmt}{n}x{stamp}.jpg"
    facecam = job.get("facecam") if v.get("kind") == "react" else None
    _save(_draw(folder, base, v, fmt, facecam), folder / name)
    old = v["files"][fmt]
    with jobs._lock:
        v["files"] = {**v["files"], fmt: name}
        st["variants"][n - 1] = v
        if st.get("chosen") == n and fmt == "v":
            clip["cover"] = name
        clip["thumbs"] = st
    jobs.save(job)
    for f in (old, look):
        (folder / f).unlink(missing_ok=True)


def remove_files(job, cid):
    """Every thumbnail file of one clip (when the clip is deleted)."""
    _remove(Path(job["dir"]) / "clips", cid)


def clip_changed(clip):
    """Called after a clip is made again: keep a chosen cover, and say when the thumbnails no longer match."""
    st = clip.get("thumbs")
    if not st or st.get("status") != "done":
        return
    if st.get("chosen"):
        v = st["variants"][st["chosen"] - 1]
        clip["frame_cover"] = clip.get("cover")
        clip["cover"] = v["files"]["v"]
    times = [v["t"] for v in st.get("variants") or []]
    st["stale"] = any(not (clip["start"] <= t <= clip["end"]) for t in times)


def interrupted(clip):
    st = clip.get("thumbs")
    if st and st.get("status") in ("queued", "making"):
        st.update(status="error", error="Interrupted by a restart. Press Make thumbnails again.")
    if st and st.get("restyling"):
        st.update(restyling=None, look_error="Interrupted by a restart. Press AI look again.")


def on_clip_ready(job, clip):
    """A new clip just finished: make its thumbnails too when the video (or its creator) asks for that."""
    from . import creators, load
    wanted = (job.get("options") or {}).get("thumbs")
    if not wanted and job.get("creator"):
        c = creators.get(job["creator"])
        wanted = bool(c and c.get("auto_thumbs"))
    if not wanted:
        return
    if load.disk_full():
        creators.log(job.get("creator"), "Thumbnails skipped: the disk is full.")
        return
    try:
        request(job, clip, by="auto")
    except Busy as e:
        creators.log(job.get("creator"), f"Thumbnails skipped: {e}")


# ---------------------------------------------------------------- the thread
def _worker():
    while True:
        uid, jid, cid, action, args = _q.get()
        key = (uid, jid, cid)
        try:
            with ctx.as_user(uid):
                usage.set_context(None, None)
                job = jobs.get(jid)
                if job:
                    usage.set_context(job.get("creator"), job.get("title"))
                if action == "look":
                    make_look(jid, cid, *args)
                else:
                    make(jid, cid)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        finally:
            with _plock:
                _pending.discard(key)


def start():
    global _started
    if _started:
        return
    _started = True
    jobs.on_clip_ready.append(on_clip_ready)
    threading.Thread(target=_worker, daemon=True).start()
