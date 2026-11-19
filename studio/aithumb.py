"""AI-designed thumbnails: an image model (gpt-image-2) draws the whole thumbnail, in the style of popular gaming
YouTube thumbnails: the streamer's face big with a glow outline, a neon background, a huge headline on brush-stroke
banners, the game's name, a channel badge and an arrow at the action.

How one is made:
  1. Frames are taken from the clip (the same way as for normal thumbnails) and the text AI looks at them and
     writes a plan: which moment, the two-line headline, what the arrow points at, where the streamer's face is.
  2. The streamer's face and the game moment are cut from that frame and sent to the image model with a written
     design that uses the creator's brand: the name on the badge, the game, the colours.
  3. The picture comes back 2:3 (or 3:2) and is cropped to exactly 9:16 (or 16:9). The design keeps the edges
     that are cut off free of anything important.

Each one costs money on the person's OpenAI account (about $0.17 at high quality), so it is only made when asked
for, counted, and stopped by the daily limit in Settings, AI."""
import io
import re
import shutil
import tempfile
import time
import traceback
from pathlib import Path

from . import brain, config, games, imagegen, jobs, safe, thumbdraw

STYLES = {
    "vice": ("a neon purple and hot-pink Vice City sunset with palm tree silhouettes and a glowing city skyline",
             "magenta"),
    "fire": ("a dramatic orange and red sky with sparks, smoke and a burning glow", "orange"),
    "ice": ("an electric blue and cyan neon city at night with light streaks", "cyan"),
    "gold": ("a luxurious black and gold background with light rays and floating sparkles", "gold"),
}
SIZE = {"v": ("1024x1536", (1080, 1920)), "h": ("1536x1024", (1280, 720))}
MAX_ITEMS = 6


class Busy(Exception):
    pass


def state(clip) -> dict:
    return clip.get("ai_thumbs") or {"status": "none", "items": []}


def style_for(brand_style: str, game: str | None) -> str:
    if brand_style in STYLES:
        return brand_style
    return "vice" if game and "grand theft auto" in game.lower() else "fire"


def game_label(game: str | None) -> str:
    """The game's name as it goes on the thumbnail: short and loud."""
    if not game:
        return ""
    short = {"Grand Theft Auto VI": "GTA 6", "Grand Theft Auto V": "GTA 5", "Counter-Strike 2": "CS2",
             "Call of Duty: Warzone": "WARZONE", "Red Dead Redemption 2": "RDR 2"}
    name = short.get(games.name_of(game) or game, games.name_of(game) or game)
    return re.sub(r"[^\w\s:.'&-]", "", name).upper()[:18].strip()


def brand_for(job) -> dict:
    """Name on the badge, game and colours: from the creator's brand settings, else from the stream itself."""
    from . import creators
    c = creators.get(job["creator"]) if job.get("creator") else None
    name = (c or {}).get("brand_name") or job.get("uploader") or ""
    game = (c or {}).get("game") or job.get("game")
    return {"name": re.sub(r"[^\w\s.'&-]", "", str(name)).strip().upper()[:24],
            "game": game_label(game), "style": style_for((c or {}).get("brand_style") or "auto", job.get("game"))}


def prompt(brief: dict, brand: dict, fmt: str, has_face: bool) -> str:
    """The written design for the image model. Every piece of text is spelled out exactly."""
    bg, glow = STYLES[brand["style"]]
    tall = fmt == "v"
    keep = ("Keep the left and right 9% of the picture free of text, faces and badges (they will be cropped)."
            if tall else "Keep the top and bottom 8% of the picture free of text, faces and badges (they will be cropped).")
    parts = [f"Create a {'vertical YouTube Shorts' if tall else 'wide YouTube'} thumbnail in the style of popular South "
             "Asian gaming YouTube thumbnails: bold, saturated, high contrast, sharp."]
    if has_face:
        parts.append(f"Image 1 is the streamer. Keep his or her exact face, identity, skin tone, hair, beard, glasses, "
                     f"headphones and clothes. Show the streamer LARGE {'in the top half' if tall else 'on the right side'} "
                     f"(head and shoulders), cut out from the room, with a bright white glowing outline. Give a "
                     f"{brief['expression']} expression, but the person must clearly stay the same person.")
        parts.append(f"Image 2 is the game moment: {brief['scene'] or 'the scene in the picture'}. Show it "
                     f"{'in the bottom half' if tall else 'on the left side'}, redrawn clean, vivid and cinematic.")
    else:
        parts.append(f"The image is the game moment: {brief['scene'] or 'the scene in the picture'}. Make it the "
                     "whole picture, redrawn clean, vivid and cinematic.")
    parts.append("Remove every game interface element: minimap, numbers, health bars, chat, viewer counts, webcam frame.")
    parts.append(f"Background behind everything: {bg}. A glowing {glow} border runs around the whole image.")
    text = ["Text (spell it exactly, add no other words):",
            f'- The headline, huge, across the {"middle" if tall else "left half"} on two brush-stroke banners: '
            f'line 1 "{brief["line1"]}" in bold italic white letters with a thick black outline on a red brush stroke; '
            f'line 2 "{brief["line2"]}" in bold italic black letters on a bright yellow brush stroke.']
    if brand["game"]:
        text.append(f'- Top corner: "{brand["game"]}" in big bold white letters with a pink-to-orange gradient glow '
                    "(an original design, not any official logo).")
    if brand["name"]:
        text.append(f'- A red banner with a white play-button icon and the name "{brand["name"]}" in bold italic white.')
    parts.append("\n".join(text))
    if brief.get("arrow_to"):
        parts.append(f"A thick curved red arrow pointing at {brief['arrow_to']}.")
    if brief.get("icon"):
        parts.append(f"A glowing round badge with {brief['icon']}, near the headline.")
    parts.append(keep + " No watermarks. Nothing sexual, gory or offensive.")
    return "\n\n".join(parts)


def _png(img) -> bytes:
    out = io.BytesIO()
    img.convert("RGB").save(out, "PNG", optimize=True)
    return out.getvalue()


def crop_to(img, fmt):
    """The model's 2:3 (or 3:2) picture as exactly 9:16 (or 16:9), cut from the middle."""
    from PIL import Image
    w, h = SIZE[fmt][1]
    target = w / h
    if img.width / img.height > target:
        cw = int(img.height * target)
        img = img.crop(((img.width - cw) // 2, 0, (img.width - cw) // 2 + cw, img.height))
    else:
        ch = int(img.width / target)
        img = img.crop((0, (img.height - ch) // 2, img.width, (img.height - ch) // 2 + ch))
    return img.resize((w, h), Image.LANCZOS)


# ---------------------------------------------------------------- asking for one
def request(job, clip, fmt, is_admin) -> None:
    """Queue one AI-designed thumbnail. Raises Busy, ValueError or imagegen.NotAllowed with a plain message."""
    from . import thumbs
    if fmt not in SIZE:
        raise ValueError("Choose the 9:16 or the 16:9 picture.")
    if clip.get("status") != "done" or not clip.get("file"):
        raise Busy("This clip isn't finished yet. Make an AI thumbnail once it is ready.")
    st = state(clip)
    if st.get("status") in ("queued", "making"):
        raise Busy("An AI thumbnail for this clip is already being made.")
    if len(st.get("items") or []) >= MAX_ITEMS:
        raise Busy(f"This clip already has {MAX_ITEMS} AI thumbnails. Delete one first.")
    imagegen.check(is_admin, imagegen.price(None, imagegen.current()["design_quality"]))
    key = (job["user"], job["id"], clip["id"])
    with thumbs._plock:
        if key in thumbs._pending:
            raise Busy("Thumbnails for this clip are being made right now. Wait for them to finish.")
        thumbs._pending.add(key)
    with jobs._lock:
        clip["ai_thumbs"] = {**st, "status": "queued", "error": None, "fmt": fmt}
    jobs.save(job)
    thumbs._q.put((*key, "ai", (fmt,)))


def make(jid: str, cid: str, fmt: str):
    """Make one AI-designed thumbnail. Runs on the thumbnail thread, as the clip's owner."""
    from PIL import Image
    from . import thumbs
    job = jobs.get(jid)
    clip = next((c for c in (job or {}).get("clips", []) if c["id"] == cid), None)
    if not clip:
        return
    with jobs._lock:
        clip["ai_thumbs"] = {**state(clip), "status": "making", "error": None}
    jobs.save(job)
    work = Path(tempfile.mkdtemp(prefix="pka"))
    try:
        path, off, burned = thumbs.source_for(job, clip)
        loud = [] if burned else thumbs.loud_times(path, off, float(clip["start"]), float(clip["end"]))
        frames = []
        for k, t in enumerate(thumbs.times_for(clip, burned, loud)):
            out = work / f"f{k}.jpg"
            if thumbs._grab(path, t - off, out):
                img = Image.open(out)
                img.load()
                frames.append({"t": round(t, 2), "img": img, "score": thumbs.score(img)})
        if not frames:
            raise RuntimeError("Could not take any pictures from this clip's video.")
        frames = sorted(sorted(frames, key=lambda f: -f["score"])[:6], key=lambda f: f["t"])
        said = " ".join(str(w.get("w", "")) for w in clip.get("fixed_words") or [])
        brief = brain.thumb_brief(thumbs.contact_sheet(frames), len(frames), title=clip.get("title") or job.get("title") or "",
                                  hook=clip.get("hook") or "", transcript=said, game=job.get("game"))
        if not brief["line1"] and not brief["line2"]:
            words = (thumbs.clean_words(clip.get("hook") or "") or "WAIT WHAT?!").split()
            brief.update(line1=" ".join(words[:max(1, len(words) // 2)]), line2=" ".join(words[max(1, len(words) // 2):]) or "!")
        fr = frames[(brief["frame"] or 1) - 1] if brief["frame"] else max(frames, key=lambda f: f["score"])
        frame = fr["img"].convert("RGB")
        face = thumbs.face_box(job, frame, brief.get("face"), job.get("facecam"))
        images = []
        region = thumbdraw.face_region(frame, face) if face else None
        if region is not None:
            images.append(("streamer.png", _png(region)))
        images.append(("scene.png", _png(thumbdraw.trim(frame) if frame.width > frame.height else frame)))
        brand = brand_for(job)
        api_size = SIZE[fmt][0]
        got = imagegen.design(images, prompt(brief, brand, fmt, region is not None), api_size)
        quality = imagegen.current()["design_quality"]
        imagegen.record(imagegen.price(None, quality), api_size, "Thumbnail: AI design", creator=job.get("creator"),
                        place=clip.get("title") or job.get("title"), quality=quality)
        art = crop_to(Image.open(io.BytesIO(got)).convert("RGB"), fmt)
        _finish(jid, cid, fmt, art, brief, fr["t"], work)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job = jobs.get(jid)
        clip = next((c for c in (job or {}).get("clips", []) if c["id"] == cid), None)
        if clip:
            with jobs._lock:
                clip["ai_thumbs"] = {**state(clip), "status": "error", "error": safe.friendly(e, 300)}
            jobs.save(job)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _finish(jid, cid, fmt, art, brief, t, work):
    job = jobs.get(jid)
    clip = next((c for c in (job or {}).get("clips", []) if c["id"] == cid), None)
    if not clip:
        return                                   # deleted meanwhile: nothing to keep (the picture was paid for, sadly)
    folder = Path(job["dir"]) / "clips"
    n = int(time.time() * 1000) % 10_000_000
    name = f"{cid}-ai{n}-{fmt}.jpg"
    art.save(work / name, "JPEG", quality=92, optimize=True)
    shutil.move(str(work / name), str(folder / name))
    item = {"id": str(n), "fmt": fmt, "file": name, "headline": f"{brief['line1']} {brief['line2']}".strip(),
            "t": t, "made_at": time.time()}
    with jobs._lock:
        st = state(clip)
        clip["ai_thumbs"] = {**st, "status": "done", "error": None, "items": (st.get("items") or []) + [item]}
    jobs.save(job)


def choose(job, clip, item_id):
    """Use one AI thumbnail (9:16) as the clip's cover, or go back with None."""
    st = state(clip)
    with jobs._lock:
        if item_id is None:
            if st.get("chosen") and clip.get("frame_cover"):
                clip["cover"] = clip.pop("frame_cover")
            st["chosen"] = None
        else:
            item = next((i for i in st.get("items") or [] if i["id"] == str(item_id)), None)
            if not item:
                raise ValueError("No such AI thumbnail.")
            if item["fmt"] != "v":
                raise ValueError("Only a 9:16 picture can be the clip's cover.")
            th = clip.get("thumbs") or {}
            if th.get("chosen"):
                th["chosen"] = None                  # one cover at a time
            if not clip.get("frame_cover"):
                clip["frame_cover"] = clip.get("cover")
            clip["cover"] = item["file"]
            st["chosen"] = item["id"]
        clip["ai_thumbs"] = st
    jobs.save(job)


def delete(job, clip, item_id):
    st = state(clip)
    item = next((i for i in st.get("items") or [] if i["id"] == str(item_id)), None)
    if not item:
        raise ValueError("No such AI thumbnail.")
    if st.get("chosen") == item["id"]:
        choose(job, clip, None)
        st = state(clip)
    with jobs._lock:
        st["items"] = [i for i in st["items"] if i["id"] != item["id"]]
        clip["ai_thumbs"] = st
    jobs.save(job)
    (Path(job["dir"]) / "clips" / item["file"]).unlink(missing_ok=True)


def remove_files(job, cid):
    for f in (Path(job["dir"]) / "clips").glob(f"{cid}-ai*.jpg"):
        if re.fullmatch(rf"{re.escape(cid)}-ai\d+-[vh]\.jpg", f.name):
            f.unlink(missing_ok=True)


def chosen_file(job, clip):
    """The chosen AI thumbnail's file (for posting and downloads), or None."""
    st = clip.get("ai_thumbs") or {}
    item = next((i for i in st.get("items") or [] if i["id"] == st.get("chosen")), None)
    return (Path(job["dir"]) / "clips" / item["file"], item) if item else (None, None)


def interrupted(clip):
    st = clip.get("ai_thumbs")
    if st and st.get("status") in ("queued", "making"):
        st.update(status="error", error="Interrupted by a restart. Press AI thumbnail again.")
