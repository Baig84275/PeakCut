"""The optional, paid "AI look" for thumbnails: an image model repaints a thumbnail's picture with brighter,
punchier light and colour. PeakCut then puts the words on it itself, so the text is never misspelled.

It has its own settings (ai_image.json in the data folder, next to ai.json), off until someone switches it on.
Every picture costs money on the person's own account, so each one is written down with what it roughly cost,
and a daily limit (per person, and for the whole server) stops a runaway bill."""
import io
import json
import os
import threading
import time

import httpx
import sqlalchemy as sa

from . import ai, config, ctx, db

QUALITIES = ("low", "medium", "high")
# Roughly what one portrait or landscape picture costs, in US dollars, by model and quality. Prices change:
# these are only for showing an estimate and for the daily limit. The service's own bill is the truth.
PRICES = {
    "gpt-image-2": {"low": 0.005, "medium": 0.041, "high": 0.165},
    "gpt-image-1.5": {"low": 0.013, "medium": 0.05, "high": 0.2},
    "gpt-image-1-mini": {"low": 0.006, "medium": 0.015, "high": 0.052},
    "gpt-image-1": {"low": 0.016, "medium": 0.063, "high": 0.25},
}
FALLBACK_PRICE = {"low": 0.02, "medium": 0.07, "high": 0.25}      # an unknown model: assume it costs more, not less
DEFAULTS = {"provider": "off", "model": "gpt-image-2", "key": "", "use_chat_key": True, "quality": "medium",
            "daily_usd": 1.0, "everyone": False, "base_url": "https://api.openai.com/v1"}
SERVER_DAILY_USD = float(config.env("PEAKCUT_AI_IMAGE_DAILY_USD", "0") or 0)    # 0 = no limit for the whole server
PROMPT = ("Make this into a scroll-stopping thumbnail: vivid, punchy colour and lighting, high contrast, crisp detail, "
          "a slight glow on the main subject. Keep the same scene and composition. Keep every person's face, body and "
          "identity exactly as they are. Do not add any text, letters, numbers, logos, emoji or watermarks.")
_lock = threading.Lock()


class NotAllowed(Exception):
    pass


def _path():
    return config.DATA_DIR / "ai_image.json"


def current() -> dict:
    try:
        saved = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}


def key() -> str:
    c = current()
    if c["use_chat_key"] and ai.provider() == "openai":
        return ai.key()
    return c["key"]


def ready() -> bool:
    c = current()
    return c["provider"] == "openai" and bool(key()) and bool(c["model"])


def price(model=None, quality=None) -> float:
    c = current()
    model, quality = model or c["model"], quality or c["quality"]
    table = next((v for k, v in PRICES.items() if model == k or model.startswith(k + "-20")), FALLBACK_PRICE)
    return table.get(quality, FALLBACK_PRICE["high"])


def public() -> dict:
    """What the settings page may see: never the key itself."""
    c = current()
    own = c["key"]
    return {"provider": c["provider"], "model": c["model"], "quality": c["quality"], "daily_usd": c["daily_usd"],
            "everyone": c["everyone"], "use_chat_key": c["use_chat_key"], "has_key": bool(own),
            "key_hint": own[-4:] if len(own) >= 8 else "", "chat_is_openai": ai.provider() == "openai",
            "ready": ready(), "price": round(price(), 3), "qualities": list(QUALITIES),
            "prices": {q: round(price(c["model"], q), 3) for q in QUALITIES},
            "spent_today": round(spent_today(), 3) if ctx.uid_or_none() else 0.0}


def save(body: dict) -> dict:
    """Save the settings from the page. A key of None (or left out) keeps the saved one."""
    import re
    c = current()
    if "provider" in body:
        if body["provider"] not in ("off", "openai"):
            raise ValueError("Choose Off or OpenAI.")
        c["provider"] = body["provider"]
    if "model" in body:
        m = str(body["model"] or "").strip()
        if not re.fullmatch(r"[\w.\-:]{1,80}", m):
            raise ValueError("That model name doesn't look right.")
        c["model"] = m
    if "quality" in body:
        if body["quality"] not in QUALITIES:
            raise ValueError("Choose low, medium or high.")
        c["quality"] = body["quality"]
    if "daily_usd" in body:
        try:
            v = float(body["daily_usd"])
        except (TypeError, ValueError):
            raise ValueError("The daily limit must be a number of dollars.")
        if not 0 <= v <= 1000:
            raise ValueError("The daily limit must be between 0 and 1000 dollars.")
        c["daily_usd"] = round(v, 2)
    for k in ("use_chat_key", "everyone"):
        if k in body:
            c[k] = bool(body[k])
    if body.get("key") is not None:
        k = str(body["key"]).strip()
        if len(k) > 300 or any(ch.isspace() for ch in k):
            raise ValueError("That key doesn't look right.")
        c["key"] = k
    with _lock:
        tmp = _path().with_suffix(".tmp")
        tmp.write_text(json.dumps(c), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(_path())
    return public()


# ---------------------------------------------------------------- what it cost
def spent_today(uid=None, everyone=False) -> float:
    midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    u = db.image_usage.c
    q = sa.select(sa.func.coalesce(sa.func.sum(u.cost_usd), 0.0)).where(u.t >= midnight)
    if not everyone:
        q = q.where(u.user_id == (uid or ctx.uid()))
    with db.engine.connect() as c:
        return float(c.execute(q).scalar() or 0.0)


def check(is_admin: bool):
    """Raise NotAllowed with a plain message when this person may not make an AI picture right now."""
    c = current()
    if not ready():
        raise NotAllowed("AI pictures are off. Switch them on in Settings, AI.")
    if not is_admin and not c["everyone"]:
        raise NotAllowed("AI pictures are only for the admin on this server.")
    cost = price()
    if c["daily_usd"] and spent_today() + cost > c["daily_usd"] + 1e-9:
        raise NotAllowed(f"Today's limit for AI pictures (${c['daily_usd']:g}) is reached. It resets at midnight, "
                         "or raise it in Settings, AI.")
    if SERVER_DAILY_USD and spent_today(everyone=True) + cost > SERVER_DAILY_USD + 1e-9:
        raise NotAllowed("The server's daily limit for AI pictures is reached. Try again tomorrow.")


def record(cost: float, size: str, purpose="Thumbnail: AI look", creator=None, place=None):
    uid = ctx.uid_or_none()
    if not uid:
        return
    c = current()
    with db.engine.begin() as conn:
        conn.execute(db.image_usage.insert().values(user_id=uid, t=time.time(), model=c["model"][:80], size=size[:20],
                                                    quality=c["quality"], cost_usd=float(cost), purpose=purpose[:60],
                                                    creator=creator, place=(place or "")[:200]))
    ctx.bump(uid)


def summary(uid=None) -> dict:
    u = db.image_usage.c
    uid = uid or ctx.uid()
    with db.engine.connect() as c:
        n, usd = c.execute(sa.select(sa.func.count(), sa.func.coalesce(sa.func.sum(u.cost_usd), 0.0)).where(u.user_id == uid)).first()
    return {"images": int(n or 0), "usd": round(float(usd or 0.0), 3), "today_usd": round(spent_today(uid), 3)}


def forget(uid: str):
    with db.engine.begin() as c:
        c.execute(db.image_usage.delete().where(db.image_usage.c.user_id == uid))


# ---------------------------------------------------------------- one picture
def _explain(r) -> str:
    try:
        msg = (r.json().get("error") or {}).get("message", "")
    except ValueError:
        msg = ""
    low = msg.lower()
    if r.status_code in (401, 403):
        return "The key for AI pictures was refused. Check it in Settings, AI."
    if r.status_code == 429 or "quota" in low or "billing" in low:
        return "The AI picture service says the account is out of credit or busy. Check your OpenAI billing."
    if "safety" in low or "moderation" in low or "rejected" in low:
        return "The AI picture service refused this picture (its safety rules). Use the normal thumbnail instead."
    if r.status_code == 404 or "model" in low and "not" in low:
        return "That picture model isn't available on this key. Pick another model in Settings, AI."
    return f"The AI picture service didn't make it (HTTP {r.status_code})."


def repaint(png: bytes, size: str) -> bytes:
    """Send one picture (PNG) to the image model and get the repainted one back as image bytes."""
    c = current()
    base = (c["base_url"] or DEFAULTS["base_url"]).rstrip("/")
    tries = [{"input_fidelity": "high", "output_format": "jpeg"}, {"output_format": "jpeg"}, {}]
    last = None
    for extra in tries:          # newer and older models accept different optional settings: drop them one by one
        data = {"model": c["model"], "prompt": PROMPT, "size": size, "quality": c["quality"], "n": "1", **extra}
        try:
            r = httpx.post(f"{base}/images/edits", headers={"Authorization": f"Bearer {key()}"}, data=data,
                           files={"image": ("thumb.png", png, "image/png")}, timeout=180)
        except httpx.HTTPError as e:
            raise RuntimeError("Couldn't reach the AI picture service. Check the internet connection.") from e
        last = r
        if r.status_code == 400 and any(p in r.text for p in ("input_fidelity", "output_format", "Unknown parameter", "unsupported")):
            continue
        break
    if last.status_code != 200:
        raise RuntimeError(_explain(last))
    import base64
    item = (last.json().get("data") or [{}])[0]
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    if item.get("url", "").startswith("https://"):
        got = httpx.get(item["url"], timeout=60)
        if got.status_code == 200 and len(got.content) < 20_000_000:
            return got.content
    raise RuntimeError("The AI picture service answered without a picture.")


def test() -> dict:
    """Check the key and model without making a picture (so it costs nothing)."""
    c = current()
    if not key():
        raise RuntimeError("There is no key for AI pictures.")
    base = (c["base_url"] or DEFAULTS["base_url"]).rstrip("/")
    r = httpx.get(f"{base}/models/{c['model']}", headers={"Authorization": f"Bearer {key()}"}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(_explain(r))
    return {"ok": True, "model": c["model"], "price": price()}


def to_png(img) -> bytes:
    out = io.BytesIO()
    img.convert("RGB").save(out, "PNG", optimize=True)
    return out.getvalue()
