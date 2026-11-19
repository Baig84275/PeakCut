"""Speech to text for live streams in any language (Hindi, Urdu, Hinglish, English...), by OpenAI.

The speech model on this computer understands English only. A streamer speaking Hindi or Urdu comes out as
nonsense, and the AI then finds nothing worth clipping. This sends each few seconds of a live stream's sound to
OpenAI instead (gpt-4o-mini-transcribe, about $0.003 a minute) and gets the words back. It needs an OpenAI key:
the one saved for the AI when the AI is OpenAI, otherwise the one saved for AI pictures.

The service gives the words of a piece, not when each one was said; they are spread over the piece by length,
which is close enough for choosing a clip (clips are cut on sentence ends later anyway)."""
import io
import re
import time
import wave

import httpx
import numpy as np

from . import ai, config, usage

MODELS = [m.strip() for m in (config.env("PEAKCUT_LIVE_SPEECH_MODELS") or "gpt-4o-mini-transcribe,gpt-transcribe").split(",") if m.strip()]
PROMPT = ("A live stream, often a game stream, in Hindi, Urdu, English or a mix. Write exactly what the streamer "
          "says. Write Hindi and Urdu in Roman letters, the way people type them (Hinglish).")
SR = 16000
_down_until = 0.0
last_error = ""


def key() -> str:
    if ai.provider() == "openai" and ai.key():
        return ai.key()
    try:
        from . import imagegen
        return imagegen.current().get("key") or ""
    except Exception:  # noqa: BLE001
        return ""


def ready() -> bool:
    return bool(key())


def _wav(audio: np.ndarray) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(audio.astype(np.int16).tobytes())
    return out.getvalue()


def _spread(text: str, audio: np.ndarray) -> list:
    """The words over the part of the piece where there is sound, each taking time by its length."""
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return []
    a = audio.astype(np.float32) / 32768.0
    hop = SR // 10
    level = np.array([np.sqrt((a[i:i + hop] ** 2).mean()) if len(a[i:i + hop]) else 0 for i in range(0, len(a), hop)])
    loud = np.where(level > max(0.004, float(level.max()) * 0.12))[0] if len(level) else []
    s0 = (loud[0] * hop / SR) if len(loud) else 0.0
    s1 = ((loud[-1] + 1) * hop / SR) if len(loud) else len(a) / SR
    if s1 - s0 < 0.3 * len(words):
        s0, s1 = 0.0, len(a) / SR
    weights = np.array([max(2, len(w)) for w in words], dtype=np.float32)
    edges = s0 + np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * (s1 - s0)
    return [{"w": w, "s": round(float(edges[i]), 2), "e": round(float(edges[i + 1]), 2)} for i, w in enumerate(words)]


def transcribe(audio: np.ndarray) -> list | None:
    """[{'w', 's', 'e'}] for a piece of 16 kHz sound, or None when the service couldn't do it (the caller then
    uses the local model). Silence is never sent, so it costs nothing."""
    global _down_until, last_error
    a = audio.astype(np.float32) / 32768.0
    if len(a) < SR // 2 or float(np.sqrt((a ** 2).mean())) < 0.003:
        return []
    if time.time() < _down_until or not key():
        return None
    body = _wav(audio)
    for model in MODELS:
        try:
            r = httpx.post("https://api.openai.com/v1/audio/transcriptions", headers={"Authorization": f"Bearer {key()}"},
                           data={"model": model, "response_format": "json", "prompt": PROMPT},
                           files={"file": ("piece.wav", body, "audio/wav")}, timeout=30)
        except httpx.HTTPError as e:
            last_error = f"Couldn't reach OpenAI for speech: {type(e).__name__}"
            _down_until = time.time() + 30
            return None
        if r.status_code == 200:
            data = r.json()
            u = data.get("usage") or {}
            tin = int(u.get("input_tokens") or u.get("prompt_tokens") or 0)
            tout = int(u.get("output_tokens") or u.get("completion_tokens") or 0)
            if tin or tout:
                usage.record(tin, tout, "Live: speech to text")
            last_error = ""
            return _spread(str(data.get("text") or ""), audio)
        if r.status_code in (400, 404) and "model" in r.text.lower():
            continue                                   # this model is gone or not on this key: try the next
        last_error = {401: "OpenAI refused the key for live speech.", 403: "OpenAI refused the key for live speech.",
                      429: "OpenAI says the account is out of credit or busy (live speech)."}.get(
                          r.status_code, f"OpenAI speech failed (HTTP {r.status_code}).")
        _down_until = time.time() + (300 if r.status_code in (401, 403) else 30)
        return None
    last_error = "None of the speech models are available on this OpenAI key."
    _down_until = time.time() + 600
    return None
