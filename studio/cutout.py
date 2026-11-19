"""Cutting a person out of a picture (for thumbnails): MODNet, a small portrait matting model, run on this
computer with onnxruntime. Nothing leaves the machine.

The model (about 6.6 MB) is downloaded once into the models folder, the first time a thumbnail needs it, and
checked against a known fingerprint. Without onnxruntime or the model, thumbnails still work: the webcam
picture is shown as a framed card instead of a cut-out."""
import hashlib
import threading
from pathlib import Path

import numpy as np

from . import config

MODEL_URL = "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model_quantized.onnx"
MODEL_SHA256 = ""          # filled in below once known; an empty value only skips the check
MODEL_NAME = "modnet-portrait.onnx"
_lock = threading.Lock()
_session = None
_failed = False


def _path() -> Path:
    return Path(config.MODELS_DIR) / MODEL_NAME


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _fetch(dest: Path):
    import httpx
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=120) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Couldn't download the cut-out model (HTTP {r.status_code}).")
        h = hashlib.sha256()
        size = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 16):
                size += len(chunk)
                if size > 60_000_000:
                    raise RuntimeError("The cut-out model download is far bigger than expected; stopped.")
                h.update(chunk)
                f.write(chunk)
    if MODEL_SHA256 and h.hexdigest() != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("The downloaded cut-out model doesn't match its fingerprint; not used.")
    tmp.replace(dest)


def _load():
    global _session, _failed
    if _session is not None or _failed:
        return _session
    with _lock:
        if _session is not None or _failed:
            return _session
        try:
            import onnxruntime as ort
            p = _path()
            if not p.is_file():
                p.parent.mkdir(parents=True, exist_ok=True)
                _fetch(p)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            _session = ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
        except Exception:  # noqa: BLE001 - no cut-out: thumbnails fall back to a framed webcam card
            import traceback
            traceback.print_exc()
            _failed = True
    return _session


def matte(img) -> np.ndarray | None:
    """How much each pixel belongs to a person, 0..1, the size of the picture. None if it can't be worked out."""
    from PIL import Image
    s = _load()
    if s is None:
        return None
    rgb = img.convert("RGB")
    w, h = rgb.size
    scale = 512 / max(w, h)
    tw, th = max(32, int(round(w * scale / 32)) * 32), max(32, int(round(h * scale / 32)) * 32)
    x = np.asarray(rgb.resize((tw, th), Image.BILINEAR), dtype=np.float32) / 255.0
    x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
    name = s.get_inputs()[0].name
    try:
        out = s.run(None, {name: x.astype(np.float32)})[0]
    except Exception:  # noqa: BLE001
        return None
    m = np.squeeze(out).astype(np.float32)
    if m.ndim != 2:
        return None
    m = np.asarray(Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    return m


def person(img, least=0.06):
    """The picture with everything but the person made transparent (RGBA), trimmed to the person, or None
    when there is no clear person in it (less than `least` of the picture)."""
    from PIL import Image
    m = matte(img)
    if m is None or float((m > 0.5).mean()) < least:
        return None
    alpha = np.clip((m - 0.08) / 0.84, 0, 1)
    rgba = img.convert("RGBA")
    rgba.putalpha(Image.fromarray((alpha * 255).astype(np.uint8), "L"))
    ys, xs = np.where(alpha > 0.1)
    return rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
