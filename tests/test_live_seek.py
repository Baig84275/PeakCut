"""Clips cut from a recorded live stream must move from their very first frame.

A recorded piece of a live stream (.ts) has a full picture only every few seconds and no index. Jumping into
it lands between two full pictures, and the clip's video then stayed frozen (with the sound playing) until the
next one, up to 5 seconds. Runs against the code directly, with its own empty data folder."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-liveseek-")
sys.path.insert(0, ".")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from studio import render, thumbs  # noqa: E402

fails = []


def check(name, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"   got {got!r}"))
    ok or fails.append(name)


work = Path(tempfile.mkdtemp(prefix="pc-liveseek-w-"))
ts = work / "piece.ts"
# 20 s of moving picture with a full picture only every 5 s, written the way live pieces are (MPEG-TS)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=20",
                "-f", "lavfi", "-i", "sine=frequency=300:duration=20", "-c:v", "libx264", "-preset", "ultrafast",
                "-g", "150", "-keyint_min", "150", "-sc_threshold", "0", "-c:a", "aac", "-f", "mpegts", str(ts)], check=True)


def frozen_for(path) -> float:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-t", "6", "-vf", "fps=10,scale=96:96,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, 96, 96).astype(np.float32)
    d = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    return next((i for i, x in enumerate(d) if x >= 0.4), len(d)) / 10


clip = {"start": 5.6, "end": 13.0, "hook": "", "style": "bounce", "layout": "blur", "captions": False}
out = render.render_clip(ts, work, "clip001", clip, [], 1280, 720)
check("a clip starting just after a full picture moves from its first frame (was frozen ~4.5 s)",
      frozen_for(work / out["file"]) <= 0.2, True)
check("...and is as long as asked", abs(float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                                               "-of", "csv=p=0", str(work / out["file"])],
                                                              capture_output=True, text=True).stdout) - 7.4) < 0.3, True)
check("a normal video file still jumps straight in (fast)", render._seek(Path("v.mp4"), 3, 5)[:2], ["-ss", "3.00"])
check("a live piece is read from its start (exact)", render._seek(Path("v.ts"), 3, 5)[:2], ["-i", "v.ts"])

# thumbnails take their frames from the same pieces: the frame must be the moment asked for, not a stale one
a, b = work / "a.jpg", work / "b.jpg"
thumbs._grab(ts, 6.0, a)
thumbs._grab(ts, 9.0, b)
diff = np.abs(np.asarray(Image.open(a).convert("L"), np.float32) - np.asarray(Image.open(b).convert("L"), np.float32)).mean()
check("thumbnail frames from a live piece are the moments asked for (not the same stale picture)", diff > 1.0, True)

print(f"{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
