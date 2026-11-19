"""Live streams in any language: speech by OpenAI when the streamer doesn't speak English, and big reactions
clipped even when the words don't explain them.

Runs against the code directly, with its own empty data folder and no network: OpenAI and the AI are replaced."""
import json
import os
import sys
import tempfile
from unittest.mock import patch

os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-livespeech-")
os.environ.update(DATABASE_URL="", PEAKCUT_ACCOUNTS="0", PEAKCUT_AI_PROVIDER="openai", PEAKCUT_AI_MODELS="test-model",
                  PEAKCUT_AI_KEY="sk-chat-key-123456", PEAKCUT_AI_KEY_FILE="", PEAKCUT_AI_BASE_URL="")
sys.path.insert(0, ".")
import numpy as np  # noqa: E402

from studio import brain, creators, live, speech_cloud, transcribe  # noqa: E402

fails = []


def check(name, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"   got {got!r}"))
    ok or fails.append(name)


class Reply:
    def __init__(self, code, data=None, text=None):
        self.status_code, self._d = code, data or {}
        self.text = text if text is not None else json.dumps(self._d)

    def json(self):
        return self._d


SR = 16000
t = np.arange(SR * 10) / SR
speech = (np.sin(2 * np.pi * 220 * t) * 8000 * ((t > 2) & (t < 8))).astype(np.int16)    # sound from 2 s to 8 s
silence = np.zeros(SR * 5, dtype=np.int16)

# ---- the creator setting
c = creators._clean({"live_speech": "cloud", "loud_clips": "yes"})
check("the setting for speech in any language is kept", (c["live_speech"], c["loud_clips"]), ("cloud", True))
check("anything else falls back to this computer", creators._clean({"live_speech": "mars"})["live_speech"], "local")
check("both are off for a new creator", (creators.DEFAULTS["live_speech"], creators.DEFAULTS["loud_clips"]), ("local", False))

# ---- speech by OpenAI
sent = []


def fake_post(url, headers=None, data=None, files=None, timeout=None):
    sent.append({"url": url, "model": data["model"], "auth": headers["Authorization"], "prompt": data.get("prompt", "")})
    if data["model"] == "gpt-4o-mini-transcribe" and len(sent) == 1:
        return Reply(404, {"error": {"message": "The model gpt-4o-mini-transcribe does not exist"}})
    return Reply(200, {"text": "bhai ye kya hua yaar", "usage": {"input_tokens": 50, "output_tokens": 8}})


check("with the AI on OpenAI, its key is used for speech", speech_cloud.key(), "sk-chat-key-123456")
with patch.object(speech_cloud.httpx, "post", side_effect=fake_post):
    words = speech_cloud.transcribe(speech)
    check("silence is never sent (it would cost money for nothing)", (speech_cloud.transcribe(silence), len(sent)), ([], 2))
check("a model that's gone is skipped for the next one", [s["model"] for s in sent], ["gpt-4o-mini-transcribe", "gpt-transcribe"])
check("the words come back", [w["w"] for w in words], ["bhai", "ye", "kya", "hua", "yaar"])
check("they are spread over the part with sound, in order", 1.8 <= words[0]["s"] < words[-1]["e"] <= 8.3
      and all(a["e"] <= b["s"] + 0.01 for a, b in zip(words, words[1:])), True)
check("the request asks for Hinglish in Roman letters", "Roman letters" in sent[-1]["prompt"], True)
speech_cloud._down_until = 0
with patch.object(speech_cloud.httpx, "post", return_value=Reply(401, {"error": {"message": "bad key"}})):
    check("a refused key gives nothing (the local model takes over)", speech_cloud.transcribe(speech), None)
    check("...with a plain reason", "refused" in speech_cloud.last_error, True)
    check("...and it isn't asked again straight away", speech_cloud._down_until > 0, True)
speech_cloud._down_until = 0


# ---- a live session uses it, and falls back to this computer when it fails
def session(**settings):
    d = tempfile.mkdtemp(prefix="pc-livespeech-s-")
    base = {"live_speech": "local", "loud_clips": False, "min_len": 15, "max_len": 60, "min_score": 60, "min_gap_min": 2,
            "live_per_hour": 8, "must_clip_min": 0, "decide_mode": "either", "ai_every_min": 3, "game": "",
            "style": "bounce", "layout": "blur", "trigger_words": ""}
    s = live.Session({"title": "GTA 6 stream", "uploader": "someone", "dir": d, "creator": "", "ai": {},
                      "source": {"url": "", "name": "x"}, "clips": [], "options": {}}, {**base, **settings})
    return s


s = session(live_speech="cloud")
s.pending = bytearray(np.concatenate([speech, speech]).tobytes())
with patch.object(speech_cloud, "transcribe", return_value=[{"w": "bhai", "s": 1.0, "e": 1.4}]) as cloud, \
        patch.object(transcribe, "transcribe", return_value=[{"w": "local", "s": 0.5, "e": 0.9}]) as local:
    s._transcribe_step()
check("a creator set to OpenAI speech is transcribed online", (cloud.called, local.called, s.words[0]["w"]), (True, False, "bhai"))
check("online pieces are longer (fewer requests)", 8 * SR <= cloud.call_args[0][0].shape[0] <= 12.5 * SR, True)
s = session(live_speech="cloud")
s.pending = bytearray(np.concatenate([speech, speech]).tobytes())
speech_cloud.last_error = "OpenAI says the account is out of credit or busy (live speech)."
with patch.object(speech_cloud, "transcribe", return_value=None), \
        patch.object(transcribe, "transcribe", return_value=[{"w": "local", "s": 0.5, "e": 0.9}]):
    s._transcribe_step()
check("when OpenAI fails, this computer's model is used for that piece", s.words[0]["w"], "local")
check("...and the reason is shown on the recording", "credit" in s.job["ai"].get("speech_error", ""), True)
s = session()
s.pending = bytearray(np.concatenate([speech, speech]).tobytes())
with patch.object(speech_cloud, "transcribe") as cloud, patch.object(transcribe, "transcribe", return_value=[]):
    s._transcribe_step()
check("a creator left on this computer never sends sound online", cloud.called, False)

# ---- big reactions are clipped even when the words judge finds nothing
s = session(loud_clips=True)
s.t_done = 400.0
s.words = [{"w": w, "s": 300 + i, "e": 300.5 + i} for i, w in enumerate("bhai ye kya hua yaar dekh dekh abhi".split())]
added = []
s._add_clip = lambda c: added.append(c) or s.job["clips"].append(c)
with patch.object(brain, "judge_transcript", return_value=([], 10)), \
        patch.object(brain, "describe", return_value={"score": 0, "hook": "BHAI KYA?!", "title": "He lost it", "category": "funny",
                                                     "description": "", "hashtags": [], "highlight_words": []}):
    s._ai_check((300.0 - 30, 300.0 + 20), "Jump: loudness +140%")
check("a big jump in loudness is clipped although the words judge said no", len(added), 1)
c = added[0]
check("the clip covers the reaction: before it and just after", (c["start"] <= 290, c["end"] >= 305, 15 <= c["end"] - c["start"] <= 60), (True, True, True))
check("the AI named it", (c["hook"], c["judges"]["why"]), ("BHAI KYA?!", "Big reaction"))
check("the recording says why it was clipped", "big reaction" in s.verdicts[-1]["result"].lower(), True)
with patch.object(brain, "judge_transcript", return_value=([], 10)), patch.object(brain, "describe", return_value={}):
    s._ai_check((310.0 - 30, 310.0 + 20), "Jump: loudness +90%")
check("but not a second one right next to it", (len(added), "too close" in s.verdicts[-1]["result"].lower()), (1, True))
s2 = session(loud_clips=False)
s2.t_done = 400.0
s2.words = s.words
added2 = []
s2._add_clip = lambda c: added2.append(c)
with patch.object(brain, "judge_transcript", return_value=([], 10)):
    s2._ai_check((270.0, 320.0), "Jump: loudness +140%")
check("with the setting off, nothing changes: the words judge decides", (len(added2), s2.verdicts[-1]["result"]), (0, "Words judge: nothing clip-worthy"))
s3 = session(loud_clips=True)
s3.t_done = 400.0
s3._add_clip = lambda c: added2.append(c) or s3.job["clips"].append(c)
with patch.object(brain, "describe", side_effect=RuntimeError("The AI did not answer")):
    s3._ai_check((270.0, 320.0), "Jump: loudness +200%")
check("a big reaction with nobody talking is clipped too, even if the AI can't name it", len(added2), 1)
s4 = session(loud_clips=True)
s4.t_done = 400.0
s4.words = s.words
added4 = []
s4._add_clip = lambda c: added4.append(c)
with patch.object(brain, "judge_transcript", return_value=([], 10)):
    s4._ai_check()
check("the every-few-minutes check is unchanged (only jumps become big-reaction clips)", len(added4), 0)

print(f"{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
