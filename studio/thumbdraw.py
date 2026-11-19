"""Drawing thumbnails: the look of the picture and of the words.

Three ideas carry most of the effect:
  - the person: a streamer's own face, cut out of its webcam picture, big, with a white outline and a glow,
    in front of the game made darker and softer, so the face is the first thing anyone sees;
  - the colour: punchier light and colour than the stream (streams are grey and flat), a dark edge;
  - the words: two or three huge words with a thick outline and a shadow, or on a bright banner.
Overlays at the edges of a stream (chat, minimap, viewer count, health bars) are trimmed off the background."""
from pathlib import Path

import numpy as np

SIZES = {"v": (1080, 1920), "h": (1280, 720)}
FONT = Path(__file__).resolve().parent / "fonts" / "Anton-Regular.ttf"
STYLES = ("outline", "banner", "duo")
TRIM = (0.07, 0.09, 0.07, 0.11)      # left, top, right, bottom: where stream overlays usually sit


# ---------------------------------------------------------------- colour
def grade(img, strength=1.0):
    """Punchier colour and light: more saturation where colour is weak (vibrance), an S-curve for contrast,
    a dark edge (vignette) and a little sharpening."""
    from PIL import Image, ImageFilter
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    grey = a.mean(axis=2, keepdims=True)
    sat = (a.max(axis=2, keepdims=True) - a.min(axis=2, keepdims=True))
    boost = 1.0 + 0.55 * strength * (1.0 - np.clip(sat * 1.6, 0, 1))          # weak colour gets more
    a = grey + (a - grey) * boost
    a = np.clip(a, 0, 1)
    k = 0.18 * strength                                                    # S-curve around the middle
    a = a + k * (a - 0.5) * (1 - np.abs(2 * a - 1))
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    a = a * (1.0 - 0.38 * strength * np.clip(r - 0.35, 0, 1)[..., None] / 0.65)
    out = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB")
    return out.filter(ImageFilter.UnsharpMask(radius=2, percent=int(70 * strength), threshold=2))


def darken(img, amount=0.35, blur=3):
    from PIL import ImageEnhance, ImageFilter
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return ImageEnhance.Brightness(img).enhance(1.0 - amount)


# ---------------------------------------------------------------- framing
def trim(img):
    """The frame without its outer edges, where chat boxes, minimaps and viewer counts usually sit."""
    l, t, r, b = TRIM
    return img.crop((int(img.width * l), int(img.height * t), int(img.width * (1 - r)), int(img.height * (1 - b))))


def cover(img, w, h, fx=0.5, fy=0.45, zoom=1.0):
    """Fill w x h from the picture, cropped around the focus point (0..1). Never blown up past 2.5x the fill."""
    from PIL import Image
    base = max(w / img.width, h / img.height)
    scale = min(base * max(1.0, zoom), base * 2.5)
    nw, nh = max(w, int(img.width * scale)), max(h, int(img.height * scale))
    big = img.resize((nw, nh), Image.LANCZOS)
    x = int(min(max(0, fx * nw - w / 2), nw - w))
    y = int(min(max(0, fy * nh - h / 2), nh - h))
    return big.crop((x, y, x + w, y + h))


def face_region(frame, face):
    """The webcam or person picture around a face box (fractions of the frame), with room for the shoulders."""
    x, y, w, h = face
    fw, fh = frame.width, frame.height
    cx, cy = (x + w / 2) * fw, (y + h / 2) * fh
    half_w = max(w * fw * 0.5 * 1.45, 24)
    top = cy - h * fh * 0.5 * 1.5
    bottom = cy + h * fh * 0.5 * 2.3                 # more room below: shoulders make a cut-out look natural
    box = (max(0, int(cx - half_w)), max(0, int(top)), min(fw, int(cx + half_w)), min(fh, int(bottom)))
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return None
    return frame.crop(box)


# ---------------------------------------------------------------- the person
def outlined(cut, accent, line=10, glow=26):
    """A cut-out person (RGBA) with a white outline and a coloured glow around it, on a transparent canvas."""
    from PIL import Image, ImageFilter
    pad = line + glow
    w, h = cut.width + 2 * pad, cut.height + 2 * pad
    alpha = Image.new("L", (w, h), 0)
    alpha.paste(cut.getchannel("A"), (pad, pad))
    small = alpha.resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR)    # thick outlines are cheaper small
    grow = small.filter(ImageFilter.MaxFilter(max(3, (line // 2) | 1))).resize((w, h), Image.BILINEAR)
    grow = grow.point(lambda v: 255 if v > 60 else 0)
    halo = grow.filter(ImageFilter.GaussianBlur(glow / 2)).point(lambda v: min(255, int(v * 1.6)))
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(Image.new("RGBA", (w, h), (*accent, 255)), (0, 0), halo)
    out.paste(Image.new("RGBA", (w, h), (255, 255, 255, 255)), (0, 0), grow)
    out.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)))
    person = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    person.paste(cut, (pad, pad))
    out.alpha_composite(person)
    return out, pad


def card(region, accent, border=10):
    """No cut-out possible: the webcam picture as a card with a white border and a coloured glow."""
    from PIL import Image, ImageDraw
    w, h = region.width + 2 * border, region.height + 2 * border
    cut = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    cut.paste(region.convert("RGB"), (border, border))
    big = Image.new("L", (w * 4, h * 4), 0)                  # drawn 4x and shrunk: smooth rounded corners
    ImageDraw.Draw(big).rounded_rectangle((0, 0, w * 4 - 1, h * 4 - 1), radius=max(32, min(w, h) // 3), fill=255)
    cut.putalpha(big.resize((w, h), Image.LANCZOS))
    return cut


# ---------------------------------------------------------------- words
def _font(size):
    from PIL import ImageFont
    return ImageFont.truetype(str(FONT), size)


def _cap(f):
    """Top and height of a capital letter for this font, from where text is drawn."""
    t, b = f.getbbox("H")[1], f.getbbox("H")[3]
    return t, b - t


def _split(text, f, max_w):
    ws = text.split()
    if len(ws) <= 1 or f.getlength(text) <= max_w:
        return [text]
    cut = min(range(1, len(ws)), key=lambda i: max(f.getlength(" ".join(ws[:i])), f.getlength(" ".join(ws[i:]))))
    return [" ".join(ws[:cut]), " ".join(ws[cut:])]


def _fit(text, max_w, max_h, gap, least=40):
    """The biggest size at which the words fit, on one line or two balanced ones (measured on real letters)."""
    size = int(max_h * 0.95)
    while True:
        f = _font(size)
        lines = _split(text, f, max_w)
        _t, cap = _cap(f)
        tall = cap * len(lines) + cap * gap * (len(lines) - 1)
        widest = max(f.getlength(l) for l in lines)
        if (widest <= max_w and tall <= max_h) or size <= least:
            return lines, f, cap, tall
        size = max(least, int(size * 0.93))


def words(img, text, accent, box, align="center", style="outline", tilt=0.0):
    """Big words on the picture. Styles: "outline" (white, last word in colour, thick outline), "banner"
    (dark words on a bright colour bar), "duo" (first line white, second line in colour). All get a soft shadow."""
    from PIL import Image, ImageDraw, ImageFilter
    if not text:
        return img
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    banner = style == "banner"
    gap = 0.62 if banner else 0.24
    lines, f, cap, tall = _fit(text, bw * (0.86 if banner else 0.97), bh * (0.78 if banner else 0.9), gap)
    top_off, _ = _cap(f)
    stroke = max(5, cap // 8)
    m = stroke + cap // 2                                           # room around the block for outline and shadow
    W, H = bw + 2 * m, bh + 2 * m
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow = Image.new("L", (W, H), 0)
    d, ds = ImageDraw.Draw(layer), ImageDraw.Draw(shadow)
    y_cap = m + (bh - tall) / 2
    n_words = len(text.split())
    for i, line in enumerate(lines):
        parts = line.split()
        widths = [f.getlength(p) for p in parts]
        space = f.getlength(" ")
        total = sum(widths) + space * (len(parts) - 1)
        x = m + (0 if align == "left" else bw - total if align == "right" else (bw - total) / 2)
        if banner:
            px, py = cap * 0.22, cap * 0.2
            x = min(max(x, m + px), m + bw - total - px)
            rect = (x - px, y_cap - py, x + total + px, y_cap + cap + py)
            ds.rounded_rectangle((rect[0] + cap * 0.08, rect[1] + cap * 0.1, rect[2] + cap * 0.08, rect[3] + cap * 0.1),
                                 radius=int(cap * 0.12), fill=170)
            d.rounded_rectangle(rect, radius=int(cap * 0.12), fill=(*accent, 255))
            ink = (20, 16, 14, 255) if sum(accent) > 330 else (255, 255, 255, 255)
        for j, part in enumerate(parts):
            if banner:
                d.text((x, y_cap - top_off), part, font=f, fill=ink)
            else:
                last_line = i == len(lines) - 1 and len(lines) > 1
                if style == "duo":
                    hot = last_line or (len(lines) == 1 and j == len(parts) - 1 and n_words > 1)
                else:
                    hot = i == len(lines) - 1 and j == len(parts) - 1 and n_words > 1
                colour = (*accent, 255) if hot else (255, 255, 255, 255)
                ds.text((x + cap * 0.06, y_cap - top_off + cap * 0.08), part, font=f, fill=200, stroke_width=stroke, stroke_fill=200)
                d.text((x, y_cap - top_off), part, font=f, fill=colour, stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            x += widths[j] + space
        y_cap += cap * (1 + gap)
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(4, cap // 10)))
    block = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    block.paste(Image.new("RGBA", (W, H), (0, 0, 0, 255)), (0, 0), shadow)
    block.alpha_composite(layer)
    if tilt:
        block = block.rotate(tilt, resample=Image.BICUBIC, expand=False)
    out = img.convert("RGBA")
    out.alpha_composite(block, (x0 - m, y0 - m)) if x0 - m >= 0 and y0 - m >= 0 else out.paste(block, (x0 - m, y0 - m), block)
    return out.convert("RGB")


# ---------------------------------------------------------------- whole layouts
def hero(scene, person, accent, fmt, cutout_ok):
    """The face-first thumbnail: the scene behind, darker and softer; the person big in front.
    Returns (picture, where the words go, alignment)."""
    from PIL import Image
    w, h = SIZES[fmt]
    bg = darken(grade(cover(scene, w, h, 0.5, 0.5), 1.0), 0.32, blur=2)
    if fmt == "v":
        target_h, target_w = int(h * 0.56), int(w * 0.98)
    else:
        target_h, target_w = int(h * 0.98), int(w * 0.56)
    scale = min(target_h / person.height, target_w / person.width, 6.0)
    person = person.resize((max(1, int(person.width * scale)), max(1, int(person.height * scale))), Image.LANCZOS)
    person = grade(person.convert("RGB"), 0.8).convert("RGBA") if not cutout_ok else _grade_rgba(person)
    if cutout_ok:
        art, pad = outlined(person, accent)
    else:
        art, pad = person, 0
        art = _glow_card(art, accent)
    canvas = bg.convert("RGBA")
    if fmt == "v":
        x = (w - art.width) // 2
        y = h - art.height + (pad if cutout_ok else -40)
        canvas.alpha_composite(art, (x, max(int(h * 0.36), y)))
        box, align = (90, 300, w - 90, max(560, min(int(h * 0.36) + 40, 820))), "center"
    else:
        x = w - art.width + (pad if cutout_ok else -24)
        y = h - art.height + (pad if cutout_ok else -24)
        canvas.alpha_composite(art, (max(int(w * 0.40), x), max(-pad, y)))
        box, align = (56, 56, int(w * 0.52), h - 56), "left"
    return canvas.convert("RGB"), box, align


def _grade_rgba(img):
    rgb = grade(img.convert("RGB"), 0.8)
    rgb.putalpha(img.getchannel("A"))
    return rgb


def _glow_card(card_img, accent):
    from PIL import Image, ImageFilter
    pad = 24
    out = Image.new("RGBA", (card_img.width + 2 * pad, card_img.height + 2 * pad), (0, 0, 0, 0))
    glow = Image.new("L", out.size, 0)
    glow.paste(card_img.getchannel("A"), (pad, pad))
    glow = glow.filter(ImageFilter.GaussianBlur(14)).point(lambda v: min(255, int(v * 1.5)))
    out.paste(Image.new("RGBA", out.size, (*accent, 255)), (0, 0), glow)
    out.alpha_composite(card_img, (pad, pad))
    return out


def action(frame, fmt, fx=0.5, fy=0.45, subject=0.5):
    """No face to show: the action itself, zoomed toward the subject, with the words away from it."""
    from PIL import Image
    w, h = SIZES[fmt]
    img = frame.convert("RGB")
    tall = img.height >= img.width
    zoom = 1.0 + max(0.0, 0.45 - min(subject, 0.45)) * 1.6       # a small subject gets closer, never a copy of itself
    if fmt == "v":
        if tall:
            out = cover(img, w, h, fx, fy)
        else:
            out = cover(trim(img), w, h, fx, fy, zoom)
        out = grade(out)
        box = (90, 300, w - 90, 780) if fy > 0.5 else (90, 1020, w - 90, 1400)
        return out, box, "center"
    if tall:
        from PIL import ImageFilter
        bg = darken(cover(img, w, h, fx), 0.25, blur=26)
        fg = img.copy()
        fg.thumbnail((int(w * 0.5), h), Image.LANCZOS)
        side = "right" if fx < 0.5 else "left"
        bg.paste(fg, (w - fg.width - 40 if side == "right" else 40, (h - fg.height) // 2))
        out = grade(bg)
        box = (56, 56, int(w * 0.5), h - 56) if side == "right" else (int(w * 0.52), 56, w - 56, h - 130)
        return out, box, ("left" if side == "right" else "right")
    out = grade(cover(trim(img), w, h, fx, fy, max(1.08, zoom)))
    box = (int(w * 0.5), 56, w - 56, h - 140) if fx < 0.5 else (56, 56, int(w * 0.52), h - 56)
    return out, box, ("right" if fx < 0.5 else "left")
