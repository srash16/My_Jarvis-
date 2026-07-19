"""One-off: build assets/jarvis_orb.gif (rotating, transparent bg) from the teal ring PNG."""

from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent
SRC = BASE / "assets" / "source_orb.png"
OUT = BASE / "assets" / "jarvis_orb.gif"

SIZE = 400            # output square px
FRAMES = 36           # rotation steps
FRAME_MS = 55         # per-frame duration
BLACK_THRESH = 34     # brightness <= this -> transparent background
TRANSPARENT_INDEX = 0


def load_square() -> Image.Image:
    img = Image.open(SRC).convert("RGB")
    w, h = img.size
    side = min(w, h)
    # center-crop on the ring (roughly vertical center of the portrait frame)
    cx, cy = w // 2, int(h * 0.50)
    left = max(0, min(cx - side // 2, w - side))
    top = max(0, min(cy - side // 2, h - side))
    sq = img.crop((left, top, left + side, top + side))
    return sq.resize((SIZE, SIZE), Image.LANCZOS)


def to_transparent_p(frame_rgb: Image.Image) -> Image.Image:
    px = frame_rgb.load()
    w, h = frame_rgb.size
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if max(r, g, b) <= BLACK_THRESH:
                px[x, y] = (0, 0, 0)
    # Reserve palette index 0 for pure black -> transparent
    p = frame_rgb.convert("P", palette=Image.ADAPTIVE, colors=255)
    p = p.point(lambda i: i + 1)  # shift indices up by 1, freeing index 0
    pal = p.getpalette()
    pal = [0, 0, 0] + pal[: 255 * 3]
    p.putpalette(pal)
    # Any pixel that is pure black in RGB -> index 0
    mask = frame_rgb.convert("L").point(lambda v: 0 if v <= BLACK_THRESH else 255, "1")
    black_layer = Image.new("P", frame_rgb.size, TRANSPARENT_INDEX)
    p.paste(black_layer, (0, 0), Image.eval(mask, lambda v: 255 - v).convert("1"))
    return p


def main():
    base = load_square()
    frames = []
    for i in range(FRAMES):
        angle = (360.0 / FRAMES) * i
        rot = base.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
        frames.append(to_transparent_p(rot))

    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_MS,
        loop=0,
        transparency=TRANSPARENT_INDEX,
        disposal=2,
        optimize=False,
    )
    print(f"Wrote {OUT} ({FRAMES} frames, {SIZE}x{SIZE})")


if __name__ == "__main__":
    main()
