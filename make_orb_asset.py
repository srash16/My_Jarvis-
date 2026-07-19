"""One-off: build assets/jarvis_orb.png (square, soft transparent bg) from the ring PNG.

Full 8-bit alpha (unlike GIF) so the glow edges stay smooth. The widget animates
scale (rings pushing outward/inward) itself, so no per-frame GIF is needed.
"""

from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent
SRC = BASE / "assets" / "source_orb.png"
OUT = BASE / "assets" / "jarvis_orb.png"

SIZE = 480
ALPHA_FLOOR = 6      # brightness <= this -> fully transparent
ALPHA_GAIN = 1.7     # how quickly glow becomes opaque above the floor


def main():
    img = Image.open(SRC).convert("RGB")
    w, h = img.size
    side = min(w, h)
    cx, cy = w // 2, int(h * 0.50)
    left = max(0, min(cx - side // 2, w - side))
    top = max(0, min(cy - side // 2, h - side))
    sq = img.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.LANCZOS)

    rgba = sq.convert("RGBA")
    px = rgba.load()
    for y in range(SIZE):
        for x in range(SIZE):
            r, g, b, _ = px[x, y]
            v = max(r, g, b)
            a = 0 if v <= ALPHA_FLOOR else min(255, int((v - ALPHA_FLOOR) * ALPHA_GAIN))
            px[x, y] = (r, g, b, a)

    rgba.save(OUT)
    print(f"Wrote {OUT} ({SIZE}x{SIZE}, RGBA soft alpha)")


if __name__ == "__main__":
    main()
