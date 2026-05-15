#!/usr/bin/env python3
"""
Hackles Comic Strip Translator
================================
Translates speech bubbles in Hackles comic strip images from English to Czech.

Requirements:
    pip install pillow pytesseract opencv-python deep-translator requests numpy

Also needs Tesseract OCR installed:
    - Ubuntu/Debian:  sudo apt install tesseract-ocr
    - openSUSE:       sudo zypper install tesseract-ocr
    - Windows:        https://github.com/UB-Mannheim/tesseract/wiki
    - macOS:          brew install tesseract

Usage:
    # Translate all 364 strips:
    python translate_comics.py

    # Translate strips 1-10 only:
    python translate_comics.py --start 1 --end 10

    # Translate a single strip:
    python translate_comics.py --strip 5

    # Dry run - show OCR text without saving:
    python translate_comics.py --dry-run --start 1 --end 5
"""

import os
import sys
import argparse
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import cv2
from deep_translator import GoogleTranslator
import textwrap
import io
import time

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_RAW    = "https://raw.githubusercontent.com/lusmo/hackles/main"
SRC_DIR     = "resources/strips/en_US/images"
DST_DIR     = "resources/strips/cs_CZ/images"
TOTAL_STRIPS = 364

# Font search paths (first found is used)
PREFERRED_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/google-droid/DroidSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

OCR_SCALE       = 2.0   # upscale factor before OCR
MIN_CONFIDENCE  = 30    # minimum Tesseract word confidence
MIN_BUBBLE_AREA = 600   # min white area (px^2) to count as speech bubble
ERASE_PADDING   = 5     # extra pixels to erase around text bbox
TRANSLATE_DELAY = 0.25  # seconds between translation API calls


# ── Font loader ────────────────────────────────────────────────────────────────
def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in PREFERRED_FONTS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── Image fetching ─────────────────────────────────────────────────────────────
def fetch_image(number: int) -> Image.Image | None:
    url = f"{REPO_RAW}/{SRC_DIR}/cartoon{number}.png"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        print(f"  [WARN] HTTP {r.status_code} for cartoon{number}.png")
    except Exception as e:
        print(f"  [ERROR] fetching cartoon{number}: {e}")
    return None


# ── Speech bubble detection ────────────────────────────────────────────────────
def detect_speech_bubbles(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """
    Returns list of (x, y, w, h) bounding boxes of white bubble regions.
    Uses OpenCV contour detection on a white-threshold mask.
    """
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Keep very bright pixels (speech bubble interiors are white/near-white)
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

    # Close gaps to merge broken bubble regions
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    # Erode a bit to separate touching bubbles
    mask = cv2.erode(mask, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = gray.shape
    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BUBBLE_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Skip full-image backgrounds and border strips
        if w > W * 0.85 or h > H * 0.85:
            continue
        aspect = w / max(h, 1)
        if aspect > 8 or aspect < 0.15:
            continue
        # Speech bubbles are usually in the upper half of the strip
        # (skip large white labels at bottom like "http://hackles.org")
        if y > H * 0.85:
            continue
        bubbles.append((x, y, w, h))

    return bubbles


# ── OCR ────────────────────────────────────────────────────────────────────────
def ocr_region(img: Image.Image, x: int, y: int, w: int, h: int) -> list[dict]:
    """
    OCR a rectangular region. Returns words with original-image coordinates.
    """
    region = img.crop((x, y, x + w, y + h))
    scaled = region.resize((int(w * OCR_SCALE), int(h * OCR_SCALE)), Image.LANCZOS)

    data = pytesseract.image_to_data(
        scaled, lang="eng", config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )

    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text or int(data["conf"][i]) < MIN_CONFIDENCE:
            continue
        words.append({
            "text": text,
            "conf": int(data["conf"][i]),
            "x": x + int(data["left"][i] / OCR_SCALE),
            "y": y + int(data["top"][i] / OCR_SCALE),
            "w": int(data["width"][i] / OCR_SCALE),
            "h": int(data["height"][i] / OCR_SCALE),
        })
    return words


def words_to_text(words: list[dict]) -> str:
    """Concatenate words into a single string, preserving line breaks."""
    if not words:
        return ""
    words_s = sorted(words, key=lambda w: (w["y"], w["x"]))
    lines, line = [], [words_s[0]]
    for word in words_s[1:]:
        last = line[-1]
        if abs((word["y"] + word["h"] / 2) - (last["y"] + last["h"] / 2)) < last["h"] * 0.9:
            line.append(word)
        else:
            lines.append(line)
            line = [word]
    lines.append(line)
    return " ".join(
        " ".join(w["text"] for w in sorted(ln, key=lambda w: w["x"]))
        for ln in lines
    )


# ── Translation ────────────────────────────────────────────────────────────────
def translate_text(text: str, src: str = "en", dest: str = "cs") -> str:
    if not text.strip():
        return text
    try:
        result = GoogleTranslator(source=src, target=dest).translate(text)
        return result if result else text
    except Exception as e:
        print(f"  [WARN] Translation error: {e}")
        return text


# ── Image editing ──────────────────────────────────────────────────────────────
def erase_region(draw: ImageDraw.Draw, words: list[dict]) -> tuple[int, int, int, int] | None:
    if not words:
        return None
    x0 = max(0, min(w["x"] for w in words) - ERASE_PADDING)
    y0 = max(0, min(w["y"] for w in words) - ERASE_PADDING)
    x1 = max(w["x"] + w["w"] for w in words) + ERASE_PADDING
    y1 = max(w["y"] + w["h"] for w in words) + ERASE_PADDING
    draw.rectangle([x0, y0, x1, y1], fill="white")
    return (x0, y0, x1, y1)


def draw_text_in_box(draw: ImageDraw.Draw, text: str, bbox: tuple,
                     words: list[dict]) -> None:
    if not text.strip() or not bbox:
        return
    x0, y0, x1, y1 = bbox
    box_w, box_h = x1 - x0, y1 - y0

    # Estimate font size from original word heights
    avg_h = (sum(w["h"] for w in words) / len(words)) if words else box_h / 3
    font_size = max(8, int(avg_h * 0.82))

    # Iteratively reduce font size until text fits
    for attempt in range(12):
        font = load_font(font_size)
        cpl = max(1, int(box_w / (font_size * 0.58)))
        wrapped = textwrap.wrap(text, width=cpl)
        line_h = font_size + 2
        if len(wrapped) * line_h <= box_h + 4:
            break
        font_size = max(7, font_size - 1)

    # Draw each wrapped line
    y = y0 + 2
    for line in wrapped:
        if y + font_size > y1 + 4:
            break
        draw.text((x0 + 2, y), line, fill="black", font=font)
        y += font_size + 2


# ── Strip pipeline ─────────────────────────────────────────────────────────────
def process_strip(number: int, dry_run: bool = False) -> bool:
    print(f"\n{'='*55}")
    print(f"  cartoon{number}.png")
    print(f"{'='*55}")

    img = fetch_image(number)
    if img is None:
        print("  [SKIP] Could not download image.")
        return False
    print(f"  Size: {img.width}x{img.height}px")

    bubbles = detect_speech_bubbles(img)
    print(f"  Bubbles found: {len(bubbles)}")

    if not bubbles:
        print("  [INFO] No speech bubbles — copying image unchanged.")
        if not dry_run:
            save_image(img, number)
        return True

    result = img.copy()
    draw   = ImageDraw.Draw(result)

    for bx, by, bw, bh in bubbles:
        words = ocr_region(img, bx, by, bw, bh)
        if not words:
            continue
        orig_text = words_to_text(words)
        if not orig_text.strip():
            continue
        print(f"  EN: {orig_text!r}")
        if dry_run:
            continue
        cs_text = translate_text(orig_text)
        print(f"  CS: {cs_text!r}")
        bbox = erase_region(draw, words)
        if bbox:
            draw_text_in_box(draw, cs_text, bbox, words)
        time.sleep(TRANSLATE_DELAY)

    if not dry_run:
        save_image(result, number)
        print(f"  [OK] Saved.")
    return True


def save_image(img: Image.Image, number: int) -> None:
    os.makedirs(DST_DIR, exist_ok=True)
    img.save(os.path.join(DST_DIR, f"cartoon{number}.png"), "PNG")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Translate Hackles comic strip speech bubbles EN → CS"
    )
    ap.add_argument("--start",  type=int, default=1,
                    help="First strip number (default: 1)")
    ap.add_argument("--end",    type=int, default=TOTAL_STRIPS,
                    help=f"Last strip number (default: {TOTAL_STRIPS})")
    ap.add_argument("--strip",  type=int, default=None,
                    help="Process a single strip number only")
    ap.add_argument("--dry-run", action="store_true",
                    help="OCR only — show detected text, do not save files")
    ap.add_argument("--no-skip", action="store_true",
                    help="Re-translate even if output file already exists")
    args = ap.parse_args()

    strips = [args.strip] if args.strip else list(range(args.start, args.end + 1))

    print("Hackles Comic Translator  EN → CS")
    print(f"Target: {len(strips)} strip(s)  [{strips[0]}..{strips[-1]}]")
    print(f"Output: ./{DST_DIR}/")
    if args.dry_run:
        print("MODE: DRY RUN (no files saved)")
    print()

    ok = 0
    for n in strips:
        if not args.dry_run and not args.no_skip:
            dst = os.path.join(DST_DIR, f"cartoon{n}.png")
            if os.path.exists(dst):
                print(f"  [SKIP] cartoon{n}.png already exists  (use --no-skip to redo)")
                ok += 1
                continue
        try:
            if process_strip(n, dry_run=args.dry_run):
                ok += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"  [ERROR] cartoon{n}: {e}")
        time.sleep(0.3)

    print(f"\nDone: {ok}/{len(strips)} strips OK")
    if not args.dry_run:
        print(f"\nImages saved to: ./{DST_DIR}/")
        print("\nTo commit to GitHub:")
        print("  git add resources/strips/cs_CZ/images/")
        print('  git commit -m "feat: add Czech (cs_CZ) translated comic images"')
        print("  git push")


if __name__ == "__main__":
    main()
