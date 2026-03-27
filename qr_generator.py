"""
qr_generator.py — Generate a print-ready QR code for the STEM Fair booth.

Run ONCE after your Streamlit app is live:
    python qr_generator.py https://your-app-name.streamlit.app

Output: assets/booth_qr.png
        Print at 8×8 inches at 300 DPI for booth display.
        Or print smaller on an A4 sheet — the QR still scans cleanly.
"""

import os
import sys

try:
    import qrcode
    from qrcode.image.styledpil import StyledPilImage
    from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Missing dependencies. Run: pip install qrcode[pil] Pillow")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────

FILL_COLOR = "#0a0f1e"   # dark navy — matches dashboard
BACK_COLOR = "#ffffff"   # white background — required for scanner contrast
TITLE_LINE = "Folsom Air Quality Monitor"
EVENT_LINE = "FLC Los Rios STEM Fair 2026"
INSTR_LINE = "Scan to view the live AQI forecast →"


def _best_font(size: int) -> ImageFont.ImageFont:
    """Try to load a clean system font; fall back to PIL default."""
    candidates = [
        # Windows
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SF-Pro-Text-Regular.otf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # PIL default (no size arg in older Pillow)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def generate_booth_qr(url: str, output_path: str = "assets/booth_qr.png"):
    """
    Generate a high-quality, print-ready QR code PNG with branding text.
    """
    print(f"Generating QR code for: {url}")

    # ── Build QR code ─────────────────────────────────────────────────────
    qr = qrcode.QRCode(
        version=None,                                  # auto-size to content
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # 30% damage tolerance
        box_size=20,                                   # large pixels for print
        border=4,                                      # 4-module quiet zone (minimum per spec)
    )
    qr.add_data(url)
    qr.make(fit=True)

    qr_img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=RoundedModuleDrawer(),
        fill_color=FILL_COLOR,
        back_color=BACK_COLOR,
    )
    qr_img = qr_img.convert("RGB")

    qr_w, qr_h = qr_img.size

    # ── Build canvas with text area below QR ──────────────────────────────
    padding     = 60                      # side padding
    text_height = 340                     # vertical space for text block
    canvas_w    = qr_w + padding * 2
    canvas_h    = qr_h + padding + text_height + 40

    canvas = Image.new("RGB", (canvas_w, canvas_h), BACK_COLOR)

    # Paste QR centred horizontally with top padding
    qr_x = (canvas_w - qr_w) // 2
    canvas.paste(qr_img, (qr_x, padding))

    draw = ImageDraw.Draw(canvas)
    cx   = canvas_w // 2             # horizontal center
    y0   = qr_h + padding + 30       # text block starts here

    # ── Decorative line separator ─────────────────────────────────────────
    line_w = qr_w // 2
    draw.rectangle(
        [cx - line_w // 2, y0, cx + line_w // 2, y0 + 3],
        fill="#0a0f1e",
    )
    y0 += 28

    # ── Title ─────────────────────────────────────────────────────────────
    f_title = _best_font(52)
    draw.text((cx, y0), TITLE_LINE, fill="#0a0f1e", anchor="mt", font=f_title)
    bbox = draw.textbbox((cx, y0), TITLE_LINE, anchor="mt", font=f_title)
    y0  += (bbox[3] - bbox[1]) + 20

    # ── Event line ────────────────────────────────────────────────────────
    f_event = _best_font(36)
    draw.text((cx, y0), EVENT_LINE, fill="#374151", anchor="mt", font=f_event)
    bbox = draw.textbbox((cx, y0), EVENT_LINE, anchor="mt", font=f_event)
    y0  += (bbox[3] - bbox[1]) + 22

    # ── Instruction line ──────────────────────────────────────────────────
    f_instr = _best_font(30)
    draw.text((cx, y0), INSTR_LINE, fill="#6b7280", anchor="mt", font=f_instr)
    bbox = draw.textbbox((cx, y0), INSTR_LINE, anchor="mt", font=f_instr)
    y0  += (bbox[3] - bbox[1]) + 18

    # ── URL (small, for manual entry) ─────────────────────────────────────
    f_url = _best_font(24)
    draw.text((cx, y0), url, fill="#9ca3af", anchor="mt", font=f_url)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path, dpi=(300, 300), quality=95)

    print(f"✅  QR code saved → {output_path}")
    print(f"    Canvas size : {canvas.size[0]} × {canvas.size[1]} px")
    print(f"    Print size  : {canvas.size[0]/300:.1f} × {canvas.size[1]/300:.1f} inches at 300 DPI")
    print(f"    URL encoded : {url}")
    print()
    print("Next steps:")
    print("  1. Commit assets/booth_qr.png to your GitHub repo")
    print("  2. Print at a copy shop — request '8×8 inches, no scaling'")
    print("  3. Ensure at least 5mm white border around the QR when mounted")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qr_generator.py https://your-app-name.streamlit.app")
        sys.exit(1)
    generate_booth_qr(sys.argv[1])
