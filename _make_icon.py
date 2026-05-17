"""Draw the Anemon app icon as a 512x512 PNG."""
from PIL import Image, ImageDraw, ImageFont, ImageFilter

S = 512
out = '/notebooks/viz-qcc/public/anemon-icon.png'

img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# Rounded-corner background
radius = 96
d.rounded_rectangle([0, 0, S - 1, S - 1], radius=radius, fill=(10, 10, 10, 255))

# Outer ring  (radius 200, stroke 18)
cx, cy = S // 2, S // 2 - 18
def ring(r, stroke, color):
    bbox = [cx - r, cy - r, cx + r, cy + r]
    d.ellipse(bbox, outline=color, width=stroke)

ring(200, 22, (107, 191, 255, 255))   # #6bf
ring(120, 22, (107, 255, 153, 255))   # #6f9

# Center dot
dot_r = 28
d.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=(107, 191, 255, 255))

# Glow under rings for some depth
glow = Image.new('RGBA', (S, S), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
gd.ellipse([cx - 220, cy - 220, cx + 220, cy + 220], outline=(107, 191, 255, 60), width=6)
glow = glow.filter(ImageFilter.GaussianBlur(8))
img = Image.alpha_composite(img, glow)

# Title text "ANEMON"
d = ImageDraw.Draw(img)
font_paths = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf',
]
font = None
for fp in font_paths:
    try:
        font = ImageFont.truetype(fp, 64)
        break
    except Exception:
        continue
if font is None:
    font = ImageFont.load_default()

text = 'ANEMON'
bbox = d.textbbox((0, 0), text, font=font)
tw = bbox[2] - bbox[0]
th = bbox[3] - bbox[1]
d.text(((S - tw) // 2, S - 110), text, font=font, fill=(232, 232, 232, 255))

img.save(out, format='PNG', optimize=True)
print(f'wrote {out} ({img.size})')

# Smaller versions
for sz in (192, 180):
    small = img.resize((sz, sz), Image.LANCZOS)
    p = out.replace('anemon-icon.png', f'anemon-icon-{sz}.png')
    small.save(p, format='PNG', optimize=True)
    print(f'wrote {p}')
