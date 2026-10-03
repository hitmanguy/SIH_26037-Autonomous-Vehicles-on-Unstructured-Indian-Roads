"""Review sheet of boxes added by pseudo_new.py (reads <out>/added.csv).
   python added_sheet.py data/v21/driveindia_p pothole      -> review/added_pothole.jpg (30 random, numbered)
"""
import sys, csv, random, os
from PIL import Image, ImageDraw
out, name = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open(os.path.join(out, 'added.csv'))) if r['name'] == name]
print(len(rows), name, 'boxes added;', len({r['image'] for r in rows}), 'images')
pick = random.Random(0).sample(rows, min(30, len(rows)))
S = 300; sheet = Image.new('RGB', (5 * S, 6 * S), 'white'); d = ImageDraw.Draw(sheet)
with open(f'review/added_{name}.txt', 'w') as f:
    for k, r in enumerate(pick, 1):
        im = Image.open(r['image']).convert('RGB'); W, H = im.size
        cx, cy, w, h = (float(r[x]) for x in ('cx', 'cy', 'w', 'h'))
        x1, y1, x2, y2 = (cx - w/2) * W, (cy - h/2) * H, (cx + w/2) * W, (cy + h/2) * H
        m = max(x2 - x1, y2 - y1) * 1.5 + 60
        cr = im.crop((max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)))
        ox, oy = max(0, x1 - m), max(0, y1 - m)
        ImageDraw.Draw(cr).rectangle((x1 - ox, y1 - oy, x2 - ox, y2 - oy), outline=(255, 0, 0), width=4)
        cr.thumbnail((S - 6, S - 24)); px, py = ((k - 1) % 5) * S, ((k - 1) // 5) * S
        sheet.paste(cr, (px + 3, py + 22)); d.text((px + 5, py + 4), f"#{k}  {float(r['conf']):.2f}", fill='black')
        f.write(f"{k} {r['image']} conf {r['conf']}\n")
sheet.save(f'review/added_{name}.jpg', quality=85); print('sheet: review/added_' + name + '.jpg')
