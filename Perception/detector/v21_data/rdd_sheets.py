"""Review sheets for v2.1: which RDD2022 countries match our car-camera view, and what D20 looks like.

Run on the server:
    python rdd_sheets.py            # -> ~/epsilon_yash/review/*.jpg + counts printed
Then copy the jpgs to the laptop:
    scp "<user>@<server>:~/epsilon_yash/review/*.jpg" <local folder>

Sheets
  view_<country>.jpg   20 random full frames (with D40 or D20), red = D40 pothole, orange = D20 alligator
  d20_<country>.jpg    30 random D20 crops (what 'damaged_road' would learn)
Counts: images and boxes per damage code per country (train split only; test has no labels).
"""
import os, glob, random, collections, argparse, xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=os.path.expanduser('~/epsilon_yash/data/rdd2022'))
ap.add_argument('--out', default=os.path.expanduser('~/epsilon_yash/review'))
ap.add_argument('--countries', nargs='+', default=None, help='default: every extracted country folder')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
COL = {'D40': (255, 0, 0), 'D20': (255, 150, 0)}

countries = a.countries or sorted(d for d in os.listdir(a.src)
                                  if os.path.isdir(os.path.join(a.src, d, d, 'train', 'annotations', 'xmls')))
print('countries found:', countries)


def parse(p):
    r = ET.parse(p).getroot()
    out = []
    for o in r.findall('object'):
        bb = o.find('bndbox')
        out.append((o.find('name').text.strip(), [float(bb.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')]))
    return r.find('filename').text, out


print(f"\n{'country':18s} {'images':>7s} {'img w/ D40':>10s} {'D40':>6s} {'img w/ D20':>10s} {'D20':>6s}  size(s)")
for c in countries:
    base = os.path.join(a.src, c, c, 'train')
    items, cnt, sizes = [], collections.Counter(), collections.Counter()
    for x in sorted(glob.glob(os.path.join(base, 'annotations', 'xmls', '*.xml'))):
        fn, objs = parse(x)
        img = os.path.join(base, 'images', fn)
        if not os.path.isfile(img):
            img = os.path.join(base, 'images', os.path.basename(x)[:-4] + '.jpg')
        names = [n for n, _ in objs]
        cnt['images'] += 1
        for code in ('D40', 'D20'):
            k = names.count(code)
            cnt[code] += k; cnt['img_' + code] += k > 0
        if any(n in COL for n in names) and os.path.isfile(img):
            items.append((img, objs))
    rnd = random.Random(0)
    # --- full-frame view sheet (4 x 5)
    pick = rnd.sample(items, min(20, len(items)))
    T = 320; sheet = Image.new('RGB', (5 * T, 4 * T + 30), 'white'); d = ImageDraw.Draw(sheet)
    d.text((8, 8), f'RDD2022 {c}: 20 random frames with damage (red D40 pothole, orange D20 alligator)', fill='black')
    for k, (img, objs) in enumerate(pick):
        im = Image.open(img).convert('RGB'); sizes[im.size] += 1
        dd = ImageDraw.Draw(im); lw = max(3, im.size[0] // 200)
        for n, b in objs:
            if n in COL:
                dd.rectangle(b, outline=COL[n], width=lw)
        im.thumbnail((T - 6, T - 6))
        sheet.paste(im, ((k % 5) * T + 3, (k // 5) * T + 33))
    sheet.save(os.path.join(a.out, f'view_{c}.jpg'), quality=85)
    # --- D20 crop sheet (5 x 6)
    d20 = [(img, b) for img, objs in items for n, b in objs if n == 'D20']
    pick = rnd.sample(d20, min(30, len(d20)))
    S = 240; sheet = Image.new('RGB', (5 * S, 6 * S + 30), 'white'); d = ImageDraw.Draw(sheet)
    d.text((8, 8), f'RDD2022 {c}: 30 random D20 (alligator cracking) boxes = would become damaged_road', fill='black')
    for k, (img, (x1, y1, x2, y2)) in enumerate(pick):
        im = Image.open(img).convert('RGB'); W, H = im.size
        m = max(x2 - x1, y2 - y1) * 0.5 + 30
        cx1, cy1 = max(0, x1 - m), max(0, y1 - m)
        cr = im.crop((cx1, cy1, min(W, x2 + m), min(H, y2 + m)))
        ImageDraw.Draw(cr).rectangle((x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1), outline=COL['D20'], width=4)
        cr.thumbnail((S - 6, S - 6))
        sheet.paste(cr, ((k % 5) * S + 3, (k // 5) * S + 33))
    sheet.save(os.path.join(a.out, f'd20_{c}.jpg'), quality=85)
    sz = ', '.join(f'{w}x{h}' for (w, h), _ in sizes.most_common(2))
    print(f"{c:18s} {cnt['images']:7d} {cnt['img_D40']:10d} {cnt['D40']:6d} {cnt['img_D20']:10d} {cnt['D20']:6d}  {sz}")
print('\nsheets in', a.out)
