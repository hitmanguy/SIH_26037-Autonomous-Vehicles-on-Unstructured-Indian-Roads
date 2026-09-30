"""Auto-box the Kaggle speed-bump photos (no labels) for v2.1 class 17 'slow_zone'.

Two steps, run on the server:
  1) python bump_autolabel.py propose
       YOLO-World (open-vocabulary YOLO, prompt "speed bump") draws candidate boxes,
       numbered review sheets -> ~/epsilon_yash/data/v21/bump/review/bump_review_*.jpg
  (or hand boxes: MATLAB Image Labeler -> bump_export.m -> bump_manual.jsonl, then
     python bump_autolabel.py finalize --cands ~/epsilon_yash/bump_manual.jsonl --reject none)
  2) Yash writes the numbers of bad ones into ~/epsilon_yash/bump_reject.txt, then
     python bump_autolabel.py finalize --reject ~/epsilon_yash/bump_reject.txt
       kept images -> YOLO labels (slow_zone + v2 auto-labels for vehicles/people at conf 0.5),
       90/10 split, train.txt / val.txt
"""
import argparse, os, glob, json, random, collections
from PIL import Image, ImageDraw

V21 = ['person', 'rider', 'car', 'bus', 'truck', 'autorickshaw', 'motorcycle', 'bicycle',
       'animal', 'traffic sign', 'traffic light', 'vehicle fallback',
       'pothole', 'pushcart', 'tractor', 'emergency vehicle', 'cone_barrier', 'slow_zone']
SLOW = 17
H = os.path.expanduser('~/epsilon_yash')

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['propose', 'finalize'])
ap.add_argument('--src', default=f'{H}/data/speed_bump_kaggle/bump')
ap.add_argument('--out', default=f'{H}/data/v21/bump')
ap.add_argument('--world', default='yolov8s-worldv2.pt')
ap.add_argument('--prompts', default='speed bump,speed breaker,road hump')
ap.add_argument('--conf', type=float, default=0.05)
ap.add_argument('--v2', default=f'{H}/runs/v2_final/weights/best.pt')
ap.add_argument('--reject', default=f'{H}/bump_reject.txt')
ap.add_argument('--cands', default=None, help='finalize from this jsonl instead (e.g. hand boxes from MATLAB bump_export.m)')
a = ap.parse_args()
cand_path = a.cands or os.path.join(a.out, 'candidates.jsonl')

from ultralytics import YOLO

if a.mode == 'propose':
    imgs = sorted(p for p in glob.glob(os.path.join(a.src, '*')) if p.lower().endswith(('.jpg', '.jpeg', '.png')))
    print(len(imgs), 'images in', a.src)
    m = YOLO(a.world)
    m.set_classes([s.strip() for s in a.prompts.split(',')])
    os.makedirs(os.path.join(a.out, 'review'), exist_ok=True)
    cands = []
    for n, p in enumerate(imgs, 1):
        r = m.predict(p, conf=a.conf, agnostic_nms=True, verbose=False)[0]
        bx = sorted(zip(r.boxes.conf.tolist(), r.boxes.xyxyn.tolist()), reverse=True)
        keep = [(c, b) for c, b in bx if c >= 0.5 * bx[0][0]][:3] if bx else []
        cands.append({'n': n, 'image': p, 'boxes': [b + [c] for c, b in keep]})
    with open(cand_path, 'w') as f:
        for c in cands:
            f.write(json.dumps(c) + '\n')
    # numbered sheets, 30 per sheet
    T, cols, per = 320, 5, 30
    for sh in range((len(cands) + per - 1) // per):
        chunk = cands[sh * per:(sh + 1) * per]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * T, rows * T), 'white'); d = ImageDraw.Draw(sheet)
        for k, c in enumerate(chunk):
            im = Image.open(c['image']).convert('RGB'); W, Hh = im.size
            dd = ImageDraw.Draw(im); lw = max(3, W // 150)
            for x1, y1, x2, y2, cf in c['boxes']:
                dd.rectangle((x1 * W, y1 * Hh, x2 * W, y2 * Hh), outline=(0, 255, 0), width=lw)
            im.thumbnail((T - 8, T - 8))
            px, py = (k % cols) * T, (k // cols) * T
            sheet.paste(im, (px + 4, py + 4))
            tag = f"#{c['n']}  " + (' '.join(f'{b[4]:.2f}' for b in c['boxes']) if c['boxes'] else 'NO BOX')
            d.rectangle((px + 4, py + 4, px + 8 + 7 * len(tag), py + 22), fill='black')
            d.text((px + 8, py + 7), tag, fill='yellow' if c['boxes'] else 'red')
        sheet.save(os.path.join(a.out, 'review', f'bump_review_{sh + 1}.jpg'), quality=85)
    nb = sum(1 for c in cands if not c['boxes'])
    print(f'{len(cands)} images, {nb} with NO BOX -> sheets in {a.out}/review')

else:
    cands = [json.loads(l) for l in open(cand_path) if l.strip()]
    rej = set()
    if os.path.isfile(a.reject):
        rej = {int(t) for t in open(a.reject).read().replace(',', ' ').split() if t.strip().isdigit()}
    items = [c for c in cands if c['boxes'] and c['n'] not in rej]
    print(f'{len(cands)} candidates, {len(rej)} rejected, {sum(1 for c in cands if not c["boxes"])} no box -> keep {len(items)}')
    v2 = YOLO(a.v2)
    stats = collections.Counter()
    random.Random(0).shuffle(items)
    nval = int(round(0.1 * len(items)))
    for split, part in (('val', items[:nval]), ('train', items[nval:])):
        idir = os.path.join(a.out, 'images', split); ldir = os.path.join(a.out, 'labels', split)
        os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
        lst = []
        for c in part:
            stem = 'kb_' + os.path.splitext(os.path.basename(c['image']))[0]
            dst = os.path.join(idir, stem + os.path.splitext(c['image'])[1])
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(c['image']), dst)
            rows = [(SLOW, ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)) for x1, y1, x2, y2, _ in c['boxes']]
            r = v2.predict(c['image'], conf=0.5, verbose=False)[0]
            for cl, b in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist()):
                rows.append((int(cl), tuple(b))); stats[f'v2_{V21[int(cl)]}'] += 1
            with open(os.path.join(ldir, stem + '.txt'), 'w') as f:
                for cl, b in rows:
                    f.write(f'{cl} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n')
            stats[f'{split}_slow_zone'] += len(c['boxes']); lst.append(dst)
        with open(os.path.join(a.out, f'{split}.txt'), 'w') as f:
            f.write('\n'.join(lst) + '\n')
        stats[f'{split}_images'] = len(lst)
    txt = '\n'.join(['== Kaggle bumps -> v2.1 =='] + [f'{k:28s} {v}' for k, v in sorted(stats.items())])
    print(txt); open(os.path.join(a.out, 'report.txt'), 'w').write(txt + '\n')
