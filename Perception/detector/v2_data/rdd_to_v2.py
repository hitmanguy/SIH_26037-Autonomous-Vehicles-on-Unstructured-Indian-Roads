"""RDD2022 (Pascal-VOC XML) -> Epsilon v2 YOLO labels. Keeps only potholes (D40 -> 'pothole').

Run on the server:
    python rdd_to_v2.py --countries India \
        --weights ~/epsilon_yash/runs/c3_yolov8s_e60/weights/best.pt

- Reads <country>/<country>/train/annotations/xmls/*.xml (RDD's own layout).
- D40 -> v2 class 12 'pothole'. Cracks (D00/D10/D20) and other codes are dropped: not our classes.
- RDD only labels road damage, so cars/people/bikes in these images are unlabelled. With
  --weights, v1 auto-labels them (classes 0-11, conf >= --pseudo-conf) so they don't teach
  "car = background". Pseudo-labels are written to the same txt; counts go in the report.
- Only images that contain >= 1 pothole are kept by default (--keep-empty to keep all).
- Deterministic 90/10 train/val split per country (--val-frac).
"""
import argparse, os, random, collections, xml.etree.ElementTree as ET

POTHOLE = 12


def parse_xml(p):
    r = ET.parse(p).getroot()
    W = float(r.find('size/width').text); H = float(r.find('size/height').text)
    boxes = []
    for o in r.findall('object'):
        name = o.find('name').text.strip()
        bb = o.find('bndbox')
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
        x1, x2 = max(0.0, min(x1, x2)), min(W, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(H, max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append((name, ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)))
    return r.find('filename').text, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.expanduser('~/epsilon_yash/data/rdd2022'))
    ap.add_argument('--out', default=os.path.expanduser('~/epsilon_yash/data/v2/rdd'))
    ap.add_argument('--countries', nargs='+', default=['India'])
    ap.add_argument('--weights', default=None, help='v1 best.pt for pseudo-labelling classes 0-11')
    ap.add_argument('--pseudo-conf', type=float, default=0.5)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--keep-empty', action='store_true')
    a = ap.parse_args()
    a.src, a.out = (os.path.abspath(os.path.expanduser(p)) for p in (a.src, a.out))

    stats = collections.Counter(); names_seen = collections.Counter()
    items = []                                           # (country, img_path, uid, [(cls, box)])
    for c in a.countries:
        base = os.path.join(a.src, c, c, 'train')
        xdir, idir = os.path.join(base, 'annotations', 'xmls'), os.path.join(base, 'images')
        for fn in sorted(os.listdir(xdir)):
            if not fn.endswith('.xml'):
                continue
            img_name, boxes = parse_xml(os.path.join(xdir, fn))
            img = os.path.join(idir, img_name)
            if not os.path.isfile(img):
                img = os.path.join(idir, fn[:-4] + '.jpg')
            if not os.path.isfile(img):
                stats['missing_image'] += 1; continue
            for n, _ in boxes:
                names_seen[n] += 1
            rows = [(POTHOLE, b) for n, b in boxes if n == 'D40']
            if not rows and not a.keep_empty:
                stats['skipped_no_pothole'] += 1; continue
            items.append((c, img, f'rdd_{c}_{os.path.splitext(img_name)[0]}', rows))
            stats[f'{c}_images'] += 1; stats[f'{c}_potholes'] += len(rows)

    if a.weights:                                         # pseudo-label road users with v1
        from ultralytics import YOLO
        model = YOLO(os.path.expanduser(a.weights))
        pc = collections.Counter()
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            res = model.predict([it[1] for it in chunk], conf=a.pseudo_conf, verbose=False)
            for it, r in zip(chunk, res):
                for cl, b in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist()):
                    it[3].append((int(cl), tuple(b))); pc[model.names[int(cl)]] += 1
            print(f'pseudo-labelling {min(i + 16, len(items))}/{len(items)}', end='\r')
        for k, v in pc.items():
            stats[f'pseudo_{k}'] = v

    rnd = random.Random(0); rnd.shuffle(items)
    nval = int(round(len(items) * a.val_frac))
    for split, part in (('val', items[:nval]), ('train', items[nval:])):
        idir = os.path.join(a.out, 'images', split); ldir = os.path.join(a.out, 'labels', split)
        os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
        lst = []
        for c, img, uid, rows in part:
            dst = os.path.join(idir, uid + os.path.splitext(img)[1])
            if not os.path.lexists(dst):
                os.symlink(img, dst)
            with open(os.path.join(ldir, uid + '.txt'), 'w') as f:
                for cl, b in rows:
                    f.write(f'{cl} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n')
            lst.append(dst)
        with open(os.path.join(a.out, f'{split}.txt'), 'w') as f:
            f.write('\n'.join(lst) + '\n')
        stats[f'{split}_images'] = len(lst)

    rep = ['== RDD2022 -> v2 (potholes only) =='] + [f'{k:28s} {v}' for k, v in sorted(stats.items())]
    rep += ['-- damage codes seen (only D40 kept) --'] + [f'  {k:6s} {v}' for k, v in sorted(names_seen.items())]
    txt = '\n'.join(rep); print('\n' + txt)
    open(os.path.join(a.out, 'report.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
