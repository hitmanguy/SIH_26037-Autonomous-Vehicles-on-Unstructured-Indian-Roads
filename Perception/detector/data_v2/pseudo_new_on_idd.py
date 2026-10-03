"""Add the 5 NEW v2 classes to IDD images, using the teacher model (trained on DriveIndia + RDD).

IDD (v1's data) never labelled potholes, pushcarts, tractors, emergency vehicles or cones, so
in IDD images those objects are unlabelled = "background" - which would fight the new classes
when everything is merged. The teacher finds them; confident boxes are added to IDD labels.
IDD's own labels for classes 0-11 are left untouched.

Run on the server after the teacher run:
    python pseudo_new_on_idd.py \
        --teacher ~/epsilon_yash/runs/v2_teacher_di_rdd/weights/best.pt \
        --idd ~/epsilon_yash/yolo_idd --out ~/epsilon_yash/data/v2/idd

Writes a copy of IDD's labels (IDD's own folder is never modified) with the extra boxes,
symlinks the images, and train.txt / val.txt lists.
"""
import argparse, os, glob, shutil, collections

NEW = {12: 'pothole', 13: 'pushcart', 14: 'tractor', 15: 'emergency vehicle', 16: 'cone_barrier'}
# per-class confidence: stricter where a false box would do more harm
CONF = {12: 0.45, 13: 0.5, 14: 0.5, 15: 0.6, 16: 0.5}
# classes an existing IDD box may already cover; skip a teacher box that sits on one
IDD_OVERLAP = {14: {11, 4},        # tractor may be IDD 'vehicle fallback' / 'truck'
               13: {11},           # pushcart may be IDD 'vehicle fallback'
               15: {2, 4, 11}}     # emergency vehicle may be IDD car / truck / fallback


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw = max(0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0, min(ay2, by2) - max(ay1, by1))
    i = iw * ih
    return i / (a[2]*a[3] + b[2]*b[3] - i + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--idd', default=os.path.expanduser('~/epsilon_yash/yolo_idd'))
    ap.add_argument('--out', default=os.path.expanduser('~/epsilon_yash/data/v2/idd'))
    ap.add_argument('--relabel-overlap', action='store_true',
                    help='replace an IDD fallback/truck/car box by the new class instead of skipping')
    ap.add_argument('--batch', type=int, default=32)
    a = ap.parse_args()
    a.idd, a.out = (os.path.abspath(os.path.expanduser(p)) for p in (a.idd, a.out))

    from ultralytics import YOLO
    model = YOLO(os.path.expanduser(a.teacher))
    stats = collections.Counter()
    for split in ('train', 'val'):
        idir = os.path.join(a.idd, 'images', split); ldir = os.path.join(a.idd, 'labels', split)
        if not os.path.isdir(idir):
            print('missing', idir); continue
        oi = os.path.join(a.out, 'images', split); ol = os.path.join(a.out, 'labels', split)
        os.makedirs(oi, exist_ok=True); os.makedirs(ol, exist_ok=True)
        imgs = sorted(p for p in glob.glob(os.path.join(idir, '*')) if p.lower().endswith(('.jpg', '.jpeg', '.png')))
        lst = []
        for i in range(0, len(imgs), a.batch):
            chunk = imgs[i:i + a.batch]
            res = model.predict(chunk, conf=min(CONF.values()), verbose=False)
            for img, r in zip(chunk, res):
                stem = os.path.splitext(os.path.basename(img))[0]
                src_lbl = os.path.join(ldir, stem + '.txt')
                rows = []
                if os.path.isfile(src_lbl):
                    for ln in open(src_lbl):
                        s = ln.split()
                        if len(s) >= 5:
                            rows.append([int(s[0]), tuple(float(v) for v in s[1:5])])
                for c, cf, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xywhn.tolist()):
                    c = int(c); b = tuple(b)
                    if c not in NEW or cf < CONF[c]:
                        continue
                    hit = [row for row in rows if row[0] in IDD_OVERLAP.get(c, set()) and iou(row[1], b) > 0.5]
                    if hit:
                        if a.relabel_overlap:
                            hit[0][0] = c; stats[f'{split}_relabel_{NEW[c]}'] += 1
                        else:
                            stats[f'{split}_skip_on_idd_box_{NEW[c]}'] += 1
                        continue
                    rows.append([c, b]); stats[f'{split}_added_{NEW[c]}'] += 1
                with open(os.path.join(ol, stem + '.txt'), 'w') as f:
                    for c, b in rows:
                        f.write(f'{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n')
                dst = os.path.join(oi, os.path.basename(img))
                if not os.path.lexists(dst):
                    os.symlink(os.path.realpath(img), dst)
                lst.append(dst)
            print(f'{split}: {min(i + a.batch, len(imgs))}/{len(imgs)}', end='\r')
        with open(os.path.join(a.out, f'{split}.txt'), 'w') as f:
            f.write('\n'.join(lst) + '\n')
        stats[f'{split}_images'] = len(lst)
        print()
    rep = ['== teacher pseudo-labels of new classes on IDD =='] + [f'{k:40s} {v}' for k, v in sorted(stats.items())]
    txt = '\n'.join(rep); print(txt)
    open(os.path.join(a.out, 'report.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
