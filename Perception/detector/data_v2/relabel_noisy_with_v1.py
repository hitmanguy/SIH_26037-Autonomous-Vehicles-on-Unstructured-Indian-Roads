"""Fix DriveIndia's noisy boxes (IDs 1 'bicycle', 5 'bus', 26 mixed vans, 27 mostly bikes) using v1.

Run on the server after di_to_v2.py:
    python relabel_noisy_with_v1.py \
        --v2dir ~/epsilon_yash/data/v2/driveindia \
        --weights ~/epsilon_yash/runs/c3_yolov8s_e60/weights/best.pt

For every noisy box: run v1 on the image, find v1's best-overlapping detection among the
allowed classes, and if IoU >= --iou take v1's class. If no v1 box matches, the box can't be
trusted -> the whole image stays held back (an unlabelled object is worse than a missing image).
Resolved images get their boxes appended to the label file and are added to train.txt / val.txt.
"""
import argparse, json, os, collections, random
from ultralytics import YOLO

# v1 / v2 shared class ids (first 12 are identical)
PERSON, RIDER, CAR, BUS, TRUCK, AUTO, MOTO, BICY, ANIMAL, SIGN, LIGHT, FALLBACK = range(12)
ALLOWED = {
    1: {MOTO, BICY},                               # 'bicycle' (mostly motorbikes)
    27: {MOTO, BICY},                              # mostly motorbikes
    5: {BUS, TRUCK, CAR, AUTO, FALLBACK},          # 'bus' mixed
    26: {CAR, BUS, TRUCK, FALLBACK},               # vans / SUVs / minibuses
}


def iou_xywhn(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2dir', default=os.path.expanduser('~/epsilon_yash/data/v2/driveindia'))
    ap.add_argument('--weights', default=os.path.expanduser('~/epsilon_yash/runs/c3_yolov8s_e60/weights/best.pt'))
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()
    a.v2dir = os.path.abspath(os.path.expanduser(a.v2dir))

    recs = [json.loads(l) for l in open(os.path.join(a.v2dir, 'noisy_boxes.jsonl')) if l.strip()]
    model = YOLO(os.path.expanduser(a.weights))
    names = model.names
    stats = collections.Counter(); moves = collections.Counter()
    ready = collections.defaultdict(list); still_held = []; samples = []

    for i in range(0, len(recs), a.batch):
        chunk = recs[i:i + a.batch]
        results = model.predict([r['image'] for r in chunk], conf=a.conf, verbose=False, stream=False)
        for r, res in zip(chunk, results):
            dets = [(int(c), tuple(float(v) for v in xywhn))
                    for c, xywhn in zip(res.boxes.cls.tolist(), res.boxes.xywhn.tolist())]
            new_rows, ok = [], True
            for nb in r['boxes']:
                src, box = nb['id'], tuple(nb['box'])
                best, best_iou = None, 0.0
                for c, d in dets:
                    if c in ALLOWED[src]:
                        v = iou_xywhn(box, d)
                        if v > best_iou:
                            best, best_iou = c, v
                if best is not None and best_iou >= a.iou:
                    new_rows.append((best, box)); moves[(src, names[best])] += 1
                    stats['boxes_resolved'] += 1
                    if len(samples) < 60:
                        samples.append({'image': r['image'], 'src_id': src, 'v1': names[best],
                                        'iou': round(best_iou, 2), 'box': box})
                else:
                    ok = False; stats['boxes_unresolved'] += 1; moves[(src, 'UNRESOLVED')] += 1
            if ok:
                with open(r['label'], 'a') as f:
                    for c, b in new_rows:
                        f.write(f'{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n')
                ready[r['split']].append(r['image']); stats['images_released'] += 1
            else:
                still_held.append(r['image']); stats['images_still_held'] += 1
        print(f'{min(i + a.batch, len(recs))}/{len(recs)} images', end='\r')

    for split, imgs in ready.items():
        with open(os.path.join(a.v2dir, f'{split}.txt'), 'a') as f:
            f.write('\n'.join(imgs) + '\n')
    with open(os.path.join(a.v2dir, 'hold_images.txt'), 'w') as f:
        f.write('\n'.join(sorted(still_held)) + '\n')
    with open(os.path.join(a.v2dir, 'relabel_samples.json'), 'w') as f:
        json.dump(samples, f, indent=1)

    rep = ['== v1-assisted relabel of noisy DriveIndia boxes ==']
    rep += [f'{k:22s} {v}' for k, v in sorted(stats.items())]
    rep.append('-- DriveIndia id -> v1 class --')
    rep += [f'  id {s:>2} -> {t:18s} {n}' for (s, t), n in sorted(moves.items())]
    txt = '\n'.join(rep); print('\n' + txt)
    with open(os.path.join(a.v2dir, 'report.txt'), 'a') as f:
        f.write('\n' + txt + '\n')


if __name__ == '__main__':
    main()
