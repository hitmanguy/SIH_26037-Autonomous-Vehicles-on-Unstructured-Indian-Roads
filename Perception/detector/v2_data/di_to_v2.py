"""DriveIndia (released YOLO labels, IDs 0-27) -> Epsilon v2 label set (17 classes).

Run on the server after unzipping DriveIndia:
    python di_to_v2.py --src ~/epsilon_yash/data/driveindia --out ~/epsilon_yash/data/v2/driveindia

What it does
  1. Maps each DriveIndia ID to a v2 class (table MAP below; verified by eye on crop sheets, 29 Sep).
  2. Rider rule: a DriveIndia 'person' box sitting on a motorcycle/bicycle box becomes 'rider'.
  3. Noisy IDs (1 bicycle, 5 bus, 26, 27) are NOT trusted: written to noisy_boxes.jsonl for
     relabel_noisy_with_v1.py; until then the image's label file gets a marker so it can't be
     trained on by accident (image listed in hold_images.txt, left out of the split lists).
  4. Potholes (ID 18) that Yash rejected in the review (pothole_reject.txt) are dropped.
  5. Images are symlinked (no copy), labels written fresh. Writes train.txt / val.txt image lists,
     a data yaml, and a stats report.
"""
import argparse, json, os, collections

# ---- v2 class list: first 12 identical to v1 (IDD) so v1 predictions map 1:1 ----
V2 = ['person', 'rider', 'car', 'bus', 'truck', 'autorickshaw', 'motorcycle', 'bicycle',
      'animal', 'traffic sign', 'traffic light', 'vehicle fallback',
      'pothole', 'pushcart', 'tractor', 'emergency vehicle', 'cone_barrier']
C = {n: i for i, n in enumerate(V2)}

NOISY = 'noisy'
DROP = None
# DriveIndia released ID -> v2 class name (or DROP / NOISY)
MAP = {
    0: 'person',            # pedestrians AND riders (rider rule below)
    1: NOISY,               # 'bicycle' but ~75% of boxes are motorcycles
    2: 'car',
    3: 'motorcycle',
    4: 'traffic sign',      # route board
    5: NOISY,               # 'bus' mixed with cars / trucks / autos
    6: 'truck',             # commercial vehicle (tempo, mini truck)
    7: 'truck',
    8: 'traffic sign',
    9: 'traffic light',
    10: 'autorickshaw',
    11: DROP,               # empty slot in release
    12: 'emergency vehicle',  # ambulance
    13: DROP,               # empty slot in release
    14: 'vehicle fallback',   # construction vehicle
    15: 'animal',
    16: DROP, 17: DROP,     # speed bumps (1 box in whole train set)
    18: 'pothole',
    19: 'emergency vehicle',  # police
    20: 'tractor',
    21: 'pushcart',
    22: 'cone_barrier',     # temporary barrier
    23: DROP,               # rumble strips (road paint)
    24: 'cone_barrier',     # traffic cone
    25: DROP,               # zebra crossing (road paint)
    26: NOISY,              # mixed vans / SUVs / minibuses
    27: NOISY,              # mostly motorcycles
}
BIKE_IDS = {1, 3, 27}       # DriveIndia boxes that can carry a rider


def rider_test(p, b):
    """p, b = (cx, cy, w, h) normalised. True if person p sits on two-wheeler b.
    DriveIndia boxes a rider's upper body so that its bottom edge usually just touches
    (or slightly overlaps) the top of the bike box, so we test stacking, not overlap."""
    px1, px2 = p[0] - p[2] / 2, p[0] + p[2] / 2
    py2 = p[1] + p[3] / 2
    bx1, bx2 = b[0] - b[2] / 2, b[0] + b[2] / 2
    by1 = b[1] - b[3] / 2
    ix = max(0.0, min(px2, bx2) - max(px1, bx1))
    if ix < 0.5 * min(p[2], b[2]):            # mostly stacked horizontally
        return False
    if p[1] >= b[1]:                          # person centre above bike centre
        return False
    # person's bottom edge near the bike's top edge: from 25% of bike height above it
    # to 60% of bike height below it
    return by1 - 0.25 * b[3] <= py2 <= by1 + 0.60 * b[3]


def read_labels(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    for ln in open(path):
        s = ln.split()
        if len(s) >= 5:
            rows.append((int(s[0]), tuple(float(v) for v in s[1:5])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.expanduser('~/epsilon_yash/data/driveindia'))
    ap.add_argument('--out', default=os.path.expanduser('~/epsilon_yash/data/v2/driveindia'))
    ap.add_argument('--pothole-reject', default=None,
                    help='txt of "<split>/<stem> <x> <y>" lines from the pothole review')
    a = ap.parse_args()
    a.src, a.out = os.path.abspath(os.path.expanduser(a.src)), os.path.abspath(os.path.expanduser(a.out))

    splits = {'train': [('train1', 'images', 'labels'), ('train2', 'images', 'labels'),
                        ('train3', 'images', 'labels')],
              'val': [('val', 'images_2500', 'labels_2500')]}
    reject = set()
    if a.pothole_reject and os.path.isfile(a.pothole_reject):
        reject = {ln.strip() for ln in open(a.pothole_reject) if ln.strip()}

    stats = collections.Counter(); per_split = collections.defaultdict(collections.Counter)
    os.makedirs(a.out, exist_ok=True)
    noisy_out = open(os.path.join(a.out, 'noisy_boxes.jsonl'), 'w')
    hold = set()

    for split, parts in splits.items():
        img_dir = os.path.join(a.out, 'images', split); lbl_dir = os.path.join(a.out, 'labels', split)
        os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
        lst = []
        for part, idir, ldir in parts:
            src_img = os.path.join(a.src, part, idir); src_lbl = os.path.join(a.src, part, ldir)
            for fn in sorted(os.listdir(src_img)):
                if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                stem = os.path.splitext(fn)[0]
                uid = f'{part}_{stem}'                       # unique across train1/2/3
                rows = read_labels(os.path.join(src_lbl, stem + '.txt'))
                if not rows:
                    stats['images_without_labels'] += 1
                out_rows = []; noisy = []
                bikes = [b for c, b in rows if c in BIKE_IDS]
                for c, b in rows:
                    tgt = MAP.get(c, 'UNKNOWN')
                    if tgt == 'UNKNOWN':
                        stats[f'unknown_id_{c}'] += 1; continue
                    if tgt is DROP:
                        stats[f'drop_id_{c}'] += 1; continue
                    if tgt == NOISY:
                        noisy.append({'id': c, 'box': b}); stats[f'noisy_id_{c}'] += 1; continue
                    if c == 18 and f'{part}/{stem} {b[0]:.4f} {b[1]:.4f}' in reject:
                        stats['pothole_rejected'] += 1; continue
                    if tgt == 'person' and any(rider_test(b, bb) for bb in bikes):
                        tgt = 'rider'
                    out_rows.append((C[tgt], b)); per_split[split][tgt] += 1
                dst = os.path.join(img_dir, uid + os.path.splitext(fn)[1])
                if not os.path.lexists(dst):
                    os.symlink(os.path.join(src_img, fn), dst)
                with open(os.path.join(lbl_dir, uid + '.txt'), 'w') as f:
                    for c, b in out_rows:
                        f.write(f'{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n')
                if noisy:
                    noisy_out.write(json.dumps({'split': split, 'uid': uid, 'image': dst,
                                                'label': os.path.join(lbl_dir, uid + '.txt'),
                                                'boxes': noisy}) + '\n')
                    hold.add(dst)                           # held back until relabelled
                else:
                    lst.append(dst)
        with open(os.path.join(a.out, f'{split}.txt'), 'w') as f:
            f.write('\n'.join(lst) + '\n')
        stats[f'{split}_images_ready'] = len(lst)
    noisy_out.close()
    with open(os.path.join(a.out, 'hold_images.txt'), 'w') as f:
        f.write('\n'.join(sorted(hold)) + '\n')
    stats['images_held_noisy'] = len(hold)

    with open(os.path.join(a.out, 'driveindia_v2.yaml'), 'w') as f:
        f.write(f'path: {a.out}\ntrain: train.txt\nval: val.txt\nnames:\n')
        for i, n in enumerate(V2):
            f.write(f'  {i}: {n}\n')

    rep = ['== DriveIndia -> v2 ==']
    for k in sorted(stats):
        rep.append(f'{k:32s} {stats[k]}')
    for split in per_split:
        rep.append(f'-- {split} boxes per v2 class --')
        for n in V2:
            rep.append(f'  {n:20s} {per_split[split][n]}')
    txt = '\n'.join(rep); print(txt)
    open(os.path.join(a.out, 'report.txt'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
