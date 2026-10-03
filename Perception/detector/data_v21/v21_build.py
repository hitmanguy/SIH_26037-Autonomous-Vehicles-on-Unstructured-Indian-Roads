"""v2.1 data build (run on the server after rdd_to_v2.py --out data/v21/rdd).

  1. DriveIndia: copy v2 labels + add class 17 'slow_zone' from DriveIndia IDs 23 (rumble strips)
     and 25 (zebra crossing) -> data/v21/driveindia (v2 folder untouched, images symlinked)
  2. init18.pt: v2 best.pt with the class head grown 17 -> 18 (old classes keep their weights,
     slow_zone starts fresh), so training starts from v2 instead of from scratch
  3. yaml files: teacher21.yaml, v21.yaml, val_*.yaml
"""
import argparse, os, glob, copy, collections
H = os.path.expanduser('~/epsilon_yash')
V21 = ['person', 'rider', 'car', 'bus', 'truck', 'autorickshaw', 'motorcycle', 'bicycle',
       'animal', 'traffic sign', 'traffic light', 'vehicle fallback',
       'pothole', 'pushcart', 'tractor', 'emergency vehicle', 'cone_barrier', 'slow_zone']
SLOW_SRC = {23: 'rumble', 25: 'zebra'}

ap = argparse.ArgumentParser()
ap.add_argument('--v2di', default=f'{H}/data/v2/driveindia')
ap.add_argument('--src', default=f'{H}/data/driveindia')
ap.add_argument('--v21', default=f'{H}/data/v21')
ap.add_argument('--v2pt', default=f'{H}/runs/v2_final/weights/best.pt')
a = ap.parse_args()
out = os.path.join(a.v21, 'driveindia')
PARTS = {'train': ['train1', 'train2', 'train3'], 'val': ['val']}
stats = collections.Counter()

# ---- 1. DriveIndia + slow_zone
for split, parts in PARTS.items():
    li = os.path.join(a.v2di, 'images', split); ll = os.path.join(a.v2di, 'labels', split)
    oi = os.path.join(out, 'images', split); ol = os.path.join(out, 'labels', split)
    os.makedirs(oi, exist_ok=True); os.makedirs(ol, exist_ok=True)
    imgs = {os.path.splitext(n)[0]: n for n in os.listdir(li)}
    for lf in glob.glob(os.path.join(ll, '*.txt')):
        uid = os.path.basename(lf)[:-4]
        part = next(p for p in parts if uid.startswith(p + '_'))
        stem = uid[len(part) + 1:]
        rows = [ln.rstrip('\n') for ln in open(lf) if ln.strip()]
        sl = os.path.join(a.src, part, 'labels_2500' if part == 'val' else 'labels', stem + '.txt')
        if os.path.isfile(sl):
            for ln in open(sl):
                s = ln.split()
                if len(s) >= 5 and int(s[0]) in SLOW_SRC:
                    rows.append('17 ' + ' '.join(s[1:5])); stats[f'{split}_slow_{SLOW_SRC[int(s[0])]}'] += 1
        with open(os.path.join(ol, uid + '.txt'), 'w') as f:
            f.write('\n'.join(rows) + ('\n' if rows else ''))
        if uid in imgs:
            dst = os.path.join(oi, imgs[uid])
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(os.path.join(li, imgs[uid])), dst)
    old_prefix, new_prefix = os.path.join(a.v2di, 'images') + '/', os.path.join(out, 'images') + '/'
    lst = [p.strip().replace(old_prefix, new_prefix) for p in open(os.path.join(a.v2di, f'{split}.txt')) if p.strip()]
    missing = sum(1 for p in lst if not os.path.lexists(p))
    with open(os.path.join(out, f'{split}.txt'), 'w') as f:
        f.write('\n'.join(lst) + '\n')
    stats[f'{split}_images_listed'] = len(lst); stats[f'{split}_list_paths_missing'] = missing

# ---- 2. init18.pt (head 17 -> 18)
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
old = YOLO(a.v2pt).model.float()
cfg = copy.deepcopy(old.yaml); cfg['nc'] = len(V21)
new = DetectionModel(cfg, nc=len(V21), verbose=False)
osd, nsd = old.state_dict(), new.state_dict()
grown = []
for k, v in nsd.items():
    if k not in osd:
        continue
    o = osd[k]
    if o.shape == v.shape:
        nsd[k] = o.clone()
    elif o.dim() >= 1 and o.shape[0] == 17 and v.shape[0] == 18 and o.shape[1:] == v.shape[1:]:
        t = v.clone(); t[:17] = o; nsd[k] = t; grown.append(k)
    else:
        stats['init_shape_mismatch_' + k] += 1
new.load_state_dict(nsd)
new.names = {i: n for i, n in enumerate(V21)}
ck = torch.load(a.v2pt, map_location='cpu', weights_only=False)
torch.save({'model': new.half(), 'train_args': ck.get('train_args', {}), 'epoch': -1,
            'version': ck.get('version'), 'date': ck.get('date')}, os.path.join(a.v21, 'init18.pt'))
stats['init_head_tensors_grown'] = len(grown)
print('grown:', grown)

# ---- 3. yamls
names = ''.join(f'  {i}: {n}\n' for i, n in enumerate(V21))
def yml(fn, train, val):
    with open(os.path.join(a.v21, fn), 'w') as f:
        f.write(f'path: {a.v21}\ntrain: [{", ".join(train)}]\nval: [{", ".join(val)}]\nnames:\n{names}')
yml('teacher21.yaml', ['driveindia/train.txt', 'rdd/train.txt', 'bump/train.txt'],
    ['driveindia/val.txt', 'rdd/val.txt', 'bump/val.txt'])
yml('v21.yaml', ['idd/train.txt', 'driveindia/train.txt', 'rdd_p/train.txt', 'bump/train.txt'],
    ['idd/val.txt', 'driveindia/val.txt', 'rdd_p/val.txt', 'bump/val.txt'])
for d in ('idd', 'driveindia', 'rdd_p', 'bump'):
    yml(f'val_{d}.yaml', [f'{d}/train.txt'], [f'{d}/val.txt'])

txt = '\n'.join(['== v2.1 build =='] + [f'{k:36s} {v}' for k, v in sorted(stats.items())])
print(txt); open(os.path.join(a.v21, 'build_report.txt'), 'w').write(txt + '\n')
