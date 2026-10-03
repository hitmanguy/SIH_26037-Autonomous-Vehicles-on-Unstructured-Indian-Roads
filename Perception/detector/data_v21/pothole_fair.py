"""Fair pothole test: v2 vs v2.1 on RDD photos that NEITHER model trained on.
Run on the server:  python pothole_fair.py
"""
import os, collections
from ultralytics import YOLO
H = os.path.expanduser('~/epsilon_yash')
V21 = ['person', 'rider', 'car', 'bus', 'truck', 'autorickshaw', 'motorcycle', 'bicycle', 'animal',
       'traffic sign', 'traffic light', 'vehicle fallback', 'pothole', 'pushcart', 'tractor',
       'emergency vehicle', 'cone_barrier', 'slow_zone']
base = lambda p: os.path.splitext(os.path.basename(p.strip()))[0]
v2_train = {base(p) for p in open(f'{H}/data/v2/rdd/train.txt') if p.strip()}
v21_train = {base(p) for p in open(f'{H}/data/v21/rdd/train.txt') if p.strip()}
cands = [p.strip() for p in open(f'{H}/data/v21/rdd/val.txt') if p.strip()]
out = f'{H}/data/fair_pothole'
for d in ('images/val', 'labels/val'):
    os.makedirs(f'{out}/{d}', exist_ok=True)
lst, per = [], collections.Counter()
for p in cands:
    u = base(p)
    if u in v2_train or u in v21_train:
        continue
    rows = [l for l in open(p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt') if l.split()[:1] == ['12']]
    if not rows:
        continue
    dst = f'{out}/images/val/{os.path.basename(p)}'
    if not os.path.lexists(dst):
        os.symlink(os.path.realpath(p), dst)
    open(f'{out}/labels/val/{u}.txt', 'w').writelines(rows)
    lst.append(dst); per[u.split('_')[1]] += len(rows)
open(f'{out}/val.txt', 'w').write('\n'.join(lst) + '\n')
print(f'fair test set: {len(lst)} images, {sum(per.values())} potholes  {dict(per)}')
for tag, w, names in (('v2  ', f'{H}/runs/v2_final/weights/best.pt', V21[:17]),
                      ('v2.1', f'{H}/runs/v21_final/weights/best.pt', V21)):
    y = f'{out}/fair_{len(names)}.yaml'
    open(y, 'w').write(f'path: {out}\ntrain: val.txt\nval: val.txt\nnames:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(names)))
    m = YOLO(w).val(data=y, batch=32, verbose=False, plots=False, project=f'{H}/runs/fair', name=tag.strip())
    i = list(m.box.ap_class_index).index(12)
    p, r, ap50, ap = m.box.class_result(i)
    print(f'{tag} pothole  P {p:.3f}  R {r:.3f}  mAP50 {ap50:.3f}  mAP50-95 {ap:.3f}')
