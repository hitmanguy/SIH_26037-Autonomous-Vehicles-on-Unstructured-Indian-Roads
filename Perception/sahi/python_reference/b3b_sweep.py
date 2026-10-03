"""Block 3b (offline analysis, no network): stricter score cut for boxes ONLY SAHI found.
Reads B3_raw_detections.mat (MATLAB v7.3) and re-scores with the same rules as test_B3_accuracy.m."""
import h5py, numpy as np, sys
f = h5py.File(sys.argv[1], 'r')
def arr(ds):
    if 'MATLAB_empty' in ds.attrs and ds.attrs['MATLAB_empty']: return np.zeros((0,))
    return np.array(ds).T
def get(ref):
    g = f[ref]; B = arr(g['B']).reshape(-1, 4) if arr(g['B']).size else np.zeros((0, 4))
    return B, arr(g['C']).ravel(), arr(g['S']).ravel() if 'S' in g else arr(g['H']).ravel()
nI, nM = f['D'].shape
D = [[None]*nI for _ in range(nM)]
for k in range(nI):
    for m in range(nM):
        D[m][k] = get(f['D'][k, m])
G = []
for k in range(nI):
    g = f[f['G'][k, 0]]
    B = arr(g['B']); B = B.reshape(-1, 4) if B.size else np.zeros((0, 4))
    G.append((B, arr(g['C']).ravel(), float(arr(g['H']).ravel()[0])))

def iou(a, b):
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:,0]+a[:,2], a[:,1]+a[:,3], b[:,0]+b[:,2], b[:,1]+b[:,3]
    iw = np.clip(np.minimum(ax2[:,None], bx2[None]) - np.maximum(a[:,0][:,None], b[:,0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:,None], by2[None]) - np.maximum(a[:,1][:,None], b[:,1][None]), 0, None)
    inter = iw*ih; return inter / np.maximum((a[:,2]*a[:,3])[:,None] + (b[:,2]*b[:,3])[None] - inter, 1e-9)
def match(B, C, S, gB, gC):
    tp = np.zeros(len(S), bool); gh = np.zeros(len(gC), bool)
    if len(S) == 0 or len(gC) == 0: return tp, gh
    M = iou(B, gB); M[C[:,None] != gC[None]] = 0
    for i in np.argsort(-S, kind='stable'):
        v = M[i].copy(); v[gh] = 0; j = np.argmax(v)
        if v[j] >= 0.5: tp[i] = True; gh[j] = True
    return tp, gh
def ap(s, t, n):
    if n == 0: return np.nan
    if len(s) == 0: return 0.0
    o = np.argsort(-s, kind='stable'); t = t[o].astype(float)
    ctp = np.cumsum(t); cfp = np.cumsum(1-t); rec = ctp/n; prec = ctp/np.maximum(ctp+cfp, 1e-12)
    mrec = np.r_[0, rec, 1]; mpre = np.r_[1, prec, 0]
    for i in range(len(mpre)-2, -1, -1): mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx]-mrec[idx-1])*mpre[idx]))
edges = [0, 16, 32, 64, 128, np.inf]; RU = {1,2,3,4,5,6,7,8,9,12}
def evaluate(dets):
    hitsRU = np.zeros(5); nRU = np.zeros(5); fp = fps = nd = ntp = 0
    S_ = {c: [] for c in range(1,13)}; T_ = {c: [] for c in range(1,13)}; nG = {c: 0 for c in range(1,13)}
    for k in range(nI):
        B, C, S = dets[k]; gB, gC, H = G[k]
        tp, gh = match(B, C, S, gB, gC)
        hg = gB[:,3]*1080/H; bk = np.digitize(hg, edges) - 1; ru = np.isin(gC, list(RU))
        for b in range(5): hitsRU[b] += np.sum(gh & ru & (bk == b)); nRU[b] += np.sum(ru & (bk == b))
        hd = B[:,3]*1080/H if len(B) else np.zeros(0)
        fp += np.sum(~tp); fps += np.sum(~tp & (hd < 32)); nd += len(tp); ntp += tp.sum()
        for c in range(1, 13):
            S_[c] += list(S[C == c]); T_[c] += list(tp[C == c]); nG[c] += int(np.sum(gC == c))
    m = np.nanmean([ap(np.array(S_[c]), np.array(T_[c], bool), nG[c]) for c in range(1, 13)])
    return 100*hitsRU/np.maximum(nRU, 1), fp/nI, fps/nI, 100*ntp/max(nd, 1), m, nRU
hdr = f"{'set-up':34s} {'<16':>6s} {'16-32':>6s} {'32-64':>6s} {'64-128':>7s} | {'FP/fr':>6s} {'smallFP':>7s} {'prec':>6s} {'mAP50':>6s}"
def row(name, r):
    rec, fp, fps, pr, m, _ = r
    print(f"{name:34s} {rec[0]:5.1f}% {rec[1]:5.1f}% {rec[2]:5.1f}% {rec[3]:6.1f}% | {fp:6.2f} {fps:7.2f} {pr:5.1f}% {m:6.3f}")
print("Check: must equal the MATLAB Block 3 log"); print(hdr)
r1 = evaluate(D[0]); row('full-frame only', r1); row('per-camera rows, 640xN (all)', evaluate(D[4]))
print(f"(GT road users per bucket: {r1[5].astype(int).tolist()})")
# tile-only = SAHI box with no same-class full-frame box at IoU >= 0.5
print("\nStricter cut ONLY for boxes that SAHI alone found (per-camera rows, 640xN):"); print(hdr)
for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
    dets = []
    for k in range(nI):
        B, C, S = D[4][k]; fB, fC, fS = D[0][k]
        M = iou(B, fB)
        if M.size: M[C[:,None] != fC[None]] = 0
        only = (M.max(1) < 0.5) if M.size else np.ones(len(S), bool)
        keep = ~only | (S >= t)
        dets.append((B[keep], C[keep], S[keep]))
    row(f'tile-only cut {t:.2f}', evaluate(dets))
